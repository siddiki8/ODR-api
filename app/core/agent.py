from typing import Any, Dict, Optional, Callable, List, Tuple, Literal, Coroutine
import logging # Import logging
import json
import re
import asyncio
from collections import Counter
from pydantic import ValidationError

# --- Internal Imports (New Structure) ---
from ..services.search import execute_batch_serper_search, SerperConfig, SearchResult
from ..services.ranking import rerank_with_together_api
from ..services.scraping import WebScraper, ExtractionResult
from ..services.chunking import Chunker # Assuming placeholder exists
from .schemas import PlannerOutput, SourceSummary, SearchRequest # Removed WritingPlan
from .prompts import (
    get_planner_prompt,
    get_summarizer_prompt,
    get_writer_initial_prompt,
    get_refiner_prompt,
    format_summaries_for_prompt as format_summaries_for_prompt_template,
    _WRITER_SYSTEM_PROMPT_BASE
)
# Import config classes and helper, but not instances
from .config import AppSettings, ApiKeys, LLMConfig, get_litellm_params 
# Import the new service function
from ..services.llm import call_litellm_acompletion

# Import Custom Exceptions - Remove unused ones
from .exceptions import (
    ConfigurationError, ValidationError, LLMError,
    ExternalServiceError, ScrapingError,
    AgentExecutionError, LLMOutputValidationError
)

# --- Constants --- (Define near the top of the class or globally)
# Estimate 800k tokens * 4 chars/token
WRITER_INPUT_CHAR_LIMIT = 3_200_000 

class DeepResearchAgent:
    """Agent responsible for conducting deep research based on a user query, adapted for API structure."""

    def __init__(
        self,
        # Require settings and api_keys to be passed in
        settings: AppSettings, 
        api_keys: ApiKeys, 
        # Allow overriding LLM configs during instantiation if needed (e.g., from request)
        planner_llm_override: Optional[LLMConfig] = None,
        summarizer_llm_override: Optional[LLMConfig] = None,
        writer_llm_override: Optional[LLMConfig] = None,
        scraper_strategies_override: Optional[List[str]] = None,
        max_search_tasks_override: Optional[int] = None, # Allow overriding max search tasks
        logger_callback: Optional[Callable[[str, int], None]] = None, # Adjusted logger callback signature expectation if needed
        # Allow overriding provider at agent instantiation
        llm_provider_override: Optional[Literal['google', 'openrouter']] = None, 
        websocket_callback: Optional[Callable[..., Coroutine]] = None # <-- Add WebSocket callback
    ):
        """
        Initializes the DeepResearchAgent using configuration objects.

        Args:
            settings: Instantiated AppSettings object.
            api_keys: Instantiated ApiKeys object.
            planner_llm_override: Optional LLMConfig to override default planner LLM settings (model name is determined by provider).
            summarizer_llm_override: Optional LLMConfig to override default summarizer LLM settings.
            writer_llm_override: Optional LLMConfig to override default writer LLM settings.
            scraper_strategies_override: Optional list of strategies to override default scraper strategies.
            max_search_tasks_override: Optional integer to override the maximum number of search tasks for the planner.
            logger_callback: Optional function to call for logging instead of print.
            llm_provider_override: Optionally override the LLM provider ('google' or 'openrouter').
            websocket_callback: Optional async function to send status updates over WebSocket.
        """
        self.settings = settings
        self.api_keys = api_keys
        # Determine the effective LLM provider
        self.llm_provider = llm_provider_override or settings.llm_provider
        
        # Use a proper logger instance for the agent
        self.logger = logging.getLogger(f"DeepResearchAgent_{id(self)}")
        # Configure logger level and handlers (assuming basic setup here, might be enhanced)
        if not self.logger.hasHandlers():
             handler = logging.StreamHandler()
             # Use a more detailed formatter for debug, simpler otherwise
             log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Consistent format
             formatter = logging.Formatter(log_format)
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
        
        # Force INFO level to reduce console noise from underlying libraries like crawl4ai
        log_level = logging.INFO # Was: logging.DEBUG if settings.scraper_debug else logging.INFO
        self.logger.setLevel(log_level)

        # Keep the callback if provided, but use the logger for internal agent logging
        self._external_log_callback = logger_callback 
        self.websocket_callback = websocket_callback # <-- Store the callback
        
        # Log initial setup using the new logger
        # Use a helper method for unified logging
        self._log(f"Initializing Agent with LLM Provider: {self.llm_provider}", level=logging.INFO)
        if self.websocket_callback:
             self._log("WebSocket callback provided.", level=logging.DEBUG)

        # --- Determine effective LLM configurations --- 
        try:
            effective_planner_config = planner_llm_override or settings.default_planner_llm
            effective_summarizer_config = summarizer_llm_override or settings.default_summarizer_llm
            effective_writer_config = writer_llm_override or settings.default_writer_llm

            # Set the correct model name based on the provider for each role
            # Use the provider determined earlier
            effective_planner_config.model = settings.get_model_name('planner') # Remove extra arg
            effective_summarizer_config.model = settings.get_model_name('summarizer') # Remove extra arg
            effective_writer_config.model = settings.get_model_name('writer') # Remove extra arg

            # Finalize LiteLLM parameters using the helper
            # This can raise ConfigurationError if keys are missing
            self.planner_llm_config = get_litellm_params(
                effective_planner_config, self.llm_provider, self.api_keys
            )
            self.summarizer_llm_config = get_litellm_params(
                effective_summarizer_config, self.llm_provider, self.api_keys
            )
            self.writer_llm_config = get_litellm_params(
                effective_writer_config, self.llm_provider, self.api_keys
            )
            
            # --- Add Logging Here --- 
            self.logger.debug("[Agent __init__] Planner LLM Config AFTER get_litellm_params:")
            self.logger.debug(f"  Model: {self.planner_llm_config.get('model')}")
            self.logger.debug(f"  API Key Type: {type(self.planner_llm_config.get('api_key'))}") 
            self.logger.debug(f"  API Key Provided: {bool(self.planner_llm_config.get('api_key'))}")
            self.logger.debug(f"  API Base: {self.planner_llm_config.get('api_base')}")
            # --- End Logging ---
            
        except ConfigurationError as e:
            self._log(f"Agent Initialization Failed: LLM configuration error - {e}", level=logging.CRITICAL)
            raise # Re-raise critical configuration error
        except Exception as e:
            self._log(f"Agent Initialization Failed: Unexpected error during LLM config setup - {e}", level=logging.CRITICAL, exc_info=True)
            raise ConfigurationError(f"Unexpected error initializing LLM configs: {e}") from e

        # --- Initialize other components --- 
        try:
            # Chunker init can raise ValueError
            self.chunker = Chunker() 
            # WebScraper init - Pass settings
            self.scraper = WebScraper(
                settings=settings, # Pass the AppSettings instance
                debug=settings.scraper_debug
            )
            # SerperConfig init can raise ConfigurationError
            self.serper_config = SerperConfig(
                api_key=api_keys.serper_api_key.get_secret_value(),
                base_url=str(settings.serper_base_url), 
                default_location=settings.serper_default_location,
                timeout=settings.serper_timeout
            )
            # Check Together API key existence
            self.together_api_key = api_keys.together_api_key.get_secret_value()
            if not self.together_api_key:
                 raise ConfigurationError("TOGETHER_API_KEY is required for reranking but not found.")
                 
        except (ConfigurationError, ValueError) as e:
            self._log(f"Agent Initialization Failed: Component configuration error - {e}", level=logging.CRITICAL)
            raise # Re-raise critical configuration error
        except Exception as e:
            self._log(f"Agent Initialization Failed: Unexpected error during component setup - {e}", level=logging.CRITICAL, exc_info=True)
            raise ConfigurationError(f"Unexpected error initializing agent components: {e}") from e

        self.reranker_model = settings.reranker_model
        # Workflow parameters from settings
        self.max_initial_search_tasks = max_search_tasks_override if max_search_tasks_override is not None else settings.max_initial_search_tasks
        self.max_refinement_iterations = settings.max_refinement_iterations

        # Initialize usage trackers
        self.token_usage = {
            'planner': Counter(),
            'summarizer': Counter(),
            'writer': Counter(),
            'refiner': Counter(), # Add counter for refiner calls
            'total': Counter()
        }
        self.estimated_cost = {
            'planner': 0.0,
            'summarizer': 0.0,
            'writer': 0.0,
            'refiner': 0.0, # Add cost for refiner calls
            'total': 0.0
        }
        self.serper_queries_used = 0

        self._log("DeepResearchAgent initialized successfully.", level=logging.INFO)
        self._log(f"- LLM Provider: {self.llm_provider}", level=logging.DEBUG)
        self._log(f"- Planner LLM Config: {self.planner_llm_config}", level=logging.DEBUG) 
        self._log(f"- Summarizer LLM Config: {self.summarizer_llm_config}", level=logging.DEBUG)
        self._log(f"- Writer LLM Config: {self.writer_llm_config}", level=logging.DEBUG)
        self._log(f"- Reranker Model: {self.reranker_model}", level=logging.DEBUG)
        self._log(f"- Chunker Config: {self.chunker.__dict__}", level=logging.DEBUG)
        self._log(f"- Scraper Strategies: {self.scraper.extraction_configs}", level=logging.DEBUG) # Show actual configs
        self._log(f"- Workflow Params: max_search_tasks={self.max_initial_search_tasks}, max_iterations={self.max_refinement_iterations}", level=logging.DEBUG)

    # --- Helper for unified logging ---
    def _log(self, message: str, level: int = logging.INFO, exc_info: bool = False):
        """Logs messages using the agent's logger and optionally calls the external callback."""
        self.logger.log(level, message, exc_info=exc_info)
        if self._external_log_callback:
            # Assuming callback expects message and level
            try:
                self._external_log_callback(message, level)
            except Exception as e:
                # Prevent callback errors from crashing the agent
                self.logger.error(f"External logger callback failed: {e}", exc_info=True)

    # --- Helper for WebSocket Updates ---
    async def _send_ws_update(self, step: str, status: str, message: str, details: dict | None = None):
        """Sends status updates over the WebSocket connection if the callback is set."""
        if self.websocket_callback:
            try:
                self._log(f"WS Update: {step}/{status} - {message}", level=logging.DEBUG)
                await self.websocket_callback(step, status, message, details)
            except Exception as e:
                # Log error but don't crash agent if WS send fails
                self._log(f"Failed to send WebSocket update ({step}/{status}): {e}", level=logging.ERROR, exc_info=True)
        else:
             self._log(f"WS Update Skipped (no callback): {step}/{status} - {message}", level=logging.DEBUG)

    def _log_and_update_usage(self, role: Literal['planner', 'summarizer', 'writer', 'refiner'], usage_info: Optional[Dict[str, int]], cost_info: Optional[Dict[str, float]]):
        """Logs usage and cost from the service call and updates agent totals for the specified role."""
        if role not in self.token_usage:
            self._log(f"Invalid role '{role}' passed to _log_and_update_usage. Skipping update.", level=logging.ERROR)
            return
            
        role_token_usage = self.token_usage[role]
        total_token_usage = self.token_usage['total']
        
        if usage_info:
            # Update role-specific counters
            role_token_usage['completion_tokens'] += usage_info['completion_tokens']
            role_token_usage['prompt_tokens'] += usage_info['prompt_tokens']
            role_token_usage['total_tokens'] += usage_info['total_tokens']
            
            # Update total counters
            total_token_usage['completion_tokens'] += usage_info['completion_tokens']
            total_token_usage['prompt_tokens'] += usage_info['prompt_tokens']
            total_token_usage['total_tokens'] += usage_info['total_tokens']
            
            self._log(f"    [{role.upper()}] Tokens Used: Prompt={usage_info['prompt_tokens']}, Completion={usage_info['completion_tokens']}, Total={usage_info['total_tokens']}", level=logging.INFO)
            self._log(f"    [{role.upper()}] Cumulative Role Tokens: {role_token_usage['total_tokens']}", level=logging.DEBUG)
            self._log(f"    Cumulative Total Tokens: {total_token_usage['total_tokens']}", level=logging.DEBUG)
        else:
             self._log(f"    [{role.upper()}] Token usage information not available for this call.", level=logging.WARNING)

        if cost_info and 'total_cost' in cost_info:
            current_cost = cost_info['total_cost']
            
            # Update role-specific cost
            self.estimated_cost[role] += current_cost
            # Update total cost
            self.estimated_cost['total'] += current_cost
            
            self._log(f"    [{role.upper()}] LLM call cost: ${current_cost:.6f}. Cumulative Role Cost: ${self.estimated_cost[role]:.6f}", level=logging.INFO)
            self._log(f"    Cumulative Total Cost: ${self.estimated_cost['total']:.6f}", level=logging.DEBUG)
        else:
             self._log(f"    [{role.upper()}] Cost information not available for this call.", level=logging.WARNING)
             
    async def _extract_search_request(self, text: str) -> Optional[SearchRequest]:
        """Extracts and validates search requests from the writer/refiner output."""
        try:
            match = re.search(r'<search_request query=["\']([^"\']*)["\'](?:\s*/?)?>', text, re.IGNORECASE)
            if not match:
                return None
            
            query = match.group(1).strip()
        except re.error as e:
            self._log(f"Regex error extracting search request: {e}", level=logging.ERROR)
            return None # Treat regex error as no match found
        
        try:
            # Validate using the SearchRequest schema (only requires query)
            return SearchRequest(query=query)
        except ValidationError as e:
            # Log as warning, not critical error
            self._log(f"Invalid search request format found in LLM output: '{query}'. Error: {e}", level=logging.WARNING)
            return None # Return None if validation fails
            
    async def _call_refiner_llm(self, previous_draft: str, search_query: str, new_info: List[Dict[str, Any]]) -> str:
        """Calls the Refiner LLM (using Summarizer config) to revise the draft."""
        self._log(f"Calling Refiner LLM ({self.summarizer_llm_config.get('model')}) to incorporate info for query: '{search_query}'", level=logging.INFO)
        await self._send_ws_update("REFINING", "IN_PROGRESS", "Calling LLM to refine draft...")

        try:
            refiner_messages = get_refiner_prompt(
                previous_draft=previous_draft,
                search_query=search_query,
                new_info=new_info # Pass the list of SourceSummary objects
            )
            # Remove search tag from draft before sending to refiner
            # cleaned_previous_draft = re.sub(r'<search_request.*?>', '', previous_draft, flags=re.IGNORECASE).strip()
            # refiner_messages[1]['content'] = refiner_messages[1]['content'].format(previous_draft=cleaned_previous_draft, ...) # More robust update needed if template changes

            input_chars = sum(len(m.get('content', '')) for m in refiner_messages if isinstance(m.get('content'), str))
            self._log(f"    - Refiner input contains {len(new_info)} new info items.")
            self._log(f"    - Refiner input size: ~{input_chars} characters.")

            response, usage, cost = await call_litellm_acompletion(
                messages=refiner_messages,
                llm_config=self.summarizer_llm_config # Use summarizer config
            )
            refined_draft = response.choices[0].message.content or ""
            self._log_and_update_usage('refiner', usage, cost)

            if not refined_draft.strip():
                self._log("Refiner LLM returned an empty response.", level=logging.WARNING)
                # Do not raise, just return empty string as per oldagent logic
                return ""

            # Clean search tags just in case refiner adds one
            refined_draft_cleaned = re.sub(r'<search_request.*?>', '', refined_draft, flags=re.IGNORECASE).strip()
            self._log(f"Refiner revised draft ({len(refined_draft_cleaned)} chars).", level=logging.INFO)
            await self._send_ws_update("REFINING", "INFO", "LLM refinement complete.")
            return refined_draft_cleaned

        except LLMError as e:
            self._log(f"Refiner LLM call failed: {e}", level=logging.ERROR)
            await self._send_ws_update("REFINING", "ERROR", f"LLM refinement call failed: {e}")
            return "" # Return empty string on failure as per oldagent logic
        except Exception as e:
            self._log(f"Unexpected error during refiner LLM call: {e}", level=logging.ERROR, exc_info=True)
            await self._send_ws_update("REFINING", "ERROR", "Unexpected error during LLM refinement.")
            return "" # Return empty string on failure

    def _estimate_writer_input_chars(self, user_query, writing_plan, source_summaries):
        """Estimates the character count of the input to the writer LLM."""
        formatted_summaries = format_summaries_for_prompt_template(source_summaries)
        writer_prompt = get_writer_initial_prompt(
            user_query=user_query,
            writing_plan=writing_plan, 
            source_summaries=formatted_summaries
        )
        # Consider system prompt length too? Approx for now.
        return len(writer_prompt) + len(_WRITER_SYSTEM_PROMPT_BASE)

    async def run_deep_research(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the full deep research process asynchronously.

        Args:
            user_query: The user's research query.

        Returns:
            A dictionary containing the final report and usage statistics.
        
        Raises:
            Various exceptions (DeepResearchError subclasses) if steps fail critically.
        """
        self._log(f"Starting deep research for query: '{user_query[:100]}...'", level=logging.INFO)
        await self._send_ws_update("STARTING", "START", "Research process initiated.")

        # --- 1. Planning Phase ---
        planner_output: Optional[PlannerOutput] = None
        try:
            self._log("--- Phase 1: Planning ---", level=logging.INFO)
            await self._send_ws_update("PLANNING", "START", "Generating initial research plan...")
            # Get the correctly formatted list of messages directly
            messages = get_planner_prompt(user_query, self.max_initial_search_tasks)
            
            # --- Add Logging Here --- 
            self.logger.debug("[Agent run_deep_research] Planner LLM Config BEFORE call_litellm_acompletion:")
            self.logger.debug(f"  Model: {self.planner_llm_config.get('model')}")
            self.logger.debug(f"  API Key Type: {type(self.planner_llm_config.get('api_key'))}") 
            self.logger.debug(f"  API Key Provided: {bool(self.planner_llm_config.get('api_key'))}")
            self.logger.debug(f"  API Base: {self.planner_llm_config.get('api_base')}")
            # --- End Logging ---

            # Use the new service function for LiteLLM calls
            response, usage_info, cost_info = await call_litellm_acompletion(
                messages=messages,
                llm_config=self.planner_llm_config,
                response_pydantic_model=PlannerOutput, # Pass Pydantic model for parsing
                num_retries=3,
                logger_callback=self.logger # Pass agent logger to service
            )
            # Log usage/cost after the call
            self._log_and_update_usage('planner', usage_info, cost_info)
            
            # --- DEBUG: Print raw response content --- 
            raw_content_debug = "(No content found in response)"
            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                raw_content_debug = response.choices[0].message.content
            print(f"\n--- RAW PLANNER LLM RESPONSE CONTENT ---\n{raw_content_debug}\n--------------------------------------\n")
            # --- END DEBUG ---

            # Check if response is valid before proceeding
            if response is None or not response.choices or not response.choices[0].message:
                raise ValueError("Planner LLM call failed or returned an invalid response.")

            # --- Correctly Parse Planner Output --- 
            planner_output_obj: Optional[PlannerOutput] = None
            # Check if LiteLLM already parsed the output
            if hasattr(response.choices[0].message, '_response_format_output') and response.choices[0].message._response_format_output:
                if isinstance(response.choices[0].message._response_format_output, PlannerOutput):
                    planner_output_obj = response.choices[0].message._response_format_output
                    self._log("✓ Planner output parsed and validated by LiteLLM's response_model feature.", level=logging.INFO)
                else:
                    self._log(f"Warning: LiteLLM's _response_format_output is not a PlannerOutput instance (type: {type(response.choices[0].message._response_format_output)}). Attempting manual parse.", level=logging.WARNING)
            
            # If not parsed by LiteLLM, attempt manual parse from content
            if planner_output_obj is None:
                raw_content = response.choices[0].message.content
                if not raw_content:
                     raise LLMOutputValidationError("Planner LLM response content is empty.")
                     
                self._log("Attempting manual JSON parse from planner response content...", level=logging.INFO)
                try:
                    # --- Markdown stripping logic --- 
                    cleaned_text = raw_content.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:] # Remove ```json
                    if cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:] # Remove ```
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3] # Remove ```
                    cleaned_text = cleaned_text.strip()
                    # --- End stripping logic --- 
                    
                    self._log(f"Cleaned text for validation: {cleaned_text[:200]}...", level=logging.DEBUG)
                    planner_output_obj = PlannerOutput.model_validate_json(cleaned_text)
                    self._log("✓ Planner output validated via manual parse.", level=logging.INFO)
                except (json.JSONDecodeError, ValidationError) as fallback_e:
                    self._log(f"Error: Fallback validation failed: {fallback_e}", level=logging.ERROR, exc_info=True)
                    self._log(f"Content attempted in fallback validation: {cleaned_text}", level=logging.ERROR)
                    raise LLMOutputValidationError(f"Planner LLM failed to return valid structured output after cleaning. Error: {fallback_e}") from fallback_e
            # --- End Parsing Logic ---

            # Ensure we have a valid object before proceeding
            if not planner_output_obj:
                 raise AgentExecutionError("Failed to obtain valid planner output object after parsing attempts.")

            # planner_output = response.choices[0].message.tool_calls[0].function.parsed_arguments # Access parsed arguments
            # self._log_and_update_usage('planner', usage_info, cost_info) # Usage already logged

            # Extract data from the validated Pydantic object
            search_tasks = planner_output_obj.search_tasks
            writing_plan = planner_output_obj.writing_plan

            if not search_tasks:
                 raise LLMOutputValidationError("Planner did not generate any search tasks.")

            self._log(f"Planner generated {len(search_tasks)} search tasks.", level=logging.INFO)
            plan_details = {
                "writing_plan": planner_output_obj.writing_plan.model_dump(), # Convert to dict
                "search_task_count": len(search_tasks),
                "search_queries": [task.query for task in search_tasks]
            }
            await self._send_ws_update("PLANNING", "END", "Research plan generated.", {"plan": plan_details})

        except (LLMError, ValidationError, json.JSONDecodeError) as e:
            error_msg = f"Planning phase failed: {type(e).__name__}"
            self._log(error_msg, level=logging.ERROR, exc_info=False)
            await self._send_ws_update("PLANNING", "ERROR", f"Failed to generate plan: {type(e).__name__}")
            raise AgentExecutionError(f"Failed to generate a valid research plan: {type(e).__name__}") from e
        except Exception as e:
            error_msg = f"Unexpected error during planning phase: {type(e).__name__}"
            self._log(error_msg, level=logging.CRITICAL, exc_info=False)
            await self._send_ws_update("PLANNING", "ERROR", "Unexpected error during planning.")
            raise AgentExecutionError(f"Unexpected error during planning: {type(e).__name__}") from e

        # Ensure planner_output is not None before proceeding
        if not planner_output_obj:
            # This should not happen if exceptions are raised correctly above, but as a safeguard:
            err_msg = "Planning phase completed without generating a plan output."
            self._log(err_msg, level=logging.CRITICAL)
            await self._send_ws_update("PLANNING", "ERROR", err_msg, {"error": "Internal agent error"})
            raise AgentExecutionError(err_msg)

        # --- 2. Initial Search Phase ---
        all_search_results: List[SearchResult] = []
        try:
            self._log("--- Phase 2: Initial Search ---", level=logging.INFO)
            search_tasks = planner_output_obj.search_tasks
            task_count = len(search_tasks)
            await self._send_ws_update("SEARCHING", "START", f"Performing initial search based on {task_count} tasks...")

            # Execute searches in batch - CORRECTED ARGUMENTS
            batch_results = await execute_batch_serper_search(
                search_tasks=search_tasks, # Pass the list of task dictionaries
                config=self.serper_config   # Pass the config object
            )
            self.serper_queries_used += len(search_tasks) # Count based on tasks sent

            # Process results (assuming execute_batch_serper_search returns List[Dict[str, Any]])
            # Flatten results might not be needed if Serper returns one dict per task
            # raw_search_results = [result for sublist in batch_results for result in sublist.get('organic', [])] # Old assumption
            raw_search_results = batch_results # Assuming it returns a list of dicts directly
            
            successful_queries = len(raw_search_results) # Count based on results received
            self._log(f"Initial search completed. Received results for {successful_queries}/{task_count} tasks.", level=logging.INFO)
            
            # Parse results into SearchResult objects if needed, or process raw dicts
            parsed_results: List[SearchResult] = []
            for task_result in raw_search_results:
                # Example parsing, adjust based on actual structure and SearchResult.from_dict
                for item in task_result.get('organic', []):
                     try:
                         # Assuming SearchResult.from_dict exists and works
                         parsed_results.append(SearchResult.from_dict(item))
                     except Exception as parse_e:
                          self._log(f"Failed to parse search result item: {item}. Error: {parse_e}", level=logging.WARNING)
            
            self._log(f"Parsed {len(parsed_results)} organic results from initial search.", level=logging.DEBUG)

            await self._send_ws_update("SEARCHING", "END", f"Initial search yielded results for {successful_queries} queries.", {"raw_result_count": len(parsed_results), "queries_executed": successful_queries})

            if not parsed_results:
                 self._log("Initial search returned no valid organic results. Proceeding without web context.", level=logging.WARNING)
                 # Allow proceeding, the writer might handle this or fail later

        except ExternalServiceError as e:
            error_msg = f"Search phase failed: {type(e).__name__}"
            self._log(error_msg, level=logging.ERROR, exc_info=False)
            await self._send_ws_update("SEARCHING", "ERROR", f"Search API error: {type(e).__name__}")
            raise # Re-raise to be caught by main handler or specific endpoint logic
        except Exception as e:
            error_msg = f"Unexpected error during initial search phase: {type(e).__name__}"
            self._log(error_msg, level=logging.CRITICAL, exc_info=False)
            await self._send_ws_update("SEARCHING", "ERROR", "Unexpected error during search.")
            raise AgentExecutionError(f"Unexpected error during initial search: {type(e).__name__}") from e

        # --- 3. Reranking Phase ---
        top_sources: List[SearchResult] = []
        if parsed_results: # Only rerank if we have parsed results
            try:
                self._log("--- Phase 3: Reranking Search Results ---", level=logging.INFO)
                # Deduplicate results based on URL before ranking
                unique_results_dict: Dict[str, SearchResult] = {}
                for result in parsed_results:
                    if result.link and result.link not in unique_results_dict:
                         unique_results_dict[result.link] = result
                unique_results = list(unique_results_dict.values())
                self._log(f"Found {len(unique_results)} unique URLs for reranking.", level=logging.DEBUG)

                await self._send_ws_update("RANKING", "START", f"Reranking {len(unique_results)} unique search results...")

                # Prepare data for reranker
                passages = [f"{r.title} {r.snippet}" for r in unique_results] # Rerank based on title + snippet
                if not passages:
                    self._log("No valid passages with URLs found for reranking.", level=logging.WARNING)
                else:
                    reranked_data = await rerank_with_together_api(
                        query=user_query,
                        documents=passages, # Pass documents (title+snippet)
                        model=self.reranker_model,
                        api_key=self.together_api_key,
                        relevance_threshold=0.2, # Apply threshold
                    )
                    # reranked_data is List[Dict[str, Any]] with 'index' and 'score'
                    
                    # --- NEW LOGIC: Split sources based on Y/2 (max 10) ---
                    num_reranked = len(reranked_data)
                    num_to_summarize = min(num_reranked // 2, 10) # Y/2, max 10
                    
                    # Get indices and then the actual SearchResult objects
                    reranked_indices_scores = [(res['index'], res['score']) for res in reranked_data]
                    
                    sources_to_summarize_indices = [idx for idx, score in reranked_indices_scores[:num_to_summarize]]
                    sources_to_chunk_indices = [idx for idx, score in reranked_indices_scores[num_to_summarize:]] # The rest
                    
                    sources_to_summarize = [unique_results[i] for i in sources_to_summarize_indices]
                    sources_to_chunk = [unique_results[i] for i in sources_to_chunk_indices]
                    
                    self._log(f"Reranking filtered {len(passages) - num_reranked} below 0.2 threshold.", level=logging.DEBUG)
                    self._log(f"Splitting {num_reranked} sources: {len(sources_to_summarize)} for summarization, {len(sources_to_chunk)} for chunking/reranking.", level=logging.INFO)
                    # --- END NEW LOGIC ---

                    # Map reranked indices back to original unique_results - REMOVED OLD LOGIC
                    # num_top_sources = min(len(reranked_data), self.settings.top_m_full_text_sources) # Use settings value
                    # top_sources_indices = [res['index'] for res in reranked_data[:num_top_sources]]
                    # top_sources = [unique_results[i] for i in top_sources_indices]
                    
                    await self._send_ws_update("RANKING", "END", f"Identified {len(sources_to_summarize)} sources for summarization, {len(sources_to_chunk)} for chunking.")
                    top_sources = sources_to_summarize # Keep variable name for later code? Or rename? Let's rename later if needed.
                    # Need to store sources_to_chunk for the next phase

            except ExternalServiceError as e:
                error_msg = f"Reranking phase failed: {type(e).__name__}"
                self._log(error_msg, level=logging.ERROR, exc_info=False)
                await self._send_ws_update("RANKING", "ERROR", f"Reranking API error: {type(e).__name__}")
                # Non-critical? Could proceed with unranked results? For now, raise.
                raise
            except Exception as e:
                error_msg = f"Unexpected error during reranking phase: {type(e).__name__}"
                self._log(error_msg, level=logging.CRITICAL, exc_info=False)
                await self._send_ws_update("RANKING", "ERROR", "Unexpected error during reranking.")
                raise AgentExecutionError(f"Unexpected error during reranking: {type(e).__name__}") from e
        else:
             self._log("Skipping reranking phase as there were no initial search results.", level=logging.INFO)
             await self._send_ws_update("RANKING", "INFO", "Skipped - No initial search results.")

        # --- 4. Content Fetching and Processing Phase --- # Renamed Phase
        processed_summaries: List[SourceSummary] = []
        processed_chunks: List[Dict[str, Any]] = [] # To store relevant chunks
        scraped_content_cache: Dict[str, str] = {} # Cache scraped content (might be redundant now)
        all_processed_urls: set[str] = set() # Track all URLs processed in this phase

        # 4a. Process Sources for Summarization
        if sources_to_summarize: 
            self._log(f"--- Phase 4a: Fetching and Summarizing {len(sources_to_summarize)} Sources ---", level=logging.INFO)
            await self._send_ws_update("PROCESSING", "START", f"Starting summarization for {len(sources_to_summarize)} sources...")

            summarization_tasks = []
            # Map URL to source for context (only for summary group)
            summary_source_map = {source.link: source for source in sources_to_summarize if source.link} 

            # --- Run Summarization Sequentially --- 
            successful_summaries = 0
            failed_summaries = 0
            for url, source in summary_source_map.items():
                self._log(f"Processing source for summarization: {url}")
                all_processed_urls.add(url) # Mark as processed
                try:
                    result = await self._fetch_and_summarize(url, user_query, source)
                    if result: # Check if result is not None (success case returns tuple)
                        summary, scraped_content = result
                        processed_summaries.append(summary)
                        scraped_content_cache[url] = scraped_content # Keep cache for now
                        successful_summaries += 1
                    else: # Handle None case from helper (should indicate failure)
                        failed_summaries += 1
                        self._log(f"[Summarization] Helper function returned None for {url}, indicating processing failure.")
                        # Error WS update should be sent by helper
                except Exception as e:
                    failed_summaries += 1
                    error_msg = f"[Summarization] Exception processing source {url}: {type(e).__name__}"
                    self._log(error_msg, level=logging.ERROR, exc_info=False)
                    await self._send_ws_update("PROCESSING", "ERROR", f"Unexpected error summarizing {url}")
                
                # Add delay *within* the sequential loop
                await asyncio.sleep(0.5) # Half-second delay

            self._log(f"Summarization complete. Successfully processed {successful_summaries}/{len(sources_to_summarize)} sources.", level=logging.INFO)
            await self._send_ws_update("PROCESSING", "INFO", f"Finished summarizing {successful_summaries} sources.")

        else:
            self._log("Skipping summarization as no sources were designated.", level=logging.INFO)
            # No WS update needed here?

        # 4b. Process Sources for Chunking and Reranking
        if sources_to_chunk:
            self._log(f"--- Phase 4b: Fetching, Chunking, Reranking Chunks for {len(sources_to_chunk)} Sources ---", level=logging.INFO)
            await self._send_ws_update("PROCESSING", "START", f"Starting chunk processing for {len(sources_to_chunk)} sources...")

            # Define chunk relevance threshold
            CHUNK_RELEVANCE_THRESHOLD = 0.5
            
            # Process sources sequentially to avoid rate limits
            self._log(f"Processing {len(sources_to_chunk)} sources sequentially to avoid rate limits")
            chunk_source_map = {source.link: source for source in sources_to_chunk if source.link}
            
            total_relevant_chunks = 0
            failed_chunking_sources = 0
            
            for url, source in chunk_source_map.items():
                try:
                    self._log(f"Processing source for chunking: {url}")
                    all_processed_urls.add(url) # Mark as processed
                    
                    chunk_result = await self._fetch_chunk_and_rerank(
                        url=url, 
                        query=user_query, # Rerank against original user query
                        source=source, 
                        threshold=CHUNK_RELEVANCE_THRESHOLD,
                        is_refinement=False # Mark as initial chunking
                    )
                    
                    if isinstance(chunk_result, list): # Helper returns list of chunk dicts on success
                        processed_chunks.extend(chunk_result) # Add the list of relevant chunks
                        total_relevant_chunks += len(chunk_result)
                        self._log(f"Found {len(chunk_result)} relevant chunks for {url}")
                    # else: Helper returned empty list (logged internally) or None (shouldn't happen)
                except Exception as e:
                    failed_chunking_sources += 1
                    error_msg = f"[Chunking] Error processing source {url}: {type(e).__name__}"
                    self._log(error_msg, level=logging.ERROR, exc_info=False)
                    await self._send_ws_update("PROCESSING", "ERROR", f"Error processing chunks for {url}")
                
                # Add a short delay between requests to respect rate limits
                await asyncio.sleep(0.5)  # Half-second delay

            self._log(f"Chunking & Reranking complete. Found {total_relevant_chunks} relevant chunks (score >= {CHUNK_RELEVANCE_THRESHOLD}) across processed sources.", level=logging.INFO)
            await self._send_ws_update("PROCESSING", "END", f"Finished chunk processing. Found {total_relevant_chunks} relevant chunks.")

        else:
            self._log("Skipping chunking/reranking as no sources were designated.", level=logging.INFO)
        
        # --- Overall Processing Phase End ---
        await self._send_ws_update("PROCESSING", "END", f"Finished processing all initial sources ({len(all_processed_urls)} unique URLs).")


        # --- 5. Assemble Writer Context --- # Renamed Phase
        self._log("--- Phase 5: Assembling Initial Writer Context ---", level=logging.INFO)
        await self._send_ws_update("FILTERING", "START", "Assembling context for initial report...")

        # Combine summaries and chunks into a unified list for the writer
        # Add a 'type' field and sequential 'rank' for citation generation
        writer_context_items = []
        current_rank = 1
        if processed_summaries:
            for s in processed_summaries:
                 writer_context_items.append({
                     "type": "summary", 
                     "content": s.content, 
                     "link": s.link, # Use link from SourceSummary
                     "title": s.title,
                     "rank": current_rank 
                 })
                 current_rank += 1
            
        if processed_chunks:
             for c in processed_chunks:
                  writer_context_items.append({
                      "type": "chunk", 
                      "content": c["content"], 
                      "link": c["link"], # Use link from chunk dict
                      "title": c["title"], 
                      "score": c.get("relevance_score"),
                      "rank": current_rank # Add sequential rank
                  })
                  current_rank += 1

        # TODO: Implement context filtering/truncation if necessary, maybe prioritize summaries?
        # For now, pass all combined context. Need to update the prompt formatter.
        final_writer_context = writer_context_items 
        context_item_count = len(final_writer_context)
        # Estimate character count (approximate)
        context_char_count = sum(len(item.get('content', '')) for item in final_writer_context)

        self._log(f"Assembled {context_item_count} items (summaries & chunks, ~{context_char_count} chars) for the initial writer prompt.", level=logging.INFO)
        await self._send_ws_update("FILTERING", "END", f"Assembled {context_item_count} context items (~{context_char_count} chars).")


        # --- 6. Initial Report Generation ---
        initial_draft = ""
        try:
            self._log("--- Phase 6: Initial Report Generation ---", level=logging.INFO)
            await self._send_ws_update("WRITING", "START", "Generating initial report draft...")

            writer_prompt_messages = get_writer_initial_prompt(
                user_query=user_query,
                writing_plan=planner_output_obj.writing_plan.model_dump(), # Pass plan dict
                source_summaries=final_writer_context # Pass combined list
            )

            # Add a delay before the writer call
            await asyncio.sleep(1.0) # 1-second delay before writer

            response, usage, cost = await call_litellm_acompletion(
                messages=writer_prompt_messages, # Use the generated messages
                llm_config=self.writer_llm_config
            )
            initial_draft = response.choices[0].message.content or ""
            self._log_and_update_usage('writer', usage, cost)

            if not initial_draft.strip():
                raise LLMOutputValidationError("Writer LLM returned an empty initial draft.")

            self._log(f"Initial report draft generated (length: {len(initial_draft)} chars).", level=logging.INFO)
            await self._send_ws_update("WRITING", "END", "Initial report draft generated.")

        except (LLMError, ValidationError) as e:
            error_msg = f"Initial report generation failed: {type(e).__name__}"
            self._log(error_msg, level=logging.ERROR, exc_info=False)
            await self._send_ws_update("WRITING", "ERROR", f"Failed to generate initial draft: {type(e).__name__}")
            raise AgentExecutionError(f"Failed to generate initial report draft: {type(e).__name__}") from e
        except Exception as e:
            error_msg = f"Unexpected error during initial report generation: {type(e).__name__}"
            self._log(error_msg, level=logging.CRITICAL, exc_info=False)
            await self._send_ws_update("WRITING", "ERROR", "Unexpected error during initial writing.")
            raise AgentExecutionError(f"Unexpected error generating initial report: {type(e).__name__}") from e

        # --- 7. Refinement Loop ---
        current_draft = initial_draft
        # Update all_processed_urls based on phase 4
        # all_processed_urls = set(s.url for s in processed_summaries if s.url) # OLD way
        # all_processed_urls should already contain all URLs from phase 4a and 4b

        if self.max_refinement_iterations > 0:
            self._log(f"--- Phase 7: Refinement Loop (Max Iterations: {self.max_refinement_iterations}) ---", level=logging.INFO)
            await self._send_ws_update("REFINING", "START", f"Starting refinement process (max {self.max_refinement_iterations} iterations)...")
        else:
             self._log("Skipping refinement loop as max_refinement_iterations is 0.", level=logging.INFO)
             await self._send_ws_update("REFINING", "INFO", "Skipped - Max iterations set to 0.")

        refinement_iteration = 0
        for i in range(self.max_refinement_iterations):
            refinement_iteration = i + 1 # Track actual iterations run
            self._log(f"--- Refinement Iteration {refinement_iteration}/{self.max_refinement_iterations} ---", level=logging.INFO)
            await self._send_ws_update("REFINING", "IN_PROGRESS", f"Starting refinement iteration {refinement_iteration}/{self.max_refinement_iterations}...")

            # 7a. Check for Search Request in Draft
            search_request = await self._extract_search_request(current_draft)
            if not search_request:
                self._log("No further search requested by the writer/refiner. Ending refinement loop.", level=logging.INFO)
                await self._send_ws_update("REFINING", "INFO", "No further search requested. Ending refinement.")
                break # Exit loop if no search tag is found

            self._log(f"Refinement search requested: '{search_request.query}'", level=logging.INFO)
            await self._send_ws_update("REFINING", "INFO", f"Refinement search requested: '{search_request.query}'")

            # Remove search tag from the current draft before potentially passing it to refiner
            current_draft = re.sub(r'<search_request.*?>', '', current_draft, flags=re.IGNORECASE).strip()

            # 7b. Execute Refinement Search
            new_search_results: List[SearchResult] = []
            try:
                await self._send_ws_update("SEARCHING", "START", f"[Refinement Iteration {refinement_iteration}] Performing search...")
                
                # Create a proper search task list for refinement
                refinement_tasks = [{
                    "query": search_request.query,
                    "endpoint": "/search", # Default to general search for refinement
                    "num_results": 10    # Get a decent number for processing
                }]
                
                batch_results = await execute_batch_serper_search(
                    search_tasks=refinement_tasks, 
                    config=self.serper_config
                )
                # batch_results = await execute_batch_serper_search(self.serper_config, [search_request.query]) # Incorrect call
                self.serper_queries_used += len(refinement_tasks)
                
                # Process results (assuming List[Dict[str, Any]])
                parsed_results: List[SearchResult] = []
                for task_result in batch_results:
                    for item in task_result.get('organic', []):
                        try:
                            parsed_results.append(SearchResult.from_dict(item))
                        except Exception as parse_e:
                             self._log(f"[Refinement] Failed to parse search result item: {item}. Error: {parse_e}", level=logging.WARNING)
                new_search_results = parsed_results # Assign parsed results

                self._log(f"Refinement search yielded {len(new_search_results)} results.", level=logging.INFO)
                await self._send_ws_update("SEARCHING", "END", f"[Refinement Iteration {refinement_iteration}] Search complete.")
            except ExternalServiceError as e:
                error_msg = f"Refinement search failed (Iteration {refinement_iteration}): {type(e).__name__}"
                self._log(error_msg, level=logging.ERROR, exc_info=False)
                await self._send_ws_update("SEARCHING", "ERROR", f"[Refinement Iteration {refinement_iteration}] Search failed.")
                break

            # 7c. Filter & Process New Search Results
            new_relevant_sources: List[SearchResult] = []
            if new_search_results:
                # Filter out URLs already processed in the initial phase
                potential_new_sources = [res for res in new_search_results if res.link and res.link not in all_processed_urls]
                self._log(f"Found {len(potential_new_sources)} potential new sources from refinement search.", level=logging.DEBUG)

                if potential_new_sources:
                    # For simplicity, let's just take the top N for now
                    MAX_SOURCES_PER_REFINEMENT = 3  # Hardcoded value instead of self.settings.max_sources_per_refinement
                    num_new_to_process = min(len(potential_new_sources), MAX_SOURCES_PER_REFINEMENT)
                    new_relevant_sources = potential_new_sources[:num_new_to_process]
                    self._log(f"Selected {len(new_relevant_sources)} new sources to process for refinement.", level=logging.INFO)
                else:
                    self._log("Refinement search did not yield any sources with new URLs.", level=logging.INFO)
            else:
                self._log("Refinement search yielded no results.", level=logging.INFO)

            # 7d. Fetch, Summarize/Chunk New Relevant Info
            new_relevant_chunks: List[Dict[str, Any]] = []
            if new_relevant_sources:
                self._log(f"[Refinement Iteration {refinement_iteration}] Processing {len(new_relevant_sources)} new sources...", level=logging.INFO)
                await self._send_ws_update("PROCESSING", "START", f"[Refinement Iteration {refinement_iteration}] Processing {len(new_relevant_sources)} new sources...")

                new_source_map = {source.link: source for source in new_relevant_sources if source.link}
                CHUNK_RELEVANCE_THRESHOLD = 0.5 # Same threshold as before
                
                # Process sources sequentially to avoid rate limits
                self._log(f"[Refinement] Processing {len(new_source_map)} sources sequentially to avoid rate limits")
                total_new_relevant_chunks = 0
                successful_new = 0
                failed_new = 0
                
                for url, source in new_source_map.items():
                    try:
                        self._log(f"[Refinement] Processing source: {url}")
                        all_processed_urls.add(url) # Mark as processed
                        
                        proc_result = await self._fetch_chunk_and_rerank(
                            url=url,
                            query=search_request.query, # Use refinement query if available
                            source=source,
                            threshold=CHUNK_RELEVANCE_THRESHOLD,
                            is_refinement=True # Mark as refinement chunking
                        )
                        
                        if isinstance(proc_result, list):
                            new_relevant_chunks.extend(proc_result)
                            total_new_relevant_chunks += len(proc_result)
                            self._log(f"[Refinement] Found {len(proc_result)} relevant chunks for {url}")
                            if proc_result: # If list is not empty, count source as successful
                                successful_new += 1
                        # else: Helper returned empty list (logged internally)
                    except Exception as e:
                        failed_new += 1
                        error_msg = f"[Refinement Chunking] Error processing source {url}: {type(e).__name__}"
                        self._log(error_msg, level=logging.ERROR, exc_info=False)
                        await self._send_ws_update("PROCESSING", "ERROR", f"[Refinement] Error processing chunks for {url}")
                    
                    # Add a short delay between requests to respect rate limits
                    await asyncio.sleep(0.5)  # Half-second delay

                self._log(f"[Refinement Iteration {refinement_iteration}] Chunk processing complete. Found {total_new_relevant_chunks} relevant chunks (score >= {CHUNK_RELEVANCE_THRESHOLD}) across {successful_new}/{len(new_relevant_sources)} sources.", level=logging.INFO)
                await self._send_ws_update("PROCESSING", "END", f"[Refinement Iteration {refinement_iteration}] Finished chunk processing. Found {total_new_relevant_chunks} relevant chunks.")
            else:
                self._log(f"[Refinement Iteration {refinement_iteration}] No new relevant sources to process.", level=logging.INFO)
                await self._send_ws_update("PROCESSING", "INFO", f"[Refinement Iteration {refinement_iteration}] No new sources to process.")

            # 7e. Call Refiner LLM
            if new_relevant_chunks: # Check if we found relevant chunks
                try:
                    # Add delay before the refiner call
                    await asyncio.sleep(1.0) # 1-second delay before refiner
                    
                    refined_draft = await self._call_refiner_llm(
                        previous_draft=current_draft,
                        search_query=search_request.query,
                        new_info=new_relevant_chunks # Pass new relevant chunk dicts
                    )
                    current_draft = refined_draft # Update draft for next iteration or final output
                    self._log(f"Draft updated after refinement iteration {refinement_iteration}.", level=logging.INFO)
                    await self._send_ws_update("REFINING", "IN_PROGRESS", f"Draft refined based on new chunk information (Iteration {refinement_iteration}).")

                except (LLMError, AgentExecutionError) as e:
                     error_msg = f"Refinement LLM call failed (Iteration {refinement_iteration}): {type(e).__name__}"
                     self._log(error_msg, level=logging.ERROR, exc_info=False)
                     await self._send_ws_update("REFINING", "ERROR", f"Refinement failed for iteration {refinement_iteration}.")
                     break
                else:
                     self._log(f"Skipping refiner LLM call as no new information was processed in iteration {refinement_iteration}.", level=logging.INFO)
                break

            # Small delay between iterations? Optional.
            # await asyncio.sleep(1)

        # After loop finishes or breaks
        if self.max_refinement_iterations > 0:
             self._log(f"--- Refinement Loop Completed after {refinement_iteration} iterations ---", level=logging.INFO)
             await self._send_ws_update("REFINING", "END", f"Refinement process completed after {refinement_iteration} iterations.")

        # --- 8. Final Report Assembly ---
        final_report = ""
        try:
             self._log("--- Phase 8: Final Report Assembly ---", level=logging.INFO)
             await self._send_ws_update("FINALIZING", "START", "Assembling final report...")

             # Use the latest draft (either initial or refined)
             # The _assemble_final_report might add citations or formatting
             final_report = self._assemble_final_report(current_draft, final_writer_context) 

             self._log(f"Final report assembled (length: {len(final_report)} chars).", level=logging.INFO)
             await self._send_ws_update("FINALIZING", "END", "Final report assembled.")
        except Exception as e:
             error_msg = f"Error during final report assembly: {type(e).__name__}"
             self._log(error_msg, level=logging.ERROR, exc_info=False)
             await self._send_ws_update("FINALIZING", "ERROR", "Failed to assemble final report.")
             final_report = current_draft
             self._log("Using latest draft as final report due to assembly error.", level=logging.WARNING)
             await self._send_ws_update("FINALIZING", "INFO", "Using latest draft due to assembly error.")

        # --- 9. Completion ---
        self._log("--- Research Process Completed ---", level=logging.INFO)

        usage_data = {
            "token_usage": self.token_usage,
            "estimated_cost": self.estimated_cost,
            "serper_queries_used": self.serper_queries_used,
            # Add other stats like num_sources_processed, refinement_iterations?
            "sources_processed_count": len(all_processed_urls),
             "refinement_iterations_run": refinement_iteration if self.max_refinement_iterations > 0 else 0
        }
        self._log(f"Final Usage: {usage_data}", level=logging.INFO)

        # Send final COMPLETE message before returning
        await self._send_ws_update("COMPLETE", "END", "Research process completed successfully.", {
            # Avoid sending the full potentially large report in the WS message details?
            # Send summary stats instead. Client can get report from return value if needed.
            # "report": final_report, # Maybe omit or truncate
            "final_report_length": len(final_report),
            "usage": usage_data
        })

        return {
            "final_report": final_report,
            "usage_statistics": usage_data
        }

    async def _fetch_and_summarize(
        self,
        url: str,
        query: str, # Pass query for context during summarization
        source: SearchResult, # Pass the original source for context (title, snippet)
        is_refinement: bool = False # Flag if called during refinement
    ) -> Optional[Tuple[SourceSummary, str]]: # Return summary and scraped content
        """Fetches content, summarizes it, and sends WS updates."""
        action_prefix = "[Refinement] " if is_refinement else ""
        stage = "PROCESSING" # Consistent stage for WS updates

        try:
            # STEP 1: Scrape content using the updated scraper
            self._log(f"{action_prefix}Fetching content from: {url}", level=logging.INFO)
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Fetching content...", {"source_url": url, "action": "Fetching"})

            # Call the refactored scrape method - it returns a single ExtractionResult
            # Add args for download_pdfs etc. if agent needs to control it, otherwise use defaults
            scrape_result: ExtractionResult = await self.scraper.scrape(url)
            
            # STEP 2: Check for successful content extraction
            scraped_content: Optional[str] = None
            scrape_source_name: str = "unknown_source" 

            if scrape_result and scrape_result.content and scrape_result.content.strip():
                scraped_content = scrape_result.content
                scrape_source_name = scrape_result.name # Get source name from result
                self._log(f"{action_prefix}Successfully scraped {len(scraped_content)} chars from {url} (source: {scrape_source_name})", level=logging.DEBUG)
            else:
                # Handle case where scrape succeeded but content is empty/None, or scrape failed implicitly
                error_detail = f"ExtractionResult content was empty or None." if scrape_result else f"scrape() returned None or failed."
                raise ScrapingError(f"No valid content extracted from {url}. {error_detail}")
            
            # STEP 3: Summarize with LLM
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Summarizing content...", {"source_url": url, "action": "Summarizing"})
            self._log(f"{action_prefix}Summarizing content for: {url}", level=logging.INFO)

            summarizer_prompt = get_summarizer_prompt(
                user_query=query,
                source_title=source.title or "Unknown Title",
                source_link=url,
                source_content=scraped_content # Pass the extracted string content
            )
            messages = [{"role": "user", "content": summarizer_prompt}]

            response, usage, cost = await call_litellm_acompletion(
                messages=messages,
                llm_config=self.summarizer_llm_config,
            )

            summary_content = response.choices[0].message.content or ""
            if not summary_content.strip():
                raise LLMError("Summarizer returned empty content.")

            # STEP 4: Log usage and update progress
            self._log_and_update_usage('summarizer', usage, cost)

            summary = SourceSummary(
                title=source.title or "Unknown Title",
                 link=url, # Use 'link' as per schema
                content=summary_content, # Use 'content' as per schema
                content_type='summary' # Add required content_type
                # strategy removed as it's not in schema
            )
            
            self._log(f"{action_prefix}Successfully summarized content from {url}: {len(summary_content)} chars", level=logging.INFO)
            await self._send_ws_update(stage, "SUCCESS", f"{action_prefix}Successfully summarized content from {url}", {"source_url": url})
            
            return summary, scraped_content # Return summary and original content
            
        except (ScrapingError, LLMError) as e:
            error_msg = f"{action_prefix}Error processing {url}: {type(e).__name__}"
            self._log(error_msg, level=logging.WARNING, exc_info=False)
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Failed to summarize {url}: {type(e).__name__}", {"source_url": url, "error": type(e).__name__})
            return None  # Indicate failure
        except Exception as e:
            error_msg = f"{action_prefix}Unexpected error processing {url}: {type(e).__name__}"
            self._log(error_msg, level=logging.ERROR, exc_info=False)
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Unexpected error processing {url}", {"source_url": url, "error": type(e).__name__})
            return None  # Indicate failure

    async def _fetch_chunk_and_rerank(
        self,
        url: str,
        query: str, # Query to rerank chunks against
        source: SearchResult, # Original source for context
        threshold: float, # Relevance threshold for chunks
        is_refinement: bool = False # Flag if called during refinement
    ) -> List[Dict[str, Any]]: # Return list of relevant chunk dicts
        """Fetches, chunks, reranks chunks, filters, and sends WS updates."""
        action_prefix = "[Refinement] " if is_refinement else ""
        stage = "PROCESSING" # Consistent stage for WS updates
        relevant_chunks_info = []

        try:
            # STEP 1: Scrape content
            self._log(f"{action_prefix}Fetching content for chunking from: {url}", level=logging.INFO)
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Fetching content...", {"source_url": url, "action": "Fetching"})

            # Call the refactored scrape method - it returns a single ExtractionResult
            scrape_result: ExtractionResult = await self.scraper.scrape(url)
            
            # STEP 2: Check for successful content extraction
            scraped_content: Optional[str] = None
            # scrape_source_name = "unknown_source" # Not needed here

            if scrape_result and scrape_result.content and scrape_result.content.strip():
                scraped_content = scrape_result.content
                # scrape_source_name = scrape_result.name \
                self._log(f"{action_prefix}Successfully scraped {len(scraped_content)} chars from {url} (source: {scrape_result.name})", level=logging.DEBUG)
            else:
                error_detail = f"ExtractionResult content was empty or None." if scrape_result else f"scrape() returned None or failed."
                raise ScrapingError(f"No valid content extracted from {url} by any strategy for chunking. {error_detail}")

            # STEP 3: Chunk the content
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Chunking content...", {"source_url": url, "action": "Chunking"})
            self._log(f"{action_prefix}Chunking content for: {url}", level=logging.INFO)
            chunker = Chunker(chunk_size=2048, chunk_overlap=100, min_chunk_size=256)
            doc_to_chunk = [{'title': source.title or "Unknown Title", 'link': url, 'content': scraped_content}]
            chunked_docs = chunker.chunk_and_label(doc_to_chunk)
            
            if not chunked_docs:
                self._log(f"{action_prefix}Chunking produced no chunks for {url}", level=logging.WARNING)
                await self._send_ws_update(stage, "WARNING", f"{action_prefix}Chunking produced no chunks for {url}")
                return []
            
            chunk_contents = [doc['content'] for doc in chunked_docs]
            self._log(f"{action_prefix}Created {len(chunk_contents)} chunks from {url} (avg length: {sum(len(c) for c in chunk_contents)/max(1, len(chunk_contents)):.0f} chars)", level=logging.INFO)
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Reranking {len(chunk_contents)} chunks...", {"source_url": url, "action": "Reranking"})

            # STEP 4: Rerank chunks against the query
            if not chunk_contents:
                # Skip empty chunking result
                self._log(f"{action_prefix}No chunks to rerank for {url}", level=logging.WARNING)
                await self._send_ws_update(stage, "WARNING", f"{action_prefix}No chunks to rerank for {url}")
                return []
                
            reranked_chunks = await rerank_with_together_api(
                query=query,
                documents=chunk_contents,
                model=self.reranker_model,
                api_key=self.together_api_key,
                relevance_threshold=threshold
            )
            self._log(f"{action_prefix}Reranking complete for {len(chunk_contents)} chunks", level=logging.DEBUG)

            # STEP 5: Filter chunks below threshold
            filtered_chunks = []
            for chunk_result in reranked_chunks:
                chunk_index = chunk_result['index']
                chunk_score = chunk_result['score']
                if 0 <= chunk_index < len(chunked_docs):
                    chunk_doc = chunked_docs[chunk_index].copy()
                    chunk_doc['relevance_score'] = chunk_score
                    chunk_doc['rank'] = len(filtered_chunks) + 1
                    filtered_chunks.append(chunk_doc)

            # Filter chunks by threshold
            self._log(f"{action_prefix}{len(filtered_chunks)}/{len(chunk_contents)} chunks passed threshold {threshold} for {url}", level=logging.INFO)
            
            # STEP 6: Return relevant chunks
            await self._send_ws_update(stage, "SUCCESS", f"{action_prefix}Processed {len(filtered_chunks)} relevant chunks from {url}", {"source_url": url})
            return filtered_chunks
            
        except ScrapingError as e:
            error_msg = f"{action_prefix}Scraping error for {url}: {type(e).__name__}"
            self._log(error_msg, level=logging.WARNING, exc_info=False)
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Failed to fetch content from {url}: {type(e).__name__}", {"source_url": url, "error": type(e).__name__})
            return []
        except Exception as e:
            error_msg = f"{action_prefix}Unexpected error processing {url}: {type(e).__name__}"
            self._log(error_msg, level=logging.ERROR, exc_info=False)
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Unexpected error processing {url}", {"source_url": url, "error": type(e).__name__})
            return []

    def _assemble_final_report(self, report_draft: str, writer_context_items: list) -> str: # Adjusted type hint for context
         """Assembles the final report with references, matching oldagent logic."""
         self._log("Assembling final report with references...", level=logging.DEBUG)
         cleaned_draft = re.sub(r'<search_request.*?>', '', report_draft, flags=re.IGNORECASE).strip()

         cited_links = set()
         reference_list_items = []

         if writer_context_items:
             self._log(f"Checking {len(writer_context_items)} processed sources/chunks for citations in draft...", level=logging.DEBUG)
             for item in writer_context_items:
                 # Removed if item['type'] == 'summary': - Check all items for rank
                 citation_marker = f"[{item['rank']}]"
                 # Consistently use 'link', assuming Phase 5 adds it correctly for both types
                 link = item.get('link') # Default to None if key missing
                 display_title = item.get('title', 'Untitled') # Default title

                 if citation_marker in cleaned_draft:
                     if link and link not in cited_links:
                         # Ensure link is converted to string if it's a Pydantic URL type from SourceSummary
                         link_str = str(link) 
                         reference_list_items.append((item['rank'], display_title, link_str))
                         cited_links.add(link_str)
                         self._log(f"  Found citation {citation_marker} for '{display_title}' ({link_str})", level=logging.DEBUG)

             if reference_list_items:
                 reference_list_items.sort(key=lambda x: x[0]) # Sort by the rank added in Phase 5

                 reference_list_str = "\n\nReferences:\n"
                 # Use the sorted list index for final numbering (1, 2, 3...)
                 for i, (_, title, url) in enumerate(reference_list_items): 
                      reference_list_str += f"{i+1}. [{title}]({url})\n"

                 final_report = cleaned_draft + "\n" + reference_list_str.strip()
                 self._log(f"Appended reference list with {len(reference_list_items)} unique, cited sources.", level=logging.INFO)
             else:
                 final_report = cleaned_draft
                 self._log("No citations found in draft. Final report has no references.", level=logging.INFO)
         else:
             final_report = cleaned_draft
             self._log("No source materials processed. Final report has no references.", level=logging.INFO)

         return final_report