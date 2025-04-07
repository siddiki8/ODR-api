from typing import Any, Dict, Optional, Callable, List, Tuple, Literal, Coroutine
import logging 
import json
import re
import asyncio
import os
from collections import Counter
from pydantic import ValidationError, HttpUrl
from datetime import datetime 

# --- Internal Imports (New Structure) ---
from ..services.search import execute_batch_serper_search, SerperConfig
from ..core.schemas import SearchResult, SearchTask
from ..services.ranking import rerank_with_together_api
from ..services.scraping import WebScraper, ExtractionResult
from ..services.chunking import chunk_and_label
from .schemas import PlannerOutput, SourceSummary, SearchRequest, Chunk, TokenUsageCounter, UsageStatistics # Removed WritingPlan
from .prompts import (
    get_planner_prompt,
    get_summarizer_prompt,
    get_writer_initial_prompt,
    format_summaries_for_prompt as format_summaries_for_prompt_template,
    _WRITER_SYSTEM_PROMPT_BASE,
    get_writer_refinement_prompt
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
    """
    Agent responsible for conducting deep research based on a user query.
    
    Orchestrates a multi-step process involving search, filtering, scraping, 
    chunking, LLM summarization, and report generation. Manages API keys,
    configurations, and provides status updates via logging and WebSockets.
    """

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
        # logger_callback: Optional[Callable[[str, int], None]] = None, # Removed logger callback
        # Allow overriding provider at agent instantiation
        llm_provider_override: Optional[Literal['google', 'openrouter']] = None, 
        websocket_callback: Optional[Callable[..., Coroutine]] = None # <-- Add WebSocket callback
    ):
        """
        Initializes the DeepResearchAgent with configuration and API settings.

        Sets up configuration, logging, API clients, and workflow parameters.

        Args:
            settings: Instantiated AppSettings object with configuration parameters.
            api_keys: Instantiated ApiKeys object with necessary API credentials.
            planner_llm_override: Optional LLMConfig to override default planner settings.
            summarizer_llm_override: Optional LLMConfig to override default summarizer settings.
            writer_llm_override: Optional LLMConfig to override default writer settings.
            scraper_strategies_override: Optional list of strategies to override default scraper strategies.
            max_search_tasks_override: Optional integer to override the maximum number of search tasks.
            # logger_callback: Optional function to call for logging instead of print. # Removed
            llm_provider_override: Optionally override the LLM provider ('google' or 'openrouter').
            websocket_callback: Optional async function to send status updates over WebSocket.
        """
        self.settings = settings
        self.api_keys = api_keys
        # Determine the effective LLM provider
        self.llm_provider = llm_provider_override or settings.llm_provider
        
        # Setup logger instance
        self.logger = logging.getLogger(f"DeepResearchAgent_{id(self)}")
        if not self.logger.hasHandlers():
             handler = logging.StreamHandler()
             log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
             formatter = logging.Formatter(log_format)
             handler.setFormatter(formatter)
             self.logger.addHandler(handler)
        log_level = logging.INFO 
        self.logger.setLevel(log_level)

        self.websocket_callback = websocket_callback 
        
        self.logger.info(f"Initializing Agent with LLM Provider: {self.llm_provider}")
        if self.websocket_callback:
             self.logger.debug("WebSocket callback provided.")

        # --- Determine effective LLM configurations --- 
        try:
            effective_planner_config = planner_llm_override or settings.default_planner_llm
            effective_summarizer_config = summarizer_llm_override or settings.default_summarizer_llm
            effective_writer_config = writer_llm_override or settings.default_writer_llm

            # Set model names based on provider
            effective_planner_config.model = settings.get_model_name('planner')
            effective_summarizer_config.model = settings.get_model_name('summarizer')
            effective_writer_config.model = settings.get_model_name('writer')

            # Finalize LiteLLM parameters
            self.planner_llm_config = get_litellm_params(
                effective_planner_config, self.llm_provider, self.api_keys,
                settings=settings # Pass settings
            )
            self.summarizer_llm_config = get_litellm_params(
                effective_summarizer_config, self.llm_provider, self.api_keys,
                settings=settings # Pass settings
            )
            self.writer_llm_config = get_litellm_params(
                effective_writer_config, self.llm_provider, self.api_keys,
                settings=settings # Pass settings
            )
            
            self.logger.debug("LLM configurations finalized.")
            # Example debug log for one config (others similar)
            self.logger.debug(f"  Planner Config Model: {self.planner_llm_config.get('model')}")
            
        except ConfigurationError as e:
            self.logger.critical(f"Agent Initialization Failed: LLM configuration error - {e}")
            raise 
        except Exception as e:
            self.logger.critical(f"Agent Initialization Failed: Unexpected error during LLM config setup - {e}", exc_info=True)
            raise ConfigurationError(f"Unexpected error initializing LLM configs: {e}") from e

        # --- Initialize other components --- 
        try:
            self.scraper = WebScraper(
                settings=settings, 
                debug=settings.scraper_debug
            )
            self.serper_config = SerperConfig(
                api_key=api_keys.serper_api_key.get_secret_value(),
                base_url=str(settings.serper_base_url), 
                default_location=settings.serper_default_location,
                timeout=settings.serper_timeout
            )
            self.together_api_key = api_keys.together_api_key.get_secret_value()
            if not self.together_api_key:
                 raise ConfigurationError("TOGETHER_API_KEY is required for reranking but not found.")
                 
        except (ConfigurationError, ValueError) as e:
            self.logger.critical(f"Agent Initialization Failed: Component configuration error - {e}")
            raise 
        except Exception as e:
            self.logger.critical(f"Agent Initialization Failed: Unexpected error during component setup - {e}", exc_info=True)
            raise ConfigurationError(f"Unexpected error initializing agent components: {e}") from e

        self.reranker_model = settings.reranker_model
        # Workflow parameters 
        self.max_initial_search_tasks = max_search_tasks_override if max_search_tasks_override is not None else settings.max_initial_search_tasks
        self.max_refinement_iterations = settings.max_refinement_iterations

        # Initialize usage trackers
        self.token_usage = {
            'planner': TokenUsageCounter(),
            'summarizer': TokenUsageCounter(), 
            'writer': TokenUsageCounter(),
            'refiner': TokenUsageCounter(), 
            'total': TokenUsageCounter()
        }
        self.estimated_cost = {
            'planner': 0.0, 'summarizer': 0.0, 'writer': 0.0,
            'refiner': 0.0, 'total': 0.0
        }
        self.serper_queries_used = 0
        self.sources_processed = set()  # Track unique URLs processed

        self.logger.info("DeepResearchAgent initialized successfully.")
        # Removed verbose logging of all configs here, rely on specific debug logs if needed

    # --- Helper for WebSocket Updates ---
    async def _send_ws_update(self, step: str, status: str, message: str, details: Dict[str, Any] | None = None):
        """
        Sends status updates over the WebSocket connection if the callback is set.
        Preprocesses details to ensure JSON serializability (e.g., converts HttpUrl to str).
        
        Args:
            step: The current step/phase of the research process (e.g., "PLANNING", "SEARCHING").
            status: The status of the step (e.g., "START", "IN_PROGRESS", "END", "ERROR").
            message: A human-readable message describing the current status.
            details: Optional dictionary with additional structured data about the status.
        """
        if self.websocket_callback:
            processed_details = {}
            if details:
                # Convert known non-serializable types
                for key, value in details.items():
                    if isinstance(value, HttpUrl):
                        processed_details[key] = str(value)
                    # Add checks for other non-serializable types if needed
                    # elif isinstance(value, SomeOtherType):
                    #    processed_details[key] = convert_to_serializable(value)
                    else:
                        processed_details[key] = value
            else:
                # Pass None if original details were None
                processed_details = None 

            try:
                self.logger.debug(f"WS Update: {step}/{status} - {message}")
                # Pass the processed details to the callback
                await self.websocket_callback(step, status, message, processed_details)
            except Exception as e:
                # Log error but don't crash agent if WS send fails
                self.logger.error(f"Failed to send WebSocket update ({step}/{status}): {e}", exc_info=True)
        else:
             self.logger.debug(f"WS Update Skipped (no callback): {step}/{status} - {message}")

    def _log_and_update_usage(self, role: Literal['planner', 'summarizer', 'writer', 'refiner'], usage_info: Optional[Dict[str, int]], cost_info: Optional[Dict[str, float]]):
        """
        Logs usage/cost and updates agent's usage tracking data for a specific LLM role.
        
        Handles immutable TokenUsageCounter objects by creating new instances with
        updated values when token counts change.
        
        Args:
            role: The LLM role (planner, summarizer, writer, refiner).
            usage_info: Dictionary containing token usage information (prompt_tokens, completion_tokens, total_tokens).
            cost_info: Dictionary containing cost information (total_cost).
        """
        if role not in self.token_usage:
            self.logger.error(f"Invalid role '{role}' passed to _log_and_update_usage. Skipping update.")
            return
            
        # Get the TokenUsageCounter objects
        role_token_usage = self.token_usage[role]  # TokenUsageCounter for this role
        total_token_usage = self.token_usage['total']  # TokenUsageCounter for total
        
        if usage_info:
            # We have immutable TokenUsageCounter objects, so we create new ones
            # with updated values
            
            # Update role-specific counter
            self.token_usage[role] = TokenUsageCounter(
                prompt_tokens=role_token_usage.prompt_tokens + usage_info['prompt_tokens'],
                completion_tokens=role_token_usage.completion_tokens + usage_info['completion_tokens'],
                total_tokens=role_token_usage.total_tokens + usage_info['total_tokens']
            )
            
            # Update total counter
            self.token_usage['total'] = TokenUsageCounter(
                prompt_tokens=total_token_usage.prompt_tokens + usage_info['prompt_tokens'],
                completion_tokens=total_token_usage.completion_tokens + usage_info['completion_tokens'],
                total_tokens=total_token_usage.total_tokens + usage_info['total_tokens']
            )
            
            # Get updated objects for logging
            updated_role_counter = self.token_usage[role]
            updated_total_counter = self.token_usage['total']
            
            self.logger.info(f"    [{role.upper()}] Tokens Used: Prompt={usage_info['prompt_tokens']}, Completion={usage_info['completion_tokens']}, Total={usage_info['total_tokens']}")
            self.logger.debug(f"    [{role.upper()}] Cumulative Role Tokens: {updated_role_counter.total_tokens}")
            self.logger.debug(f"    Cumulative Total Tokens: {updated_total_counter.total_tokens}")
        else:
            self.logger.warning(f"    [{role.upper()}] Token usage information not available for this call.")

        if cost_info and 'total_cost' in cost_info:
            current_cost = cost_info['total_cost']
            
            # Update role-specific cost
            self.estimated_cost[role] += current_cost
            # Update total cost
            self.estimated_cost['total'] += current_cost
            
            self.logger.info(f"    [{role.upper()}] LLM call cost: ${current_cost:.6f}. Cumulative Role Cost: ${self.estimated_cost[role]:.6f}")
            self.logger.debug(f"    Cumulative Total Cost: ${self.estimated_cost['total']:.6f}")
        else:
            self.logger.warning(f"    [{role.upper()}] Cost information not available for this call.")
             
    async def _extract_search_request(self, text: str) -> Optional[SearchRequest]:
        """Extracts a <search_request query=...> tag from text."""
        try:
            match = re.search(r'<search_request query=["\']([^"\']*)["\'](?:\s*/?)?>', text, re.IGNORECASE)
            if not match:
                return None
            
            query = match.group(1).strip()
        except re.error as e:
            self.logger.error(f"Regex error extracting search request: {e}")
            return None # Treat regex error as no match found
        
        try:
            # Validate using the SearchRequest schema (only requires query)
            return SearchRequest(query=query)
        except ValidationError as e:
            # Log as warning, not critical error
            self.logger.warning(f"Invalid search request format found in LLM output: '{query}'. Error: {e}")
            return None # Return None if validation fails
            
    def _estimate_writer_input_chars(self, user_query, writing_plan, source_materials):
        """DEPRECATED/UNUSED? Estimates the character count for the writer prompt."""
        # This method seems unused, consider removing if WRITER_INPUT_CHAR_LIMIT is not enforced.
        formatted_materials = format_summaries_for_prompt_template(source_materials)
        writer_prompt_messages = get_writer_initial_prompt(
            user_query=user_query,
            writing_plan=writing_plan,
            source_materials=formatted_materials
        )
        # Consider system prompt length too? Approx for now.
        # Need to calculate length of messages list, not just a single string
        total_chars = len(_WRITER_SYSTEM_PROMPT_BASE)
        for msg in writer_prompt_messages:
            if isinstance(msg.get('content'), str):
                 total_chars += len(msg['content'])
        return total_chars

    # --- Phase-Specific Methods --- #
    
    async def _run_planning_phase(self, user_query: str) -> PlannerOutput:
        """Generates the research plan using the Planner LLM.
        
        Handles LLM call, response validation (Pydantic model), and error reporting.
        
        Returns:
            PlannerOutput: The validated research plan.
        Raises:
            AgentExecutionError: If planning fails critically.
        """
        self.logger.info("--- Phase 1: Planning ---")
        await self._send_ws_update("PLANNING", "START", "Generating initial research plan...")
        try:
            messages = get_planner_prompt(user_query, self.max_initial_search_tasks)
            
            self.logger.debug("[Agent _run_planning_phase] Planner LLM Config BEFORE call_litellm_acompletion:")
            self.logger.debug(f"  Model: {self.planner_llm_config.get('model')}")
            self.logger.debug(f"  API Key Provided: {bool(self.planner_llm_config.get('api_key'))}")
            self.logger.debug(f"  API Base: {self.planner_llm_config.get('api_base')}")

            response, usage_info, cost_info = await call_litellm_acompletion(
                messages=messages,
                llm_config=self.planner_llm_config,
                num_retries=3,
                logger_callback=self.logger
            )
            self._log_and_update_usage('planner', usage_info, cost_info)

            # Debug Raw Response
            raw_content_debug = "(No content found)"
            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                raw_content_debug = response.choices[0].message.content
            self.logger.debug(f"--- RAW PLANNER LLM RESPONSE CONTENT ---\n{raw_content_debug}\n--------------------------------------")

            if response is None or not response.choices or not response.choices[0].message:
                raise ValueError("Planner LLM call failed or returned an invalid response structure.")

            planner_output_obj: Optional[PlannerOutput] = None
            if hasattr(response.choices[0].message, '_response_format_output') and response.choices[0].message._response_format_output:
                if isinstance(response.choices[0].message._response_format_output, PlannerOutput):
                    planner_output_obj = response.choices[0].message._response_format_output
                    self.logger.info("✓ Planner output parsed and validated by LiteLLM response_model.")
                else:
                    self.logger.warning(f"LiteLLM _response_format_output is not PlannerOutput (type: {type(response.choices[0].message._response_format_output)}). Attempting manual parse.")
            
            if planner_output_obj is None:
                raw_content = response.choices[0].message.content
                if not raw_content:
                     raise LLMOutputValidationError("Planner LLM response content is empty.")
                self.logger.info("Attempting manual JSON parse from planner response content...")
                try:
                    cleaned_text = raw_content.strip()
                    if cleaned_text.startswith("```json"):
                        cleaned_text = cleaned_text[7:]
                    if cleaned_text.startswith("```"):
                        cleaned_text = cleaned_text[3:]
                    if cleaned_text.endswith("```"):
                        cleaned_text = cleaned_text[:-3]
                    cleaned_text = cleaned_text.strip()
                    self.logger.debug(f"Cleaned text for validation: {cleaned_text[:200]}...")
                    planner_output_obj = PlannerOutput.model_validate_json(cleaned_text)
                    self.logger.info("✓ Planner output validated via manual parse.")
                except (json.JSONDecodeError, ValidationError) as fallback_e:
                    self.logger.error(f"Fallback validation failed: {fallback_e}", exc_info=True)
                    self.logger.error(f"Content attempted: {cleaned_text}")
                    raise LLMOutputValidationError(f"Planner failed to return valid structured output. Error: {fallback_e}") from fallback_e

            if not planner_output_obj:
                 raise AgentExecutionError("Failed to obtain valid planner output after parsing.")

            if not planner_output_obj.search_tasks:
                 raise LLMOutputValidationError("Planner did not generate any search tasks.")

            self.logger.info(f"Planner generated {len(planner_output_obj.search_tasks)} search tasks.")
            plan_details = {
                "writing_plan": planner_output_obj.writing_plan.model_dump(), 
                "search_task_count": len(planner_output_obj.search_tasks),
                "search_queries": [task.query for task in planner_output_obj.search_tasks]
            }
            await self._send_ws_update("PLANNING", "END", "Research plan generated.", {"plan": plan_details})
            return planner_output_obj
            
        except (LLMError, ValueError, ValidationError, json.JSONDecodeError, AgentExecutionError) as e:
            error_msg = f"Planning phase failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg, exc_info=False) # Don't need full trace for validation errors
            await self._send_ws_update("PLANNING", "ERROR", f"Failed to generate plan: {type(e).__name__}")
            raise AgentExecutionError("Failed to generate a valid research plan.") from e
        except Exception as e:
            error_msg = f"Unexpected error during planning phase: {type(e).__name__}: {e}"
            self.logger.critical(error_msg, exc_info=True) 
            await self._send_ws_update("PLANNING", "ERROR", "Unexpected error during planning.")
            raise AgentExecutionError("Unexpected error during planning.") from e
            
    async def _run_initial_search_phase(self, planner_output: PlannerOutput) -> List[SearchResult]:
        """Executes initial searches based on the plan via Serper API.
        
        Returns:
            List[SearchResult]: A list of parsed search results.
        Raises:
            AgentExecutionError: If the search phase fails critically.
        """
        self.logger.info("--- Phase 2: Initial Search ---")
        search_tasks = planner_output.search_tasks
        task_count = len(search_tasks)
        await self._send_ws_update("SEARCHING", "START", f"Performing initial search based on {task_count} tasks...")
        try:
            batch_results = await execute_batch_serper_search(
                search_tasks=search_tasks, 
                config=self.serper_config
            )
            self.serper_queries_used += task_count

            raw_search_results = batch_results 
            successful_queries = len(raw_search_results)
            self.logger.info(f"Initial search completed. Received results for {successful_queries}/{task_count} tasks.")
            
            parsed_results: List[SearchResult] = []
            for task_result in raw_search_results:
                for item in task_result.get('organic', []):
                     try:
                         parsed_results.append(SearchResult.from_dict(item))
                     except Exception as parse_e:
                          self.logger.warning(f"Failed to parse search result item: {item}. Error: {parse_e}")
            
            self.logger.debug(f"Parsed {len(parsed_results)} organic results from initial search.")
            await self._send_ws_update("SEARCHING", "END", f"Initial search yielded results for {successful_queries} queries.", {"raw_result_count": len(parsed_results), "queries_executed": successful_queries})

            if not parsed_results:
                 self.logger.warning("Initial search returned no valid organic results. Proceeding without web context.")
            
            return parsed_results
            
        except ExternalServiceError as e:
            error_msg = f"Search phase failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg, exc_info=False)
            await self._send_ws_update("SEARCHING", "ERROR", f"Search API error: {type(e).__name__}")
            raise # Re-raise for higher level handling or specific logic
        except Exception as e:
            error_msg = f"Unexpected error during initial search phase: {type(e).__name__}: {e}"
            self.logger.critical(error_msg, exc_info=True)
            await self._send_ws_update("SEARCHING", "ERROR", "Unexpected error during search.")
            raise AgentExecutionError("Unexpected error during initial search.") from e
            
    async def _run_reranking_phase(self, search_results: List[SearchResult], user_query: str) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Deduplicates and reranks search results against the user query.
        
        Splits results into sources for summarization and sources for chunking based on rank.
        
        Returns:
            Tuple[List[SearchResult], List[SearchResult]]: (sources_to_summarize, sources_to_chunk)
        Raises:
            AgentExecutionError: If the reranking phase fails critically.
        """
        self.logger.info("--- Phase 3: Reranking Search Results ---")
        sources_to_summarize: List[SearchResult] = []
        sources_to_chunk: List[SearchResult] = []

        if not search_results:
            self.logger.info("Skipping reranking phase as there were no initial search results.")
            await self._send_ws_update("RANKING", "INFO", "Skipped - No initial search results.")
            return sources_to_summarize, sources_to_chunk

        try:
            # Deduplicate
            unique_results_dict: Dict[str, SearchResult] = {}
            for result in search_results:
                if result.link and result.link not in unique_results_dict:
                     unique_results_dict[result.link] = result
            unique_results = list(unique_results_dict.values())
            self.logger.debug(f"Found {len(unique_results)} unique URLs for reranking.")
            
            if not unique_results:
                self.logger.warning("No unique search results with valid links found for reranking.")
                await self._send_ws_update("RANKING", "INFO", "No unique results to rerank.")
                return sources_to_summarize, sources_to_chunk

            await self._send_ws_update("RANKING", "START", f"Reranking {len(unique_results)} unique search results...")

            passages = [f"{r.title} {r.snippet}" for r in unique_results] 
            if not passages:
                self.logger.warning("No valid passages (title/snippet) found for reranking.")
            else:
                reranked_data = await rerank_with_together_api(
                    query=user_query,
                    documents=passages,
                    model=self.reranker_model,
                    api_key=self.together_api_key,
                    relevance_threshold=0.2,
                )
                
                num_reranked = len(reranked_data)
                num_to_summarize = min(num_reranked // 2, 10)
                
                reranked_indices_scores = [(res['index'], res['score']) for res in reranked_data]
                sources_to_summarize_indices = [idx for idx, score in reranked_indices_scores[:num_to_summarize]]
                sources_to_chunk_indices = [idx for idx, score in reranked_indices_scores[num_to_summarize:]]
                
                sources_to_summarize = [unique_results[i] for i in sources_to_summarize_indices]
                sources_to_chunk = [unique_results[i] for i in sources_to_chunk_indices]
                
                self.logger.debug(f"Reranking filtered {len(passages) - num_reranked} below threshold.")
                self.logger.info(f"Splitting {num_reranked} sources: {len(sources_to_summarize)} for summarization, {len(sources_to_chunk)} for chunking.")
            
            await self._send_ws_update("RANKING", "END", f"Identified {len(sources_to_summarize)} sources for summarization, {len(sources_to_chunk)} for chunking.")
            return sources_to_summarize, sources_to_chunk

        except ExternalServiceError as e:
            error_msg = f"Reranking phase failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg, exc_info=False)
            await self._send_ws_update("RANKING", "ERROR", f"Reranking API error: {type(e).__name__}")
            raise # Non-critical? For now, raise.
        except Exception as e:
            error_msg = f"Unexpected error during reranking phase: {type(e).__name__}: {e}"
            self.logger.critical(error_msg, exc_info=True)
            await self._send_ws_update("RANKING", "ERROR", "Unexpected error during reranking.")
            raise AgentExecutionError("Unexpected error during reranking.") from e

    async def _run_content_processing_phase(self, sources_to_summarize: List[SearchResult], sources_to_chunk: List[SearchResult], user_query: str) -> Tuple[List[SourceSummary], List[Chunk], set[str]]:
        """
        Fetches, summarizes, and chunks content from selected sources sequentially.
        
        Args:
            sources_to_summarize: List of sources to fetch and summarize.
            sources_to_chunk: List of sources to fetch, chunk, and rerank chunks.
            user_query: The original user query for context.
            
        Returns:
             Tuple containing:
                - List[SourceSummary]: Successfully generated summaries.
                - List[Chunk]: Relevant chunks extracted and reranked.
                - set[str]: All URLs processed in this phase.
        """
        processed_summaries: List[SourceSummary] = []
        processed_chunks: List[Chunk] = []  # Updated type hint
        all_processed_urls: set[str] = set()
        scraped_content_cache: Dict[str, str] = {}
        CHUNK_RELEVANCE_THRESHOLD = 0.5

        # --- 4a. Summarization --- 
        if sources_to_summarize: 
            self.logger.info(f"--- Phase 4a: Fetching and Summarizing {len(sources_to_summarize)} Sources ---")
            await self._send_ws_update("PROCESSING", "START", f"Starting summarization for {len(sources_to_summarize)} sources...")
            summary_source_map = {source.link: source for source in sources_to_summarize if source.link} 
            successful_summaries = 0
            failed_summaries = 0
            for url, source in summary_source_map.items():
                self.logger.info(f"Processing source for summarization: {url}")
                all_processed_urls.add(url)
                try:
                    result = await self._fetch_and_summarize(url, user_query, source)
                    if result:
                        summary, scraped_content = result
                        processed_summaries.append(summary)
                        scraped_content_cache[url] = scraped_content
                        successful_summaries += 1
                    else:
                        failed_summaries += 1
                        self.logger.warning(f"Summarization helper returned None for {url}.")
                except Exception as e:
                    failed_summaries += 1
                    self.logger.error(f"Exception summarizing {url}: {e}", exc_info=False)
                    await self._send_ws_update("PROCESSING", "ERROR", f"Unexpected error summarizing {url}")
                await asyncio.sleep(0.5) # Delay
            self.logger.info(f"Summarization complete. Success: {successful_summaries}/{len(sources_to_summarize)}.")
            await self._send_ws_update("PROCESSING", "INFO", f"Finished summarizing {successful_summaries} sources.")
        else:
            self.logger.info("Skipping summarization as no sources were designated.")

        # --- 4b. Chunking --- 
        if sources_to_chunk:
            self.logger.info(f"--- Phase 4b: Fetching, Chunking, Reranking Chunks for {len(sources_to_chunk)} Sources ---")
            await self._send_ws_update("PROCESSING", "START", f"Starting chunk processing for {len(sources_to_chunk)} sources...")
            chunk_source_map = {source.link: source for source in sources_to_chunk if source.link}
            total_relevant_chunks = 0
            failed_chunking_sources = 0
            for url, source in chunk_source_map.items():
                self.logger.info(f"Processing source for chunking: {url}")
                all_processed_urls.add(url)
                try:
                    chunk_result = await self._fetch_chunk_and_rerank(
                        url=url, 
                        query=user_query, 
                        source=source, 
                        threshold=CHUNK_RELEVANCE_THRESHOLD,
                        is_refinement=False
                    )
                    if isinstance(chunk_result, list):
                        processed_chunks.extend(chunk_result)
                        total_relevant_chunks += len(chunk_result)
                        self.logger.info(f"Found {len(chunk_result)} relevant chunks for {url}")
                except Exception as e:
                    failed_chunking_sources += 1
                    self.logger.error(f"Exception chunking {url}: {e}", exc_info=False)
                    await self._send_ws_update("PROCESSING", "ERROR", f"Error processing chunks for {url}")
                await asyncio.sleep(0.5) # Delay
            self.logger.info(f"Chunking & Reranking complete. Found {total_relevant_chunks} relevant chunks.")
            await self._send_ws_update("PROCESSING", "END", f"Finished chunk processing. Found {total_relevant_chunks} relevant chunks.")
        else:
            self.logger.info("Skipping chunking/reranking as no sources were designated.")
        
        await self._send_ws_update("PROCESSING", "END", f"Finished processing all initial sources ({len(all_processed_urls)} unique URLs).")
        # --- Add Logging --- #
        self.logger.info(f"Content Processing Result: {len(processed_summaries)} summaries, {len(processed_chunks)} relevant chunks obtained.")
        # ----------------- #
        return processed_summaries, processed_chunks, all_processed_urls
        
    def _assemble_writer_context(self, processed_summaries: List[SourceSummary], processed_chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        Assembles and ranks the combined list of summaries and chunks for the writer.
        
        Assigns a sequential 'rank' used for citation markers.
        
        Args:
            processed_summaries: List of SourceSummary objects containing summaries.
            processed_chunks: List of Chunk objects containing relevant text chunks.
            
        Returns:
            List[Dict[str, Any]]: The ranked list of context items ready for the writer.
        """
        self.logger.info("--- Phase 5: Assembling Initial Writer Context ---")
        writer_context_items = []
        current_rank = 1
        
        # Add summaries
        if processed_summaries:
            for s in processed_summaries:
                writer_context_items.append({
                    "type": "summary", 
                    "content": s.content, 
                    "link": str(s.link),  # Convert HttpUrl to string
                    "title": s.title,
                    "rank": current_rank 
                })
                current_rank += 1
        
        # Add chunks
        if processed_chunks:
            for c in processed_chunks:
                writer_context_items.append({
                    "type": "chunk", 
                    "content": c.content, 
                    "link": str(c.link),  # Convert HttpUrl to string
                    "title": c.title, 
                    "score": c.relevance_score,
                    "rank": current_rank
                })
                current_rank += 1
        
        context_item_count = len(writer_context_items)
        context_char_count = sum(len(str(item.get('content', ''))) for item in writer_context_items)
        self.logger.info(f"Assembled {context_item_count} items (~{context_char_count} chars) for writer.")
        return writer_context_items

    async def _run_initial_writing_phase(self, user_query: str, planner_output: PlannerOutput, writer_context: List[Dict[str, Any]]) -> str:
        """Generates the initial report draft using the Writer LLM.
        
        Returns:
            str: The generated initial draft.
        Raises:
            AgentExecutionError: If writing fails critically.
        """
        self.logger.info("--- Phase 6: Initial Report Generation ---")
        await self._send_ws_update("WRITING", "START", "Generating initial report draft...")
        try:
            self.logger.info("Preparing messages for initial writer LLM call...")
            writer_prompt_messages = get_writer_initial_prompt(
                user_query=user_query,
                writing_plan=planner_output.writing_plan.model_dump(),
                source_materials=writer_context
            )

            self.logger.info(f"Attempting to generate writer prompt with {len(writer_context)} context items.")

            await asyncio.sleep(1.0) # Delay
            response, usage, cost = await call_litellm_acompletion(
                messages=writer_prompt_messages,
                llm_config=self.writer_llm_config
            )
            initial_draft = response.choices[0].message.content or ""
            self._log_and_update_usage('writer', usage, cost)
            if not initial_draft.strip():
                raise LLMOutputValidationError("Writer LLM returned an empty initial draft.")
            self.logger.info(f"Initial report draft generated (length: {len(initial_draft)} chars).")
            await self._send_ws_update("WRITING", "END", "Initial report draft generated.")
            return initial_draft
        except (LLMError, ValidationError) as e:
            error_msg = f"Initial report generation failed: {type(e).__name__}: {e}"
            self.logger.error(error_msg, exc_info=False)
            await self._send_ws_update("WRITING", "ERROR", f"Failed to generate initial draft: {type(e).__name__}")
            raise AgentExecutionError("Failed to generate initial report draft.") from e
        except Exception as e:
            error_msg = f"Unexpected error during initial report generation: {type(e).__name__}: {e}"
            self.logger.critical(error_msg, exc_info=True)
            await self._send_ws_update("WRITING", "ERROR", "Unexpected error during initial writing.")
            raise AgentExecutionError("Unexpected error generating initial report.") from e

    async def _run_refinement_loop(self, user_query: str, planner_output: PlannerOutput, initial_draft: str, initial_context: List[Dict[str, Any]], processed_urls: set[str]) -> Tuple[str, int, List[Dict[str, Any]], set[str]]:
        """Runs the iterative refinement loop using the Writer LLM.
        
        Performs searches based on <search_request> tags, processes new sources,
        adds them to the context, and calls the Writer LLM with the full updated context.
        
        Args:
            user_query: The original user query.
            planner_output: The initial planner output (for writing plan).
            initial_draft: The draft to start refining.
            initial_context: The list of source materials used for the initial draft.
            processed_urls: Set of URLs already processed to avoid re-fetching.

        Returns:
            Tuple[str, int, List[Dict[str, Any]], set[str]]: The latest draft, the number of iterations run, the final updated context list, and the final set of processed URLs.
        """
        current_draft = initial_draft
        # Start with the context used for the initial draft
        all_source_materials = list(initial_context) # Make a mutable copy
        all_processed_urls = set(processed_urls) # Make a mutable copy
        refinement_iteration = 0
        CHUNK_RELEVANCE_THRESHOLD = 0.5 # Define threshold

        if self.max_refinement_iterations <= 0:
            self.logger.info("Skipping refinement loop as max_refinement_iterations is 0.")
            await self._send_ws_update("REFINING", "INFO", "Skipped - Max iterations set to 0.")
            # Return initial draft, 0 iterations, initial context
            return current_draft, 0, all_source_materials, all_processed_urls 

        self.logger.info(f"--- Phase 7: Refinement Loop (Max Iterations: {self.max_refinement_iterations}) ---")
        await self._send_ws_update("REFINING", "START", f"Starting refinement process (max {self.max_refinement_iterations} iterations)...")

        for i in range(self.max_refinement_iterations):
            refinement_iteration = i + 1
            self.logger.info(f"--- Refinement Iteration {refinement_iteration}/{self.max_refinement_iterations} ---")
            await self._send_ws_update("REFINING", "IN_PROGRESS", f"Starting refinement iteration {refinement_iteration}/{self.max_refinement_iterations}...")

            # 7a. Check for Search Request
            search_request = await self._extract_search_request(current_draft)
            if not search_request:
                self.logger.info("No further search requested. Ending refinement loop.")
                await self._send_ws_update("REFINING", "INFO", "No further search requested. Ending refinement.")
                break

            self.logger.info(f"Refinement search requested: '{search_request.query}'")
            await self._send_ws_update("REFINING", "INFO", f"Refinement search requested: '{search_request.query}'")
            # Keep the search request tag in the draft sent to the Writer LLM for context?
            # For now, remove it before sending to avoid confusing the LLM?
            # Let's remove it for now to prevent potential loops if the LLM includes it again.
            draft_for_refinement_call = re.sub(r'<search_request.*?>', '', current_draft, flags=re.IGNORECASE).strip()

            # 7b. Execute Refinement Search
            new_search_results: List[SearchResult] = []
            try:
                await self._send_ws_update("SEARCHING", "START", f"[Refinement {refinement_iteration}] Performing search...")
                # Instantiate SearchTask object instead of dict
                refinement_tasks = [
                    SearchTask(
                        query=search_request.query, 
                        endpoint="/search", # Defaulting to /search for refinement
                        num_results=10, 
                        reasoning=f"Refinement search iteration {refinement_iteration} based on writer request for '{search_request.query[:30]}...'" # Add required reasoning
                    )
                ]
                batch_results = await execute_batch_serper_search(search_tasks=refinement_tasks, config=self.serper_config)
                self.serper_queries_used += len(refinement_tasks)
                parsed_results: List[SearchResult] = []
                for task_result in batch_results:
                    for item in task_result.get('organic', []):
                        try:
                            parsed_results.append(SearchResult.from_dict(item))
                        except Exception as parse_e: self.logger.warning(f"[Refinement] Failed parse: {item}. Error: {parse_e}")
                new_search_results = parsed_results
                self.logger.info(f"[Refinement {refinement_iteration}] Search yielded {len(new_search_results)} results.")
                await self._send_ws_update("SEARCHING", "END", f"[Refinement {refinement_iteration}] Search complete.")
            except ExternalServiceError as e:
                self.logger.error(f"[Refinement {refinement_iteration}] Search failed: {e}", exc_info=False)
                await self._send_ws_update("SEARCHING", "ERROR", f"[Refinement {refinement_iteration}] Search failed.")
                break # Stop refinement on search failure

            # 7c. Filter & Process New Results (Chunking only for refinement)
            new_relevant_chunks: List[Chunk] = [] # Store Chunk objects
            newly_processed_urls_this_iter = set()
            if new_search_results:
                potential_new_sources = [res for res in new_search_results if res.link and str(res.link) not in all_processed_urls]
                self.logger.debug(f"Found {len(potential_new_sources)} potential new sources.")
                if potential_new_sources:
                    MAX_SOURCES_PER_REFINEMENT = 3
                    new_relevant_sources = potential_new_sources[:MAX_SOURCES_PER_REFINEMENT]
                    self.logger.info(f"Selected {len(new_relevant_sources)} new sources for refinement processing.")
                    await self._send_ws_update("PROCESSING", "START", f"[Refinement {refinement_iteration}] Processing {len(new_relevant_sources)} new sources (chunking only)..." )
                    
                    successful_new = 0
                    for source in new_relevant_sources:
                        if not source.link: continue
                        url_str = str(source.link)
                        self.logger.info(f"[Refinement {refinement_iteration}] Processing source: {url_str}")
                        all_processed_urls.add(url_str)
                        newly_processed_urls_this_iter.add(url_str)
                        try:
                            # Only chunking & reranking for refinement
                            proc_result = await self._fetch_chunk_and_rerank(
                                url=url_str,
                                query=search_request.query,
                                source=source,
                                threshold=CHUNK_RELEVANCE_THRESHOLD,
                                is_refinement=True
                            )
                            if isinstance(proc_result, list):
                                new_relevant_chunks.extend(proc_result)
                                self.logger.info(f"[Refinement] Found {len(proc_result)} relevant chunks for {url_str}")
                                if proc_result: successful_new += 1
                        except Exception as e:
                            self.logger.error(f"[Refinement Chunking] Error processing {url_str}: {e}", exc_info=False)
                            await self._send_ws_update("PROCESSING", "ERROR", f"[Refinement] Error chunking {url_str}")
                        await asyncio.sleep(0.5) # Delay
                    self.logger.info(f"[Refinement {refinement_iteration}] Chunk processing complete. Success: {successful_new}/{len(new_relevant_sources)} sources yielded relevant chunks.")
                    await self._send_ws_update("PROCESSING", "END", f"[Refinement {refinement_iteration}] Finished processing {successful_new} sources.")
                else:
                    self.logger.info("[Refinement] No new source URLs found.")
                    await self._send_ws_update("PROCESSING", "INFO", f"[Refinement {refinement_iteration}] No new sources found.")
            else:
                self.logger.info("[Refinement] Search yielded no results.")
                await self._send_ws_update("PROCESSING", "INFO", f"[Refinement {refinement_iteration}] No new sources found.")

            # 7d. Update Master Context and Re-rank for Citations
            if new_relevant_chunks:
                self.logger.info(f"Adding {len(new_relevant_chunks)} new chunks to the master context.")
                current_max_rank = max((item.get('rank', 0) for item in all_source_materials), default=0)
                for chunk in new_relevant_chunks:
                    current_max_rank += 1
                    # Convert Chunk object to dict format expected by _assemble/_format
                    all_source_materials.append({
                        "type": "chunk",
                        "content": chunk.content,
                        "link": str(chunk.link),
                        "title": chunk.title,
                        "score": chunk.relevance_score,
                        "rank": current_max_rank # Assign sequential rank
                    })
                self.logger.debug(f"Master context now contains {len(all_source_materials)} items.")
            else:
                 # If no new chunks were found relevant, stop refining this path
                 self.logger.info(f"Skipping Writer LLM call as no new relevant chunks were found in iteration {refinement_iteration}.")
                 self.logger.info("Ending refinement loop as no new relevant info found for search request.")
                 await self._send_ws_update("REFINING", "INFO", "Ending refinement - no new info found.")
                 break
            
            # 7e. Call Writer LLM for Refinement (using get_writer_refinement_prompt)
            try:
                self.logger.info(f"Calling Writer LLM ({self.summarizer_llm_config.get('model')}) for refinement iteration {refinement_iteration}.")
                await self._send_ws_update("REFINING", "IN_PROGRESS", "Calling LLM to refine draft..." )
                await asyncio.sleep(1.0) # Delay
                
                # Get the newly added context items (which are dicts)
                new_context_items = all_source_materials[-len(new_relevant_chunks):]

                # Generate messages using the WRITER refinement prompt
                # Pass the list of dicts directly
                refinement_messages = get_writer_refinement_prompt(
                    user_query=user_query,
                    writing_plan=planner_output.writing_plan.model_dump(), # Pass the original plan
                    previous_draft=draft_for_refinement_call,
                    refinement_topic=search_request.query, # Topic is the search query itself
                    new_summaries=new_context_items, # Pass the list of new context item dicts
                    all_summaries=all_source_materials # Pass the FULL updated context list
                )
                
                # Use the SUMMARIZER/PLANNER config for the LLM call
                response, usage, cost = await call_litellm_acompletion(
                    messages=refinement_messages,
                    llm_config=self.summarizer_llm_config # Use the summarizer config
                )
                refined_draft_content = response.choices[0].message.content or ""
                # Log usage under the 'refiner' key
                self._log_and_update_usage('refiner', usage, cost)

                if refined_draft_content:
                    # Remove search tags just in case the LLM adds one back
                    current_draft = re.sub(r'<search_request.*?>', '', refined_draft_content, flags=re.IGNORECASE).strip()
                    self.logger.info(f"Draft updated after refinement iteration {refinement_iteration}.")
                    await self._send_ws_update("REFINING", "IN_PROGRESS", f"Draft refined (Iteration {refinement_iteration}).")
                else:
                    self.logger.warning(f"Refinement call (using Writer prompt with Summarizer config) returned empty draft for iteration {refinement_iteration}. Keeping previous draft.")
                    await self._send_ws_update("REFINING", "WARNING", f"Refinement LLM call returned empty draft (Iteration {refinement_iteration}).")
                    # Optionally break here if empty refiner output is critical
            
            except LLMError as e:
                 self.logger.error(f"Refinement LLM call failed (Iteration {refinement_iteration}): {e}")
                 await self._send_ws_update("REFINING", "ERROR", f"Refinement failed for iteration {refinement_iteration}: {e}")
                 break # Stop refinement on LLM failure
            except Exception as e: # Catch other unexpected errors during refinement call
                 self.logger.error(f"Unexpected error during refinement LLM call (Iteration {refinement_iteration}): {e}", exc_info=True)
                 await self._send_ws_update("REFINING", "ERROR", f"Unexpected error during refinement call for iteration {refinement_iteration}.")
                 break # Stop refinement on LLM failure

        # After loop
        if self.max_refinement_iterations > 0:
             final_message = f"Refinement process completed after {refinement_iteration} iterations."
             self.logger.info(f"--- Refinement Loop Completed after {refinement_iteration} iterations --- ")
             await self._send_ws_update("REFINING", "END", final_message)

        # Return latest draft, iteration count, the final context list, and FINAL PROCESSED URLS
        return current_draft, refinement_iteration, all_source_materials, all_processed_urls

    async def run_deep_research(self, user_query: str) -> Dict[str, Any]:
        """
        Orchestrates the entire deep research process.

        Calls phase-specific methods, handles overall error flow, 
        and returns the final report and usage statistics.

        Args:
            user_query: The user's research query.

        Returns:
            A dictionary containing the final report and usage statistics.
        
        Raises:
            AgentExecutionError: If any phase fails critically.
        """
        self.logger.info(f"Starting deep research for query: '{user_query[:100]}...'")
        await self._send_ws_update("STARTING", "START", "Research process initiated.")

        final_report_content = "Error: Research did not complete." # Default in case of early exit
        final_context = [] # Will hold the list of dicts for writer/final assembly
        latest_draft = ""
        refinement_iterations_run = 0
        all_processed_urls_final: set[str] = set()

        try:
            # Phase 1: Planning
            planner_output = await self._run_planning_phase(user_query)

            # Phase 2: Initial Search
            initial_search_results = await self._run_initial_search_phase(planner_output)

            # Phase 3: Reranking
            sources_to_summarize, sources_to_chunk = await self._run_reranking_phase(initial_search_results, user_query)

            # Phase 4: Content Processing (Summaries & Chunks)
            processed_summaries, processed_chunks, all_processed_urls_final = await self._run_content_processing_phase(sources_to_summarize, sources_to_chunk, user_query)

            # Phase 5: Assemble Initial Context for Writer
            await self._send_ws_update("FILTERING", "START", "Assembling context for initial report...")
            initial_writer_context = self._assemble_writer_context(processed_summaries, processed_chunks)
            await self._send_ws_update("FILTERING", "END", f"Assembled {len(initial_writer_context)} context items for initial draft.")

            # Phase 6: Initial Writing
            initial_draft = await self._run_initial_writing_phase(user_query, planner_output, initial_writer_context)
            latest_draft = initial_draft # Start with initial draft
            
            # Phase 7: Refinement Loop (Pass necessary context and capture final URL set)
            latest_draft, refinement_iterations_run, final_context, all_processed_urls_final = await self._run_refinement_loop(
                user_query=user_query,
                planner_output=planner_output,
                initial_draft=initial_draft,
                initial_context=initial_writer_context, # Pass the context list
                processed_urls=all_processed_urls_final # Pass the set of processed URLs
            )
            # The loop now returns the final updated context list and final URL set

            # Phase 8: Final Assembly (Use the final context from the refinement loop)
            self.logger.info("--- Phase 8: Final Report Assembly ---")
            await self._send_ws_update("FINALIZING", "START", "Assembling final report...")
            try:
                # Use the potentially updated final_context list returned by the loop
                final_report_content = self._assemble_final_report(latest_draft, final_context) 
                self.logger.info(f"Final report assembled (length: {len(final_report_content)} chars).")
                # Include the actual report content in the FINALIZING/END message details
                await self._send_ws_update("FINALIZING", "END", "Final report assembled.", {"final_report": final_report_content})
            except Exception as e:
                 self.logger.error(f"Error during final report assembly: {e}", exc_info=True)
                 # Send error status for FINALIZING, but keep the latest draft as fallback content
                 await self._send_ws_update("FINALIZING", "ERROR", f"Failed to assemble final report: {e}")
                 final_report_content = latest_draft # Use latest draft as fallback
                 self.logger.warning("Using latest draft as final report due to assembly error.")
                 # Still send FINALIZING/END, but indicate fallback (maybe add detail?)
                 await self._send_ws_update("FINALIZING", "INFO", "Using latest draft due to assembly error.") # Changed status to INFO

        except AgentExecutionError as e:
            # Logged within the phase method where it originated
            final_report_content = f"Error: Research process failed during execution. {e}" 
            # Ensure we send a final error status if we reach here
            await self._send_ws_update("ERROR", "FATAL", f"Agent execution failed: {e}")
        except Exception as e:
            # Catch any other unexpected errors during orchestration
            self.logger.critical(f"Unexpected critical error during research orchestration: {e}", exc_info=True)
            final_report_content = f"Error: An unexpected critical error occurred: {e}"
            await self._send_ws_update("ERROR", "FATAL", f"Critical error: {e}")
        finally:
            # Phase 9: Completion & Stats (Use final count of processed URLs)
            self.logger.info("--- Research Process Completed ---")
            # Use the final set of URLs returned by the loop (or the initial set if loop didn't run)
            final_processed_url_count = len(all_processed_urls_final) 
            
            usage_statistics = UsageStatistics(
                token_usage=self.token_usage,
                estimated_cost=self.estimated_cost,
                serper_queries_used=self.serper_queries_used,
                sources_processed_count=final_processed_url_count, # Use updated count
                refinement_iterations_run=refinement_iterations_run
            )
            
            self.logger.info(f"Final Usage: {usage_statistics.model_dump()}")

            # Prepare the final result dictionary including the context
            result_dict = {
                "final_report": final_report_content,
                "usage_statistics": usage_statistics.model_dump(),
                "final_context": final_context # Include the final context list
            }

            # Send final COMPLETE/ERROR message (without the report content, it was sent in FINALIZING)
            final_status = "COMPLETE" if not final_report_content.startswith("Error:") else "ERROR"
            final_ws_status = "END" if final_status == "COMPLETE" else "FATAL"
            final_ws_message = "Research process completed successfully." if final_status == "COMPLETE" else "Research process failed."
            
            await self._send_ws_update(final_status, final_ws_status, final_ws_message, {
                "final_report_length": len(final_report_content),
                "usage": usage_statistics.model_dump(),  # Convert to dict for JSON serialization
                # REMOVED final_report content here
            })

            # Return the dictionary containing report, stats, and context
            return result_dict

    # --- Existing Helper Methods --- #
    
    async def _fetch_and_summarize(
        self,
        url: str,
        query: str, # Pass query for context during summarization
        source: SearchResult, # Pass the original source for context (title, snippet)
        is_refinement: bool = False # Flag if called during refinement
    ) -> Optional[Tuple[SourceSummary, str]]: # Return summary and scraped content
        """Fetches content, summarizes it using an LLM, and sends WS updates.
        
        Args:
            url: The URL of the source to fetch and summarize.
            query: The original user query for contextual relevance.
            source: The SearchResult object for metadata (title).
            is_refinement: Flag indicating if called during refinement phase (unused).
            
        Returns:
            A tuple (SourceSummary, scraped_content) on success, None on failure.
        """
        action_prefix = "[Refinement] " if is_refinement else ""
        stage = "PROCESSING" 

        try:
            # STEP 1: Scrape content using the updated scraper
            self.logger.info(f"{action_prefix}Fetching content from: {url}")
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Fetching content...", {"source_url": url, "action": "Fetching"})

            # Call the refactored scrape method - it returns a single ExtractionResult
            scrape_result: ExtractionResult = await self.scraper.scrape(url)
            
            # STEP 2: Check for successful content extraction
            scraped_content: Optional[str] = None
            scrape_source_name: str = "unknown_source" 

            if scrape_result and scrape_result.content and scrape_result.content.strip():
                scraped_content = scrape_result.content
                scrape_source_name = scrape_result.name # Get source name from result
                self.logger.debug(f"{action_prefix}Successfully scraped {len(scraped_content)} chars from {url} (source: {scrape_source_name})")
            else:
                # Handle case where scrape succeeded but content is empty/None, or scrape failed implicitly
                # CHANGED: Log warning, send WS WARNING, return None instead of raising error
                error_detail = f"ExtractionResult content was empty or None." if scrape_result else f"scrape() returned None or failed."
                warning_msg = f"No valid content extracted from {url}. {error_detail}. Skipping summarization."
                self.logger.warning(f"{action_prefix}{warning_msg}")
                await self._send_ws_update(stage, "WARNING", f"{action_prefix}Failed to fetch content from {url}. Skipping.", {"source_url": url, "reason": "No content returned by scraper"})
                return None
                # REMOVED: raise ScrapingError(f"No valid content extracted from {url}. {error_detail}")
            
            # STEP 3: Summarize with LLM
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Summarizing content...", {"source_url": url, "action": "Summarizing"})
            self.logger.info(f"{action_prefix}Summarizing content for: {url}")

            summarizer_prompt_messages = get_summarizer_prompt(
                user_query=query,
                source_title=source.title or "Unknown Title",
                source_link=url,
                source_content=scraped_content # Pass the extracted string content
            )
            # Note: get_summarizer_prompt returns a list of dicts already
            # messages = [{"role": "user", "content": summarizer_prompt}]
            # Directly use the result from get_summarizer_prompt
            messages = summarizer_prompt_messages 

            response, usage, cost = await call_litellm_acompletion(
                messages=messages,
                llm_config=self.summarizer_llm_config,
            )
            
            # Safely extract content with checks
            summary_content = "" # Default to empty string
            if response and response.choices and response.choices[0].message and hasattr(response.choices[0].message, 'content'):
                summary_content = response.choices[0].message.content or ""
            else:
                 # Log a warning if the expected structure isn't found
                 self.logger.warning(f"{action_prefix}LLM response structure unexpected or content missing for {url}. Response object keys: {list(response.__dict__.keys()) if response else 'None'}")
                 # Depending on requirements, could raise an error or proceed with empty summary

            if not summary_content.strip():
                # Raise error only if content is empty after attempting access
                # NOTE: This error now implies the LLM failed, not the scraping. Keep as LLMError? Yes.
                raise LLMError("Summarizer returned empty content or failed to extract content from response.")

            # STEP 4: Log usage and update progress
            self._log_and_update_usage('summarizer', usage, cost)

            summary = SourceSummary(
                title=source.title or "Unknown Title",
                 link=url, # Use 'link' as per schema
                content=summary_content, # Use 'content' as per schema
                content_type='summary' # Add required content_type
                # strategy removed as it's not in schema
            )
            
            self.logger.info(f"{action_prefix}Successfully summarized content from {url}: {len(summary_content)} chars")
            await self._send_ws_update(stage, "SUCCESS", f"{action_prefix}Successfully summarized content from {url}", {"source_url": url})
            
            return summary, scraped_content # Return summary and original content
            
        except (ScrapingError, LLMError) as e:
            # This block now primarily catches LLM errors or UNEXPECTED ScrapingErrors (e.g., from summarizer LLM call if it raised one somehow)
            error_msg = f"{action_prefix}Error processing {url} during summarization phase: {type(e).__name__}"
            self.logger.warning(error_msg, exc_info=False)
            # Keep sending ERROR here as it's likely an LLM or unexpected issue now
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Failed to summarize {url}: {type(e).__name__}", {"source_url": url, "error": type(e).__name__})
            return None  # Indicate failure
        except Exception as e:
            error_msg = f"{action_prefix}Unexpected error processing {url}: {type(e).__name__}"
            self.logger.error(error_msg, exc_info=True) # Log full traceback
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Unexpected error processing {url}", {"source_url": url, "error": type(e).__name__})
            return None  # Indicate failure

    async def _fetch_chunk_and_rerank(
        self,
        url: str,
        query: str,
        source: SearchResult,
        threshold: float,
        is_refinement: bool = False
    ) -> List[Chunk]:  # Updated return type
        """
        Fetches content, chunks it, reranks chunks, filters by threshold, and sends WS updates.
        
        Args:
            url: The URL of the source.
            query: The query to use for reranking chunks.
            source: The SearchResult object for metadata.
            threshold: The relevance score threshold for keeping chunks.
            is_refinement: Flag indicating if called during refinement phase.
            
        Returns:
            A list of Chunk objects representing the relevant text chunks.
        """
        action_prefix = "[Refinement] " if is_refinement else ""
        stage = "PROCESSING" 

        try:
            # STEP 1: Scrape content
            self.logger.info(f"{action_prefix}Fetching content for chunking from: {url}")
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Fetching content...", {"source_url": url, "action": "Fetching"})

            # Call the refactored scrape method - it returns a single ExtractionResult
            scrape_result: ExtractionResult = await self.scraper.scrape(url)
            
            # STEP 2: Check for successful content extraction
            scraped_content: Optional[str] = None

            if scrape_result and scrape_result.content and scrape_result.content.strip():
                scraped_content = scrape_result.content
                self.logger.debug(f"{action_prefix}Successfully scraped {len(scraped_content)} chars from {url} (source: {scrape_result.name})")
            else:
                # Handle case where scrape succeeded but content is empty/None, or scrape failed implicitly
                # CHANGED: Log warning, send WS WARNING, return [] instead of raising error
                error_detail = f"ExtractionResult content was empty or None." if scrape_result else f"scrape() returned None or failed."
                warning_msg = f"No valid content extracted from {url} by any strategy for chunking. {error_detail}. Skipping chunking."
                self.logger.warning(f"{action_prefix}{warning_msg}")
                await self._send_ws_update(stage, "WARNING", f"{action_prefix}Failed to fetch content from {url}. Skipping.", {"source_url": url, "reason": "No content returned by scraper"})
                return []
                # REMOVED: raise ScrapingError(f"No valid content extracted from {url} by any strategy for chunking. {error_detail}")

            # STEP 3: Chunk the content
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Chunking content...", {"source_url": url, "action": "Chunking"})
            self.logger.info(f"{action_prefix}Chunking content for: {url}")
            # Use the imported chunk_and_label function directly
            # Pass chunking parameters (could get from self.settings if configurable)
            doc_to_chunk = [{'title': source.title or "Unknown Title", 'link': url, 'content': scraped_content}]
            chunked_docs_dicts = chunk_and_label( # Renamed var to avoid confusion
                documents=doc_to_chunk, 
                chunk_size=2048, # Example values, consider making configurable
                chunk_overlap=100,
                min_chunk_size=256 
            )
            
            if not chunked_docs_dicts:
                self.logger.warning(f"{action_prefix}Chunking produced no chunks for {url}")
                await self._send_ws_update(stage, "WARNING", f"{action_prefix}Chunking produced no chunks for {url}")
                return []
            
            # Extract content for reranking
            chunk_contents = [doc['content'] for doc in chunked_docs_dicts]
            self.logger.info(f"{action_prefix}Created {len(chunk_contents)} chunks from {url} (avg length: {sum(len(c) for c in chunk_contents)/max(1, len(chunk_contents)):.0f} chars)")
            await self._send_ws_update(stage, "IN_PROGRESS", f"{action_prefix}Reranking {len(chunk_contents)} chunks...", {"source_url": url, "action": "Reranking"})

            # STEP 4: Rerank chunks against the query
            reranked_chunks = await rerank_with_together_api(
                query=query,
                documents=chunk_contents,
                model=self.reranker_model,
                api_key=self.together_api_key,
                relevance_threshold=threshold
            )
            self.logger.debug(f"{action_prefix}Reranking complete for {len(chunk_contents)} chunks")

            # STEP 5: Convert to Chunk objects
            filtered_chunks: List[Chunk] = []
            for chunk_result in reranked_chunks:
                chunk_index = chunk_result['index']
                chunk_score = chunk_result['score']
                if 0 <= chunk_index < len(chunked_docs_dicts): # Use the dict list here
                    chunk_doc_dict = chunked_docs_dicts[chunk_index] # Get the dictionary
                    # Create a Chunk object using data from the dict
                    try:
                        chunk = Chunk(
                            content=chunk_doc_dict['content'],
                            link=chunk_doc_dict['link'], # Assumes link is in metadata
                            title=chunk_doc_dict['title'], # Assumes title is in metadata
                            relevance_score=chunk_score,
                            rank=len(filtered_chunks) + 1
                        )
                        filtered_chunks.append(chunk)
                    except KeyError as e:
                         self.logger.warning(f"Missing expected key '{e}' in chunk dictionary from chunk_and_label for URL {url}. Chunk dict: {chunk_doc_dict}")
                    except ValidationError as e:
                         self.logger.warning(f"Pydantic validation error creating Chunk object for URL {url}: {e}. Chunk dict: {chunk_doc_dict}")

            # Filter chunks by threshold implicitly done by rerank_with_together_api
            self.logger.info(f"{action_prefix}{len(filtered_chunks)}/{len(chunk_contents)} chunks passed threshold {threshold} for {url}")
            
            # STEP 6: Return relevant chunks
            await self._send_ws_update(stage, "SUCCESS", f"{action_prefix}Processed {len(filtered_chunks)} relevant chunks from {url}", {"source_url": url})
            return filtered_chunks
            
        except ScrapingError as e:
            # This block now catches unexpected ScrapingErrors potentially raised during chunking/reranking itself
            error_msg = f"{action_prefix}Scraping error during chunk/rerank for {url}: {type(e).__name__}"
            self.logger.warning(error_msg, exc_info=False)
            # Keep sending ERROR here for unexpected issues during chunk/rerank
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Failed during chunk/rerank for {url}: {type(e).__name__}", {"source_url": url, "error": type(e).__name__})
            return []
        except Exception as e:
            error_msg = f"{action_prefix}Unexpected error processing {url}: {type(e).__name__}"
            self.logger.error(error_msg, exc_info=True) # Log full traceback
            await self._send_ws_update(stage, "ERROR", f"{action_prefix}Unexpected error processing {url}", {"source_url": url, "error": type(e).__name__})
            return []

    def _assemble_final_report(self, report_draft: str, writer_context_items: List[Dict[str, Any]]) -> str:
        """
        Assembles the final report, processing [[CITATION:rank]] markers and appending
        a numbered list of all sources provided to the writer.

        Removes any remaining <search_request> tags from the draft.
        Groups sources by unique link for the final list but uses the rank for citation numbers.

        Args:
            report_draft: The drafted report text generated by the LLM, potentially
                          containing [[CITATION:rank]] markers.
            writer_context_items: A list of dictionaries containing assembled context items
                                (converted from SourceSummary and Chunk objects) that were
                                provided to the writer, each with a unique 'rank'.

        Returns:
            The final report string with processed citations and a "Sources Consulted" list appended.
        """
        self.logger.debug("Assembling final report: Processing citations and adding source list...")

        # --- 1. Process Citations --- #
        processed_draft = report_draft
        max_rank = max((item.get('rank', 0) for item in writer_context_items), default=0)
        citation_errors = []

        def replace_citation_marker(match):
            ranks_str = match.group(1)
            try:
                # Parse potentially comma-separated ranks
                ranks = [int(r.strip()) for r in ranks_str.split(',')]
                valid_ranks = []
                for rank in ranks:
                    if 1 <= rank <= max_rank:
                        valid_ranks.append(rank)
                    else:
                        # Log invalid rank found
                        citation_errors.append(f"Invalid rank {rank} found (max is {max_rank}).")
                if not valid_ranks:
                    # If all ranks in a marker are invalid, return empty string or marker?
                    # Let's return empty for now to avoid showing invalid citations.
                    self.logger.warning(f"Citation marker '{match.group(0)}' contained only invalid ranks. Removing.")
                    return ""
                else:
                    # Format valid ranks into [1] or [1, 2] style
                    return f"[{', '.join(map(str, sorted(valid_ranks)))}]"""
            except ValueError:
                # Log error if rank is not an integer
                citation_errors.append(f"Non-integer rank found: '{ranks_str}'.")
                self.logger.warning(f"Citation marker '{match.group(0)}' contained non-integer rank. Removing.")
                return "" # Remove invalid marker

        # Regex to find [[CITATION:rank1,rank2,...]] markers
        # It allows spaces around the numbers and commas
        citation_pattern = re.compile(r"\[\[CITATION:\s*([\d\s,]+?)\s*\]\]")
        processed_draft = citation_pattern.sub(replace_citation_marker, report_draft)

        if citation_errors:
            self.logger.warning(f"Found {len(citation_errors)} issues during citation processing: {'; '.join(citation_errors)}")

        # --- 2. Clean Remaining Tags --- #
        # Remove any stray search request tags
        cleaned_draft = re.sub(r'<search_request.*?>', '', processed_draft, flags=re.IGNORECASE).strip()

        # --- 3. Assemble Source List --- #
        consulted_sources = {}
        if writer_context_items:
            self.logger.debug(f"Processing {len(writer_context_items)} context items to compile final source list...")
            # Use rank as the primary key now for sorting and display
            ranked_sources = {}
            for item in writer_context_items:
                rank = item.get('rank')
                link = item.get('link') # Should be a string already
                title = item.get('title', 'Untitled')

                if rank is not None and link is not None:
                    # Store title and link associated with each rank
                    # If multiple items have the same rank (shouldn't happen), last one wins
                    ranked_sources[rank] = {
                        'title': title,
                        'link': link
                    }
                else:
                    self.logger.warning(f"Skipping item for source list due to missing rank or link: {item}")

            if ranked_sources:
                # Sort by rank (key of the dict)
                sorted_ranks = sorted(ranked_sources.keys())

                reference_list_str = "\n\nSources Consulted:\n"
                for rank in sorted_ranks:
                    source_info = ranked_sources[rank]
                    # Format: 1. [Title](link)
                    reference_list_str += f"{rank}. [{source_info['title']}]({source_info['link']})\n"

                final_report = cleaned_draft + "\n" + reference_list_str.strip()
                self.logger.info(f"Appended list of {len(sorted_ranks)} sources consulted, ordered by rank.")
            else:
                final_report = cleaned_draft
                self.logger.info("No valid ranked sources found in context items. Final report has no source list.")
        else:
            final_report = cleaned_draft
            self.logger.info("No source materials provided to writer. Final report has no source list.")

        return final_report