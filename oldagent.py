from typing import Any, Dict, Optional, Callable, List, Tuple, Literal
import logging # Import logging
import litellm
import json
import re
import asyncio
import traceback
from collections import Counter
from pydantic import ValidationError

# --- Internal Imports (New Structure) ---
from ..services.search import execute_batch_serper_search, SerperConfig, SearchResult
from ..services.ranking import rerank_with_together_api
from ..services.scraping import WebScraper
from ..services.chunking import Chunker # Assuming placeholder exists
from .schemas import PlannerOutput, SourceSummary, RefinementRequest, SearchRequest
from .prompts import (
    get_planner_prompt,
    get_summarizer_prompt,
    get_writer_initial_prompt,
    get_writer_refinement_prompt,
    get_refiner_prompt,
    format_summaries_for_prompt as format_summaries_for_prompt_template,
    _WRITER_SYSTEM_PROMPT_BASE
)
# Import config classes and helper, but not instances
from .config import AppSettings, ApiKeys, LLMConfig, get_litellm_params 
# Import the new service function
from ..services.llm import call_litellm_acompletion

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
        logger_callback: Optional[Callable[[str], None]] = None,
        # Allow overriding provider at agent instantiation
        llm_provider_override: Optional[Literal['google', 'openrouter']] = None 
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
        """
        self.settings = settings
        self.api_keys = api_keys
        # Determine the effective LLM provider
        self.llm_provider = llm_provider_override or settings.llm_provider
        
        # Use a proper logger instance for the agent
        self.logger = logging.getLogger(f"DeepResearchAgent_{id(self)}")
        if settings.scraper_debug:
            # If debug is on, set level to DEBUG and add a handler if not already configured
            if not self.logger.hasHandlers():
                 handler = logging.StreamHandler()
                 formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                 handler.setFormatter(formatter)
                 self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)
        else:
             # Otherwise, set to INFO or higher (or keep default)
             self.logger.setLevel(logging.INFO) # Or WARNING, ERROR as needed
             # Add handler if needed for non-debug levels
             if not self.logger.hasHandlers():
                 handler = logging.StreamHandler()
                 formatter = logging.Formatter('%(levelname)s - %(message)s') # Simpler format for INFO
                 handler.setFormatter(formatter)
                 self.logger.addHandler(handler)

        # Keep the callback if provided, but use the logger for internal agent logging
        self._external_log_callback = logger_callback 
        
        # Internal log method now uses the logger
        self.log = self.logger.info # Default to info level logging
        if settings.scraper_debug:
             self.log = self.logger.debug # Use debug level if verbose

        # Log initial setup using the new logger
        self.log(f"Initializing Agent with LLM Provider: {self.llm_provider}")

        # --- Determine effective LLM configurations --- 
        # Start with defaults from settings
        effective_planner_config = planner_llm_override or settings.default_planner_llm
        effective_summarizer_config = summarizer_llm_override or settings.default_summarizer_llm
        effective_writer_config = writer_llm_override or settings.default_writer_llm

        # Set the correct model name based on the provider for each role
        # The LLMConfig object is mutable, so we update its model field
        effective_planner_config.model = settings.get_model_name('planner')
        effective_summarizer_config.model = settings.get_model_name('summarizer')
        effective_writer_config.model = settings.get_model_name('writer')

        # Finalize LiteLLM parameters using the helper, passing the provider and keys
        self.planner_llm_config = get_litellm_params(
            effective_planner_config, self.llm_provider, self.api_keys
        )
        self.summarizer_llm_config = get_litellm_params(
            effective_summarizer_config, self.llm_provider, self.api_keys
        )
        self.writer_llm_config = get_litellm_params(
            effective_writer_config, self.llm_provider, self.api_keys
        )

        # --- Initialize other components --- 
        self.chunker = Chunker() # Initialize Chunker 
        self.scraper_strategies = scraper_strategies_override or settings.scraper_strategies
        self.scraper = WebScraper(
            strategies=self.scraper_strategies,
            debug=settings.scraper_debug
        )
        self.serper_config = SerperConfig(
            api_key=api_keys.serper_api_key.get_secret_value(),
            base_url=str(settings.serper_base_url), 
            default_location=settings.serper_default_location,
            timeout=settings.serper_timeout
        )
        self.reranker_model = settings.reranker_model
        self.together_api_key = api_keys.together_api_key.get_secret_value()

        # Workflow parameters from settings
        self.max_initial_search_tasks = max_search_tasks_override if max_search_tasks_override is not None else settings.max_initial_search_tasks
        self.max_refinement_iterations = settings.max_refinement_iterations

        # Logging setup (moved provider check up earlier)
        self.verbose = settings.scraper_debug # Tie verbose logging to scraper_debug for now
        
        # Initialize usage trackers (now as dictionaries with roles)
        self.token_usage = {
            'planner': Counter(),
            'summarizer': Counter(),
            'writer': Counter(),
            'total': Counter() # Keep a total counter as well
        }
        self.estimated_cost = {
            'planner': 0.0,
            'summarizer': 0.0,
            'writer': 0.0,
            'total': 0.0 # Keep a total cost as well
        }
        self.serper_queries_used = 0

        self.log("DeepResearchAgent initialized.")
        self.log(f"- LLM Provider: {self.llm_provider}")
        self.log(f"- Planner LLM Config: {self.planner_llm_config}") # Now shows finalized params
        self.log(f"- Summarizer LLM Config: {self.summarizer_llm_config}")
        self.log(f"- Writer LLM Config: {self.writer_llm_config}")
        self.log(f"- Reranker Model: {self.reranker_model}")
        self.log(f"- Chunker: {type(self.chunker).__name__}")
        self.log(f"- Scraper Strategies: {self.scraper_strategies}")
        self.log(f"- Workflow Params: max_search_tasks={self.max_initial_search_tasks}, max_iterations={self.max_refinement_iterations}")

    # Helper to update usage (Replaced by direct updates after service call)
    # def _update_usage(self, response):
    #     if hasattr(response, 'usage') and response.usage is not None:
    #         self.token_usage['completion_tokens'] += response.usage.completion_tokens
    #         self.token_usage['prompt_tokens'] += response.usage.prompt_tokens
    #         self.token_usage['total_tokens'] += response.usage.total_tokens
    #         if hasattr(response, 'cost') and isinstance(response.cost, dict) and 'total_cost' in response.cost:
    #              current_cost = response.cost['total_cost']
    #              self.estimated_cost += current_cost
    #              self.log(f"    LLM call cost: ${current_cost:.6f}. Cumulative cost: ${self.estimated_cost:.6f}")
    #         self.log(f"    Tokens Used: Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")
    #         self.log(f"    Cumulative Tokens: {self.token_usage['total_tokens']}")
    #     else:
    #         self.log("    Usage/cost information not available in LLM response.")
    
    def _log_and_update_usage(self, role: Literal['planner', 'summarizer', 'writer'], usage_info: Optional[Dict[str, int]], cost_info: Optional[Dict[str, float]]):
        """Logs usage and cost from the service call and updates agent totals for the specified role."""
        if role not in self.token_usage:
            self.log(f"Error: Invalid role '{role}' passed to _log_and_update_usage. Skipping update.")
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
            
            self.log(f"    [{role.upper()}] Tokens Used: Prompt={usage_info['prompt_tokens']}, Completion={usage_info['completion_tokens']}, Total={usage_info['total_tokens']}")
            self.log(f"    [{role.upper()}] Cumulative Role Tokens: {role_token_usage['total_tokens']}")
            self.log(f"    Cumulative Total Tokens: {total_token_usage['total_tokens']}")
        else:
             self.log(f"    [{role.upper()}] Token usage information not available for this call.")

        if cost_info and 'total_cost' in cost_info:
            current_cost = cost_info['total_cost']
            
            # Update role-specific cost
            self.estimated_cost[role] += current_cost
            # Update total cost
            self.estimated_cost['total'] += current_cost
            
            self.log(f"    [{role.upper()}] LLM call cost: ${current_cost:.6f}. Cumulative Role Cost: ${self.estimated_cost[role]:.6f}")
            self.log(f"    Cumulative Total Cost: ${self.estimated_cost['total']:.6f}")
        else:
             self.log(f"    [{role.upper()}] Cost information not available for this call.")
             
    async def _extract_search_request(self, text: str) -> Optional[SearchRequest]:
        """Extracts and validates search requests from the writer/refiner output."""
        # Use regex to find the new tag format
        match = re.search(r'<search_request query=["\']([^"\']*)["\'](?:\s*/?)?>', text, re.IGNORECASE)
        if not match:
            return None
        
        query = match.group(1).strip()
        
        try:
            # Validate using the SearchRequest schema (only requires query)
            return SearchRequest(query=query)
        except ValidationError as e:
            self.log(f"Warning: Invalid search request format: {e}")
            return None
            
    async def _call_refiner_llm(self, previous_draft: str, search_query: str, new_chunks: list[dict[str, str]]) -> str:
        """Calls the Refiner LLM (using Summarizer config) to revise the draft."""
        self.log(f"  Calling Refiner LLM ({self.summarizer_llm_config.get('model')}) to incorporate info for query: '{search_query}'")
        try:
            refiner_messages = get_refiner_prompt(
                previous_draft=previous_draft,
                search_query=search_query,
                new_chunks=new_chunks
            )
            input_chars = sum(len(m.get('content', '')) for m in refiner_messages if isinstance(m.get('content'), str))
            self.log(f"    - Refiner input contains {len(new_chunks)} new chunks.")
            self.log(f"    - Estimated refiner input size: {input_chars} characters.")
            
            response, usage_info, cost_info = await call_litellm_acompletion(
                messages=refiner_messages,
                # Use summarizer config as the "Refiner" LLM
                llm_config=self.summarizer_llm_config, 
                num_retries=3,
                logger_callback=self.logger
            )
            # Log usage under the 'summarizer' role as it uses that config
            self._log_and_update_usage('summarizer', usage_info, cost_info) 

            # Log raw output for debugging
            raw_content = "(Response or content not available)"
            if response and response.choices and response.choices[0].message:
                 raw_content = response.choices[0].message.content
            self.log(f"    Raw Refiner Output Content: >>>\n{raw_content}\n<<<" ) 

            if response is None or not response.choices or not response.choices[0].message:
                self.log("Error: Refiner LLM call failed or returned structurally invalid response.")
                return "" # Return empty string on failure to revise
            
            revised_draft = raw_content.strip()
            if not revised_draft:
                 self.log("Warning: Refiner LLM returned empty or whitespace-only content.")
                 return "" # Return empty string
            else:
                 self.log(f"Refiner revised draft ({len(revised_draft)} chars).")
                 return revised_draft

        except Exception as e:
            self.log(f"Error during Refiner LLM call: {e}", exc_info=True)
            return "" # Return empty string on error

    # --- Helper for input estimation --- 
    def _estimate_writer_input_chars(self, user_query, writing_plan, source_summaries):
        """Estimates the character count for the writer prompt."""
        # Base prompt structure estimation (approximate)
        # System prompt is large, user templates have placeholders
        # Rough estimate for fixed parts + query + plan
        base_chars = len(_WRITER_SYSTEM_PROMPT_BASE) + len(user_query) + len(json.dumps(writing_plan)) + 200 # Extra buffer for template text
        
        # Estimate formatted summaries size
        # Use the actual formatting logic for accurate sizing
        formatted_summaries_str = format_summaries_for_prompt_template(source_summaries)
        summaries_chars = len(formatted_summaries_str)
        
        return base_chars + summaries_chars

    async def run_deep_research(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the deep research process.

        Args:
            user_query: The user's research query.

        Returns:
            A dictionary containing the final report and usage statistics,
            matching the structure expected by ResearchResponse.
        """
        # Reset counters at the start of each run
        self.token_usage = {
            'planner': Counter(),
            'summarizer': Counter(),
            'writer': Counter(),
            'total': Counter() # Keep a total counter as well
        }
        self.estimated_cost = {
            'planner': 0.0,
            'summarizer': 0.0,
            'writer': 0.0,
            'total': 0.0 # Keep a total cost as well
        }
        self.serper_queries_used = 0

        self.log("--- Starting Deep Research for query: '{}' ---".format(user_query))

        # Initialize result dictionary
        result = {
            "report": "",
            "llm_token_usage": self.token_usage,
            "estimated_llm_cost": self.estimated_cost,
            "serper_queries_used": self.serper_queries_used
        }

        # == Step 1: Planning Phase ==
        self.log("\n[Step 1/8] Calling Planner LLM...")
        try:
            planner_messages = get_planner_prompt(
                user_query=user_query, 
                max_search_tasks=self.max_initial_search_tasks # Pass the configured max value
            )
            self.log(f"Calling Planner LLM ({self.planner_llm_config.get('model')}) with max_search_tasks={self.max_initial_search_tasks}...")
            
            # Use the new service function for LiteLLM calls
            response, usage_info, cost_info = await call_litellm_acompletion(
                messages=planner_messages,
                llm_config=self.planner_llm_config,
                response_pydantic_model=PlannerOutput, # Pass Pydantic model for parsing
                num_retries=3,
                logger_callback=self.logger # Pass agent logger to service
            )
            # Log usage/cost after the call
            self._log_and_update_usage('planner', usage_info, cost_info)

            # Check if response is valid before proceeding
            if response is None or not response.choices or not response.choices[0].message:
                raise ValueError("Planner LLM call failed or returned an invalid response.")

            # Access the parsed content directly
            # LiteLLM adds a ._response_format_output attribute for Pydantic models
            if hasattr(response.choices[0].message, '_response_format_output') and response.choices[0].message._response_format_output:
                planner_output_obj = response.choices[0].message._response_format_output
                self.log("✓ Planner output parsed and validated by LiteLLM.")
            else:
                 # Fallback or handle case where structured output isn't directly available 
                 # This might happen if the model doesn't fully support it or LiteLLM had issues
                 self.log("Warning: Structured output not directly available in response, attempting manual parse.")
                 raw_content = response.choices[0].message.content
                 try:
                    # --- Add markdown stripping logic here --- 
                    cleaned_text = raw_content.strip()
                    if "```json" in cleaned_text or "```" in cleaned_text:
                        self.log("Applying markdown stripping in fallback...")
                        cleaned_text = cleaned_text.replace("```json", "```")
                        parts = cleaned_text.split("```")
                        if len(parts) >= 3:
                            cleaned_text = parts[1].strip()
                            self.log(f"Fallback: Extracted content from code block. Length: {len(cleaned_text)}")
                        else:
                            self.log("Warning: Incomplete code block markers found in fallback content, attempting parse on original.")
                            # If markers are incomplete, try parsing original content just in case
                            cleaned_text = raw_content.strip() 
                    else:
                        # No markers found, use original stripped content
                        cleaned_text = raw_content.strip()
                    # --- End stripping logic --- 
                    
                    self.log(f"Fallback: Attempting to validate content: {cleaned_text[:200]}...")
                    # Attempt manual validation as a fallback on the *cleaned* text
                    planner_output_obj = PlannerOutput.model_validate_json(cleaned_text)
                    self.log("✓ Planner output validated via manual fallback.")
                 except (json.JSONDecodeError, ValidationError) as fallback_e:
                    self.log(f"Error: Fallback validation failed: {fallback_e}")
                    # Log the text we *tried* to parse
                    self.log(f"Content attempted in fallback validation: {cleaned_text}")
                    raise ValueError("Planner LLM failed to return valid structured output.") from fallback_e

            # Convert Pydantic object back to dict for the rest of the workflow
            planner_output = planner_output_obj.model_dump()
            search_tasks = planner_output['search_tasks']
            writing_plan = planner_output['writing_plan']

            self.log("Planner LLM call complete.")
            self.log(f"- Planned Search Tasks ({len(search_tasks)}): {json.dumps(search_tasks, indent=2)}")
            self.log(f"- Writing Plan: {json.dumps(writing_plan, indent=2)}")

        except Exception as e:
            self.log(f"Error during Planning Phase: {e}")
            result["report"] = f"Failed during planning phase: {e}"
            # Update usage before returning
            result["llm_token_usage"] = dict(self.token_usage)
            result["estimated_llm_cost"] = self.estimated_cost
            result["serper_queries_used"] = self.serper_queries_used
            return result

        # == Step 2: Initial Search Execution ==
        self.log("\n[Step 2/8] Executing initial Serper searches...")
        if not search_tasks:
            self.log("No search tasks planned. Skipping search execution.")
            raw_search_results = []
        else:
            self.log(f"Executing {len(search_tasks)} Serper search tasks...")
            # NOTE: execute_batch_serper_search is synchronous in its current form.
            # If it becomes async, use 'await' here.
            search_result_obj: SearchResult[List[Dict[str, Any]]] = execute_batch_serper_search(
                search_tasks=search_tasks,
                config=self.serper_config
            )
            self.serper_queries_used += len(search_tasks)

            if search_result_obj.failed:
                self.log(f"Error during Serper batch search: {search_result_obj.error}")
                result["report"] = f"Failed to execute search: {search_result_obj.error}"
                result["llm_token_usage"] = dict(self.token_usage)
                result["estimated_llm_cost"] = self.estimated_cost
                result["serper_queries_used"] = self.serper_queries_used
                return result

            raw_search_results = search_result_obj.data or []
            self.log(f"Initial Serper searches complete. Received results for {len(raw_search_results)} tasks.")

        # == Step 3: Initial Content Processing & Source Reranking ==
        self.log("\n[Step 3/8] Processing sources and reranking...")
        unique_sources_map: Dict[str, Dict[str, Any]] = {}
        for task_result in raw_search_results:
            for source in task_result.get('organic', []):
                link = source.get('link')
                if link and link not in unique_sources_map:
                    unique_sources_map[link] = {
                        'title': source.get('title', 'No Title'),
                        'link': link,
                        'snippet': source.get('snippet', ''),
                        # Store original task index if needed later, example:
                        # 'task_index': raw_search_results.index(task_result) 
                    }

        all_sources_list = list(unique_sources_map.values()) # Keep the original list for indexing
        self.log(f"Consolidated {len(all_sources_list)} unique sources from search results.")

        # reranked_results will hold dicts with {'index': original_index, 'score': score}
        # sorted by score descending, only including those >= 0.2
        reranked_results: List[Dict[str, Any]] = [] 

        if not all_sources_list:
            self.log("Warning: No unique sources found to process.")
        else:
            try:
                # Prepare documents using title + snippet for initial reranking
                docs_for_rerank = [f"{s.get('title', '')} {s.get('snippet', '')}" for s in all_sources_list]

                self.log(f"Reranking {len(docs_for_rerank)} sources using Together Rerank API ({self.reranker_model})... Filtering below 0.2 relevance.")
                
                # Call the reranker API - this returns results already filtered by the threshold
                api_reranked_results = rerank_with_together_api(
                    query=user_query,
                    documents=docs_for_rerank,
                    relevance_threshold=0.2, # Filter sources below 0.2
                    model=self.reranker_model,
                    api_key=self.together_api_key,
                    verbose=self.verbose 
                )

                if not api_reranked_results:
                    self.log("Warning: Reranking returned no results above the 0.2 threshold.")
                else:
                    # Store the filtered and sorted results
                    reranked_results = api_reranked_results 
                    self.log(f"Reranking complete. {len(reranked_results)} sources passed 0.2 threshold.")
                    # Log the sources that passed the filter
                    self.log(f"Sources passing 0.2 threshold ({len(reranked_results)} total, ordered by relevance):")
                    for i, res in enumerate(reranked_results):
                        source_info = all_sources_list[res['index']]
                        self.log(f"  Rank {i+1}: {source_info.get('title')} ({source_info.get('link')}) - Score: {res['score']:.4f}")

            except Exception as e:
                self.log(f"Error during source reranking: {e}. Cannot proceed with fetching/summarization.")
                result["report"] = f"Failed during source reranking phase: {e}"
                result["llm_token_usage"] = dict(self.token_usage)
                result["estimated_llm_cost"] = self.estimated_cost
                result["serper_queries_used"] = self.serper_queries_used
                return result

        # == Step 4 & 5: Sequential Fetch and Process (Summarize or Chunk/Rerank) ==
        self.log("\n[Step 4&5 / 8] Sequentially fetching and processing filtered sources...")
        source_summaries = [] # Collect final summaries and relevant chunks

        if not reranked_results:
            self.log("No sources passed the relevance filter. Skipping fetching and processing.")
        else:
            # Determine split point and calculate dynamic max_tokens for summarizer
            num_filtered = len(reranked_results)
            # Original split point is half, but cap summarization at 10 sources
            original_split_point = max(1, num_filtered // 2) if num_filtered > 0 else 0 
            num_sources_to_summarize = min(original_split_point, 10) 
            
            total_summary_token_budget = 400000 # User increased budget
            max_tokens_per_summary = 0 # Default if no summaries needed
            if num_sources_to_summarize > 0:
                # Calculate based on the *actual* number being summarized (max 10)
                max_tokens_per_summary = max(500, total_summary_token_budget // num_sources_to_summarize) # Ensure at least 500 tokens
            
            self.log(f"Processing {num_filtered} filtered sources:")
            self.log(f"- Top {num_sources_to_summarize} sources -> Full Summarization (max_tokens_per_summary: {max_tokens_per_summary})")
            self.log(f"- Remaining {num_filtered - num_sources_to_summarize} sources -> Chunk & Rerank") # Log based on actual split

            # Loop through the FILTERED and SORTED results
            for i, res_info in enumerate(reranked_results):
                rank = i + 1
                original_index = res_info['index']
                source_info = all_sources_list[original_index]
                link = source_info.get('link')
                title = source_info.get('title', 'No Title')
                
                self.log(f"\nProcessing Source Rank {rank}/{num_filtered}: {title} ({link})")

                if not link:
                    self.log("  Skipping source: Missing link.")
                    continue

                # --- 4a: Fetch Content (for this source) ---
                self.log(f"  Fetching content...")
                scraped_content = None # Initialize before try block
                try:
                    scrape_result_dict = await self.scraper.scrape(url=link)                     
                    strategy_used = "None"
                    if 'no_extraction' in scrape_result_dict and scrape_result_dict['no_extraction'].success:
                        scraped_content = scrape_result_dict['no_extraction'].content
                        strategy_used = 'no_extraction'
                    else:
                        for strategy_name, extract_result in scrape_result_dict.items():
                            if extract_result.success and extract_result.content:
                                scraped_content = extract_result.content
                                strategy_used = strategy_name
                                break 
                    
                    if not scraped_content:
                         self.log(f"  Failed to fetch or extract content (Strategy: {strategy_used}). Skipping source.")
                         continue                    
                    self.log(f"  Successfully fetched content ({len(scraped_content)} chars) using strategy: {strategy_used}")
                except Exception as e_scrape:
                    self.log(f"  Error scraping source {link}: {e_scrape}. Skipping source.")
                    continue

                # --- 5a: Process Content (Summarize or Chunk/Rerank) ---
                if rank <= num_sources_to_summarize:
                    # --- Top Group (Max 10): Summarize Sequentially ---
                    self.log(f"  Rank {rank} <= {num_sources_to_summarize}: Summarizing sequentially...")
                    try:
                        summarizer_messages = get_summarizer_prompt(
                            user_query=user_query,
                            source_title=title,
                            source_link=link,
                            source_content=scraped_content 
                        )
                        self.log(f"    Calling Summarizer LLM ({self.summarizer_llm_config.get('model')})...")
                        
                        # Add dynamic max_tokens to summarizer config
                        current_summarizer_config = self.summarizer_llm_config.copy()
                        if max_tokens_per_summary > 0: # Only add if calculated
                           current_summarizer_config['max_tokens'] = max_tokens_per_summary
                           self.log(f"      Setting max_tokens: {max_tokens_per_summary}")
                        else:
                            self.log("      Not setting max_tokens for summarizer (0 sources to summarize).")
                            
                        # Await the summarization call directly
                        response, usage_info, cost_info = await call_litellm_acompletion(
                            messages=summarizer_messages,
                            llm_config=current_summarizer_config, # Use config potentially with max_tokens
                            num_retries=3,
                            logger_callback=self.logger
                        )
                        self._log_and_update_usage('summarizer', usage_info, cost_info)
                        
                        if response is None or not response.choices or not response.choices[0].message:
                            self.log("    Warning: Summarizer call failed or returned empty response.")
                            summary_text = "" # Ensure summary_text is empty string on failure
                        else:
                            summary_text = response.choices[0].message.content.strip()
                            
                        if summary_text:
                            # Store summary info as a dict, including original title
                            if title and link: # Basic validation
                                source_summaries.append({
                                    'title': title,
                                    'link': link,
                                    'summary': summary_text,
                                    'original_title': title # Add original_title for consistency
                                })
                                self.log(f"    + Summary generated and stored ({len(summary_text)} chars).")
                            else:
                                self.log(f"    Warning: Skipping summary due to missing title/link.")
                        else:
                            self.log(f"    Warning: Summarizer returned empty content. Skipping.")
                    except Exception as e_sum:
                         self.log(f"    Error during summarization for {link}: {e_sum}. Skipping.")
                
                else:
                    # --- Remaining Sources: Chunk & Rerank (remains sequential) ---
                    self.log(f"  Rank {rank} > {num_sources_to_summarize}: Processing with chunk & rerank.")
                    try:
                        # Initialize chunker with larger chunk size
                        chunker = Chunker(chunk_size=2048, chunk_overlap=100, min_chunk_size=256)
                        self.log(f"    Using chunker with size={chunker.chunk_size}, overlap={chunker.chunk_overlap}")
                        
                        # Prepare single document for chunking
                        doc_to_chunk = [{'title': title, 'link': link, 'content': scraped_content}]
                        
                        self.log(f"    Chunking document...")
                        chunked_docs_all = chunker.chunk_and_label(doc_to_chunk)
                        self.log(f"    Created {len(chunked_docs_all)} total chunks.")
                        
                        if not chunked_docs_all:
                            self.log("    No chunks generated. Skipping reranking for this source.")
                            continue

                        # Limit chunks to 1024 for the reranker API
                        max_chunks_for_api = 1024
                        if len(chunked_docs_all) > max_chunks_for_api:
                            self.log(f"    Warning: Generated {len(chunked_docs_all)} chunks, exceeding reranker limit of {max_chunks_for_api}. Truncating list.")
                            chunked_documents = chunked_docs_all[:max_chunks_for_api]
                        else:
                            chunked_documents = chunked_docs_all
                        chunk_contents = [doc['content'] for doc in chunked_documents]

                        if chunk_contents:
                            # Rerank chunks with a higher threshold
                            self.log(f"    Reranking {len(chunk_contents)} chunks (threshold=0.5)...")
                            chunk_rerank_results = rerank_with_together_api(
                                query=user_query,
                                documents=chunk_contents,
                                relevance_threshold=0.5, 
                                model=self.reranker_model,
                                api_key=self.together_api_key,
                                verbose=self.verbose
                            )
                            if chunk_rerank_results:
                                top_chunk_indices = [result['index'] for result in chunk_rerank_results]
                                # Map indices back to the (potentially truncated) chunked_documents list
                                relevant_chunks = [chunked_documents[idx] for idx in top_chunk_indices if idx < len(chunked_documents)]
                                self.log(f"    Selected {len(relevant_chunks)} relevant chunks passing 0.5 threshold.")
                                for chunk in relevant_chunks:
                                    # Use chunk content directly as the 'summary'
                                    chunk_summary_data = {
                                         'title': f"{chunk['title']} (chunk {chunk.get('chunk_id', '')})",
                                         'link': chunk['link'],
                                         'summary': chunk['content'] ,
                                         'original_title': chunk['title'] # Add the original title
                                    }
                                    # Basic validation (ensure summary content exists)
                                    if chunk_summary_data['summary'] and chunk_summary_data['link']:
                                        source_summaries.append(chunk_summary_data)
                                        self.log(f"      + Added chunk {chunk.get('chunk_id', '')} from {chunk['title']}")
                                    else:
                                         self.log(f"      - Skipping empty or linkless chunk {chunk.get('chunk_id', '')}")
                            else:
                                self.log("    No chunks passed the 0.5 relevance threshold after reranking.")
                        else:
                            self.log("    No valid chunk content generated for reranking.")
                    except Exception as e_chunk:
                         self.log(f"    Error during chunking/reranking for {link}: {e_chunk}. Skipping.")
            
            # Removed the concurrent summarization block            

        self.log(f"\nSource fetching and processing complete. Generated {len(source_summaries)} total source materials (summaries + chunks).")

        # == Step 6: Initial Report Generation ==
        self.log("\n[Step 6/8] Generating initial report...")
        report_draft = ""

        # Check writing plan and source materials (summaries + chunks)
        if not writing_plan:
            self.log("Cannot generate report: Missing writing plan from Planner.")
            result["report"] = "Failed: No writing plan was generated."
            # Convert usage breakdown for the return format
            usage_breakdown_dict = {role: dict(counts) for role, counts in self.token_usage.items()}
            cost_breakdown_dict = {role: cost for role, cost in self.estimated_cost.items()}
            result["llm_token_usage"] = usage_breakdown_dict.get('total', {})
            result["estimated_llm_cost"] = cost_breakdown_dict.get('total', 0.0)
            result["serper_queries_used"] = self.serper_queries_used
            result["llm_usage_breakdown"] = {
                 "tokens": {role: usage for role, usage in usage_breakdown_dict.items() if role != 'total'},
                 "cost": {role: cost for role, cost in cost_breakdown_dict.items() if role != 'total'}
            }
            return result
        if not source_summaries:
            self.log("Warning: No source materials (summaries or relevant chunks) available. Report generation might be limited.")
            # Proceed, but writer will have less to work with

        try:
            # Use the potentially filtered list
            writer_messages = get_writer_initial_prompt(
                user_query=user_query,
                writing_plan=writing_plan,
                source_summaries=source_summaries 
            )
            
            # --- Log Input Size Estimate Unconditionally ---
            input_chars = sum(len(m.get('content', '')) for m in writer_messages if isinstance(m.get('content'), str))
            self.log(f"Calling Writer LLM ({self.writer_llm_config.get('model')}) for initial draft.")
            self.log(f"  - Input contains {len(source_summaries)} source materials.")
            self.log(f"  - Estimated input size: {input_chars} characters.")
            # --- End Input Size Estimate Logging ---
            
            response, usage_info, cost_info = await call_litellm_acompletion(
                messages=writer_messages,
                llm_config=self.writer_llm_config, # Use original config without max_tokens
                num_retries=3,
                logger_callback=self.logger
            )
            self._log_and_update_usage('writer', usage_info, cost_info)
            
            # --- Log Raw Writer Output --- 
            raw_content = "(Response or content not available)" # Default if checks fail
            if response and response.choices and response.choices[0].message:
                 raw_content = response.choices[0].message.content
            self.log(f"Raw Writer Output Content (Initial Draft): >>>\n{raw_content}\n<<<" ) 
            # --- End Raw Output Log ---
            
            if response is None or not response.choices or not response.choices[0].message:
                # Handle potential failure from LLM call
                self.log("Error: Writer LLM call failed or returned structurally invalid response during initial draft.")
                raise ValueError("Writer LLM failed to generate initial draft structure.")
            
            # Assign potentially empty content *after* logging raw content
            report_draft = raw_content.strip() 
            
            if not report_draft:
                 self.log("Warning: Writer LLM returned empty or whitespace-only content for initial draft.")
            else:
                 self.log(f"Initial report draft generated ({len(report_draft)} chars).")

        except Exception as e:
            self.log(f"Error during Initial Report Generation Phase: {e}")
            result["report"] = f"Failed during initial report generation phase: {e}"
            # Convert usage breakdown for the return format
            usage_breakdown_dict = {role: dict(counts) for role, counts in self.token_usage.items()}
            cost_breakdown_dict = {role: cost for role, cost in self.estimated_cost.items()}
            result["llm_token_usage"] = usage_breakdown_dict.get('total', {})
            result["estimated_llm_cost"] = cost_breakdown_dict.get('total', 0.0)
            result["serper_queries_used"] = self.serper_queries_used
            result["llm_usage_breakdown"] = {
                 "tokens": {role: usage for role, usage in usage_breakdown_dict.items() if role != 'total'},
                 "cost": {role: cost for role, cost in cost_breakdown_dict.items() if role != 'total'}
            }
            return result

        # all_summaries_unfiltered now holds both LLM summaries and selected raw chunks
        all_summaries_unfiltered = list(source_summaries)

        # == Step 7: Refinement Loop (Max `max_refinement_iterations`, Chunk Only, Direct Search) ==
        self.log(f"\n[Step 7/{8+self.max_refinement_iterations}] Starting refinement loop (max {self.max_refinement_iterations} iterations)...")
        current_iteration = 0
        while current_iteration < self.max_refinement_iterations:
            iteration_num = current_iteration + 1
            self.log(f"-- Refinement Iteration {iteration_num}/{self.max_refinement_iterations} --")
            
            search_request = await self._extract_search_request(report_draft) # Use new function
            if not search_request:
                self.log("No valid search request found in the draft. Ending refinement loop.")
                break # Exit loop if no request
                
            self.log(f"LLM requested search: '{search_request.query}'")

            # --- Step 7a: Execute Direct Search --- 
            refinement_search_task = [{
                "query": search_request.query, # Use the query from the LLM
                "endpoint": "/search", # Default to general search for refinement
                "num_results": 10 # Get a decent number for chunking
            }]
            self.log(f"  [7a] Executing direct search for: '{search_request.query}'")
            refinement_search_result_obj: SearchResult[List[Dict[str, Any]]] = execute_batch_serper_search(
                search_tasks=refinement_search_task,
                config=self.serper_config
            )
            self.serper_queries_used += len(refinement_search_task)

            if refinement_search_result_obj.failed:
                self.log(f"Error during refinement search: {refinement_search_result_obj.error}. Skipping refinement for this iteration.")
                # Don't break the loop entirely, maybe next iteration works if writer asks again
                current_iteration += 1 
                continue 

            refinement_raw_results = refinement_search_result_obj.data or []
            if not refinement_raw_results or not refinement_raw_results[0].get('organic'):
                 self.log("Refinement search yielded no results. Skipping refinement for this iteration.")
                 current_iteration += 1
                 continue
            self.log(f"Refinement search complete. Processing results...")
            
            # --- Step 7b: Process Results (Chunk & Rerank ONLY) ---
            self.log("  [7b] Processing refinement results (Chunk & Rerank only)...")
            new_refinement_chunks = [] # Collect relevant chunks
            refinement_sources = refinement_raw_results[0].get('organic', []) # Results from the single query

            for source in refinement_sources:
                link = source.get('link')
                title = source.get('title', 'No Title')
                self.log(f"    Processing source: {title} ({link})")
                if not link: continue
                
                # Fetch
                ref_scraped_content = None
                try:
                    # ... (Standard scraping logic) ...
                    scrape_result_dict = await self.scraper.scrape(url=link)
                    strategy_used = "None" # Simplified scraping logic representation
                    if 'no_extraction' in scrape_result_dict and scrape_result_dict['no_extraction'].success:
                        ref_scraped_content = scrape_result_dict['no_extraction'].content
                        strategy_used = 'no_extraction'
                    else:
                        for strategy_name, extract_result in scrape_result_dict.items():
                            if extract_result.success and extract_result.content:
                                ref_scraped_content = extract_result.content
                                strategy_used = strategy_name
                                break
                    if not ref_scraped_content: continue
                    self.log(f"      Successfully fetched refinement content ({len(ref_scraped_content)} chars) using strategy: {strategy_used}")
                except Exception as e_ref_scrape:
                    self.log(f"      Error scraping refinement source {link}: {e_ref_scrape}. Skipping source.")
                    continue
                    
                # Chunk & Rerank
                try:
                    chunker = Chunker(chunk_size=2048, overlap=100, min_chunk_size=256)
                    doc_to_chunk = [{'title': title, 'link': link, 'content': ref_scraped_content}]
                    chunked_docs_all = chunker.chunk_and_label(doc_to_chunk)
                    if not chunked_docs_all: continue

                    max_chunks_for_api = 1024 
                    chunked_documents = chunked_docs_all[:max_chunks_for_api]
                    chunk_contents = [doc['content'] for doc in chunked_documents]

                    if chunk_contents:
                        chunk_rerank_results = rerank_with_together_api(
                            query=search_request.query, # Rerank based on the requested query
                            documents=chunk_contents,
                            relevance_threshold=0.5, 
                            model=self.reranker_model,
                            api_key=self.together_api_key,
                            verbose=self.verbose
                        )
                        if chunk_rerank_results:
                            top_chunk_indices = [result['index'] for result in chunk_rerank_results]
                            relevant_chunks = [chunked_documents[idx] for idx in top_chunk_indices if idx < len(chunked_documents)]
                            self.log(f"      Selected {len(relevant_chunks)} relevant chunks.")
                            for chunk in relevant_chunks:
                                chunk_data = {
                                    'title': f"{chunk['title']} (chunk {chunk.get('chunk_id', '')})",
                                    'link': chunk['link'],
                                    'summary': chunk['content'],
                                    'original_title': chunk['title']
                                }
                                if chunk_data['summary'] and chunk_data['link']:
                                    new_refinement_chunks.append(chunk_data)
                except Exception as e_ref_chunk:
                     self.log(f"      Error during refinement chunking/reranking for {link}: {e_ref_chunk}. Skipping source.")
            # --- End Source Loop for Refinement --- 
            
            self.log(f"  Generated {len(new_refinement_chunks)} new relevant chunks for query '{search_request.query}'.")

            # --- Step 7c: Call Refiner LLM --- 
            if new_refinement_chunks:
                self.log("  [7c] Calling Refiner LLM to revise draft...")
                # No input size check needed here as Refiner prompt is much smaller
                revised_draft = await self._call_refiner_llm(
                    previous_draft=report_draft,
                    search_query=search_request.query,
                    new_chunks=new_refinement_chunks
                )
                
                if revised_draft:
                    report_draft = revised_draft # Update the main draft
                    # Add new chunks to unfiltered list *only if revision was successful*
                    # This prevents adding useless chunks if refiner fails
                    all_summaries_unfiltered.extend(new_refinement_chunks)
                else:
                    self.log("Refiner LLM failed to produce a revised draft. Keeping previous draft.")
                    # Do not add chunks to all_summaries_unfiltered if refinement failed
            else:
                 self.log("    No new relevant chunks generated. Skipping revision call.")
                 
            # --- End Refinement Iteration --- 
            current_iteration += 1
            # Loop continues if current_iteration < max_refinement_iterations
            
        if current_iteration >= self.max_refinement_iterations:
             self.log(f"Reached maximum refinement iterations ({self.max_refinement_iterations}).")

        self.log(f"-- Refinement loop complete (Completed {current_iteration} iterations). --")

        # == Step 8: Final Output ==
        self.log("\n[Step 8/8] Finalizing report...")
        # Use the `all_summaries_unfiltered` list which contains both LLM summaries and raw chunks
        cited_links = set()
        reference_list_items = [] # Use a list to maintain order somewhat

        if all_summaries_unfiltered: 
            for summary_info in all_summaries_unfiltered:
                link = summary_info.get('link')
                original_title = summary_info.get('original_title') # Get the original title

                # Use original title if available, otherwise fall back to 'title'
                display_title = original_title if original_title else summary_info.get('title', 'Untitled') 
                
                if link and link not in cited_links:
                    reference_list_items.append(f"[{display_title}]({link})")
                    cited_links.add(link)
            
            if reference_list_items:
                reference_list_str = "\n\nReferences:\n"
                for i, item_str in enumerate(reference_list_items):
                    reference_list_str += f"{i+1}. {item_str}\n"
                
                final_report = report_draft.strip() + "\n" + reference_list_str.strip()
                self.log(f"Appended reference list with {len(reference_list_items)} unique sources to the final report.")
            else:
                final_report = report_draft # No valid references found
                self.log("No valid unique sources found for references.")

        else:
            final_report = report_draft # No summaries/chunks
            self.log("No summaries or relevant chunks generated, final report has no references.")

        # --- Log Final Usage ---
        self.log("\n--- Usage Summary ---")
        self.log(f"Total Serper Queries Used: {self.serper_queries_used}")
        self.log("LLM Token Usage:")
        for role, usage in self.token_usage.items():
            self.log(f"  - {role}:")
            for token_type, count in usage.items():
                self.log(f"    - {token_type}: {count}")
        self.log(f"Estimated LLM Cost: ${sum(self.estimated_cost.values()):.6f}")
        self.log("---------------------")

        self.log("--- Deep Research complete! --- Delivering final report.")

        # Prepare final result dictionary with breakdown
        # Convert Counters to simple dicts for JSON serialization
        usage_breakdown = {role: dict(counts) for role, counts in self.token_usage.items()}
        cost_breakdown = {role: cost for role, cost in self.estimated_cost.items()}
        
        result = {
            "report": final_report,
            # Keep totals for backward compatibility / top-level view
            "llm_token_usage": usage_breakdown.get('total', {}), 
            "estimated_llm_cost": cost_breakdown.get('total', 0.0),
            "serper_queries_used": self.serper_queries_used,
            # Add the detailed breakdown
            "llm_usage_breakdown": {
                "tokens": {role: usage for role, usage in usage_breakdown.items() if role != 'total'}, # Exclude total from here
                "cost": {role: cost for role, cost in cost_breakdown.items() if role != 'total'} # Exclude total from here
            }
        }

        return result 

    def _assemble_final_report(self, report_draft: str, all_summaries: list[dict[str, str]]) -> str:
        """Helper to assemble the final report with references."""
        self.log("\n[Step 8/8] Finalizing report...")
        cited_links = set()
        reference_list_items = []

        if all_summaries:
            for idx, summary_info in enumerate(all_summaries):
                link = summary_info.get('link')
                # Use original title if available, otherwise fall back to 'title' 
                # (Strip chunk info for display)
                display_title = summary_info.get('original_title', summary_info.get('title', 'Untitled'))
                
                # Ensure citation markers match the list used by the writer eventually
                citation_marker = f"[{idx+1}]" # Matches the format_summaries_for_prompt
                
                # Check if this marker is actually present in the report draft
                # Simple check, might need refinement for robustness (e.g., regex)
                if citation_marker in report_draft:
                    if link and link not in cited_links:
                        reference_list_items.append(f"{citation_marker} [{display_title}]({link})")
                        cited_links.add(link)
                #else:
                #    self.log(f"  Note: Citation {citation_marker} for '{display_title}' not found in final draft.")
            
            if reference_list_items:
                # Sort references numerically based on marker
                reference_list_items.sort(key=lambda x: int(x.split(']')[0].strip('['))) 
                reference_list_str = "\n\nReferences:\n"
                # Re-number based on sorted order for final display
                for i, item_str in enumerate(reference_list_items):
                     # Extract original marker and rest of string
                     parts = item_str.split('] ', 1)
                     if len(parts) == 2:
                         #original_marker = parts[0] + ']'
                         content = parts[1]
                         reference_list_str += f"{i+1}. {content}\n"
                     else: # Fallback if split fails unexpectedly
                          reference_list_str += f"{i+1}. {item_str}\n"
                          
                final_report = report_draft.strip() + "\n" + reference_list_str.strip()
                self.log(f"Appended reference list with {len(reference_list_items)} unique, cited sources to the final report.")
            else:
                final_report = report_draft
                self.log("No valid, cited sources found for references.")
        else:
            final_report = report_draft
            self.log("No source materials available, final report has no references.")
            
        return final_report 