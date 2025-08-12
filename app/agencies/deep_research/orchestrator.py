from __future__ import annotations
import asyncio
import logging
from typing import List, Dict, Any, Optional
import tiktoken

# Import agency components
from . import agents
from . import schemas
from .config import DeepResearchConfig

# --- Import Helpers --- #
from .helpers import (
    execute_search_queries,
    rerank_search_results_helper,
    rerank_chunks_helper,
    batch_scrape_urls_helper,
    chunk_content_helper,
    format_report_citations,
    generate_reference_list
)

# --- Import Callback Handler ---
from .callbacks import WebSocketUpdateHandler

from ..services.search import SearchResult, SerperConfig # Removed SearchTask import, using schemas.SearchTask
from ..services.scraper import ExtractionResult
from app.core.config import AppSettings
# Import the new RunUsage tracker
from app.core.schemas import RunUsage, UsageStatistics, SearchTask

# --- Add Firestore DocumentReference type --- #
from google.cloud.firestore_v1.document import DocumentReference
from firebase_admin import firestore # To access SERVER_TIMESTAMP

logger = logging.getLogger(__name__)

# --- Define Constants --- #
# Max tokens allowed for the combined context (summaries + chunks) for the writer
MAX_WRITER_CONTEXT_TOKENS = 600_000

async def run_deep_research_orchestration(
    user_query: str,
    agents_collection: agents.AgencyAgents,
    config: DeepResearchConfig,
    app_settings: AppSettings,
    # Add the callback handler parameter
    update_callback: Optional[WebSocketUpdateHandler] = None,
    # Add Firestore parameters
    task_doc_ref: Optional[DocumentReference] = None,
    firestore_available: bool = False
) -> schemas.ResearchResponse:
    """
    Orchestrates the deep research multi-agent process.
    """
    # Initialize the new RunUsage tracker
    usage_tracker = RunUsage()
    all_source_materials: List[Dict[str, Any]] = [] # Will store summaries/chunks with source_ref_num
    processed_links = set() # Tracks URLs added to unique_sources
    # New: Map to store unique sources and their stable reference number
    unique_sources: Dict[str, Dict[str, Any]] = {} # link -> {"title": ..., "ref_num": ...}
    source_counter = 0 # Counter for assigning stable reference numbers

    logger.info(f"Starting deep research orchestration for query: '{user_query}'")
    if update_callback:
        await update_callback.orchestration_start()

    # === Step 1: Planning ===
    logger.info("Calling Planner Agent...")
    if update_callback:
        await update_callback.planning_start()

    try:
        # Run planner and directly get usage
        planner_result = await agents_collection.planner.run(
            agents._PLANNER_USER_MESSAGE_TEMPLATE.format(user_query=user_query)
            # No longer passing usage tracker directly to the agent run
        )
        # Update usage tracker after the call
        usage_tracker.update_agent_usage("planner", planner_result.usage())

        # Revert: Expect PlannerOutput directly now that result_type is set
        planner_output: schemas.PlannerOutput = planner_result.data

        # Type check for safety (optional, but good practice)
        if not isinstance(planner_output, schemas.PlannerOutput):
             logger.error(f"Planner agent returned unexpected type: {type(planner_output)}")
             logger.error(f"Raw data: {planner_result.data}")
             # Log the model name used for the planner agent if available
             try:
                 model_name = agents_collection.planner.model.model if hasattr(agents_collection.planner.model, 'model') else 'N/A'
                 logger.error(f"Model used: {model_name}")
             except Exception: # Catch potential issues accessing model info
                 logger.error("Could not retrieve model name.")
             raise TypeError(
                 f"Planner agent was expected to return schemas.PlannerOutput, "
                 f"but returned {type(planner_output)}. Check model compatibility and prompt compliance."
             )

        # --- Update Firestore with Plan --- #
        if firestore_available and task_doc_ref:
            try:
                plan_update_data = {
                    "writingPlan": planner_output.model_dump(mode='json'),
                    "initialSearchTaskCount": len(planner_output.search_tasks),
                    "status": "PLANNING_COMPLETE", # Or a more specific status
                    "updatedAt": firestore.SERVER_TIMESTAMP
                }
                task_doc_ref.update(plan_update_data)
                logger.info(f"Updated Firestore with research plan.")
            except Exception as fs_e:
                logger.error(f"Failed to update Firestore with plan: {fs_e}")
        # --- End Firestore Update --- #

        search_tasks: List[SearchTask] = planner_output.search_tasks
        writing_plan: schemas.WritingPlan = planner_output.writing_plan
        logger.info(f"Planner generated {len(search_tasks)} search tasks and a writing plan.")
        if update_callback:
            # Extract query strings for the callback
            query_strings = [task.query for task in search_tasks if task.query]
            plan_details = {
                "plan": {
                    "writing_plan": writing_plan.model_dump(mode='json'),
                    "search_task_count": len(search_tasks),
                    "search_queries": query_strings # Add the actual queries here
                }
            }
            await update_callback.planning_end(plan_details)
    except Exception as e:
        logger.error(f"Planner agent failed: {e}", exc_info=True)
        if update_callback:
            await update_callback.planning_error(e)
        # Return error response with current usage stats
        return schemas.ResearchResponse(
            report=f"Planner Agent Error: {e}",
            usage_statistics=usage_tracker.get_statistics()
        )

    # === Step 2: Initial Search & Rerank ===
    logger.info("Executing initial search tasks...")
    if update_callback:
        await update_callback.initial_search_start(len(search_tasks))

    initial_search_results: List[SearchResult] = []
    reranked_initial_search_results: List[SearchResult] = []
    try:
        serper_cfg = SerperConfig.from_env()
        search_results_map = await execute_search_queries(
            search_tasks=search_tasks,
            config=serper_cfg,
            logger=logger
        )
        # Increment Serper query usage
        usage_tracker.increment_serper_queries(len(search_tasks))

        # Flatten & Deduplicate
        initial_search_links_seen = set()
        temp_results = []
        for query, query_results in search_results_map.items():
            for result in query_results:
                if result.link not in initial_search_links_seen:
                    temp_results.append(result)
                    initial_search_links_seen.add(result.link)
        initial_search_results = temp_results
        logger.info(f"Initial search yielded {len(initial_search_results)} unique results before reranking.")

        # Send search end update before reranking starts
        if update_callback:
            await update_callback.initial_search_end(
                unique_result_count=len(initial_search_results),
                queries_executed=len(search_tasks)
            )

        # Rerank Search Results
        if initial_search_results:
            if update_callback:
                await update_callback.initial_rerank_start(len(initial_search_results))
            api_key = config.together_api_key.get_secret_value()
            model = config.reranker_model
            reranked_initial_search_results = await rerank_search_results_helper(
                 query=user_query,
                 search_results=initial_search_results,
                 model=model,
                 api_key=api_key,
                 threshold=config.rerank_relevance_threshold,
                 logger=logger
            )
        else:
            # If no results to rerank, send a simplified end update? Or skip?
            # Let's send an end update indicating 0 results.
            if update_callback:
                 await update_callback.initial_rerank_start(0)
            logger.info("No initial search results to rerank.")

        if not reranked_initial_search_results:
             logger.warning("Initial search and reranking yielded no results.")

    # Catch config errors separately as they prevent continuation
    except schemas.ConfigurationError as e: # Assuming ConfigurationError is in schemas or core
         logger.error(f"Search configuration error: {e}", exc_info=True)
         if update_callback:
             # Use a generic search error or a specific config error?
             await update_callback.initial_search_error(e) # Or initial_rerank_error(e)
         return schemas.ResearchResponse(report=f"Configuration Error: {e}", usage_statistics=usage_tracker.get_statistics())
    except Exception as e:
        logger.error(f"Initial search/rerank failed critically: {e}", exc_info=True)
        if update_callback:
            # Determine if error was during search or rerank based on progress?
            # For simplicity, send a generic search error for now
            await update_callback.initial_search_error(e) # Or initial_rerank_error(e)
        return schemas.ResearchResponse(report=f"Search Execution/Reranking Error: {e}", usage_statistics=usage_tracker.get_statistics())

    # --- Calculate Dynamic TOP_N and Split Reranked Results --- #
    top_results_for_summary: List[SearchResult] = []
    secondary_results_for_chunking: List[SearchResult] = []
    if reranked_initial_search_results:
        total_reranked_count = len(reranked_initial_search_results)
        top_n_calculated = min(total_reranked_count // 2, 10)
        if total_reranked_count == 1: 
            top_n_calculated = 1 
        logger.info(f"Dynamically calculated TOP_N for summary: {top_n_calculated}...")
        top_results_for_summary = reranked_initial_search_results[:top_n_calculated]
        secondary_results_for_chunking = reranked_initial_search_results[top_n_calculated:]
        logger.info(f"Split results: {len(top_results_for_summary)} for summary, {len(secondary_results_for_chunking)} for chunking/reranking.")
    else:
         logger.warning("Proceeding without any reranked search results.")

    # Send rerank end update (even if results were empty)
    if update_callback:
        await update_callback.initial_rerank_end(
            results_for_summary=len(top_results_for_summary),
            results_for_chunking=len(secondary_results_for_chunking)
        )

    # === Step 3: Source Processing ===
    logger.info("Starting source processing...")

    # --- Batch Scrape --- #
    urls_to_scrape = list(set([str(res.link) for res in top_results_for_summary] + 
                              [str(res.link) for res in secondary_results_for_chunking]))
    scraped_data: Dict[str, ExtractionResult] = {}
    if urls_to_scrape:
        logger.info(f"Batch scraping {len(urls_to_scrape)} unique URLs...")
        # --- Call scraping_start callback ---
        if update_callback:
            await update_callback.scraping_start(len(urls_to_scrape))
        # --- End call ---
        scraped_data = await batch_scrape_urls_helper(
            urls=urls_to_scrape,
            settings=app_settings,
            logger=logger
        )
    else:
        logger.info("No URLs to scrape.")
        # Optional: Send a scraping update even if none? Or skip?
        # if update_callback:
        #     await update_callback.scraping_start(0)

    # --- Call processing_start AFTER scraping --- #
    # Determine how many sources *might* be processed (those successfully scraped)
    potentially_processable_sources = set()
    for url, result in scraped_data.items():
        if result.status == 'success' and result.content:
            potentially_processable_sources.add(url)

    if update_callback:
        await update_callback.processing_start(len(potentially_processable_sources))
    # --- End call --- #

    # --- Summarization Path (Top Results) --- #
    logger.info(f"Processing {len(top_results_for_summary)} sources for summary...")
    processed_for_summary = set() # Tracks links processed *in this section* to avoid duplicate agent calls if link appears multiple times in top_results_for_summary
    for result in top_results_for_summary:
        link_str = str(result.link)
        if link_str in processed_for_summary: # Avoid reprocessing same link within this loop
            continue

        processed_for_summary.add(link_str) # Mark as processed for this loop

        extraction_result = scraped_data.get(link_str)
        fetching_done = False
        if extraction_result and extraction_result.status == 'success' and extraction_result.content:
            if update_callback:
                await update_callback.processing_source_fetching(link_str)
            fetching_done = True # Mark fetching as attempted (for later warning)

            user_message = agents._SUMMARIZER_USER_MESSAGE_TEMPLATE.format(
                user_query=user_query,
                source_title=result.title or "Unknown Title",
                source_link=link_str,
                source_content=extraction_result.content
            )
            # --- Add logging for input content ---
            input_log_prefix = extraction_result.content[:200].replace("\n", " ") + ("..." if len(extraction_result.content) > 200 else "")
            logger.debug(f"Sending content to summarizer for {link_str}: '{input_log_prefix}'")
            # --- End logging --- 
            try:
                # Run summarizer 
                if update_callback:
                    await update_callback.processing_source_summarizing(link_str)

                summarizer_result = await agents_collection.summarizer.run(
                    user_message
                )
                
                # Attempt to access and log the summary content first
                summary_content = None
                if summarizer_result and hasattr(summarizer_result, 'data'):
                    summary_content = summarizer_result.data
                    if isinstance(summary_content, str):
                         # Log the beginning of the summary
                         log_prefix = summary_content[:200].replace("\n", " ") + ("..." if len(summary_content) > 200 else "")
                         logger.info(f"Summarizer generated content for {link_str}: '{log_prefix}'")
                    else:
                         logger.warning(f"Summarizer for {link_str} returned data of unexpected type: {type(summary_content)}")
                         summary_content = None # Treat as failure if not string
                else:
                    logger.warning(f"Summarizer for {link_str} did not return a result with data.")

                # Now, try to update usage statistics, handling potential TypeError
                if summarizer_result and hasattr(summarizer_result, 'usage'):
                    try:
                         usage_data = summarizer_result.usage()
                         usage_tracker.update_agent_usage("summarizer", usage_data)
                    except TypeError as usage_error:
                         logger.warning(f"Could not calculate usage statistics for summarizer ({link_str}) due to TypeError (likely missing 'created' timestamp): {usage_error}", exc_info=False)
                         # Continue without usage stats for this call
                    except Exception as usage_error:
                         logger.error(f"Unexpected error getting usage statistics for summarizer ({link_str}): {usage_error}", exc_info=True)
                
                # Process the summary content if we successfully retrieved it
                if not summary_content or not isinstance(summary_content, str) or not summary_content.strip():
                     logger.warning(f"Summarizer content for {link_str} is empty or invalid after potential retrieval. Skipping.")
                     continue

                # --- Assign Stable Reference Number ---
                if link_str not in processed_links:
                    source_counter += 1
                    unique_sources[link_str] = {"title": result.title, "link": link_str, "ref_num": source_counter}
                    processed_links.add(link_str)
                    usage_tracker.increment_sources_processed() # Count unique sources
                    source_ref_num = source_counter
                    logger.debug(f"Assigned new reference number {source_ref_num} to source: {link_str}")
                else:
                    source_ref_num = unique_sources[link_str]["ref_num"]
                    logger.debug(f"Using existing reference number {source_ref_num} for source: {link_str}")

                # --- Add to all_source_materials with source_ref_num ---
                all_source_materials.append({
                    "source_ref_num": source_ref_num, # Use stable ref num
                    "title": result.title, "link": link_str,
                    "content": summary_content, "type": "summary"
                })
                logger.debug(f"Added summary for {link_str} (Ref: {source_ref_num}) to materials.")

                if update_callback:
                    await update_callback.processing_source_summary_success(link_str)

            except Exception as e:
                 # This outer block catches errors during agent.run() itself or accessing .data
                 logger.error(f"Error during summarization processing for {link_str}: {e}", exc_info=True)
                 if update_callback:
                     # Send a warning, but continue processing other sources
                     await update_callback.processing_source_warning(link_str, f"Summarizer error: {e}")
                 continue 
        else:
             status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
             error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
             logger.warning(f"Skipping summary for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

    # --- Chunking & Rerank Path (Secondary Results) --- #
    logger.info(f"Processing {len(secondary_results_for_chunking)} sources for chunking & reranking...")
    documents_for_chunking = []
    links_for_chunking = set() # Track links intended for chunking to avoid redundant work
    for result in secondary_results_for_chunking:
        link_str = str(result.link)
        if link_str in processed_for_summary: # Already processed (e.g., summarized or added via another chunk earlier)
              logger.debug(f"Skipping chunking for {link_str} as it's already in unique_sources.")
              continue
        if link_str in links_for_chunking: # Avoid duplicate chunking setup if link appears multiple times here
              continue

        extraction_result = scraped_data.get(link_str)
        fetching_done_chunk = False
        if extraction_result and extraction_result.status == 'success' and extraction_result.content:
            if update_callback:
                await update_callback.processing_source_fetching(link_str)
            fetching_done_chunk = True

            documents_for_chunking.append({
                'content': extraction_result.content,
                'metadata': {'link': link_str, 'title': result.title}
            })
            links_for_chunking.add(link_str)
        else:
             status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
             error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
             logger.warning(f"Skipping chunking for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

    if documents_for_chunking:
        try:
            # Chunk content and group by source URL
            grouped_chunk_dicts: Dict[str, List[Dict[str, Any]]] = await chunk_content_helper(
                documents_to_chunk=documents_for_chunking,
                chunk_settings=config.default_chunk_settings.model_dump(),
                max_chunks=config.max_total_chunks, # Overall max chunks still applies
                logger=logger
            )

            # Iterate through each source and rerank its chunks
            relevant_chunks_from_all_sources: List[Dict[str, Any]] = []
            api_key = config.together_api_key.get_secret_value()
            model = config.reranker_model
            rerank_threshold = config.chunk_rerank_relevance_threshold

            for source_url_str, source_chunks in grouped_chunk_dicts.items():
                if not source_chunks:
                      continue # Skip if no chunks for this source

                if update_callback:
                    await update_callback.processing_source_chunking(source_url_str)

                logger.info(f"Reranking {len(source_chunks)} chunks for source: {source_url_str}")
                try:
                    reranked_source_chunks = await rerank_chunks_helper(
                        query=user_query, # Rerank based on the main query
                        chunk_dicts=source_chunks,
                        model=model,
                        api_key=api_key,
                        threshold=rerank_threshold, # Use the chunk-specific threshold
                        logger=logger
                    )

                    if reranked_source_chunks:
                        relevant_chunks_from_all_sources.extend(reranked_source_chunks) # Add all returned chunks
                        logger.info(f"Selected {len(reranked_source_chunks)} chunks for source {source_url_str} meeting threshold {config.chunk_rerank_relevance_threshold}.")
                        if update_callback:
                            await update_callback.processing_source_chunks_success(source_url_str, len(reranked_source_chunks))
                    else:
                        logger.info(f"No chunks from source {source_url_str} met the rerank threshold ({config.chunk_rerank_relevance_threshold}).")
                        if update_callback:
                            await update_callback.processing_source_chunks_success(source_url_str, 0) # Send success with 0 count
                except Exception as rerank_err:
                    logger.error(f"Error reranking chunks for source {source_url_str}: {rerank_err}", exc_info=False)
                    if update_callback:
                        await update_callback.processing_source_warning(source_url_str, f"Chunk reranking error: {rerank_err}")
                    # Continue to the next source

            # Now process the combined list of top relevant chunks from all sources
            if relevant_chunks_from_all_sources:
                 logger.info(f"Adding {len(relevant_chunks_from_all_sources)} total relevant chunks from {len(grouped_chunk_dicts)} secondary sources.")
                 # Optional: Sort the final combined list by score if desired
                 relevant_chunks_from_all_sources.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                 for chunk_dict in relevant_chunks_from_all_sources:
                     link_str_chunk = chunk_dict.get('metadata', {}).get('link')
                     title = chunk_dict.get('metadata', {}).get('title')
                     content = chunk_dict.get('content')
                     score = chunk_dict.get('score')

                     if link_str_chunk and title and content:
                             # --- Assign Stable Reference Number ---
                             if link_str_chunk not in processed_links:
                                 source_counter += 1
                                 unique_sources[link_str_chunk] = {"title": title, "link": link_str_chunk, "ref_num": source_counter}
                                 processed_links.add(link_str_chunk)
                                 usage_tracker.increment_sources_processed() # Count unique sources
                                 source_ref_num = source_counter
                                 logger.debug(f"Assigned new reference number {source_ref_num} to source: {link_str_chunk}")
                             else:
                                 # This case should ideally not happen if we skip chunking for processed_links above,
                                 # but handle defensively.
                                 if link_str_chunk in unique_sources:
                                     source_ref_num = unique_sources[link_str_chunk]["ref_num"]
                                     logger.debug(f"Using existing reference number {source_ref_num} for source: {link_str_chunk}")
                                 else:
                                      logger.error(f"Logic error: Link {link_str_chunk} was processed but not in unique_sources. Skipping chunk.")
                                      continue


                             # --- Add to all_source_materials with source_ref_num ---
                             all_source_materials.append({
                                  "source_ref_num": source_ref_num, # Use stable ref num
                                  "title": title, "link": link_str_chunk,
                                  "content": content, "type": "chunk", "score": score
                             })
                             score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
                             logger.debug(f"Added relevant chunk for {link_str_chunk} (Ref: {source_ref_num}, Score: {score_str}) to materials.")
                     else:
                         logger.warning(f"Skipping selected chunk due to missing link/title/content")
        except Exception as e:
            logger.error(f"Error during chunking/reranking of secondary sources: {e}", exc_info=True)
            if update_callback:
                # Send a general processing warning/error if chunking/reranking fails globally
                await update_callback.processing_source_warning("Multiple Sources", f"Chunking/Reranking Block Error: {e}")

    logger.info(f"Source processing complete. Collected {len(all_source_materials)} items from {len(unique_sources)} unique sources.")

    # --- Update Firestore with Sources --- #
    if firestore_available and task_doc_ref:
        try:
            # Convert unique_sources dict to a list of dicts for Firestore
            sources_list = list(unique_sources.values()) if unique_sources else []

            source_update_data = {
                "sources": sources_list,
                "sourceCount": len(sources_list),
                "status": "PROCESSING_COMPLETE", # Or a more specific status
                "updatedAt": firestore.SERVER_TIMESTAMP
            }
            task_doc_ref.update(source_update_data)
            logger.info(f"Updated Firestore with {len(sources_list)} unique sources.")
        except Exception as fs_e:
            logger.error(f"Failed to update Firestore with sources: {fs_e}")
    # --- End Firestore Update --- #

    # === Context Token Limiting ===
    if all_source_materials:
        logger.info(f"Limiting writer context to ~{MAX_WRITER_CONTEXT_TOKENS} tokens...")
        original_material_count = len(all_source_materials)
        try:
            # TODO: Make the encoding name configurable or dependent on the writer model
            encoding = tiktoken.get_encoding("cl100k_base") # Common encoding
        except Exception as enc_err:
            logger.error(f"Could not get tiktoken encoding 'cl100k_base'. Proceeding without token limiting. Error: {enc_err}", exc_info=True)
            # If encoding fails, we skip limiting and proceed with all materials
        else:
            summaries = []
            chunks = []
            total_summary_tokens = 0
            total_initial_chunk_tokens = 0

            # Calculate tokens and split materials
            for item in all_source_materials:
                content = item.get('content', '')
                token_count = len(encoding.encode(content))
                item['token_count'] = token_count # Store token count

                if item.get('type') == 'summary':
                    summaries.append(item)
                    total_summary_tokens += token_count
                elif item.get('type') == 'chunk':
                    chunks.append(item)
                    total_initial_chunk_tokens += token_count
                else:
                    # Should not happen, but handle defensively
                    logger.warning(f"Unknown material type found: {item.get('type')}. Treating as chunk for token counting.")
                    chunks.append(item)
                    total_initial_chunk_tokens += token_count

            logger.info(f"Initial context tokens: Summaries={total_summary_tokens}, Chunks={total_initial_chunk_tokens}, Total={total_summary_tokens + total_initial_chunk_tokens}")

            # Check if total tokens exceed the limit
            if (total_summary_tokens + total_initial_chunk_tokens) > MAX_WRITER_CONTEXT_TOKENS:
                logger.warning(f"Total context tokens ({total_summary_tokens + total_initial_chunk_tokens}) exceed limit ({MAX_WRITER_CONTEXT_TOKENS}). Trimming chunks...")

                allowed_chunk_tokens = MAX_WRITER_CONTEXT_TOKENS - total_summary_tokens
                if allowed_chunk_tokens < 0:
                    logger.error(f"Summary tokens ({total_summary_tokens}) alone exceed the total limit ({MAX_WRITER_CONTEXT_TOKENS}). Cannot add any chunks.")
                    allowed_chunk_tokens = 0 # Cannot add any chunks

                # Sort chunks by score (descending, higher is better). Handle missing scores (treat as 0).
                chunks.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                final_chunks = []
                current_chunk_tokens = 0
                original_chunk_count = len(chunks)
                for chunk in chunks:
                    chunk_tokens = chunk.get('token_count', 0)
                    if (current_chunk_tokens + chunk_tokens) <= allowed_chunk_tokens:
                        final_chunks.append(chunk)
                        current_chunk_tokens += chunk_tokens
                    else:
                        # Stop adding chunks once the budget is exceeded
                        break

                trimmed_count = original_chunk_count - len(final_chunks)
                logger.info(f"Trimmed {trimmed_count} chunks based on score and token limit. Final chunk tokens: {current_chunk_tokens} (Allowed: {allowed_chunk_tokens})")

                # Reconstruct all_source_materials with summaries and trimmed chunks
                all_source_materials = summaries + final_chunks
                final_total_tokens = total_summary_tokens + current_chunk_tokens
                logger.info(f"Final context size: {len(all_source_materials)} items, ~{final_total_tokens} tokens.")

            else:
                 logger.info("Total context tokens are within the limit. No trimming needed.")
    else:
         logger.info("No source materials generated, skipping token limiting step.")
    # === End Context Token Limiting ===

    # Send overall processing end update
    if update_callback:
        await update_callback.processing_end(len(unique_sources), len(all_source_materials))

    # === Step 4: Initial Writing ===
    logger.info("Calling Writer Agent...")
    if update_callback:
        await update_callback.writing_start(len(all_source_materials))

    if not all_source_materials:
        logger.warning("No source materials processed, cannot generate report.")
        # Use get_statistics() for the final response
        return schemas.ResearchResponse(report="Could not generate report: No information gathered.", usage_statistics=usage_tracker.get_statistics())

    current_draft = ""
    requested_searches: Optional[List[schemas.SearchRequest]] = None
    writer_result = None # Initialize writer_result
    writer_output: Optional[schemas.WriterOutput] = None
    writer_exception: Optional[Exception] = None # Store exception if all retries fail

    max_writer_retries = 3 # Define max retries
    writer_retry_delay_seconds = 2 # Define delay between retries

    # --- Retry Loop for Initial Writer ---
    for attempt in range(max_writer_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_writer_retries} to call Writer Agent...")
            # Now uses source_ref_num internally
            formatted_context = agents.format_summaries_for_prompt(all_source_materials)
            writer_user_prompt = agents._WRITER_USER_MESSAGE_TEMPLATE_INITIAL.format(
                user_query=user_query,
                writing_plan_json=writing_plan.model_dump_json(indent=2),
                formatted_summaries=formatted_context
            )

            writer_result = await agents_collection.writer.run(writer_user_prompt)

            # --- Success Handling (moved inside try block) ---
            if writer_result and hasattr(writer_result, 'data'):
                if isinstance(writer_result.data, schemas.WriterOutput):
                    writer_output = writer_result.data
                    if writer_output.report_content:
                        log_prefix = writer_output.report_content[:200].replace("\n", " ") + ("..." if len(writer_output.report_content) > 200 else "")
                        logger.info(f"Initial Writer generated content: '{log_prefix}'")
                    else:
                        logger.warning("Initial Writer returned output object but report_content is empty.")
                    # Successfully got output, break the retry loop
                    writer_exception = None # Reset exception on success
                    break # Exit retry loop on success
                else:
                    # Treat as critical failure if type is wrong - log and store exception before retry
                    logger.error(f"Initial Writer returned data of unexpected type: {type(writer_result.data)}. Retrying...")
                    writer_exception = TypeError(f"Writer agent returned unexpected type: {type(writer_result.data)}")
                    # Fall through to retry delay below

            else:
                 logger.error("Initial Writer did not return a result with data. Retrying...")
                 writer_exception = ValueError("Writer agent did not return data.") # Store error
                 # Fall through to retry delay below

        except TypeError as e:
            writer_exception = e # Store the exception
            logger.warning(f"Writer agent call failed with TypeError (Attempt {attempt + 1}/{max_writer_retries}): {e}. Retrying...")
            # Check if it's the specific NoneType error, just for logging clarity
            if "'NoneType' object cannot be interpreted as an integer" in str(e):
                logger.warning("Specific TypeError related to 'response.created' detected.")
        except Exception as e:
            writer_exception = e # Store the exception
            logger.error(f"Writer agent failed with unexpected error (Attempt {attempt + 1}/{max_writer_retries}): {e}", exc_info=True)
            # Decide if this type of error is retryable. Let's retry general exceptions too for now.

        # If not the last attempt, wait before retrying
        if attempt < max_writer_retries - 1:
            logger.info(f"Waiting {writer_retry_delay_seconds} seconds before next writer attempt...")
            await asyncio.sleep(writer_retry_delay_seconds)
        else:
            logger.error(f"Writer agent failed after {max_writer_retries} attempts.")
            # Final attempt failed, loop will exit

    # --- Check results after retry loop ---
    if writer_output: # Successfully got output from the loop
        # Now, try to update usage statistics, handling potential TypeError
        if writer_result and hasattr(writer_result, 'usage'):
             try:
                  usage_data = writer_result.usage()
                  usage_tracker.update_agent_usage("writer_initial", usage_data) # Specific tag for initial write
             except TypeError as usage_error:
                  # This might still occur if the successful response *still* lacks 'created'
                  logger.warning(f"Could not calculate usage statistics for initial writer due to TypeError (potentially missing 'created'): {usage_error}", exc_info=False)
             except Exception as usage_error:
                  logger.error(f"Unexpected error getting usage statistics for initial writer: {usage_error}", exc_info=True)

        # Proceed with the retrieved writer output
        current_draft = writer_output.report_content
        requested_searches = writer_output.requested_searches
        logger.info("Initial draft processing complete.") # Adjusted log message slightly
        if update_callback:
            await update_callback.writing_end(len(requested_searches) if requested_searches else 0)
        if requested_searches:
             logger.info(f"Writer requested {len(requested_searches)} additional searches.")

    else: # All retries failed
        # Log the final error, send callback, set placeholder draft, and prevent refinement
        final_error_msg = f"Writer Agent Critical Error (Initial Draft after {max_writer_retries} retries): {writer_exception}"
        logger.error(final_error_msg, exc_info=writer_exception if isinstance(writer_exception, Exception) else False)
        if update_callback:
            # Pass the last known exception to the callback
            await update_callback.writing_error(writer_exception or Exception("Writer failed after retries"))

        # Set placeholder draft and prevent refinement loop
        current_draft = f"**Warning:** The report writer failed to generate content after {max_writer_retries} attempts. The error was: {writer_exception}. Please see the reference list below for collected sources."
        requested_searches = None
        logger.warning("Proceeding to final assembly without a generated report draft due to writer failure.")
        # NOTE: Do not return here, continue to final assembly

    # === Step 5: Refinement Loop ===
    refinement_loop_count = 0
    while requested_searches and refinement_loop_count < config.max_refinement_loops:
        refinement_loop_count += 1
        # Increment refinement loop counter
        usage_tracker.increment_refinement_iterations()
        logger.info(f"--- Starting Refinement Loop {refinement_loop_count} ---")

        if update_callback:
            await update_callback.refinement_loop_start(refinement_loop_count, config.max_refinement_loops)

        new_materials_in_this_loop: List[Dict[str, Any]] = [] # Stores materials *added* this loop for refiner context

        # --- Execute & Rerank Additional Searches --- #
        logger.info(f"Executing {len(requested_searches)} additional searches...")

        refinement_query_topic = requested_searches[0].query if requested_searches else "Unknown"
        if update_callback:
            await update_callback.refinement_search_start(refinement_loop_count, refinement_query_topic)

        reranked_refinement_search_results: List[SearchResult] = []
        try:
            refinement_tasks = [
                SearchTask(query=req.query, endpoint="/search", num_results=5, reasoning="Requested by agent.") 
                for req in requested_searches if req.query
            ]
            if not refinement_tasks:
                 logger.warning("Agent requested searches, but queries were empty. Skipping refinement search.")
                 requested_searches = None
                 continue # Skip to next loop iteration (effectively ending if no more loops)

            serper_cfg = SerperConfig.from_env()
            refinement_results_map = await execute_search_queries(refinement_tasks, serper_cfg, logger)
            # Increment Serper query usage for refinement searches
            usage_tracker.increment_serper_queries(len(refinement_tasks))

            # Flatten, deduplicate against *all* processed links (in unique_sources)
            temp_results = []
            newly_added_links_temp = set() # Track links found in *this* search batch
            for query, query_results in refinement_results_map.items():
                 for result in query_results:
                      link_str = str(result.link)
                      # Check if link is truly new (not in unique_sources) and not already seen in this batch
                      if link_str not in processed_links and link_str not in newly_added_links_temp:
                           temp_results.append(result)
                           newly_added_links_temp.add(link_str)

            logger.info(f"Refinement search yielded {len(temp_results)} new unique results before reranking.")

            # Rerank these new results
            if temp_results:
                 api_key = config.together_api_key.get_secret_value()
                 model = config.reranker_model
                 # Combine user query and refinement topic? For now, use user query.
                 rerank_query = user_query # requested_searches[0].query if requested_searches else user_query
                 reranked_refinement_search_results = await rerank_search_results_helper(
                     query=rerank_query,
                     search_results=temp_results,
                     model=model, api_key=api_key,
                     threshold=config.rerank_relevance_threshold,
                     logger=logger
                 )
            else:
                 reranked_refinement_search_results = []

        except schemas.ConfigurationError as e: # Assuming ConfigurationError is available
             logger.error(f"Search config error during refinement: {e}", exc_info=True)
             if update_callback:
                 await update_callback.refinement_search_error(refinement_loop_count, e)
             requested_searches = None; break # Exit loop
        except Exception as e:
            logger.error(f"Refinement search/rerank failed: {e}", exc_info=True)
            if update_callback:
                await update_callback.refinement_search_error(refinement_loop_count, e)
            requested_searches = None; break # Exit loop

        # --- Process Refinement Results (Scrape, Chunk, Rerank Chunks) --- #
        if not reranked_refinement_search_results:
             logger.warning("Refinement search/rerank yielded no usable new results. Ending refinement loop.")
             if update_callback:
                 # Send an info update? Or rely on loop end message?
                 pass # Rely on loop end message
             requested_searches = None; break # Exit loop

        logger.info(f"Processing {len(reranked_refinement_search_results)} new refinement results for chunking...")

        if update_callback:
            await update_callback.refinement_processing_start(refinement_loop_count, len(reranked_refinement_search_results))

        urls_for_refinement_scraping = [str(res.link) for res in reranked_refinement_search_results]
        refinement_scraped_data = await batch_scrape_urls_helper(urls_for_refinement_scraping, app_settings, logger)

        documents_for_refinement_chunking = []
        links_for_refinement_chunking = set() # Track links intended for chunking this loop
        for result in reranked_refinement_search_results:
            link_str = str(result.link)
            # Double check it's not already processed (shouldn't be if reranking worked on new links)
            if link_str in processed_links or link_str in links_for_refinement_chunking:
                 continue

            extraction_result = refinement_scraped_data.get(link_str)
            if extraction_result and extraction_result.status == 'success' and extraction_result.content:
                 documents_for_refinement_chunking.append({
                    'content': extraction_result.content,
                    'metadata': {'link': link_str, 'title': result.title}
                 })
                 links_for_refinement_chunking.add(link_str)
            else:
                 status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
                 error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
                 logger.warning(f"Skipping refinement chunking for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

        newly_added_material_in_loop_flag = False
        if documents_for_refinement_chunking:
             try:
                 # Calculate remaining chunks based on *all_source_materials* length vs max_total_chunks?
                 # This needs careful thought - maybe limit chunks *per refinement source* instead?
                 # For now, apply overall limit loosely.
                 remaining_chunks_overall = config.max_total_chunks - len(all_source_materials) if config.max_total_chunks else None
                 if remaining_chunks_overall is not None and remaining_chunks_overall <= 0:
                      logger.warning(f"Max total chunks ({config.max_total_chunks}) reached. Skipping refinement chunking.")
                 else:
                    # Chunk content and group by source URL
                    refinement_grouped_chunk_dicts = await chunk_content_helper(
                          documents_to_chunk=documents_for_refinement_chunking,
                          chunk_settings=config.default_chunk_settings.model_dump(),
                          max_chunks=remaining_chunks_overall, # Apply overall limit here
                          logger=logger
                    )

                    # Rerank chunks for each source URL found in this refinement batch
                    reranked_refinement_chunks_all: List[Dict[str, Any]] = []
                    api_key = config.together_api_key.get_secret_value()
                    model = config.reranker_model
                    for source_url_str, source_chunks in refinement_grouped_chunk_dicts.items():
                         if not source_chunks: continue
                         logger.info(f"Reranking {len(source_chunks)} refinement chunks for {source_url_str}")
                         try:
                            reranked_src_chunks = await rerank_chunks_helper(
                                query=user_query, # Or maybe use refinement topic query?
                                chunk_dicts=source_chunks,
                                model=model, api_key=api_key,
                                threshold=config.chunk_rerank_relevance_threshold,
                                logger=logger
                            )
                            if reranked_src_chunks:
                                 reranked_refinement_chunks_all.extend(reranked_src_chunks)
                                 logger.info(f"Selected {len(reranked_src_chunks)} refinement chunks for {source_url_str}")
                         except Exception as rerank_err:
                              logger.error(f"Error reranking refinement chunks for {source_url_str}: {rerank_err}", exc_info=False)

                    # Process the combined list of relevant refinement chunks
                    if reranked_refinement_chunks_all:
                          logger.info(f"Adding {len(reranked_refinement_chunks_all)} total relevant refinement chunks.")
                          # Optional: Sort combined list by score
                          reranked_refinement_chunks_all.sort(key=lambda x: x.get('score', 0.0), reverse=True)

                          for chunk_dict in reranked_refinement_chunks_all:
                               link_str_chunk = chunk_dict.get('metadata', {}).get('link')
                               title = chunk_dict.get('metadata', {}).get('title')
                               content = chunk_dict.get('content')
                               score = chunk_dict.get('score')

                               if link_str_chunk and title and content:
                                    # --- Assign Stable Reference Number ---
                                    if link_str_chunk not in processed_links:
                                        source_counter += 1
                                        unique_sources[link_str_chunk] = {"title": title, "link": link_str_chunk, "ref_num": source_counter}
                                        processed_links.add(link_str_chunk)
                                        usage_tracker.increment_sources_processed() # Count unique sources
                                        source_ref_num = source_counter
                                        logger.debug(f"Assigned new reference number {source_ref_num} to refinement source: {link_str_chunk}")
                                    else:
                                        # Should not happen if initial search worked correctly
                                        if link_str_chunk in unique_sources:
                                             source_ref_num = unique_sources[link_str_chunk]["ref_num"]
                                             logger.debug(f"Using existing reference number {source_ref_num} for refinement source: {link_str_chunk}")
                                        else:
                                             logger.error(f"Logic error: Refinement link {link_str_chunk} processed but not in unique_sources. Skipping chunk.")
                                             continue

                                    # --- Add to all_source_materials & track for refiner context ---
                                    added_material = {
                                        "source_ref_num": source_ref_num,
                                        "title": title, "link": link_str_chunk,
                                        "content": content, "type": "chunk", "score": score
                                    }
                                    all_source_materials.append(added_material)
                                    new_materials_in_this_loop.append(added_material) # Add to list for refiner

                                    score_str = f"{score:.4f}" if isinstance(score, float) else "N/A"
                                    logger.debug(f"Added refinement chunk for {link_str_chunk} (Ref: {source_ref_num}, Score: {score_str}) to materials.")
                                    newly_added_material_in_loop_flag = True # Mark that we added something
                               else:
                                    logger.warning(f"Skipping refinement chunk due to missing link/title/content")

             except Exception as e:
                 logger.error(f"Error during refinement chunking/reranking: {e}", exc_info=True)
                 # Continue loop, but might not have new materials

        if not newly_added_material_in_loop_flag:
             logger.warning("Refinement processing added no new materials. Ending refinement loop.")
             requested_searches = None; break # Exit loop

        # --- Call Refiner Agent --- #
        logger.info("Calling Refiner Agent...")
        if update_callback:
            await update_callback.refinement_refiner_start(refinement_loop_count)

        try:
            # Pass only the newly added materials for context
            formatted_new_context = agents.format_summaries_for_prompt(new_materials_in_this_loop)
            refinement_topic = requested_searches[0].query if requested_searches else "requested information"
            refiner_user_prompt = agents._REFINEMENT_USER_MESSAGE_TEMPLATE.format(
                 user_query=user_query,
                 writing_plan_json=writing_plan.model_dump_json(indent=2),
                 previous_draft=current_draft,
                 refinement_topic=refinement_topic,
                 formatted_new_summaries=formatted_new_context,
            )
            # Run refiner
            refiner_result = await agents_collection.refiner.run(refiner_user_prompt)
            
            # Attempt to access and log the refiner output first
            refiner_output: Optional[schemas.WriterOutput] = None
            if refiner_result and hasattr(refiner_result, 'data'):
                if isinstance(refiner_result.data, schemas.WriterOutput):
                    refiner_output = refiner_result.data
                    if refiner_output.report_content:
                        log_prefix = refiner_output.report_content[:200].replace("\n", " ") + ("..." if len(refiner_output.report_content) > 200 else "")
                        logger.info(f"Refiner (Loop {refinement_loop_count}) generated content: '{log_prefix}'")
                    else:
                         logger.warning(f"Refiner (Loop {refinement_loop_count}) returned output object but report_content is empty.")
                else:
                    logger.error(f"Refiner (Loop {refinement_loop_count}) returned data of unexpected type: {type(refiner_result.data)}")
                    # Treat as critical failure for this loop if type is wrong
                    raise TypeError(f"Refiner agent returned unexpected type: {type(refiner_result.data)}")
            else:
                 logger.error(f"Refiner (Loop {refinement_loop_count}) did not return a result with data.")
                 # Treat as critical failure for this loop if no data
                 raise ValueError("Refiner agent did not return data.")

            # Now, try to update usage statistics, handling potential TypeError
            if refiner_result and hasattr(refiner_result, 'usage'):
                 try:
                      usage_data = refiner_result.usage()
                      usage_tracker.update_agent_usage(f"refiner_loop_{refinement_loop_count}", usage_data) # Tagged usage
                 except TypeError as usage_error:
                      logger.warning(f"Could not calculate usage statistics for refiner (Loop {refinement_loop_count}) due to TypeError: {usage_error}", exc_info=False)
                 except Exception as usage_error:
                      logger.error(f"Unexpected error getting usage statistics for refiner (Loop {refinement_loop_count}): {usage_error}", exc_info=True)
            
            # Proceed with the retrieved refiner output
            current_draft = refiner_output.report_content
            requested_searches = refiner_output.requested_searches
            logger.info(f"Refiner Agent (Loop {refinement_loop_count}) processing complete.")
            if update_callback:
                await update_callback.refinement_refiner_end(refinement_loop_count, len(requested_searches) if requested_searches else 0)

            if requested_searches:
                logger.info(f"Refiner requested {len(requested_searches)} further searches.")
            else:
                logger.info("Refiner did not request further searches.")
        except (TypeError, ValueError, Exception) as e: # Catch specific errors raised above and general exceptions
            logger.error(f"Refiner agent failed critically during loop {refinement_loop_count}: {e}", exc_info=True)
            if update_callback:
                await update_callback.refinement_refiner_error(refinement_loop_count, e)
            requested_searches = None; break # Exit the refinement loop on critical failure

    if refinement_loop_count >= config.max_refinement_loops and requested_searches:
        logger.warning(f"Reached max refinement loops ({config.max_refinement_loops}) with searches still requested.")

    # --- Refinement Loop End Update --- #
    loop_end_reason = "Unknown"
    if refinement_loop_count >= config.max_refinement_loops and requested_searches:
        loop_end_reason = "Max iterations reached"
    elif not requested_searches:
        loop_end_reason = "Completed normally"
    # Add other reasons like "No new info found" if applicable based on break conditions
    if refinement_loop_count > 0 and update_callback: # Only send if loop ran at least once
        await update_callback.refinement_loop_end(refinement_loop_count, loop_end_reason)

    # === Step 6: Final Assembly ===
    logger.info("Assembling final report and bibliography...")
    if update_callback:
        await update_callback.finalizing_start()

    final_report_content = current_draft # Start with the raw content from writer/refiner

    # Use helper function to add clickable citation links
    processed_report_content = format_report_citations(final_report_content, logger)
    
    # Use helper function to generate reference list
    reference_list_markup = generate_reference_list(unique_sources, logger)
    
    # Combine processed content and reference list
    final_report = processed_report_content + reference_list_markup

    if update_callback:
        await update_callback.finalizing_end(final_report)

    # === Step 7: Return Result ===
    logger.info("Deep research orchestration complete.")

    # Get final statistics using the new method
    final_usage_stats = usage_tracker.get_statistics()
    # Log the structured stats (optional, UsageStatistics has __str__)
    logger.info(f"Final Usage Statistics: {final_usage_stats}")

    # Use the generated UsageStatistics object in the response
    # Send final completion update via callback before returning
    if update_callback:
        await update_callback.orchestration_complete(
            final_report_length=len(final_report),
            usage_stats=final_usage_stats.model_dump(mode='json')
        )

    return schemas.ResearchResponse(
        report=final_report,
        usage_statistics=final_usage_stats
    )

# Add a global try...except around the main logic to catch unexpected errors
# and send a final orchestration_error callback.
async def run_deep_research_orchestration_wrapper(
    user_query: str,
    agents_collection: agents.AgencyAgents,
    config: DeepResearchConfig,
    app_settings: AppSettings,
    update_callback: Optional[WebSocketUpdateHandler] = None,
    # Add Firestore parameters to wrapper
    task_doc_ref: Optional[DocumentReference] = None,
    firestore_available: bool = False
) -> schemas.ResearchResponse:
    try:
        return await run_deep_research_orchestration(
            user_query=user_query,
            agents_collection=agents_collection,
            config=config,
            app_settings=app_settings,
            update_callback=update_callback,
            # Pass Firestore parameters through
            task_doc_ref=task_doc_ref,
            firestore_available=firestore_available
        )
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED ERROR during orchestration: {e}", exc_info=True)
        if update_callback:
            await update_callback.orchestration_error(e)
        # Re-raise or return an error response? Returning response for now.
        # This error means something went wrong OUTSIDE the handled flow.
        return schemas.ResearchResponse(
            report=f"Critical Orchestration Error: {e}. Please check server logs.",
            # Attempt to get usage stats if possible, otherwise create default
            usage_statistics=UsageStatistics( # Use the base UsageStatistics here
                token_usage={},
                estimated_cost={},
                serper_queries_used=0, # Might be inaccurate
                sources_processed_count=0, # Might be inaccurate
                refinement_iterations_run=0 # Might be inaccurate
            )
        )