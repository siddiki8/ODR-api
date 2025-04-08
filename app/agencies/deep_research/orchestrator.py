import asyncio
import logging
from typing import List, Dict, Any, Optional
import os

# Pydantic AI imports
from pydantic_ai.usage import Usage

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
    chunk_content_helper
)

from ..services.search import SearchResult, SerperConfig, SearchTask
from ..services.scraper import ExtractionResult
from app.core.config import AppSettings 


logger = logging.getLogger(__name__)

async def run_deep_research_orchestration(
    user_query: str,
    agents_collection: agents.AgencyAgents,
    config: DeepResearchConfig,
    app_settings: AppSettings
) -> schemas.ResearchResponse:
    """
    Orchestrates the deep research multi-agent process.
    """
    usage_tracker = Usage()
    all_source_materials: List[Dict[str, Any]] = []
    processed_links = set()
    source_metadata_map: Dict[int, Dict[str, Any]] = {}

    logger.info(f"Starting deep research orchestration for query: '{user_query}'")

    # === Step 1: Planning ===
    logger.info("Calling Planner Agent...")
    try:
        planner_result = await agents_collection.planner.run(
            agents._PLANNER_USER_MESSAGE_TEMPLATE.format(user_query=user_query),
            usage=usage_tracker
        )
        planner_output: schemas.PlannerOutput = planner_result.data
        search_tasks: List[SearchTask] = planner_output.search_tasks
        writing_plan: schemas.WritingPlan = planner_output.writing_plan
        logger.info(f"Planner generated {len(search_tasks)} search tasks and a writing plan.")
    except Exception as e:
        logger.error(f"Planner agent failed: {e}", exc_info=True)
        # TODO: Handle planner failure gracefully (e.g., return error response)
        raise

    # === Step 2: Initial Search & Rerank ===
    logger.info("Executing initial search tasks...")
    initial_search_results: List[SearchResult] = []
    reranked_initial_search_results: List[SearchResult] = []
    try:
        serper_cfg = SerperConfig.from_env()
        search_results_map = await execute_search_queries(
            search_tasks=search_tasks,
            config=serper_cfg,
            logger=logger
        )

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

        # Rerank Search Results
        if initial_search_results:
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
             logger.info("No initial search results to rerank.")

        if not reranked_initial_search_results:
             logger.warning("Initial search and reranking yielded no results.")

    # Catch config errors separately as they prevent continuation
    except schemas.ConfigurationError as e: # Assuming ConfigurationError is in schemas or core
         logger.error(f"Search configuration error: {e}", exc_info=True)
         return schemas.ResearchResponse(report=f"Configuration Error: {e}", usage_statistics=usage_tracker.get_statistics())
    except Exception as e:
        logger.error(f"Initial search/rerank failed critically: {e}", exc_info=True)
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

    # === Step 3: Source Processing ===
    logger.info("Starting source processing...")

    # --- Batch Scrape --- #
    urls_to_scrape = list(set([str(res.link) for res in top_results_for_summary] + 
                              [str(res.link) for res in secondary_results_for_chunking]))
    scraped_data: Dict[str, ExtractionResult] = {}
    if urls_to_scrape:
        logger.info(f"Batch scraping {len(urls_to_scrape)} unique URLs...")
        scraped_data = await batch_scrape_urls_helper(
            urls=urls_to_scrape,
            settings=app_settings,
            logger=logger
        )
    else:
        logger.info("No URLs to scrape.")

    # --- Summarization Path (Top Results) --- #
    logger.info(f"Processing {len(top_results_for_summary)} sources for summary...")
    processed_for_summary = set()
    for result in top_results_for_summary:
        link_str = str(result.link)
        if link_str in processed_links:
            continue
        
        extraction_result = scraped_data.get(link_str)
        if extraction_result and extraction_result.status == 'success' and extraction_result.content:
            user_message = agents._SUMMARIZER_USER_MESSAGE_TEMPLATE.format(
                user_query=user_query,
                source_title=result.title or "Unknown Title",
                source_link=link_str,
                source_content=extraction_result.content
            )
            try:
                summarizer_result = await agents_collection.summarizer.run(
                    user_message,
                    usage=usage_tracker
                )
                summary_content = summarizer_result.data
                if not summary_content or not isinstance(summary_content, str) or not summary_content.strip():
                     logger.warning(f"Summarizer returned empty content for {link_str}. Skipping.")
                     continue

                rank = len(all_source_materials) + 1
                all_source_materials.append({
                    "rank": rank, "title": result.title, "link": link_str,
                    "content": summary_content, "type": "summary"
                })
                if link_str not in processed_links:
                    source_metadata_map[rank] = {"title": result.title, "link": link_str}
                processed_links.add(link_str)
                processed_for_summary.add(link_str)
                logger.debug(f"Added summary for {link_str} with rank {rank}")
            except Exception as e:
                 logger.error(f"Error during summarization LLM call for {link_str}: {e}", exc_info=True)
                 continue 
        else:
             status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
             error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
             logger.warning(f"Skipping summary for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

    # --- Chunking & Rerank Path (Secondary Results) --- #
    logger.info(f"Processing {len(secondary_results_for_chunking)} sources for chunking & reranking...")
    documents_for_chunking = []
    for result in secondary_results_for_chunking:
        link_str = str(result.link)
        if link_str in processed_links: # Already summarized
              continue
        extraction_result = scraped_data.get(link_str)
        if extraction_result and extraction_result.status == 'success' and extraction_result.content:
              documents_for_chunking.append({
                'content': extraction_result.content,
                'metadata': {'link': link_str, 'title': result.title}
              })
        else:
             status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
             error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
             logger.warning(f"Skipping chunking for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

    if documents_for_chunking:
        try:
            all_chunk_dicts = await chunk_content_helper(
                documents_to_chunk=documents_for_chunking,
                chunk_settings=config.default_chunk_settings.model_dump(),
                max_chunks=config.max_total_chunks,
                logger=logger
            )

            reranked_chunk_dicts = []
            if all_chunk_dicts:
                api_key = config.together_api_key.get_secret_value()
                model = config.reranker_model
                reranked_chunk_dicts = await rerank_chunks_helper(
                    query=user_query,
                    chunk_dicts=all_chunk_dicts,
                    model=model,
                    api_key=api_key,
                    threshold=config.rerank_relevance_threshold,
                    logger=logger
                )
            
            if reranked_chunk_dicts:
                 logger.info(f"Adding {len(reranked_chunk_dicts)} relevant chunks from secondary sources.")
                 for chunk_dict in reranked_chunk_dicts:
                     link = chunk_dict.get('metadata', {}).get('link')
                     title = chunk_dict.get('metadata', {}).get('title')
                     content = chunk_dict.get('content')
                     score = chunk_dict.get('score') 

                     if link and title and content:
                             rank = len(all_source_materials) + 1
                             all_source_materials.append({
                                  "rank": rank, "title": title, "link": link,
                                  "content": content, "type": "chunk", "score": score
                             })
                             link_str_chunk = str(link)
                             if link_str_chunk not in processed_links:
                                 source_metadata_map[rank] = {"title": title, "link": link_str_chunk}
                                 processed_links.add(link_str_chunk)
                             logger.debug(f"Added relevant chunk for {link_str_chunk} (Score: {score:.4f if score else 'N/A'}) with rank {rank}")
                     else:
                         logger.warning(f"Skipping selected chunk due to missing link/title/content")
        except Exception as e:
            logger.error(f"Error during chunking/reranking of secondary sources: {e}", exc_info=True)

    logger.info(f"Source processing complete. Context size: {len(all_source_materials)} items.")

    # === Step 4: Initial Writing ===
    logger.info("Calling Writer Agent...")
    if not all_source_materials:
        logger.warning("No source materials processed, cannot generate report.")
        return schemas.ResearchResponse(report="Could not generate report: No information gathered.", usage_statistics=usage_tracker.get_statistics())

    current_draft = ""
    requested_searches: Optional[List[schemas.SearchRequest]] = None
    try:
        formatted_context = agents.format_summaries_for_prompt(all_source_materials)
        writer_user_prompt = agents._WRITER_USER_MESSAGE_TEMPLATE_INITIAL.format(
            user_query=user_query,
            writing_plan_json=writing_plan.model_dump_json(indent=2),
            formatted_summaries=formatted_context
        )

        writer_result = await agents_collection.writer.run(writer_user_prompt, usage=usage_tracker)
        writer_output: schemas.WriterOutput = writer_result.data
        current_draft = writer_output.report_content
        requested_searches = writer_output.requested_searches
        logger.info("Initial draft generated.")
        if requested_searches:
             logger.info(f"Writer requested {len(requested_searches)} additional searches.")

    except Exception as e:
        logger.error(f"Writer agent failed during initial draft: {e}", exc_info=True)
        # TODO: Handle writer failure more gracefully
        raise

    # === Step 5: Refinement Loop ===
    refinement_loop_count = 0
    while requested_searches and refinement_loop_count < config.max_refinement_loops:
        refinement_loop_count += 1
        logger.info(f"--- Starting Refinement Loop {refinement_loop_count} ---")

        new_materials_in_this_loop: List[Dict[str, Any]] = []

        # --- Execute & Rerank Additional Searches --- #
        logger.info(f"Executing {len(requested_searches)} additional searches...")
        reranked_refinement_search_results: List[SearchResult] = []
        try:
            refinement_tasks = [
                schemas.SearchTask(query=req.query, endpoint="/search", num_results=5, reasoning="Requested by agent.") 
                for req in requested_searches if req.query
            ]
            if not refinement_tasks:
                 logger.warning("Agent requested searches, but queries were empty. Skipping refinement.")
                 requested_searches = None
                 continue 

            serper_cfg = SerperConfig.from_env()
            refinement_results_map = await execute_search_queries(refinement_tasks, serper_cfg, logger)

            # Flatten, deduplicate against *all* processed links
            temp_results = []
            newly_added_links = set()
            for query, query_results in refinement_results_map.items():
                 for result in query_results:
                      link_str = str(result.link)
                      if link_str not in processed_links and link_str not in newly_added_links:
                           temp_results.append(result)
                           newly_added_links.add(link_str)
            
            logger.info(f"Refinement search yielded {len(temp_results)} new results before reranking.")

            # Rerank these new results
            if temp_results:
                 api_key = config.together_api_key.get_secret_value()
                 model = config.reranker_model
                 reranked_refinement_search_results = await rerank_search_results_helper(
                     query=user_query, # Or refinement_topic?
                     search_results=temp_results,
                     model=model, api_key=api_key,
                     threshold=config.rerank_relevance_threshold,
                     logger=logger
                 )
            else:
                 reranked_refinement_search_results = []

        except schemas.ConfigurationError as e: # Assuming ConfigurationError is available
             logger.error(f"Search config error during refinement: {e}", exc_info=True)
             requested_searches = None; break
        except Exception as e:
            logger.error(f"Refinement search/rerank failed: {e}", exc_info=True)
            requested_searches = None; break

        # --- Process Refinement Results (Scrape, Chunk, Rerank Chunks) --- #
        if not reranked_refinement_search_results:
             logger.warning("Refinement search/rerank yielded no usable results. Ending refinement.")
             requested_searches = None; break

        logger.info(f"Processing {len(reranked_refinement_search_results)} new refinement results...")
        urls_for_refinement_scraping = [str(res.link) for res in reranked_refinement_search_results]
        refinement_scraped_data = await batch_scrape_urls_helper(urls_for_refinement_scraping, app_settings, logger)
        
        documents_for_refinement_chunking = []
        for result in reranked_refinement_search_results:
            link_str = str(result.link)
            extraction_result = refinement_scraped_data.get(link_str)
            if extraction_result and extraction_result.status == 'success' and extraction_result.content:
                 documents_for_refinement_chunking.append({
                    'content': extraction_result.content,
                    'metadata': {'link': link_str, 'title': result.title}
                 })
            else:
                 status_msg = f"status '{extraction_result.status if extraction_result else 'N/A'}'"
                 error_msg = f"error: {extraction_result.error_message}" if extraction_result and extraction_result.error_message else "no content"
                 logger.warning(f"Skipping refinement chunking for {link_str} due to scraping failure ({status_msg}, {error_msg}).")

        newly_added_material_in_loop = False
        if documents_for_refinement_chunking:
             try:
                 remaining_chunks = config.max_total_chunks - len(all_source_materials) if config.max_total_chunks else None
                 refinement_chunk_dicts = await chunk_content_helper(
                      documents_to_chunk=documents_for_refinement_chunking,
                      chunk_settings=config.default_chunk_settings.model_dump(),
                      max_chunks=remaining_chunks,
                      logger=logger
                 )

                 reranked_refinement_chunk_dicts = []
                 if refinement_chunk_dicts:
                      api_key = config.together_api_key.get_secret_value()
                      model = config.reranker_model
                      reranked_refinement_chunk_dicts = await rerank_chunks_helper(
                           query=user_query,
                           chunk_dicts=refinement_chunk_dicts,
                           model=model, api_key=api_key,
                           threshold=config.rerank_relevance_threshold,
                           logger=logger
                      )
                 
                 if reranked_refinement_chunk_dicts:
                      logger.info(f"Adding {len(reranked_refinement_chunk_dicts)} relevant refinement chunks.")
                      for chunk_dict in reranked_refinement_chunk_dicts:
                           link = chunk_dict.get('metadata', {}).get('link')
                           title = chunk_dict.get('metadata', {}).get('title')
                           content = chunk_dict.get('content')
                           score = chunk_dict.get('score')
                           if link and title and content:
                                rank = len(all_source_materials) + 1
                                added_material = { "rank": rank, "title": title, "link": link, "content": content, "type": "chunk", "score": score }
                                all_source_materials.append(added_material)
                                new_materials_in_this_loop.append(added_material)
                                link_str_chunk = str(link)
                                if link_str_chunk not in processed_links:
                                     source_metadata_map[rank] = {"title": title, "link": link_str_chunk}
                                     processed_links.add(link_str_chunk)
                                logger.debug(f"Added refinement chunk for {link_str_chunk} (Score: {score:.4f if score else 'N/A'}) rank {rank}")
                                newly_added_material_in_loop = True
                           else:
                                logger.warning(f"Skipping refinement chunk due to missing link/title/content")

             except Exception as e:
                 logger.error(f"Error during refinement chunking/reranking: {e}", exc_info=True)

        if not newly_added_material_in_loop:
             logger.warning("Refinement processing added no new chunks. Ending refinement loop.")
             requested_searches = None; break

        # --- Call Refiner Agent --- #
        logger.info("Calling Refiner Agent...")
        try:
            formatted_new_context = agents.format_summaries_for_prompt(new_materials_in_this_loop)
            refinement_topic = requested_searches[0].query if requested_searches else "requested information"
            refiner_user_prompt = agents._REFINEMENT_USER_MESSAGE_TEMPLATE.format(
                 user_query=user_query,
                 writing_plan_json=writing_plan.model_dump_json(indent=2),
                 previous_draft=current_draft,
                 refinement_topic=refinement_topic,
                 formatted_new_summaries=formatted_new_context,
            )
            refiner_result = await agents_collection.refiner.run(refiner_user_prompt, usage=usage_tracker)
            refiner_output: schemas.WriterOutput = refiner_result.data
            current_draft = refiner_output.report_content
            requested_searches = refiner_output.requested_searches
            logger.info("Refiner Agent generated revised draft.")
            if requested_searches:
                logger.info(f"Refiner requested {len(requested_searches)} further searches.")
            else:
                logger.info("Refiner did not request further searches.")
        except Exception as e:
            logger.error(f"Refiner agent failed during loop {refinement_loop_count}: {e}", exc_info=True)
            requested_searches = None; break

    if refinement_loop_count >= config.max_refinement_loops and requested_searches:
        logger.warning(f"Reached max refinement loops ({config.max_refinement_loops}) with searches still requested.")

    # === Step 6: Final Assembly ===
    logger.info("Assembling final report...")
    final_report = current_draft # Start with the latest draft

    try:
        reference_list = "\n\n## References\n\n"
        if source_metadata_map:
            # Sort by rank for consistent order
            sorted_ranks = sorted(source_metadata_map.keys())
            for rank in sorted_ranks:
                 metadata = source_metadata_map[rank]
                 # Ensure title and link exist, provide defaults if not (shouldn't happen with current logic)
                 title = metadata.get('title', 'Unknown Title')
                 link = metadata.get('link', '#')
                 reference_list += f"{rank}. {title} ({link})\n"
            final_report += reference_list
            logger.info("Appended reference list to the report.")
        else:
             logger.info("No sources were successfully processed to generate a reference list.")
    except Exception as e:
         logger.error(f"Failed to generate reference list: {e}", exc_info=True)
         # Report will not have the list, but is still usable

    # === Step 7: Return Result ===
    logger.info("Deep research orchestration complete.")
    final_usage_stats = usage_tracker.get_statistics() # Get final usage
    logger.info(f"Final Usage: {final_usage_stats}")

    return schemas.ResearchResponse(\
        report=final_report,\
        usage_statistics=final_usage_stats\
    )