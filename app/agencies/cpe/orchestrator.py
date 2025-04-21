import logging
from urllib.parse import urlparse # Needed to extract domain
from typing import List, Dict, Any, Set, Optional # Added Optional

from .config import CPEConfig
from .schemas import (
    CPERequest, CPEResponse, CPEPlannerOutput, UsageStatistics, ExtractedCompanyData, CompanyProfile
)
from .helpers import (
    aggregate_html, 
    filter_and_validate_search_results,
    crawl_for_email_pages,
    make_company_profile
)
from .agents import get_cpe_agents, AgencyAgents
from .prompts import EXTRACTOR_USER_MESSAGE_TEMPLATE, CPE_PLANNER_USER_TEMPLATE
from app.core.schemas import RunUsage, SearchTask
from app.core.config import AppSettings, SerperConfig
from app.agencies.services.search import SearchResult, execute_batch_serper_search
# --- Add Firestore/Callback types for wrapper --- #
from google.cloud.firestore_v1.document import DocumentReference # For type hinting
from firebase_admin import firestore # To access SERVER_TIMESTAMP
from .callbacks import CpeWebSocketUpdateHandler # Import the handler

logger = logging.getLogger(__name__)

async def run_cpe(request: CPERequest, config: CPEConfig, serper_config: SerperConfig,
                   # --- Add Callback/Firestore args to main function --- #
                   update_callback: Optional[CpeWebSocketUpdateHandler] = None,
                   task_doc_ref: Optional[DocumentReference] = None,
                   firestore_available: bool = False
                   ) -> CPEResponse:
    """
    Executes the Company Profile Extractor workflow using planning and search.
    Handles callbacks and Firestore updates for progress.
    """
    agents = get_cpe_agents(config)
    usage_tracker = RunUsage()
    profiles: List[CompanyProfile] = []
    all_search_results: List[SearchResult] = []
    processed_domains: set[str] = set()

    logger.info(f"Starting CPE orchestration for query: '{request.query}', location: '{request.location}'")
    if update_callback: await update_callback.orchestration_start()
    
    # === Step 1: Planning ===
    search_tasks: List[SearchTask] = []
    try:
        logger.info("Calling Planner Agent...")
        if update_callback: await update_callback.planning_start()
        
        planner_user_prompt = CPE_PLANNER_USER_TEMPLATE.format(
            query=request.query,
            location=request.location or "Not specified",
            max_search_tasks=request.max_search_tasks
        )
        
        planner_result = await agents.planner.run(planner_user_prompt)
        usage_tracker.update_agent_usage("planner", planner_result.usage())

        planner_output: CPEPlannerOutput = planner_result.data
        search_tasks = planner_output.search_tasks
        logger.info(f"Planner generated {len(search_tasks)} search tasks.")
        
        # --- Update Firestore with Plan --- #
        if firestore_available and task_doc_ref:
            try:
                plan_update_data = {
                    "searchTasks": [t.model_dump(mode='json') for t in search_tasks],
                    "initialSearchTaskCount": len(search_tasks),
                    "status": "PLANNING_COMPLETE", 
                    "updatedAt": firestore.SERVER_TIMESTAMP
                }
                task_doc_ref.update(plan_update_data)
                logger.info(f"Updated Firestore with CPE plan.")
            except Exception as fs_e:
                logger.error(f"Failed to update Firestore with CPE plan: {fs_e}")
        # --- End Firestore Update --- #

        if update_callback: await update_callback.planning_end(len(search_tasks))

    except Exception as e:
        logger.error(f"Planner agent failed: {e}", exc_info=True)
        if update_callback: await update_callback.planning_error(e)
        # Return error response with current usage stats
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())

    # === Step 2: Execute Search Tasks ===
    if not search_tasks:
        logger.warning("Planner returned no search tasks. Cannot proceed.")
        # No need for error callback, this is expected flow
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())
        
    logger.info(f"Executing {len(search_tasks)} search tasks...")
    if update_callback: await update_callback.search_start(len(search_tasks))
    try:
        # Execute batch search using the real search service
        search_results_map = await execute_batch_serper_search(
            search_tasks=search_tasks,
            config=serper_config
        )
        usage_tracker.increment_serper_queries(len(search_tasks))
        
        # Use helper to filter and validate search results
        all_search_results, seen_links = filter_and_validate_search_results(search_results_map)
        
        # --- Update Firestore with Search Results (Links) --- #
        if firestore_available and task_doc_ref:
            try:
                search_update_data = {
                    "uniqueUrlsFound": list(seen_links),
                    "uniqueUrlCount": len(seen_links),
                    "status": "SEARCH_COMPLETE", 
                    "updatedAt": firestore.SERVER_TIMESTAMP
                }
                task_doc_ref.update(search_update_data)
                logger.info(f"Updated Firestore with {len(seen_links)} unique URLs.")
            except Exception as fs_e:
                logger.error(f"Failed to update Firestore with CPE search results: {fs_e}")
        # --- End Firestore Update --- #
        
        if update_callback: await update_callback.search_end(len(all_search_results))

    except Exception as e:
        logger.error(f"Search execution failed: {e}", exc_info=True)
        if update_callback: await update_callback.search_error(e)
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())

    # === Step 3: Process Search Results (Streaming Crawl & Extract) === 
    if not all_search_results:
        logger.warning("No valid URLs found after search execution. Cannot extract profiles.")
        # No need for error callback
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())
        
    logger.info(f"Processing {len(all_search_results)} unique search results sequentially...")
    if update_callback: await update_callback.extraction_start(len(seen_links))
    
    unique_start_urls = list(seen_links)
    for start_url_str in unique_start_urls:
        domain = "unknown" # Initialize domain for logging in case of early error
        try:
            parsed_url = urlparse(start_url_str)
            domain = parsed_url.netloc
            if not domain:
                 logger.warning(f"Could not parse domain from start URL: {start_url_str}. Skipping.")
                 if update_callback: await update_callback.extraction_domain_skipped(domain, "Could not parse domain")
                 continue
                 
            if domain in processed_domains:
                 logger.info(f"Domain {domain} (from {start_url_str}) already processed. Skipping crawl.")
                 if update_callback: await update_callback.extraction_domain_skipped(domain, "Already processed")
                 continue

            logger.info(f"Initiating processing for domain: {domain} (starting from: {start_url_str})")
            if update_callback: await update_callback.extraction_domain_start(domain, start_url_str)

            # --- 3a: Crawl & Find Emails using helper --- 
            if update_callback: await update_callback.extraction_crawling(domain)
            pages_with_new_emails = await crawl_for_email_pages(start_url_str, config)

            # Handle crawl failure or no emails found
            if pages_with_new_emails is None: # Indicates crawling failure
                if update_callback: await update_callback.extraction_domain_error(domain, Exception("Crawling failed"))
                processed_domains.add(domain) # Mark as processed to avoid retry
                continue
            if not pages_with_new_emails: # Indicates no new emails found
                logger.info(f"No pages with *new* emails found starting from {start_url_str}. Domain {domain} processing skipped.")
                if update_callback: await update_callback.extraction_domain_skipped(domain, "No new emails found")
                processed_domains.add(domain) # Mark as processed
                continue

            # --- 3b: Aggregate HTML for this domain --- 
            logger.info(f"Aggregating HTML from {len(pages_with_new_emails)} pages with new emails for domain {domain}.")
            if update_callback: await update_callback.extraction_aggregating(domain, len(pages_with_new_emails))
            html_blob = aggregate_html(pages_with_new_emails, max_bytes=200000) 
            
            if not html_blob.strip():
                logger.warning(f"Skipping domain {domain} due to empty aggregated HTML after filtering.")
                if update_callback: await update_callback.extraction_domain_skipped(domain, "Empty aggregated HTML")
                processed_domains.add(domain)
                continue
            
            del pages_with_new_emails # Memory Optimization

            # --- 3c: Extract Data using LLM --- 
            user_prompt = EXTRACTOR_USER_MESSAGE_TEMPLATE.format(
                html_blob=html_blob # Pass the aggregated HTML blob
            )
            del html_blob # Memory Optimization
            
            try:
                logger.debug(f"Calling extractor agent for domain: {domain}")
                if update_callback: await update_callback.extraction_calling_llm(domain)
                
                extractor_result = await agents.extractor.run(user_prompt)
                usage_tracker.update_agent_usage("extractor", extractor_result.usage())

                extracted_data: ExtractedCompanyData = extractor_result.data
                logger.debug(f"Extractor agent returned data for {domain}")

                # --- 3d: Create Profile --- 
                profile = make_company_profile(domain, extracted_data)
                
                if profile:
                    profiles.append(profile)
                    processed_domains.add(domain) # Mark domain as successfully processed
                    if update_callback: await update_callback.extraction_profile_success(domain)
                    
                    # --- Update Firestore with profile --- #
                    if firestore_available and task_doc_ref:
                        try:
                             # Use array_union to add profile to list if it exists, or create list
                            task_doc_ref.update({
                                "profiles": firestore.ArrayUnion([profile.model_dump(mode='json')]),
                                "status": "EXTRACTING", # Keep status as extracting during loop
                                "updatedAt": firestore.SERVER_TIMESTAMP
                            })
                            logger.info(f"Appended profile for {domain} to Firestore.")
                        except Exception as fs_e:
                            logger.error(f"Failed to append profile for {domain} to Firestore: {fs_e}")
                    # --- End Firestore Update --- #
                    
                else:
                    # Profile creation failed (error logged in helper)
                    if update_callback: await update_callback.extraction_domain_error(domain, Exception("Profile creation/validation failed"))
                    processed_domains.add(domain) # Mark domain processed even on failure
                    continue # Continue to next domain
            
            # --- Catch potential errors during agent execution --- 
            except TypeError as e:
                if "'NoneType' object cannot be interpreted as an integer" in str(e):
                    logger.error(f"Extractor agent failed for domain {domain} due to incomplete API response (likely missing 'created' timestamp). Skipping. Error: {e}", exc_info=False)
                    if update_callback: await update_callback.extraction_domain_error(domain, e)
                else:
                    logger.error(f"Extractor agent failed for domain {domain} with unexpected TypeError: {e}", exc_info=True)
                    if update_callback: await update_callback.extraction_domain_error(domain, e)
                    # Decide if we should re-raise or just continue? Let's continue for now.
                processed_domains.add(domain) # Mark domain as processed (due to error)
                continue # Continue to next domain
            except Exception as agent_error:
                logger.error(f"Extractor agent run failed for domain {domain}: {agent_error}", exc_info=True)
                if update_callback: await update_callback.extraction_domain_error(domain, agent_error)
                processed_domains.add(domain) # Mark domain processed even on agent failure
                continue # Continue to the next start URL

        except Exception as outer_loop_error:
             # Catch errors in domain parsing or the main loop logic for this URL
             logger.error(f"Unexpected error processing start URL {start_url_str} (domain: {domain}): {outer_loop_error}", exc_info=True)
             if update_callback: await update_callback.extraction_domain_error(domain, outer_loop_error)
             # Attempt to mark domain if possible, otherwise just continue
             try:
                 domain_to_mark = urlparse(start_url_str).netloc or domain
                 if domain_to_mark and domain_to_mark != "unknown":
                     processed_domains.add(domain_to_mark)
             except Exception: pass
             continue # Continue to the next start URL
             
    # === Final Steps === 
    if update_callback: await update_callback.extraction_end(len(processed_domains), len(profiles))
    if update_callback: await update_callback.finalizing_start()
    
    final_usage_stats: UsageStatistics = usage_tracker.get_statistics()
    
    logger.info(f"CPE run complete. Extracted {len(profiles)} profiles from {len(processed_domains)} processed domains.")
    logger.info(f"Final Usage: {final_usage_stats}")
    
    # --- Final Firestore Update --- #
    if firestore_available and task_doc_ref:
        try:
            final_update_data = {
                "status": "COMPLETED_PROCESSING", # More specific status before final return
                "profileCount": len(profiles),
                "processedDomainCount": len(processed_domains),
                "usageStatistics": final_usage_stats.model_dump(mode='json'),
                "completedProcessingAt": firestore.SERVER_TIMESTAMP
            }
            task_doc_ref.update(final_update_data)
            logger.info(f"Updated Firestore with final CPE processing stats.")
        except Exception as fs_e:
            logger.error(f"Failed to update Firestore with final CPE stats: {fs_e}")
    # --- End Firestore Update --- #

    if update_callback: await update_callback.finalizing_end()

    final_response = CPEResponse(profiles=profiles, usage_statistics=final_usage_stats)
    
    # Orchestrator callback for overall completion is handled by the route/wrapper now
    
    return final_response


# --- Wrapper Function --- #
async def run_cpe_wrapper(
    request: CPERequest,
    agents_collection: AgencyAgents,
    config: CPEConfig,
    app_settings: AppSettings,
    update_callback: Optional[CpeWebSocketUpdateHandler] = None,
    task_doc_ref: Optional[DocumentReference] = None,
    firestore_available: bool = False
) -> CPEResponse:
    """
    Wrapper for run_cpe to handle SerperConfig creation, errors, and callbacks.
    """
    # Construct SerperConfig from application settings
    try:
        serper_config = SerperConfig(
            api_key=app_settings.serper_api_key,
            base_url=app_settings.serper_base_url,
            default_location=app_settings.serper_default_location,
            timeout=app_settings.serper_timeout
        )
    except Exception as e:
        logger.critical(f"Failed to configure SerperConfig for CPE: {e}", exc_info=True)
        if update_callback:
            await update_callback.orchestration_error(ConfigurationError(f"Serper configuration failed: {e}"))
        if firestore_available and task_doc_ref:
            task_doc_ref.update({
                "status": "ERROR",
                "error": "Serper Config Error",
                "completedAt": firestore.SERVER_TIMESTAMP
            })
        return CPEResponse(profiles=[], usage_statistics=UsageStatistics())
    # Run main orchestration
    try:
        result = await run_cpe(
            request=request,
            config=config,
            serper_config=serper_config,
            update_callback=update_callback,
            task_doc_ref=task_doc_ref,
            firestore_available=firestore_available
        )
        return result
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED ERROR during CPE orchestration: {e}", exc_info=True)
        if update_callback:
            await update_callback.orchestration_error(e)
        if firestore_available and task_doc_ref:
            task_doc_ref.update({
                "status": "ERROR",
                "error": f"Critical Orchestration Error: {e}",
                "completedAt": firestore.SERVER_TIMESTAMP
            })
        return CPEResponse(profiles=[], usage_statistics=UsageStatistics()) 