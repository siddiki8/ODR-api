import logging
from urllib.parse import urlparse # Needed to extract domain
from pydantic import HttpUrl, ValidationError
from typing import List, Dict, Any

from .config import CPEConfig
from .schemas import (
    CPERequest, CPEResponse, CPEPlannerOutput, 
    ExtractedCompanyData, CompanyProfile, UsageStatistics
)
from .helpers import group_by_domain, aggregate_html
from .agents import get_cpe_agents
from .prompts import EXTRACTOR_USER_MESSAGE_TEMPLATE, CPE_PLANNER_USER_TEMPLATE
from app.core.schemas import RunUsage, SearchTask
from app.core.config import SerperConfig
from app.agencies.services.search import SearchResult, execute_batch_serper_search
from app.agencies.services.scraper_utils.emailfinder import find_emails_deep, EmailPageResult

logger = logging.getLogger(__name__)

async def run_cpe(request: CPERequest, config: CPEConfig, serper_config: SerperConfig) -> CPEResponse:
    """
    Executes the Company Profile Extractor workflow using planning and search.
    """
    agents = get_cpe_agents(config)
    usage_tracker = RunUsage()
    profiles: List[CompanyProfile] = []
    all_search_results: List[SearchResult] = []

    logger.info(f"Starting CPE orchestration for query: '{request.query}', location: '{request.location}'")
    
    # === Step 1: Planning ===
    search_tasks: List[SearchTask] = []
    try:
        logger.info("Calling Planner Agent...")
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

    except Exception as e:
        logger.error(f"Planner agent failed: {e}", exc_info=True)
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())

    # === Step 2: Execute Search Tasks ===
    if not search_tasks:
        logger.warning("Planner returned no search tasks. Cannot proceed.")
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())
        
    logger.info(f"Executing {len(search_tasks)} search tasks...")
    try:
        # Execute batch search using the real search service
        search_results_map = await execute_batch_serper_search(
            search_tasks=search_tasks,
            config=serper_config
        )
        usage_tracker.increment_serper_queries(len(search_tasks))
        
        seen_links = set()
        for task_query, results in search_results_map.items():
            for result in results:
                if hasattr(result, 'link') and result.link and result.link not in seen_links:
                    try:
                        validated_link = str(HttpUrl(result.link))
                        all_search_results.append(result)
                        seen_links.add(validated_link)
                    except ValidationError as link_val_error:
                        logger.warning(f"Skipping invalid search result link '{result.link}': {link_val_error}")
                elif not hasattr(result, 'link') or not result.link:
                     logger.debug(f"Skipping search result without a valid link: {result}")
        
        logger.info(f"Search tasks yielded {len(all_search_results)} unique valid results.")

    except Exception as e:
        logger.error(f"Search execution failed: {e}", exc_info=True)
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())

    # === Step 3: Process Search Results (Streaming Crawl & Extract) === 
    if not all_search_results:
        logger.warning("No valid URLs found after search execution. Cannot extract profiles.")
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())
        
    logger.info(f"Processing {len(all_search_results)} unique search results sequentially...")
    
    # Keep track of domains already fully processed to avoid redundant extraction 
    # if multiple search results point to the same domain
    processed_domains: set[str] = set()

    unique_start_urls = list(seen_links)
    for start_url_str in unique_start_urls:
        try:
            parsed_url = urlparse(start_url_str)
            domain = parsed_url.netloc
            if not domain:
                 logger.warning(f"Could not parse domain from start URL: {start_url_str}. Skipping.")
                 continue
                 
            if domain in processed_domains:
                 logger.info(f"Domain {domain} (from {start_url_str}) already processed. Skipping crawl.")
                 continue

            logger.info(f"Initiating processing for domain: {domain} (starting from: {start_url_str})")

            # --- 3a: Crawl & Find Emails for this specific start URL --- 
            pages_with_new_emails: List[EmailPageResult] = []
            try:
                # find_emails_deep now only returns pages with new emails and their HTML
                pages_with_new_emails, _ = await find_emails_deep(
                    start_url_str,
                    max_depth=config.max_crawl_depth,
                    max_pages=config.max_crawl_pages
                )
            except Exception as crawl_err:
                logger.error(f"Crawling/Email finding failed for start URL {start_url_str}: {crawl_err}", exc_info=True)
                # Decide if we should continue to the next URL or stop
                continue # Continue to next start URL on crawl failure

            if not pages_with_new_emails:
                logger.info(f"No pages with *new* emails found starting from {start_url_str}. Domain {domain} processing skipped.")
                # Mark domain as processed even if no emails found to avoid retrying
                processed_domains.add(domain)
                continue

            # --- 3b: Aggregate HTML for this domain --- 
            logger.info(f"Aggregating HTML from {len(pages_with_new_emails)} pages with new emails for domain {domain}.")
            # aggregate_html expects a list of EmailPageResult which have 'html'
            html_blob = aggregate_html(pages_with_new_emails, max_bytes=200000) 
            
            if not html_blob.strip():
                logger.warning(f"Skipping domain {domain} due to empty aggregated HTML after filtering.")
                processed_domains.add(domain)
                continue
            
            # Memory Optimization: Allow collected pages/HTML to be garbage collected sooner
            del pages_with_new_emails 

            # --- 3c: Extract Data using LLM --- 
            user_prompt = EXTRACTOR_USER_MESSAGE_TEMPLATE.format(
                html_blob=html_blob # Pass the aggregated HTML blob
            )
            
            # Memory Optimization: Allow large HTML blob to be garbage collected after use
            del html_blob 
            
            try:
                logger.debug(f"Calling extractor agent for domain: {domain}")
                extractor_result = await agents.extractor.run(user_prompt)
                usage_tracker.update_agent_usage("extractor", extractor_result.usage())

                extracted_data: ExtractedCompanyData = extractor_result.data
                logger.debug(f"Extractor agent returned data for {domain}")

                # --- 3d: Create Profile --- 
                try:
                    # Use the derived domain for the profile, ensuring it's a valid URL
                    domain_url_str = f"https://{domain}" # Assume HTTPS
                    domain_url = HttpUrl(domain_url_str)
                except ValidationError as url_val_error:
                    logger.error(f"Failed to construct valid HttpUrl for domain '{domain}' (derived from {domain_url_str}): {url_val_error}")
                    # Mark domain processed even on URL validation failure
                    processed_domains.add(domain) 
                    continue

                profile = CompanyProfile(
                    domain=domain_url,
                    **extracted_data.model_dump()
                )
                profiles.append(profile)
                logger.info(f"Successfully created CompanyProfile for domain: {domain}")
                processed_domains.add(domain) # Mark domain as successfully processed
            
            # --- Catch potential errors during agent execution --- 
            except TypeError as e:
                # Specifically catch the error caused by response.created being None
                if "'NoneType' object cannot be interpreted as an integer" in str(e):
                    logger.error(f"Extractor agent failed for domain {domain} due to incomplete API response (likely missing 'created' timestamp). Skipping. Error: {e}", exc_info=False)
                else:
                    # Re-raise other TypeErrors if they occur
                    logger.error(f"Extractor agent failed for domain {domain} with unexpected TypeError: {e}", exc_info=True)
                    raise e 
                processed_domains.add(domain) # Mark domain as processed (due to error) 
            except Exception as agent_error:
                # Catch any other unexpected errors during agent run or profile creation
                logger.error(f"Extractor agent run or profile creation failed for domain {domain}: {agent_error}", exc_info=True)
                # Mark domain processed even on agent failure to avoid retrying
                processed_domains.add(domain) 
                # Continue to the next start URL

        except Exception as outer_loop_error:
             # Catch errors in domain parsing or processing logic
             logger.error(f"Unexpected error processing start URL {start_url_str}: {outer_loop_error}", exc_info=True)
             # Attempt to mark domain if possible, otherwise just continue
             try:
                 domain_to_mark = urlparse(start_url_str).netloc
                 if domain_to_mark:
                     processed_domains.add(domain_to_mark)
             except Exception: pass
             continue # Continue to the next start URL
             
    # === Final Steps === 
    final_usage_stats: UsageStatistics = usage_tracker.get_statistics()
    
    logger.info(f"CPE run complete. Extracted {len(profiles)} profiles from {len(processed_domains)} processed domains.")
    logger.info(f"Final Usage: {final_usage_stats}")

    return CPEResponse(profiles=profiles, usage_statistics=final_usage_stats) 