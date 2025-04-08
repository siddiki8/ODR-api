import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import HttpUrl
import re

# --- Internal Imports ---
from app.core.exceptions import ScrapingError, ConfigurationError, SearchAPIError, RankingAPIError, ChunkingError
from ..services.scraper import WebScraper, ExtractionResult
from ..services.search import SearchResult, SearchTask
from ..services.ranking import rerank_with_together_api, RankedItem

# --- Service Imports ---
from ..services.search import execute_batch_serper_search, SerperConfig
from ..services import chunking as chunking_service

# --- AppSettings Import --- #
from app.core.config import AppSettings


# --- Helper Functions --- #
async def chunk_content_helper(
    documents_to_chunk: List[Dict[str, Any]], # Expects list of {'content': str, 'metadata': {'link': str, ...}}
    chunk_settings: Dict[str, int], # e.g., {'chunk_size': ..., 'chunk_overlap': ..., 'min_chunk_size': ...}
    max_chunks: Optional[int],
    logger: logging.Logger
) -> Dict[str, List[Dict[str, Any]]]: # Return Dict[url_str, List[chunk_dict]]
    """Chunks content from multiple documents using the chunking service.
       Groups the resulting chunks by their original source URL.

    Args:
        documents_to_chunk: List of dictionaries, each with 'content' and 'metadata' (must include 'link').
        chunk_settings: Dictionary with chunking parameters.
        max_chunks: Optional overall limit on the number of chunks generated.
        logger: Logger instance.

    Returns:
        A dictionary where keys are source URLs (str) and values are lists of
        chunk dictionaries associated with that URL.
        Returns empty dict on failure or if no chunks are generated.
    """
    if not documents_to_chunk:
        logger.info("No documents provided to chunk_content_helper.")
        return {}

    # Ensure required keys are present in chunk_settings with defaults if necessary
    chunk_size = chunk_settings.get('chunk_size', 2048)
    chunk_overlap = chunk_settings.get('chunk_overlap', 100)
    min_chunk_size = chunk_settings.get('min_chunk_size', 256)

    logger.info(f"Starting chunking for {len(documents_to_chunk)} documents...")
    try:
        # Call the chunking service function
        all_chunked_dicts = chunking_service.chunk_and_label(
            documents=documents_to_chunk,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunks=max_chunks
        )
        logger.info(f"Chunking service generated {len(all_chunked_dicts)} total chunk dictionaries before grouping.")

        # Group chunks by source URL
        grouped_chunks: Dict[str, List[Dict[str, Any]]] = {}
        skipped_count = 0
        for chunk in all_chunked_dicts:
            source_link = chunk.get('metadata', {}).get('link')
            if source_link:
                url_str = str(source_link) # Ensure it's a string key
                if url_str not in grouped_chunks:
                    grouped_chunks[url_str] = []
                grouped_chunks[url_str].append(chunk)
            else:
                skipped_count += 1
                logger.warning(f"Chunk missing source link in metadata, skipping: {chunk.get('content', '')[:50]}...")

        if skipped_count > 0:
             logger.warning(f"Skipped {skipped_count} chunks due to missing source link metadata.")

        logger.info(f"Grouped chunks by source URL. Result has {len(grouped_chunks)} sources.")
        return grouped_chunks
    except ChunkingError as e:
        logger.error(f"Chunking service failed: {e}", exc_info=False)
        return {}
    except Exception as e:
        logger.error(f"Unexpected error during chunking helper execution: {e}", exc_info=True)
        return {}


# --- Search Helper --- #

async def execute_search_queries(
    search_tasks: List[SearchTask],
    config: SerperConfig,
    logger: logging.Logger
) -> Dict[str, List[SearchResult]]:
    """Executes batch search queries using the search service and returns parsed results.
    
    Args:
        search_tasks: List of SearchTask objects defining the queries.
        config: SerperConfig instance for the search service.
        logger: Logger instance.

    Returns:
        A dictionary mapping original query strings to lists of parsed SearchResult objects.
        Returns an empty dictionary if the service call fails critically.
    """
    stage = "SEARCHING_HELPER"
    if not search_tasks:
        logger.warning(f"[{stage}] No search tasks provided.")
        return {}

    logger.info(f"[{stage}] Executing {len(search_tasks)} search tasks...")
    try:
        # Call the service function which now returns Dict[str, List[SearchResult]]
        search_results_map = await execute_batch_serper_search(
            search_tasks=search_tasks,
            config=config
        )
        
        total_results_count = sum(len(v) for v in search_results_map.values())
        logger.info(f"[{stage}] Search service returned results for {len(search_results_map)} queries, {total_results_count} total results parsed.")
        return search_results_map
        
    except (ConfigurationError, SearchAPIError, ValueError) as e:
        # Log errors encountered during the search service call
        logger.error(f"[{stage}] Search service failed: {type(e).__name__}: {e}", exc_info=False)
        # Return empty dict to indicate failure to the caller (agent)
        return {}
    except Exception as e:
        # Catch unexpected errors
        logger.error(f"[{stage}] Unexpected error during search execution: {e}", exc_info=True)
        return {}


# --- Content Fetching Helper --- #

async def fetch_source_content(
    url: Union[str, HttpUrl],
    source: SearchResult,  # Provides title etc.
    scraper: WebScraper,  # Passed in
    logger: logging.Logger # Passed in
) -> Optional[ExtractionResult]: # Returns the result from the scraper
    """Fetches content from a URL using the scraper service.

    Handles basic error logging but primarily delegates to the scraper.

    Args:
        url: The URL of the source to fetch.
        source: The SearchResult object for metadata (mainly for logging context).
        scraper: An initialized WebScraper instance.
        logger: Logger instance for logging messages.

    Returns:
        An ExtractionResult object from the scraper service, or None if scraping fails critically.
    """
    url_str = str(url)
    stage = "FETCHING_HELPER" # More specific stage name

    try:
        # STEP 1: Scrape content
        logger.info(f"[{stage}] Fetching content from: {url_str}")
        # Directly call the scraper service
        scrape_result: ExtractionResult = await scraper.scrape(url_str)

        # STEP 2: Check status and log
        if scrape_result and scrape_result.status == "success":
             content_len = len(scrape_result.content) if scrape_result.content else 0
             logger.debug(f"[{stage}] Successfully scraped {content_len} chars from {url_str} (source: {scrape_result.extraction_source})")
        elif scrape_result and scrape_result.status == "empty":
             logger.warning(f"[{stage}] Scraper returned empty content for {url_str} (source: {scrape_result.extraction_source}).")
        elif scrape_result and scrape_result.status == "error":
             logger.error(f"[{stage}] Scraper returned error for {url_str}: {scrape_result.error_message}")
        else:
            # This case handles if scrape_result is None or status is unexpected
            logger.error(f"[{stage}] Scraping did not return a valid ExtractionResult for {url_str}. Result: {scrape_result}")
            return None # Indicate critical failure

        # Return the entire result object, letting the caller decide how to handle status/content
        return scrape_result

    except ScrapingError as e:
        # Errors raised directly by scraper.scrape (e.g., invalid URL type before call)
        logger.error(f"[{stage}] Scraping error calling scraper for {url_str}: {e}", exc_info=False)
        return None
    except Exception as e:
        # Catch-all for unexpected errors during the fetch call
        logger.error(f"[{stage}] Unexpected error fetching content for {url_str}: {e}", exc_info=True)
        return None


# --- Reranking Helpers --- #

async def rerank_search_results_helper(
    query: str,
    search_results: List[SearchResult],
    model: str,
    api_key: str,
    threshold: float,
    logger: logging.Logger
) -> List[SearchResult]:
    """Reranks SearchResult objects based on query relevance using the ranking service.

    Args:
        query: The user query.
        search_results: The list of SearchResult objects to rerank.
        model: Reranker model identifier.
        api_key: API key for the ranking service.
        threshold: Relevance score threshold.
        logger: Logger instance.

    Returns:
        A list of SearchResult objects filtered and sorted by relevance score, 
        or the original list if reranking fails.
    """
    if not search_results:
        return []

    # Create document strings for reranking (Title + Snippet)
    docs_for_reranking = [
        f"{res.title}\n{res.snippet}" \
        for res in search_results if res.snippet # Only use results with snippets
    ]
    # Map original indices to results actually used for reranking
    indices_used = [i for i, res in enumerate(search_results) if res.snippet]

    if not docs_for_reranking:
        logger.warning("No search results had snippets suitable for reranking. Returning original list.")
        return search_results

    logger.info(f"Reranking {len(docs_for_reranking)} search results (Threshold: {threshold})...")
    try:
        # Call the ranking service
        relevant_ranked_items: List[RankedItem] = await rerank_with_together_api(
            query=query,
            documents=docs_for_reranking,
            model=model,
            api_key=api_key,
            relevance_threshold=threshold
        )
        logger.info(f"Search result reranking yielded {len(relevant_ranked_items)} items above threshold.")

        # Select original SearchResult objects based on reranked indices
        reranked_results_list = []
        for item in relevant_ranked_items:
            original_index_in_docs_list = item.index
            # Map back to the index in the original search_results list
            original_list_index = indices_used[original_index_in_docs_list]
            if 0 <= original_list_index < len(search_results):
                selected_result = search_results[original_list_index]
                reranked_results_list.append(selected_result)
                # Optionally add score to result or log it
                logger.debug(f"Kept search result index {original_list_index} (Score: {item.score:.4f}): {selected_result.title}")
            else:
                logger.warning(f"Reranker returned invalid mapped index {original_list_index}")
        return reranked_results_list

    except RankingAPIError as e:
        logger.error(f"Ranking API error during search result reranking: {e}", exc_info=False)
        logger.warning("Falling back to using original (unranked) search results.")
        return search_results # Fallback
    except Exception as e:
        logger.error(f"Unexpected error during search result reranking: {e}", exc_info=True)
        logger.warning("Falling back to using original (unranked) search results.")
        return search_results # Fallback


async def rerank_chunks_helper(
    query: str,
    chunk_dicts: List[Dict[str, Any]],
    model: str,
    api_key: str,
    threshold: float,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Reranks text chunks based on query relevance using the ranking service.

    Args:
        query: The user query.
        chunk_dicts: List of dictionaries, each representing a chunk 
                     (must contain 'content' key).
        model: Reranker model identifier.
        api_key: API key for the ranking service.
        threshold: Relevance score threshold.
        logger: Logger instance.

    Returns:
        A list of chunk dictionaries filtered and sorted by relevance score, 
        with the score added to the dictionary under the key 'score'.
        Returns an empty list if reranking fails critically or no chunks pass.
    """
    if not chunk_dicts:
        return []

    docs_for_reranking = [c.get('content', '') for c in chunk_dicts]
    indices_used = [i for i, doc in enumerate(docs_for_reranking) if doc]
    docs_for_reranking = [doc for doc in docs_for_reranking if doc]

    if not docs_for_reranking:
        logger.warning("No chunk dictionaries had content suitable for reranking.")
        return []

    logger.info(f"Reranking {len(docs_for_reranking)} chunks (Threshold: {threshold})...")
    try:
        relevant_ranked_items: List[RankedItem] = await rerank_with_together_api(
            query=query,
            documents=docs_for_reranking,
            model=model,
            api_key=api_key,
            relevance_threshold=threshold
        )
        logger.info(f"Chunk reranking yielded {len(relevant_ranked_items)} items above threshold.")

        reranked_chunks_list = []
        for item in relevant_ranked_items:
            original_index_in_docs_list = item.index
            original_list_index = indices_used[original_index_in_docs_list]
            if 0 <= original_list_index < len(chunk_dicts):
                selected_chunk = chunk_dicts[original_list_index].copy() # Copy to avoid modifying original
                selected_chunk['score'] = item.score # Add the score
                reranked_chunks_list.append(selected_chunk)
                link = selected_chunk.get('metadata', {}).get('link', 'N/A')
                logger.debug(f"Kept chunk index {original_list_index} (Score: {item.score:.4f}) from {link}")
            else:
                logger.warning(f"Reranker returned invalid mapped chunk index {original_list_index}")
        return reranked_chunks_list
    except RankingAPIError as e:
        logger.error(f"Chunk reranking service failed: {e}", exc_info=False)
        # On failure, return empty list as we can't determine relevance
        return []
    except Exception as e:
        logger.error(f"Unexpected error during chunk reranking: {e}", exc_info=True)
        return []


# --- Scraping Helper --- #

async def batch_scrape_urls_helper(
    urls: List[str],
    settings: AppSettings, # Pass AppSettings
    logger: logging.Logger
) -> Dict[str, ExtractionResult]:
    """Scrapes a list of URLs using the WebScraper service.

    Args:
        urls: A list of URL strings to scrape.
        settings: The application settings instance.
        logger: Logger instance.

    Returns:
        A dictionary mapping each URL to its ExtractionResult.
        Handles errors internally, returning ExtractionResult with error status.
    """
    if not urls:
        return {}

    logger.info(f"Initializing WebScraper for batch scraping {len(urls)} URLs...")
    try:
        # Initialize scraper within the helper
        # Assuming AppSettings contains necessary paths like scraper_pdf_save_dir
        scraper = WebScraper(settings=settings)
        # Call the scraper's batch method
        results = await scraper.scrape_many(urls)
        logger.info(f"Batch scraping helper completed. Results count: {len(results)}")
        return results
    except ConfigurationError as e:
        logger.error(f"Configuration error initializing WebScraper for batch job: {e}")
        # Return errors for all requested URLs
        return {url: ExtractionResult(source_url=url, status="error", error_message=f"Scraper config error: {e}") for url in urls}
    except Exception as e:
        logger.error(f"Unexpected error during batch scraping helper execution: {e}", exc_info=True)
        return {url: ExtractionResult(source_url=url, status="error", error_message=f"Batch helper error: {e}") for url in urls}


# --- Citation Processing Helpers (for Orchestrator Step 6) --- #

def format_report_citations(report_content: str, logger: logging.Logger) -> str:
    """Replaces [[CITATION:num1,num2]] markers with clickable Markdown links.

    Args:
        report_content: The content of the report.
        logger: Logger instance for logging warnings and errors.

    Returns:
        The processed content with clickable citation links.
    """
    def replace_citation_marker(match):
        numbers_str = match.group(1)
        links = []
        for num_str in map(str.strip, numbers_str.split(',')):
            if num_str.isdigit():
                links.append(f"[{num_str}](#ref-{num_str})")
            else:
                logger.warning(f"Found non-digit citation number '{num_str}' in marker: {match.group(0)}")
                links.append(f"[{num_str}]") # Include as non-link
        return ", ".join(links)

    citation_pattern = r"\[\[CITATION:(\d+(?:s*,\s*\d+)*)\]\]"
    try:
        processed_content = re.sub(citation_pattern, replace_citation_marker, report_content)
        logger.info("Processed report content to add clickable citation links.")
        return processed_content
    except Exception as regex_err:
        logger.error(f"Error processing citation links with regex: {regex_err}", exc_info=True)
        return report_content # Fallback to original content

def generate_reference_list(unique_sources: Dict[str, Dict[str, Any]], logger: logging.Logger) -> str:
    """Generates a Markdown formatted reference list with HTML anchors.

    Args:
        unique_sources: A dictionary mapping source URLs to their metadata.
        logger: Logger instance for logging warnings and errors.

    Returns:
        The generated reference list content.
    """
    reference_list_content = "\n\n## References\n\n"
    if unique_sources:
        try:
            # Sort sources by their assigned reference number
            sorted_sources = sorted(unique_sources.values(), key=lambda item: item['ref_num'])
            for source_info in sorted_sources:
                ref_num = source_info['ref_num']
                title = source_info.get('title', 'Unknown Title')
                link = source_info.get('link', '#')
                # Add HTML anchor tag before the number (Fixed f-string)
                reference_list_content += f"<a name=\"ref-{ref_num}\"></a>{ref_num}. {title} ({link})\n"
            logger.info(f"Generated reference list with {len(sorted_sources)} unique sources and anchors.")
        except Exception as e:
            logger.error(f"Failed during reference list generation: {e}", exc_info=True)
            # Return a minimal list or indicate error?
            reference_list_content += "*Error generating reference list.*\n"
    else:
        logger.info("No unique sources were processed to generate a reference list.")
        reference_list_content += "*No sources cited.*\n"
    return reference_list_content
