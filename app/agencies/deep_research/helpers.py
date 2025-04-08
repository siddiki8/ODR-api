import logging
from typing import List, Optional, Dict, Any, Union
from pydantic import HttpUrl

# --- Internal Imports ---
from ..core.exceptions import ScrapingError, ConfigurationError, SearchAPIError, RankingAPIError, ChunkingError
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
    documents_to_chunk: List[Dict[str, Any]], # Expects list of {'content': str, 'metadata': {...}}
    chunk_settings: Dict[str, int], # e.g., {'chunk_size': ..., 'chunk_overlap': ..., 'min_chunk_size': ...}
    max_chunks: Optional[int],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Chunks content from multiple documents using the chunking service.

    Args:
        documents_to_chunk: List of dictionaries, each with 'content' and 'metadata'.
        chunk_settings: Dictionary with chunking parameters.
        max_chunks: Optional overall limit on the number of chunks generated.
        logger: Logger instance.

    Returns:
        A list of chunk dictionaries, ready for reranking or context assembly.
        Returns empty list on failure.
    """
    if not documents_to_chunk:
        logger.info("No documents provided to chunk_content_helper.")
        return []

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
        logger.info(f"Chunking service generated {len(all_chunked_dicts)} total chunk dictionaries.")
        return all_chunked_dicts
    except ChunkingError as e:
        logger.error(f"Chunking service failed: {e}", exc_info=False)
        return []
    except Exception as e:
        logger.error(f"Unexpected error during chunking helper execution: {e}", exc_info=True)
        return []


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
