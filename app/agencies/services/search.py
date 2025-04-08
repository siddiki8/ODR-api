import os
import httpx # Using httpx for async compatibility eventually, and aligns with potential agent use
import asyncio # For potential async conversion or testing
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

# Pydantic imports for schemas
import pydantic
from pydantic import BaseModel, Field, HttpUrl, ConfigDict, SecretStr

# Import custom exceptions from the core module
from app.core.exceptions import ConfigurationError, SearchAPIError, ValidationError
# Import SearchTask schema for type hinting
from app.core.schemas import SearchTask
from app.core.config import SerperConfig # Import from core.config

# --- Search Schema Definitions ---

class SearchResult(BaseModel):
    """
    Represents a single processed search result item from an external API like Serper.
    Provides a structured way to access common fields.
    Adapts to different search types (web, scholar, news) by making specific fields optional.
    """
    model_config = ConfigDict(extra='ignore', populate_by_name=True)
    title: str = Field(..., description="The title of the search result.")
    link: HttpUrl = Field(..., description="The primary URL of the search result.")
    snippet: Optional[str] = Field(None, description="A brief snippet or description from the search result.")
    position: Optional[int] = Field(None, description="The rank/position of the result (mainly for web search).")

    # Fields potentially available in scholar/news results
    publicationInfo: Optional[str] = Field(None, description="Publication information (e.g., authors, journal, year).")
    year: Optional[int] = Field(None, description="Publication year.")
    citedBy: Optional[int] = Field(None, description="Number of citations.")
    pdfUrl: Optional[HttpUrl] = Field(None, description="Direct link to a PDF version, if available.")
    resourceId: Optional[str] = Field(None, alias='id', description="Unique identifier for the result item (e.g., Google Scholar ID).")

class SearchResultList(BaseModel):
    """Represents a list of search results for a given query."""
    query: str = Field(..., description="The original search query.")
    results: List[SearchResult] = Field(default_factory=list, description="A list of search result items.")

# --- End Search Schema Definitions ---

# Import the underlying batch execution function
from .search_services.serper import search_serper_batch_service

logger = logging.getLogger(__name__)

# --- Removed SerperConfig Definition --- 
# @dataclass
# class SerperConfig:
#     ...
# --- End Removed SerperConfig Definition --- 

async def execute_batch_serper_search(
    search_tasks: List[SearchTask], # Expecting validated SearchTask objects
    config: Optional[SerperConfig] = None
) -> Dict[str, List[SearchResult]]: # Return Dict[query, List[SearchResult]]
    """
    Executes a batch search request using the configured Serper service and parses results.

    Args:
        search_tasks: A list of SearchTask Pydantic objects, each defining a query and endpoint.
        config: Optional SerperConfig instance. If None, loads settings from environment variables.

    Returns:
        A dictionary mapping the original query string to a list of parsed SearchResult objects.
        Returns an empty dictionary if no results are found or errors occur during parsing for all queries.

    Raises:
        ValueError: If search_tasks list is empty.
        ConfigurationError: If Serper API key is not configured.
        SearchAPIError: If there's a critical error during the API request or processing.
        Exception: For any other unexpected errors during the process.
    """
    if not search_tasks:
        raise ValueError("Search tasks list cannot be empty")

    # Load config from env if not provided, raises ConfigurationError if key missing
    cfg = config or SerperConfig.from_env()

    parsed_results_map: Dict[str, List[SearchResult]] = {task.query: [] for task in search_tasks}

    try:
        logger.info(f"Delegating batch search for {len(search_tasks)} tasks to serper service.")
        # Call the imported function from serper.py, passing the config
        # This returns List[Dict[str, Any]] - one dict per task
        raw_results_list: List[Dict[str, Any]] = await search_serper_batch_service(search_tasks=search_tasks, config=cfg)
        logger.info("Successfully received raw results from serper service.")

        # --- Parse Raw Results --- # 
        if len(raw_results_list) != len(search_tasks):
             logger.warning(f"Mismatch between number of search tasks ({len(search_tasks)}) and raw results received ({len(raw_results_list)}). Parsing available results.")
             # Continue parsing what we received, association might be off

        for i, raw_result_dict in enumerate(raw_results_list):
             # Try to associate based on index, fallback if lengths mismatch
             task = search_tasks[i] if i < len(search_tasks) else None
             query_key = task.query if task else f"unknown_query_{i}" # Use original query as key
             endpoint = task.endpoint if task else "/search"
             
             if query_key not in parsed_results_map: # Handle case where query wasn't in original map
                  parsed_results_map[query_key] = []
                  
             if not isinstance(raw_result_dict, dict):
                  logger.warning(f"Expected a dictionary for result index {i}, but got {type(raw_result_dict)}. Skipping.")
                  continue

             # Determine the key containing the list of results based on endpoint
             results_list_key = 'organic' # Default for /search and /scholar
             if endpoint == '/news':
                  results_list_key = 'news'
             
             raw_items = raw_result_dict.get(results_list_key, [])
             if not isinstance(raw_items, list):
                  logger.warning(f"Expected a list under key '{results_list_key}' for query '{query_key}', but got {type(raw_items)}. Skipping results for this query.")
                  continue

             parsed_count = 0
             for item_dict in raw_items:
                 if not isinstance(item_dict, dict):
                      logger.warning(f"Skipping non-dictionary item in results for query '{query_key}': {item_dict}")
                      continue
                 try:
                     # Rely directly on Pydantic model initialization for parsing and validation
                     parsed_item = SearchResult(**item_dict)
                     parsed_results_map[query_key].append(parsed_item)
                     parsed_count += 1
                 except pydantic.ValidationError as e: # Catch validation errors
                     # Log validation errors but don't stop processing other items
                     logger.warning(f"Validation error parsing search result item for query '{query_key}': {e}. Skipping item: {item_dict}")
                 except Exception as e:
                      # Catch other unexpected errors during instantiation
                      logger.error(f"Unexpected error creating SearchResult for query '{query_key}': {e}. Item: {item_dict}", exc_info=False)

             logger.debug(f"Parsed {parsed_count} valid result items for query: '{query_key}'.")

        logger.info(f"Finished parsing batch search results. Returning map with {len(parsed_results_map)} queries.")
        return parsed_results_map

    except (ConfigurationError, SearchAPIError, ValueError) as e:
        logger.error(f"Error during batch search execution or parsing: {type(e).__name__}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during batch search execution/parsing: {e}", exc_info=True)
        raise SearchAPIError(f"Unexpected error during batch search execution/parsing: {e}") from e

# Example Usage (Async)
async def _test_async():
    logging.basicConfig(level=logging.DEBUG)
    try:
        cfg = SerperConfig.from_env()
        tasks = [
            {"query": "latest AI research trends", "endpoint": "/search", "num_results": 3},
            {"query": "fastapi websocket tutorial", "endpoint": "/search"},
            {"query": "nonexistent query that might fail", "endpoint": "/search"} # Example
        ]
        tasks_obj = [SearchTask(query=t['query'], endpoint=t.get('endpoint', '/search'), num_results=t.get('num_results', 10)) for t in tasks]
        
        # Now expects Dict[str, List[SearchResult]]
        results_map: Dict[str, List[SearchResult]] = await execute_batch_serper_search(tasks_obj, cfg)
        
        print(f"\n--- Parsed Search Results Map ({len(results_map)} queries) ---")
        for query, results_list in results_map.items():
             print(f"  Query: '{query}' ({len(results_list)} results)")
             for idx, res in enumerate(results_list[:2]): # Print first 2 results
                  print(f"    {idx+1}. {res.title} ({res.link})")
             if len(results_list) > 2:
                  print(f"    ... ({len(results_list) - 2} more)")
             if not results_list:
                  print("    (No valid results parsed for this query)")

    except (ConfigurationError, SearchAPIError, ValueError) as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(_test_async())

# --- Removed web_search tool function --- 

# --- ADDED DEFINITIONS --- #
class SearchTask(BaseModel):
    """Represents a single search query task."""
    model_config = ConfigDict(extra='ignore')
    query: str = Field(..., description="The search query string.")
    endpoint: Optional[str] = Field("/search", description="API endpoint, e.g., /search or /images")
    num_results: Optional[int] = Field(10, description="Number of results to request.")
    reasoning: Optional[str] = Field(None, description="Why this search is being performed (for logging/tracing).")

class SearchResult(BaseModel):
    """Represents a single search result item."""
    model_config = ConfigDict(extra='ignore')
    title: str = Field(..., description="The title of the search result.")
    link: HttpUrl = Field(..., description="The URL link of the search result.")
    snippet: Optional[str] = Field(None, description="A brief snippet or description from the search result.")
    position: Optional[int] = Field(None, description="The rank/position of the result.")
# --- END ADDED DEFINITIONS --- #

# --- Service Logic --- #

logger = logging.getLogger(__name__)

# ... (rest of file remains the same) ... 