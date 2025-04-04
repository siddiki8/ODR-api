import os
import httpx # Using httpx for async compatibility eventually, and aligns with potential agent use
import asyncio # For potential async conversion or testing
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

# Import custom exceptions from the core module
from ..core.exceptions import ConfigurationError, SearchAPIError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result item."""
    title: str
    link: str
    snippet: str
    position: int
    # Add other relevant fields if needed, e.g., source, date
    raw: Dict[str, Any] # Store the original dictionary for flexibility

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Creates a SearchResult instance from a dictionary."""
        # Basic validation/extraction - adjust based on actual Serper response keys
        return cls(
            title=data.get('title', 'N/A'),
            link=data.get('link', '#'),
            snippet=data.get('snippet', ''),
            position=data.get('position', -1),
            raw=data
        )

@dataclass
class SerperConfig:
    """Configuration for Serper API"""
    api_key: str
    base_url: str = "https://google.serper.dev" # Use base URL for flexibility
    default_location: str = 'us'
    timeout: int = 15 # Increased timeout for batch potentially

    @classmethod
    def from_env(cls) -> 'SerperConfig':
        """Create config from environment variables"""
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            # Raise ConfigurationError instead of custom exception
            raise ConfigurationError("SERPER_API_KEY environment variable not set")
        base_url = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
        # Add validation for timeout from env? Example:
        timeout_str = os.getenv("SERPER_TIMEOUT", "15")
        try:
            timeout = int(timeout_str)
            if timeout <= 0:
                raise ValueError("Timeout must be positive")
        except ValueError:
            logger.warning(f"Invalid SERPER_TIMEOUT value '{timeout_str}'. Using default 15.")
            timeout = 15

        return cls(api_key=api_key, base_url=base_url, timeout=timeout)

async def execute_batch_serper_search(
    search_tasks: List[Dict[str, Any]], # Keep Dict hint for compatibility, but handle SearchTask objects
    config: Optional[SerperConfig] = None
) -> List[Dict[str, Any]]: # Return list directly, raise exceptions on error
    """
    Executes a batch search request to the Serper API asynchronously.

    Args:
        search_tasks: A list of dictionaries or SearchTask Pydantic objects, where each task
                      must contain 'query', 'endpoint', and optionally 'num_results'.
                      'endpoint' should be one of '/search', '/scholar', '/news'.
        config: Optional SerperConfig instance. If None, loads from environment variables.

    Returns:
        List containing the search results (one dict per task).

    Raises:
        ValueError: If search_tasks is empty or contains invalid task format.
        ConfigurationError: If Serper API key is not configured.
        SearchAPIError: If there's an error during the API request (network, timeout, non-2xx status) 
                      or if the API response format is unexpected.
        Exception: For any other unexpected errors.
    """
    if not search_tasks:
        # Raise ValueError for invalid input
        raise ValueError("Search tasks list cannot be empty")

    # Load config or raise ConfigurationError if env var missing
    cfg = config or SerperConfig.from_env()

    headers = {
        'X-API-KEY': cfg.api_key,
        'Content-Type': 'application/json'
    }
    search_endpoint = f"{cfg.base_url.rstrip('/')}/search"

    batch_payload_list = []
    for task in search_tasks:
        # Handle both Dict and Pydantic object access
        try:
            query = getattr(task, 'query') if hasattr(task, 'query') else task.get('query')
            endpoint = getattr(task, 'endpoint') if hasattr(task, 'endpoint') else task.get('endpoint')
            num_results = getattr(task, 'num_results', 10) if hasattr(task, 'num_results') else task.get('num_results', 10)
        except AttributeError as e:
             raise ValueError(f"Invalid task format: {task}. Could not access required attributes. Error: {e}")

        if not query or not endpoint:
            # Raise ValueError for invalid task format
            raise ValueError(f"Invalid task format: {task}. Must include 'query' and 'endpoint'.")

        endpoint_path = endpoint.lower()
        search_type = "search" # Default
        if endpoint_path == "/scholar":
            search_type = "scholar"
        elif endpoint_path == "/news":
            search_type = "news"
        elif endpoint_path != "/search":
             logger.warning(f"Unsupported endpoint '{endpoint}' in batch search. Defaulting to /search type.")

        payload_item = {
            "q": query,
            "type": search_type,
            "num": num_results, # Use extracted value
            "gl": cfg.default_location
        }
        batch_payload_list.append(payload_item)

    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        try:
            logger.debug(f"Sending batch search request to {search_endpoint} with {len(batch_payload_list)} tasks.")
            response = await client.post(
                search_endpoint,
                headers=headers,
                json=batch_payload_list
            )
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

            results_list = response.json()
            logger.debug("Received successful response from Serper batch API.")

            # Validate response structure
            if not isinstance(results_list, list) or len(results_list) != len(search_tasks):
                 error_msg = f"Unexpected response format from Serper batch API. Expected list of length {len(search_tasks)}, got {type(results_list).__name__} len {len(results_list) if isinstance(results_list, list) else 'N/A'}."
                 logger.error(error_msg)
                 # Raise SearchAPIError for unexpected format
                 raise SearchAPIError(error_msg)

            return results_list

        except httpx.TimeoutException as e:
            logger.error(f"Serper batch API request timed out to {search_endpoint}: {e}")
            raise SearchAPIError(f"Request to Serper API timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            response_text = e.response.text
            logger.error(f"Serper batch API request failed with status {e.response.status_code} to {search_endpoint}: {response_text}", exc_info=False)
            raise SearchAPIError(f"Serper API returned status {e.response.status_code}: {response_text}") from e
        except httpx.RequestError as e:
            # Catch other request errors like connection refused, DNS errors etc.
            logger.error(f"Serper batch API request failed for {search_endpoint}: {e}", exc_info=True)
            raise SearchAPIError(f"Serper API request failed: {e}") from e
        except Exception as e:
            # Catch JSONDecodeError or other unexpected errors
            logger.error(f"Unexpected error during Serper batch search processing: {e}", exc_info=True)
            # Re-raise as a generic SearchAPIError or let it propagate if specific handling isn't needed
            raise SearchAPIError(f"Unexpected error processing Serper response: {e}") from e

# Example Usage (Async)
async def _test_async():
    logging.basicConfig(level=logging.DEBUG)
    try:
        # Ensure SERPER_API_KEY is set in your .env file
        cfg = SerperConfig.from_env()
        tasks = [
            {"query": "latest AI research trends", "endpoint": "/search", "num_results": 3},
            {"query": "fastapi websocket tutorial", "endpoint": "/search"},
        ]
        results = await execute_batch_serper_search(tasks, cfg)
        print(f"Successfully retrieved {len(results)} sets of results.")
        # print(results)
    except (ConfigurationError, SearchAPIError, ValueError) as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(_test_async()) 