import os
import httpx # Using httpx for async compatibility eventually, and aligns with potential agent use
import asyncio # For potential async conversion or testing
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

# Import custom exceptions from the core module
from ..core.exceptions import ConfigurationError, SearchAPIError, ValidationError
# Import SearchTask schema for type hinting
from ..core.schemas import SearchTask

logger = logging.getLogger(__name__)

@dataclass
class SerperConfig:
    """Configuration settings for the Serper Search API client."""
    api_key: str
    base_url: str = "https://google.serper.dev" # Base URL for the Serper API
    default_location: str = 'us' # Default geographical location for searches
    timeout: int = 15 # Default timeout in seconds for API requests

    @classmethod
    def from_env(cls) -> 'SerperConfig':
        """
        Creates a SerperConfig instance by loading settings from environment variables.
        
        Reads:
            SERPER_API_KEY (required)
            SERPER_BASE_URL (optional, default: https://google.serper.dev)
            SERPER_TIMEOUT (optional, default: 15)
            
        Raises:
            ConfigurationError: If SERPER_API_KEY is not set.
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ConfigurationError("SERPER_API_KEY environment variable not set")
        
        base_url = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
        
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
    search_tasks: List[SearchTask], # Expecting validated SearchTask objects
    config: Optional[SerperConfig] = None
) -> List[Dict[str, Any]]: # Returns raw response list from Serper API
    """
    Executes a batch search request to the Serper API asynchronously using httpx.

    Args:
        search_tasks: A list of SearchTask Pydantic objects, each defining a query and endpoint.
        config: Optional SerperConfig instance. If None, loads settings from environment variables.

    Returns:
        A list of dictionaries, where each dictionary corresponds to the raw JSON response 
        from the Serper API for one of the input search tasks.

    Raises:
        ValueError: If search_tasks list is empty.
        ConfigurationError: If Serper API key is not configured.
        SearchAPIError: If there's an error during the API request (network, timeout, non-2xx status) 
                      or if the API response format is unexpected.
        Exception: For any other unexpected errors during the process.
    """
    if not search_tasks:
        raise ValueError("Search tasks list cannot be empty")

    # Load config from env if not provided, raises ConfigurationError if key missing
    cfg = config or SerperConfig.from_env()

    headers = {
        'X-API-KEY': cfg.api_key,
        'Content-Type': 'application/json'
    }
    # Ensure base_url does not have a trailing slash before appending
    search_endpoint = f"{cfg.base_url.rstrip('/')}/search"

    batch_payload_list = []
    for task in search_tasks:
        # Directly access attributes from the validated SearchTask object
        query = task.query
        endpoint = task.endpoint # Already validated by SearchTask schema
        num_results = task.num_results
        
        # Map the validated endpoint path to Serper's 'type' parameter
        search_type = "search" # Default
        if endpoint == "/scholar":
            search_type = "scholar"
        elif endpoint == "/news":
            search_type = "news"
        # No need to check for unsupported endpoints here, SearchTask validation handles it

        payload_item = {
            "q": query,
            "type": search_type,
            "num": num_results,
            "gl": cfg.default_location
        }
        batch_payload_list.append(payload_item)

    async with httpx.AsyncClient(timeout=cfg.timeout) as client:
        try:
            logger.info(f"Sending batch search request to {search_endpoint} with {len(batch_payload_list)} tasks.")
            logger.debug(f"Payload: {batch_payload_list}") # Log payload for debugging
            response = await client.post(
                search_endpoint,
                headers=headers,
                json=batch_payload_list
            )
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

            results_list = response.json()
            logger.info("Received successful response from Serper batch API.")

            # Basic validation on the overall response structure
            if not isinstance(results_list, list) or len(results_list) != len(search_tasks):
                error_msg = (
                    f"Unexpected response format from Serper batch API. "
                    f"Expected list of length {len(search_tasks)}, got {type(results_list).__name__} "
                    f"length {len(results_list) if isinstance(results_list, list) else 'N/A'}."
                )
                logger.error(error_msg)
                raise SearchAPIError(error_msg)

            # Return the raw list of results; parsing individual items happens upstream
            return results_list

        except httpx.TimeoutException as e:
            logger.error(f"Serper batch API request timed out to {search_endpoint}: {e}")
            raise SearchAPIError(f"Request to Serper API timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            # Try to get more info from response if possible
            response_text = "(No response body)"
            try:
                 response_text = e.response.text
            except Exception:
                 pass # Ignore if reading response text fails
            logger.error(f"Serper batch API request failed with status {e.response.status_code} to {search_endpoint}: {response_text}", exc_info=False)
            raise SearchAPIError(f"Serper API returned status {e.response.status_code}: {response_text}") from e
        except httpx.RequestError as e:
            logger.error(f"Serper batch API request failed for {search_endpoint}: {e}", exc_info=False)
            raise SearchAPIError(f"Serper API request failed: {e}") from e
        except Exception as e:
            # Catch JSONDecodeError or other unexpected errors
            logger.error(f"Unexpected error during Serper batch search processing: {e}", exc_info=True)
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