import os
import httpx
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Import schemas needed - Adjust path if SearchTask/exceptions are elsewhere
from app.core.schemas import SearchTask
from app.core.exceptions import ConfigurationError, SearchAPIError

# Import the shared config class
from ..search import SerperConfig

logger = logging.getLogger(__name__)

# --- Removed local SerperConfig --- #

# --- Removed single search function (search_serper_service) --- #
# (Assuming batch is the primary interface needed by the service)

async def search_serper_batch_service(
    search_tasks: List[SearchTask],
    # api_key: str, # Removed, get from config
    config: SerperConfig # Now expects the config object from search.py
) -> List[Dict[str, Any]]:
    """
    Executes a batch search request to the Serper API.
    Called by the search service in search.py.
    """
    if not search_tasks:
        raise ValueError("Search tasks list cannot be empty")
    if not config or not config.api_key:
        raise ConfigurationError("Serper configuration with API key was not provided.")

    headers = {
        'X-API-KEY': config.api_key, # Use config object
        'Content-Type': 'application/json'
    }
    batch_request_url = f"{config.base_url.rstrip('/')}/search"

    batch_payload_list = []
    for task in search_tasks:
        search_type = "search"
        if task.endpoint == "/scholar": search_type = "scholar"
        elif task.endpoint == "/news": search_type = "news"
        
        payload_item = {
            "q": task.query,
            "type": search_type,
            "num": task.num_results,
            "gl": config.default_location # Use config object
        }
        batch_payload_list.append(payload_item)

    async with httpx.AsyncClient(timeout=config.timeout) as client: # Use config object
        try:
            logger.info(f"Sending batch search request to {batch_request_url} with {len(batch_payload_list)} tasks.")
            logger.debug(f"Batch Payload: {batch_payload_list}")
            response = await client.post(batch_request_url, headers=headers, json=batch_payload_list)
            response.raise_for_status()
            results_list = response.json()
            logger.info("Received successful response from Serper batch API.")

            if not isinstance(results_list, list) or len(results_list) != len(search_tasks):
                error_msg = (
                    f"Unexpected response format from Serper batch API. Expected list of "
                    f"length {len(search_tasks)}, got {type(results_list).__name__} length "
                    f"{len(results_list) if isinstance(results_list, list) else 'N/A'}."
                )
                logger.error(error_msg)
                raise SearchAPIError(error_msg)
                
            return results_list

        except httpx.TimeoutException as e:
            logger.error(f"Serper batch API request timed out to {batch_request_url}: {e}")
            raise SearchAPIError(f"Request to Serper batch API timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            response_text = "(No response body)"
            try: response_text = e.response.text
            except Exception: pass
            logger.error(f"Serper batch API request failed with status {e.response.status_code}: {response_text}", exc_info=False)
            raise SearchAPIError(f"Serper batch API returned status {e.response.status_code}: {response_text}") from e
        except httpx.RequestError as e:
            logger.error(f"Serper batch API request failed for {batch_request_url}: {e}", exc_info=False)
            raise SearchAPIError(f"Serper batch API request failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during Serper batch search: {e}", exc_info=True)
            raise SearchAPIError(f"Unexpected error processing Serper batch response: {e}") from e

# Optional test function remains similar
async def _test_service():
    # ... (Test function might need adjustments to create SerperConfig)
    logging.basicConfig(level=logging.DEBUG)
    # api_key = os.getenv("SERPER_API_KEY") # Removed direct use
    # if not api_key:
    #     print("Skipping test: SERPER_API_KEY not set.")
    #     return
    try:
        cfg = SerperConfig.from_env() # Test creating config here
        print("\n--- Testing Batch Search ---")
        tasks = [
            SearchTask(query="Python decorators", endpoint="/search", num_results=2),
            SearchTask(query="latest fastapi features", endpoint="/search", num_results=3)
        ]
        # Pass the config object
        batch_results = await search_serper_batch_service(tasks, config=cfg)
        print(f"Batch search successful. Retrieved {len(batch_results)} result sets.")
    except (ConfigurationError, SearchAPIError, ValueError) as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(_test_service()) 