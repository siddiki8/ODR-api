import os
import requests
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypeVar, Generic

T = TypeVar('T')

# Adapted from src.opendeepsearch.serp_search.serp_search
class SerperAPIException(Exception):
    """Custom exception for Serper API related errors"""
    pass

# Adapted from src.opendeepsearch.serp_search.serp_search
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
            raise SerperAPIException("SERPER_API_KEY environment variable not set")
        base_url = os.getenv("SERPER_BASE_URL", "https://google.serper.dev")
        return cls(api_key=api_key, base_url=base_url)

# Adapted from src.opendeepsearch.serp_search.serp_search
class SearchResult(Generic[T]):
    """Container for search results with error handling"""
    def __init__(self, data: Optional[T] = None, error: Optional[str] = None):
        self.data = data
        self.error = error
        self.success = error is None

    @property
    def failed(self) -> bool:
        return not self.success

def execute_batch_serper_search(
    search_tasks: List[Dict[str, Any]],
    config: Optional[SerperConfig] = None
) -> SearchResult[List[Dict[str, Any]]]:
    """
    Executes a batch search request to the Serper API.

    Args:
        search_tasks: A list of dictionaries, where each dictionary represents a search task
                      and must contain 'query', 'endpoint', and optionally 'num_results'.
                      'endpoint' should be one of '/search', '/scholar', '/news'.
        config: Optional SerperConfig instance. If None, loads from environment variables.

    Returns:
        SearchResult containing the list of search results (one dict per task) or an error.
        Note: Serper's batch response structure might vary. This function assumes it returns
        a list where each element corresponds to the result of a query in the request list.
    """
    if not search_tasks:
        return SearchResult(error="Search tasks list cannot be empty")

    try:
        cfg = config or SerperConfig.from_env()
        headers = {
            'X-API-KEY': cfg.api_key,
            'Content-Type': 'application/json'
        }
        # Corrected endpoint: Use /search for batch by sending a list
        search_endpoint = f"{cfg.base_url.rstrip('/')}/search"

        # Corrected payload: Send a list of query objects directly
        batch_payload_list = []
        for task in search_tasks:
            if not task.get('query') or not task.get('endpoint'):
                return SearchResult(error=f"Invalid task format: {task}. Must include 'query' and 'endpoint'.")

            # Map endpoint path to Serper type parameter
            endpoint_path = task['endpoint'].lower()
            search_type = "search" # Default
            if endpoint_path == "/scholar":
                search_type = "scholar"
            elif endpoint_path == "/news":
                search_type = "news"
            # Removed /images as it wasn't in the plan and might complicate batching
            elif endpoint_path != "/search":
                 print(f"Warning: Unsupported endpoint '{task['endpoint']}' in batch search. Defaulting to /search type.")
                 # Keep search_type as "search"

            payload_item = {
                "q": task['query'],
                "type": search_type,
                "num": task.get('num_results', 10), # Default to 10 results if not specified
                "gl": cfg.default_location # Add default location, can be overridden if needed
            }
            batch_payload_list.append(payload_item)

        response = requests.post(
            search_endpoint, # Use the correct endpoint
            headers=headers,
            json=batch_payload_list, # Send the list directly
            timeout=cfg.timeout
        )
        response.raise_for_status()
        # Assuming the response is a list of results, one for each query
        results_list = response.json()

        # Basic validation: Check if the number of results matches the number of tasks
        if not isinstance(results_list, list) or len(results_list) != len(search_tasks):
             raise SerperAPIException(f"Unexpected response format from batch API. Expected list of length {len(search_tasks)}, got {type(results_list)} len {len(results_list) if isinstance(results_list, list) else 'N/A'}.")

        # Potentially add more processing here if needed, e.g., adapting fields like in the original SerperAPI
        # For now, return the raw list of result dicts.

        return SearchResult(data=results_list)

    except requests.RequestException as e:
        return SearchResult(error=f"Serper batch API request failed: {str(e)}")
    except SerperAPIException as e:
         return SearchResult(error=f"Serper configuration or API error: {str(e)}")
    except Exception as e:
        return SearchResult(error=f"Unexpected error during batch Serper search: {str(e)}") 