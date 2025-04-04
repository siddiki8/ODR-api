"""
Service for reranking search results using Together AI Rerank API.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx # Using httpx for consistency and potential async needs
import asyncio # For testing
from litellm import completion # Assuming litellm is used for the ranking model

# Import custom exceptions from the core module
from ..core.exceptions import ConfigurationError, RankingAPIError

# Configure logger
logger = logging.getLogger(__name__)

# Default Base URL for Together API
TOGETHER_API_BASE = os.getenv("TOGETHER_API_BASE", "https://api.together.ai")

async def rerank_with_together_api(
    query: str,
    documents: List[str],
    model: str,
    api_key: str,
    relevance_threshold: float = 0.1,
    top_n: Optional[int] = None, # Allow specifying top_n if needed
    timeout: int = 20 # Default timeout for the API call
) -> List[Dict[str, Any]]:
    """
    Reranks documents based on a query using the Together Rerank API asynchronously.

    Args:
        query: The search query.
        documents: A list of document strings to rerank.
        model: The reranker model identifier (e.g., 'Salesforce/Llama-Rank-V1').
        api_key: Together API key (must be provided).
        relevance_threshold: Documents with scores below this threshold will be filtered out (default: 0.1).
        top_n: Optional number of top documents to return after reranking (before threshold filtering).
               If None, the API defaults (usually all documents scored).
        timeout: Timeout in seconds for the API request.

    Returns:
        A list of reranked results (dictionaries with 'index', 'score'),
        sorted by relevance score (descending), filtered by threshold.

    Raises:
        ValueError: If query or documents are empty/invalid.
        ConfigurationError: If the api_key is not provided.
        RankingAPIError: If the API call fails (network, timeout, non-2xx status) or returns an unexpected format.
    """
    if not api_key:
        raise ConfigurationError("Together API key must be provided.")
    if not query:
        raise ValueError("Query cannot be empty for reranking.")
    if not documents:
        raise ValueError("Documents list cannot be empty for reranking.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n
    else:
         # Explicitly ask for all scores if top_n is not specified by caller
         # Note: Check Together API docs if this behavior changes
         payload["top_n"] = len(documents)

    rerank_endpoint = f"{TOGETHER_API_BASE.rstrip('/')}/v1/rerank"

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            logger.debug(f"Sending rerank request to {rerank_endpoint} for {len(documents)} docs, model: {model}")
            response = await client.post(rerank_endpoint, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx

            response_data = response.json()
            logger.debug("Received successful response from Together Rerank API.")

            # Validate response structure
            if not isinstance(response_data, dict) or 'results' not in response_data or not isinstance(response_data['results'], list):
                error_msg = f"Unexpected response format from Together Rerank API. Missing 'results' list. Response: {response_data}"
                logger.error(error_msg)
                raise RankingAPIError(error_msg)

            # Format results and filter by threshold
            formatted_results = []
            raw_results = response_data['results']

            for result_item in raw_results:
                if not isinstance(result_item, dict) or 'index' not in result_item or 'relevance_score' not in result_item:
                    logger.warning(f"Skipping malformed result item in Together Rerank response: {result_item}")
                    continue

                score = result_item['relevance_score']
                index = result_item['index']

                if score >= relevance_threshold:
                    formatted_results.append({"index": index, "score": score})
                else:
                     logger.debug(f"Filtering out document index {index} with score {score} (below threshold {relevance_threshold})")

            # Ensure results are sorted by score (API usually does this, but verify)
            formatted_results.sort(key=lambda x: x['score'], reverse=True)

            logger.info(f"Reranking complete. Returning {len(formatted_results)} documents above threshold {relevance_threshold}.")
            return formatted_results

        except httpx.TimeoutException as e:
            logger.error(f"Together Rerank API request timed out to {rerank_endpoint}: {e}")
            raise RankingAPIError(f"Request to Together Rerank API timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            response_text = e.response.text
            logger.error(f"Together Rerank API request failed with status {e.response.status_code} to {rerank_endpoint}: {response_text}", exc_info=False)
            raise RankingAPIError(f"Together Rerank API returned status {e.response.status_code}: {response_text}") from e
        except httpx.RequestError as e:
            logger.error(f"Together Rerank API request failed for {rerank_endpoint}: {e}", exc_info=True)
            raise RankingAPIError(f"Together Rerank API request failed: {e}") from e
        except Exception as e:
            # Catch JSONDecodeError or other unexpected errors
            logger.error(f"Unexpected error during Together Rerank processing: {e}", exc_info=True)
            raise RankingAPIError(f"Unexpected error processing Together Rerank response: {e}") from e

# Example Usage (Async)
async def _test_async():
    logging.basicConfig(level=logging.INFO)
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("Skipping test: TOGETHER_API_KEY not set in environment.")
        return

    test_query = "What is the capital of the United States?"
    test_docs = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    ]
    model_name = "Salesforce/Llama-Rank-V1" # Or another valid reranker model

    try:
        print(f"Running rerank test with model: {model_name}")
        reranked = await rerank_with_together_api(
            query=test_query,
            documents=test_docs,
            model=model_name,
            api_key=api_key,
            relevance_threshold=0.1 # Example threshold
        )
        print("Reranked Results (Above Threshold):")
        if not reranked:
            print("  No results above threshold.")
        for result in reranked:
            print(f"  Index: {result['index']}, Score: {result['score']:.4f} - Document: {test_docs[result['index']]}")

    except (ConfigurationError, RankingAPIError) as e:
        print(f"Test failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {type(e).__name__}: {e}")

if __name__ == '__main__':
    asyncio.run(_test_async()) 