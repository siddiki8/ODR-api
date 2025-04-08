"""
Service for reranking search results using Together AI Rerank API.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import httpx # Using httpx for async API calls
import asyncio # For testing
import random # For jitter

# Pydantic for local type definition
from pydantic import BaseModel

# Import custom exceptions using absolute path
from app.core.exceptions import ConfigurationError, RankingAPIError

# Configure logger
logger = logging.getLogger(__name__)

# Default Base URL for Together API
TOGETHER_API_BASE = os.getenv("TOGETHER_API_BASE", "https://api.together.ai")

# Define the output structure for a ranked item
class RankedItem(BaseModel):
    index: int
    score: float

async def rerank_with_together_api(
    query: str,
    documents: List[str],
    model: str,
    api_key: str,
    relevance_threshold: float = 0.1,
    top_n: Optional[int] = None,
    timeout: int = 20,
    max_retries: int = 2, # e.g., 0=initial try, 1=retry1, 2=retry2 (total 3 attempts)
    initial_delay_s: float = 1.0
) -> List[RankedItem]: # Updated return type hint
    """
    Reranks a list of document strings based on a query using the Together Rerank API.

    Filters results based on a relevance score threshold.

    Args:
        query: The search query used for relevance comparison.
        documents: A list of document strings to be reranked.
        model: The identifier for the Together AI reranker model (e.g., 'Salesforce/Llama-Rank-V1').
        api_key: Your Together AI API key (must be provided).
        relevance_threshold: Minimum relevance score for a document to be included in the results.
        top_n: Optional. If specified, requests the API to return only the top N scored documents 
               (before threshold filtering). If None, requests scores for all documents.
        timeout: Timeout in seconds for the HTTP request to the Together API.
        max_retries: Maximum number of retries on transient errors.
        initial_delay_s: Initial delay in seconds before the first retry.

    Returns:
        A list of RankedItem instances, each containing the original 'index' of the document 
        in the input list and its 'score'. The list is sorted by score (descending) 
        and only includes documents meeting the relevance_threshold.

    Raises:
        ValueError: If query or documents list is empty.
        ConfigurationError: If the api_key is not provided.
        RankingAPIError: If the API call fails with a non-retryable error, 
                       or if the API returns an unexpected response format.
                       (Returns [] if retries are exhausted on transient errors).
    """
    if not api_key:
        raise ConfigurationError("Together API key must be provided.")
    if not query:
        raise ValueError("Query cannot be empty for reranking.")
    if not documents:
        # Return empty list instead of raising error? Agent might handle empty results gracefully.
        # Raising ValueError for now, as it indicates invalid input to *this* function.
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
        # Explicitly ask for all scores if top_n is not specified by caller.
        payload["top_n"] = len(documents)

    rerank_endpoint = f"{TOGETHER_API_BASE.rstrip('/')}/v1/rerank"

    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Sending rerank request (Attempt {attempt + 1}/{max_retries + 1}) to {rerank_endpoint} for {len(documents)} docs, model: {model}")
                logger.debug(f"Rerank payload (excluding docs): {{model: {model}, query: {query[:50]}..., top_n: {payload.get('top_n')}}}")
                response = await client.post(rerank_endpoint, headers=headers, json=payload)
                response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx

                response_data = response.json()
                logger.debug("Received successful response from Together Rerank API.")

                # Validate response structure
                if not isinstance(response_data, dict) or 'results' not in response_data or not isinstance(response_data['results'], list):
                    error_msg = f"Unexpected response format from Together Rerank API. Missing 'results' list. Response: {response_data}"
                    logger.error(error_msg)
                    raise RankingAPIError(error_msg) # Non-retryable format error

                # Format results and filter by threshold
                formatted_results = []
                raw_results = response_data['results']

                for result_item in raw_results:
                    if not isinstance(result_item, dict) or 'index' not in result_item or 'relevance_score' not in result_item:
                        logger.warning(f"Skipping malformed result item in Together Rerank response: {result_item}")
                        continue

                    score = result_item['relevance_score']
                    index = result_item['index']

                    # Ensure score is float before comparison
                    try:
                        score_float = float(score)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert score '{score}' to float for index {index}. Skipping.")
                        continue
                        
                    if score_float >= relevance_threshold:
                        # Create RankedItem instance
                        formatted_results.append(RankedItem(index=index, score=score_float))
                    else:
                        logger.debug(f"Filtering out document index {index} with score {score_float:.4f} (below threshold {relevance_threshold})")

                # Defensive sort: ensure results are sorted by score (API usually does this)
                formatted_results.sort(key=lambda x: x.score, reverse=True)

                logger.info(f"Reranking complete. Returning {len(formatted_results)} documents above threshold {relevance_threshold}.")
                return formatted_results # Success!

        except (httpx.TimeoutException, httpx.ReadError) as e:
            last_exception = e
            logger.warning(f"Rerank Attempt {attempt + 1} failed (Retryable Network Error): {type(e).__name__}: {e}")
        except httpx.HTTPStatusError as e:
            last_exception = e
            # Retry on 5xx server errors AND 429 rate limit errors
            if 500 <= e.response.status_code <= 599:
                logger.warning(f"Rerank Attempt {attempt + 1} failed (Retryable Server Error {e.response.status_code}): {e.response.text[:200]}...")
            elif e.response.status_code == 429:
                logger.warning(f"Rerank Attempt {attempt + 1} failed (Rate Limit Error 429). Retrying...")
                # Optionally parse retry-after header if present, but simple backoff is fine too
            else:
                # Non-retryable client error (other 4xx) or other status
                response_text = e.response.text
                logger.error(f"Together Rerank API request failed with non-retryable status {e.response.status_code}: {response_text}", exc_info=False)
                raise RankingAPIError(f"Together Rerank API returned status {e.response.status_code}: {response_text}") from e
        except httpx.RequestError as e:
            # General request errors (e.g., DNS resolution failed) - potentially retryable?
            # Let's retry these for now.
            last_exception = e
            logger.warning(f"Rerank Attempt {attempt + 1} failed (Retryable Request Error): {type(e).__name__}: {e}")
        except Exception as e:
            # Catch unexpected errors during processing, but don't retry these
            logger.error(f"Unexpected error during Together Rerank processing: {e}", exc_info=True)
            raise RankingAPIError(f"Unexpected error processing Together Rerank response: {e}") from e

        # If we are here, a retryable error occurred
        if attempt < max_retries:
            # Calculate delay with exponential backoff and jitter
            delay = (initial_delay_s * (2 ** attempt)) + random.uniform(0, 0.5)
            logger.info(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
        else:
            logger.error(f"Reranking failed after {max_retries + 1} attempts. Last error: {type(last_exception).__name__}: {last_exception}")
            # Return empty list instead of raising an error after exhausting retries
            return []

    # Should technically be unreachable if max_retries >= 0, but as a fallback:
    logger.error("Exited retry loop unexpectedly in rerank_with_together_api")
    return []

# Example Usage (Async)
async def _test_async():
    # Setup basic logging FOR THE TEST RUN ONLY if not already configured
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__) # Get logger again after basicConfig
    else:
        log = logger # Use existing logger
        log.setLevel(logging.INFO) # Ensure level is appropriate for test
        
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        log.warning("Skipping test: TOGETHER_API_KEY not set in environment.")
        return

    test_query = "What is the capital of the United States?"
    test_docs = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        "Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    ]
    model_name = os.getenv("RERANKER_MODEL", "Salesforce/Llama-Rank-V1") 

    try:
        log.info(f"--- Starting Rerank Service Test (Model: {model_name}) --- ")
        reranked = await rerank_with_together_api(
            query=test_query,
            documents=test_docs,
            model=model_name,
            api_key=api_key,
            relevance_threshold=0.1 # Example threshold
        )
        log.info("Reranked Results (Above Threshold):")
        if not reranked:
            log.info("  No results above threshold.")
        for result in reranked:
            # Use logger for output
            log.info(f"  Index: a {result['index']}, Score: {result['score']:.4f} - Document: {test_docs[result['index']]}")

    except (ConfigurationError, RankingAPIError, ValueError) as e:
        log.error(f"Test failed: {type(e).__name__}: {e}")
    except Exception as e:
        log.error(f"Test failed with unexpected error: {type(e).__name__}: {e}", exc_info=True)
    log.info("--- Finished Rerank Service Test --- ")

if __name__ == '__main__':
    asyncio.run(_test_async()) 