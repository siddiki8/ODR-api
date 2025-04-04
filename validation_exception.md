# Validation and Exception Handling Plan

This document outlines a plan to enhance validation and exception handling throughout the `deep-research-api` codebase to ensure robustness and provide informative error responses.

## 1. Goals

-   Ensure all external inputs (API requests, environment variables, API responses from external services) are rigorously validated.
-   Implement consistent and informative exception handling across all modules (`core` and `services`).
-   Catch potential errors early in the workflow to prevent unexpected failures later.
-   Provide clear error messages to the client (via HTTP responses or WebSocket messages later) when issues occur.
-   Define custom exception classes for better error categorization.

## 2. Strategy by Module/File

### 2.1. `app/main.py` (API Entrypoint)

-   **Current:** Relies on FastAPI's built-in Pydantic validation for `ResearchRequest`.
-   **Enhancements:**
    -   **Global Exception Handler:** Implement a FastAPI global exception handler (`@app.exception_handler(...)`) to catch unhandled exceptions from anywhere in the application (including agent and service layers).
        -   This handler should log the full traceback for debugging.
        -   It should return a standardized JSON error response (e.g., `{"detail": "Internal server error. Check logs for ID: XYZ"}`) with an appropriate HTTP status code (e.g., 500).
        -   Define custom exception classes (see section 3) and handle them specifically in the global handler to return more specific status codes and messages (e.g., 4xx for validation errors, 503 for external service unavailable).
    -   **Dependency Injection Errors:** Ensure errors during dependency injection (like failing to load `AppSettings` or `ApiKeys`) are caught and result in a 500 error.

### 2.2. `app/core/config.py`

-   **Current:** Uses Pydantic for loading settings and API keys, which provides good initial validation (type checks, required fields via `...`). `get_litellm_params` includes `ValueError` for missing keys.
-   **Enhancements:**
    -   **Refine Validators:** Add more specific `field_validator`s if needed (e.g., check if `max_tokens` is positive, validate choices for `scraper_strategies` against allowed values).
    -   **API Key Checks:** The `get_litellm_params` check for missing keys is good. Ensure similar explicit checks exist *before* attempting to use *any* API key (Serper, Together) if it's required for a configured component.
    -   **Custom Exceptions:** Raise specific custom exceptions (e.g., `ConfigurationError`) instead of generic `ValueError` within helper functions like `get_litellm_params` or during `AppSettings`/`ApiKeys` initialization if manual checks fail. This allows the global handler in `main.py` to respond more appropriately.

### 2.3. `app/core/schemas.py`

-   **DONE:** Strengthen Validation using Pydantic constraints (`min_length`, `min_items`, `HttpUrl`).
-   **Note:** LLM Output Validation relies on Pydantic models passed to `call_litellm_acompletion` and specific checks in `agent.py`.

### 2.4. `app/core/agent.py` (`DeepResearchAgent`)

-   **DONE:** Wrap Service Calls in try/except.
-   **DONE:** Catch Specific Custom Exceptions from services.
-   **DONE:** Implement Workflow Logic Error Handling (e.g., empty results, failed validation).
-   **DONE:** Add Contextual Error Logging.
-   **DONE:** Implement Error Propagation Strategy (Recoverable vs. Critical/AgentExecutionError).
-   **DONE:** Handle Refinement Loop Errors Gracefully.
-   **Note:** Resource management is generally handled by underlying libraries (`httpx`, `litellm`).
-   **Note:** LLM JSON output validation (`_validate_llm_json_output`) wasn't explicitly found, but Pydantic models are used directly with `call_litellm_acompletion` which handles parsing/validation. Planner output is also explicitly validated after the call.

### 2.5. `app/services/llm.py`

-   **DONE:** Catch LiteLLM Exceptions
-   **DONE:** Raise Custom Exceptions
-   **DONE:** Input Validation

### 2.6. `app/services/search.py`

-   **Enhancements:**
    -   **HTTP Client Errors:** Wrap calls to the Serper API client (likely using `httpx` or `requests`) in `try...except` blocks.
    -   **Catch Specific HTTP Errors:** Catch specific exceptions like `httpx.RequestError`, `httpx.TimeoutException`, `httpx.HTTPStatusError`.
    -   **Raise Custom Exceptions:** Convert these into a `SearchAPIError` or similar, including details like the status code, query attempted, and error message from the API if available.
    -   **Response Validation:** Validate the structure of the response received from the Serper API. Raise an error if it's malformed or missing expected fields.

### 2.7. `app/services/ranking.py`

-   **DONE:** Wrap API calls (switched to httpx).
-   **DONE:** Catch Specific HTTP/Client Library Exceptions.
-   **DONE:** Raise Custom Exceptions (`RankingAPIError`, `ConfigurationError`, `ValueError`).
-   **DONE:** Validate API responses.

### 2.8. `app/services/scraping.py`

-   **DONE:** Wrap scraping attempts (crawl4ai, helpers) in try/except.
-   **DONE:** Catch Specific Errors (httpx, parsing, quality filter, etc.).
-   **DONE:** Raise Custom Exceptions (`ScrapingError`, `ConfigurationError`, `ValueError`).
-   **DONE:** Implement basic retries/timeouts via `httpx` and `crawl4ai` config.

### 2.9. `app/services/chunking.py`

-   **DONE:** Input Validation (text, document list).
-   **DONE:** Wrap chunking logic in try/except.
-   **DONE:** Raise Custom Exceptions (`ChunkingError`, `ValueError`).

## 3. Custom Exception Classes

Define a set of custom exception classes, potentially inheriting from a base `DeepResearchError`. This allows for more granular error handling.

```python
# Example in a new file like app/core/exceptions.py

class DeepResearchError(Exception):
    """Base class for exceptions in this application."""
    pass

class ConfigurationError(DeepResearchError):
    """Errors related to application configuration."""
    pass

class ValidationError(DeepResearchError, ValueError):
    """Errors related to data validation."""
    pass

class LLMError(DeepResearchError):
    """Base class for LLM related errors."""
    pass

class LLMCommunicationError(LLMError):
    """Errors during communication with the LLM API."""
    pass

class LLMOutputValidationError(LLMError, ValidationError):
    """Errors when LLM output fails schema validation."""
    pass

class LLMRateLimitError(LLMCommunicationError):
    """Specific error for rate limiting."""
    pass

class ExternalServiceError(DeepResearchError):
    """Base class for errors from external APIs."""
    pass

class SearchAPIError(ExternalServiceError):
    """Errors related to the Search API (Serper)."""
    pass

class RankingAPIError(ExternalServiceError):
    """Errors related to the Ranking API (Together)."""
    pass

class ScrapingError(ExternalServiceError):
    """Errors during web scraping."""
    pass

class ChunkingError(DeepResearchError):
    """Errors during text chunking."""
    pass

class AgentExecutionError(DeepResearchError):
    """Errors during the execution of the agent workflow."""
    pass

```