"""Custom exception classes for the Deep Research API application."""

class DeepResearchError(Exception):
    """Base class for exceptions in this application."""
    pass

class ConfigurationError(DeepResearchError):
    """Errors related to application configuration (e.g., missing API keys, invalid settings)."""
    pass

class ValidationError(DeepResearchError, ValueError):
    """Errors related to data validation (e.g., invalid input formats, failed schema checks)."""
    pass

class LLMError(DeepResearchError):
    """Base class for LLM related errors."""
    pass

class LLMCommunicationError(LLMError):
    """Errors during communication with the LLM API (network issues, timeouts, API errors)."""
    pass

class LLMOutputValidationError(LLMError, ValidationError):
    """Errors when LLM output fails schema validation or parsing after retries."""
    pass

class LLMRateLimitError(LLMCommunicationError):
    """Specific error for rate limiting by the LLM API."""
    pass

class ExternalServiceError(DeepResearchError):
    """Base class for errors originating from external APIs (Search, Ranking, Scraping)."""
    pass

class SearchAPIError(ExternalServiceError):
    """Errors related to the Search API (e.g., Serper)."""
    pass

class RankingAPIError(ExternalServiceError):
    """Errors related to the Ranking API (e.g., Together)."""
    pass

class ScrapingError(ExternalServiceError):
    """Errors encountered during web scraping."""
    pass

class ChunkingError(DeepResearchError):
    """Errors during the text chunking process."""
    pass

class AgentExecutionError(DeepResearchError):
    """Errors occurring during the execution logic of the DeepResearchAgent workflow."""
    pass 