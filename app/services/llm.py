import litellm
from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import BaseModel
import logging

# Import custom exceptions
from ..core.exceptions import (
    LLMCommunicationError, LLMRateLimitError, LLMError, ConfigurationError
    # Removed LLMOutputValidationError as it's handled upstream
)

# Configure logging for the service
logger = logging.getLogger(__name__)

# Define type hints for clarity
UsageDict = Dict[str, int]
CostDict = Dict[str, float]
MessagesList = List[Dict[str, Any]]
LLMConfigDict = Dict[str, Any]
LiteLLMResponse = Any # Using Any as litellm.ModelResponse is common but not strictly enforced

async def call_litellm_acompletion(
    messages: MessagesList,
    llm_config: LLMConfigDict,
    response_pydantic_model: Optional[Type[BaseModel]] = None, # Passed to litellm
    num_retries: int = 3,
    logger_callback: Optional[logging.Logger] = None
) -> Tuple[LiteLLMResponse, Optional[UsageDict], Optional[CostDict]]:
    """
    Makes an asynchronous call to litellm.acompletion with retry logic and 
    extracts the response object, token usage, and cost information.

    This function acts as a standardized wrapper for interacting with LiteLLM's 
    completion endpoint within the application.

    Args:
        messages: The list of message dictionaries for the chat completion.
        llm_config: A dictionary containing LiteLLM compatible parameters 
                    (e.g., model, api_key, temperature, api_base).
        response_pydantic_model: Optional Pydantic model passed directly to 
                                 litellm for structured response parsing (if supported by model).
                                 This function does not perform parsing itself.
        num_retries: Number of retries for the LiteLLM API call.
        logger_callback: Optional logger instance to use instead of the default module logger.

    Returns:
        A tuple containing:
        - The raw litellm response object (typically includes choices, usage, etc.).
        - A dictionary containing token usage (completion_tokens, prompt_tokens, total_tokens) or None.
        - A dictionary containing cost information ('total_cost') or None.

    Raises:
        ConfigurationError: If the llm_config is missing the 'model' key.
        LLMRateLimitError: If the API returns a rate limit error.
        LLMCommunicationError: For API connection errors, timeouts, or service unavailability.
        LLMError: For other LiteLLM API errors or unexpected issues during the call.
        TypeError: If the messages list is empty.
    """
    log = logger_callback or logger # Use provided logger or default module logger

    if not messages:
         raise TypeError("Messages list cannot be empty.")
    if not llm_config or not llm_config.get('model'):
        raise ConfigurationError("LLM config must include at least the 'model' name.")

    # Prepare parameters for the LiteLLM call
    call_params = llm_config.copy()
    call_params['messages'] = messages
    call_params['num_retries'] = num_retries
    if response_pydantic_model:
        call_params['response_model'] = response_pydantic_model

    try:
        model_for_log = call_params.get('model', 'unknown')
        
        # Log call details at DEBUG level
        log.debug("--- Attempting LiteLLM API Call ---")
        log.debug(f"  Model: {model_for_log}")
        log.debug(f"  Retries: {num_retries}")
        # Log only presence and length of API key for security
        api_key_present = bool(call_params.get('api_key'))
        api_key_len = len(call_params.get('api_key', ''))
        log.debug(f"  API Key Provided: {api_key_present} (Length: {api_key_len})")
        log.debug(f"  API Base: {call_params.get('api_base')}")
        log.debug(f"  Temperature: {call_params.get('temperature')}")
        log.debug(f"  Max Tokens: {call_params.get('max_tokens')}")
        if response_pydantic_model:
             log.debug(f"  Response Model: {response_pydantic_model.__name__}")
        log.debug("----------------------------------")
        
        response = await litellm.acompletion(**call_params)
        log.info(f"LiteLLM call successful for model: {model_for_log}") # Log success at INFO

        # Extract usage and cost safely
        usage_info: Optional[UsageDict] = None
        cost_info: Optional[CostDict] = None

        if hasattr(response, 'usage') and response.usage is not None:
            # Ensure all expected keys are integers, default to 0 if missing/None
            usage_info = {
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0) or 0,
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) or 0,
                'total_tokens': getattr(response.usage, 'total_tokens', 0) or 0,
            }
            log.debug(f"Extracted usage: {usage_info}")
        else:
            log.debug("Usage information not found in LiteLLM response.")

        # LiteLLM v1.33+ returns cost dict directly
        if hasattr(response, 'cost') and isinstance(response.cost, dict) and 'total_cost' in response.cost:
             cost_info = {'total_cost': response.cost['total_cost'] or 0.0}
             log.debug(f"Extracted cost (v1.33+): {cost_info}")
        elif hasattr(response, '_response_cost'): # Older litellm versions might store it here
            # Ensure cost is float, default to 0.0
            cost_val = getattr(response, '_response_cost', 0.0) or 0.0
            cost_info = {'total_cost': float(cost_val)}
            log.debug(f"Extracted cost (legacy _response_cost): {cost_info}")
        else:
             log.debug("Cost information not found in LiteLLM response.")

        return response, usage_info, cost_info

    # --- Exception Handling --- #
    except litellm.exceptions.RateLimitError as e:
        error_type = type(e).__name__
        log.warning(f"LiteLLM RateLimitError for model {model_for_log}: {error_type}")
        raise LLMRateLimitError(f"LLM service rate limit exceeded for model {model_for_log}. Original error type: {error_type}") from e
    except litellm.exceptions.APIConnectionError as e:
        error_type = type(e).__name__
        log.error(f"LiteLLM APIConnectionError for model {model_for_log}: {error_type}")
        raise LLMCommunicationError(f"Could not connect to LLM service for model {model_for_log}. Original error type: {error_type}") from e
    except litellm.exceptions.APIError as e: # Catch broader API errors (like auth, bad request etc)
        error_type = type(e).__name__
        log.error(f"LiteLLM APIError for model {model_for_log}: {error_type} - {e}")
        # Include the error message from LiteLLM if available
        raise LLMError(f"LLM service returned an API error for model {model_for_log}. Error: {e}. Original error type: {error_type}") from e
    except Exception as e:
        error_type = type(e).__name__
        log.error(f"Unexpected error during LiteLLM call for model {model_for_log}: {error_type}", exc_info=True) # Log full traceback for unexpected
        raise LLMError(f"An unexpected error occurred during the LLM call for model {model_for_log}. Original error type: {error_type}") from e

# --- Test Function --- # 
# (Remains largely unchanged, uses logger now)
async def _test():
    import os
    from dotenv import load_dotenv
    load_dotenv() # Load .env for keys

    # Setup basic logging FOR THE TEST RUN ONLY if not already configured
    if not logger.hasHandlers():
        logging.basicConfig(level=logging.DEBUG)
        log = logging.getLogger(__name__) # Get logger again after basicConfig
    else:
        log = logger # Use existing logger
        log.setLevel(logging.DEBUG) # Ensure level is appropriate for test

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        log.warning("Skipping test: OPENROUTER_API_KEY not found in environment.")
        return

    test_config = {
        "model": "openrouter/google/gemini-flash-1.5",
        "api_key": api_key,
        "temperature": 0.1
    }
    test_messages = [{"role": "user", "content": "Say 'test successful'"}]

    log.info("--- Starting LLM Service Test --- ")
    try:
        response, usage, cost = await call_litellm_acompletion(
            messages=test_messages,
            llm_config=test_config,
            logger_callback=log # Pass logger explicitly
        )
        log.info("Test Call Successful!")
        log.info(f"Response Content: {response.choices[0].message.content}")
        log.info(f"Usage: {usage}")
        log.info(f"Cost: {cost}")

    except (LLMError, ConfigurationError, TypeError) as e: # Catch our custom exceptions
        log.error(f"Test Call Failed: {type(e).__name__}: {e}")
    except Exception as e: # Catch any unexpected errors during test
        log.error(f"Test Call Failed with unexpected error: {type(e).__name__}: {e}", exc_info=True)
    log.info("--- Finished LLM Service Test --- ")

if __name__ == "__main__":
    import asyncio
    asyncio.run(_test()) 