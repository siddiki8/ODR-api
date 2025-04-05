import litellm
from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import BaseModel
import logging

# Import custom exceptions
from ..core.exceptions import (
    LLMCommunicationError, LLMRateLimitError, LLMError, ConfigurationError,
    LLMOutputValidationError # Though parsing validation is often done in agent
)

# Configure logging for the service
logger = logging.getLogger(__name__)
# Set default logging level if not configured elsewhere
# logger.setLevel(logging.INFO) 
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# Define a type hint for the usage dictionary
UsageDict = Dict[str, int]
CostDict = Dict[str, float]

async def call_litellm_acompletion(
    messages: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    response_pydantic_model: Optional[Type[BaseModel]] = None,
    num_retries: int = 3,
    logger_callback: Optional[logging.Logger] = None # Allow passing custom logger
) -> Tuple[Any, Optional[UsageDict], Optional[CostDict]]: # Response is no longer optional if success
    """
    Makes an asynchronous call to litellm.acompletion with structured output support 
    and returns the response object, usage dictionary, and cost dictionary.

    Args:
        messages: The list of messages for the chat completion.
        llm_config: A dictionary containing LiteLLM compatible parameters 
                    (model, api_key, temperature, etc.).
        response_pydantic_model: Optional Pydantic model for structured response parsing.
        num_retries: Number of retries for the API call.
        logger_callback: Optional logger instance to use instead of the default.

    Returns:
        A tuple containing:
        - The litellm response object (contains choices, usage, etc.).
        - A dictionary containing token usage (completion, prompt, total) or None if unavailable.
        - A dictionary containing cost information ('total_cost') or None if unavailable.

    Raises:
        ConfigurationError: If the API key is missing in the config.
        LLMRateLimitError: If the API returns a rate limit error.
        LLMCommunicationError: For other API errors (non-rate limit), timeouts, or service unavailability.
        LLMError: For other unexpected LiteLLM errors.
        TypeError: If messages are not in the expected format.
    """
    log = logger_callback or logger # Use provided logger or default

    if not messages:
         raise TypeError("Messages list cannot be empty.")
    if not llm_config or not llm_config.get('model'):
        raise ConfigurationError("LLM config must include at least the 'model' name.")

    call_params = llm_config.copy()
    call_params['messages'] = messages
    call_params['num_retries'] = num_retries

    try:
        model_for_log = call_params.get('model', 'unknown')
        print("\n=== LiteLLM API Call Details ===")
        print(f"Model: {model_for_log}")
        print(f"Retries: {num_retries}")
        
        # Show part of the API key for debugging but hide most for security
        api_key = call_params.get('api_key', '')
        if api_key:
            key_prefix = api_key[:4] if len(api_key) > 8 else "****"
            key_suffix = api_key[-4:] if len(api_key) > 8 else "****"
            print(f"API Key: {key_prefix}...{key_suffix} (length: {len(api_key)})")
        else:
            print("API Key: NONE/MISSING")
            
        print(f"API Base: {call_params.get('api_base')}")
        print(f"Temperature: {call_params.get('temperature')}")
        print(f"Max Tokens: {call_params.get('max_tokens')}")
        print("=============================\n")
        
        response = await litellm.acompletion(**call_params)
        log.debug(f"LiteLLM call successful for model: {model_for_log}")

        # Extract usage and cost safely
        usage_info: Optional[UsageDict] = None
        cost_info: Optional[CostDict] = None

        if hasattr(response, 'usage') and response.usage is not None:
            usage_info = {
                'completion_tokens': response.usage.completion_tokens or 0,
                'prompt_tokens': response.usage.prompt_tokens or 0,
                'total_tokens': response.usage.total_tokens or 0,
            }

        # LiteLLM v1.33+ returns cost dict directly
        if hasattr(response, 'cost') and isinstance(response.cost, dict) and 'total_cost' in response.cost:
             cost_info = {'total_cost': response.cost['total_cost'] or 0.0}
        elif hasattr(response, '_response_cost'): # Older litellm versions might store it here
            cost_info = {'total_cost': response._response_cost or 0.0}

        # If using response_model, the validated Pydantic object is often in response.choices[0].message.content
        # or directly as the response object itself if structure_response=True was used implicitly?
        # The caller (agent) will handle extracting the structured content.

        return response, usage_info, cost_info

    except litellm.exceptions.RateLimitError as e:
        error_type = type(e).__name__
        logger.warning(f"LiteLLM RateLimitError for model {model_for_log}: {error_type}")
        raise LLMRateLimitError(f"LLM service rate limit exceeded for model {model_for_log}. Original error type: {error_type}") from e
    except litellm.exceptions.APIConnectionError as e:
        error_type = type(e).__name__
        logger.error(f"LiteLLM APIConnectionError for model {model_for_log}: {error_type}")
        raise LLMCommunicationError(f"Could not connect to LLM service for model {model_for_log}. Original error type: {error_type}") from e
    except litellm.exceptions.APIError as e:
        error_type = type(e).__name__
        logger.error(f"LiteLLM APIError for model {model_for_log}: {error_type}")
        raise LLMError(f"LLM service returned an API error for model {model_for_log}. Original error type: {error_type}") from e
    except Exception as e:
        error_type = type(e).__name__
        logger.error(f"Unexpected error during LiteLLM call for model {model_for_log}: {error_type}", exc_info=False) # Avoid full trace unless debugging needed
        raise LLMError(f"An unexpected error occurred during the LLM call for model {model_for_log}. Original error type: {error_type}") from e

# Example usage (for testing if needed)
async def _test():
    import os
    from dotenv import load_dotenv
    load_dotenv() # Load .env for keys

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Skipping test: OPENROUTER_API_KEY not found in environment.")
        return

    test_config = {
        "model": "openrouter/google/gemini-flash-1.5",
        "api_key": api_key,
        "temperature": 0.1
    }
    test_messages = [{"role": "user", "content": "Say 'test successful'"}]

    try:
        response, usage, cost = await call_litellm_acompletion(
            messages=test_messages,
            llm_config=test_config,
            logger_callback=logger # Pass logger explicitly for testing
        )
        # response is no longer Optional on success path
        print("Test Call Successful!")
        print(f"Response Content: {response.choices[0].message.content}")
        print(f"Usage: {usage}")
        print(f"Cost: {cost}")

    except (LLMError, ConfigurationError, TypeError) as e: # Catch our custom exceptions
        print(f"Test Call Failed: {type(e).__name__}: {e}")
    except Exception as e: # Catch any unexpected errors during test
         print(f"Test Call Failed with unexpected error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    import asyncio
    # Setup basic logging for the test run
    logging.basicConfig(level=logging.DEBUG) 
    asyncio.run(_test()) 