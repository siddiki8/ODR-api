import litellm
from typing import List, Dict, Any, Optional, Type, Tuple
from pydantic import BaseModel
import logging

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
) -> Tuple[Optional[Any], Optional[UsageDict], Optional[CostDict]]:
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
        - The litellm response object (contains choices, usage, etc.) or None if error.
        - A dictionary containing token usage (completion, prompt, total) or None if error/unavailable.
        - A dictionary containing cost information ('total_cost') or None if error/unavailable.
        
    Raises:
        Propagates exceptions from litellm.acompletion (e.g., APIError, Timeout, NotFoundError).
    """
    log = logger_callback or logger # Use provided logger or default
    
    call_params = llm_config.copy() # Avoid modifying the original config dict
    call_params['messages'] = messages
    call_params['num_retries'] = num_retries

    if response_pydantic_model:
        call_params['response_format'] = response_pydantic_model

    try:
        log.debug(f"Calling LiteLLM acompletion with model: {call_params.get('model')}, retries: {num_retries}")
        response = await litellm.acompletion(**call_params)
        log.debug(f"LiteLLM call successful for model: {call_params.get('model')}")

        # Extract usage and cost safely
        usage_info: Optional[UsageDict] = None
        cost_info: Optional[CostDict] = None

        if hasattr(response, 'usage') and response.usage is not None:
            usage_info = {
                'completion_tokens': response.usage.completion_tokens or 0,
                'prompt_tokens': response.usage.prompt_tokens or 0,
                'total_tokens': response.usage.total_tokens or 0,
            }
        
        if hasattr(response, 'cost') and isinstance(response.cost, dict) and 'total_cost' in response.cost:
             cost_info = {'total_cost': response.cost['total_cost'] or 0.0}

        return response, usage_info, cost_info

    except Exception as e:
        log.error(f"LiteLLM acompletion failed: {e}", exc_info=True) # Log traceback
        # Re-raise the exception to be handled by the caller (Agent)
        raise e 

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
        if response:
            print("Test Call Successful!")
            print(f"Response Content: {response.choices[0].message.content}")
            print(f"Usage: {usage}")
            print(f"Cost: {cost}")
        else:
             print("Test Call returned None response.")

    except Exception as e:
        print(f"Test Call Failed: {e}")

if __name__ == "__main__":
    import asyncio
    # Setup basic logging for the test run
    logging.basicConfig(level=logging.DEBUG) 
    asyncio.run(_test()) 