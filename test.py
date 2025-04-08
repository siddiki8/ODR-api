# test.py
import asyncio
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import Agency and Core Components ---
# Assuming 'app' is a package accessible from the root
try:
    from app.agencies.deep_research.agents import get_agency_agents
    from app.agencies.deep_research.orchestrator import run_deep_research_orchestration
    from app.agencies.deep_research.config import DeepResearchConfig
    from app.core.config import AppSettings
    from pydantic import ValidationError
except ImportError as e:
    logger.error(f"Import Error: {e}. Make sure you are running this script from the project root "
                 f"and the necessary packages are installed and accessible.")
    exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during imports: {e}")
    exit(1)


async def main():
    """Runs the deep research orchestration test."""
    logger.info("--- Starting Deep Research Test ---")

    # --- Configuration Loading ---
    try:
        logger.info("Loading AppSettings...")
        # Note: Adjust AppSettings instantiation if it requires specific args/env vars
        app_settings = AppSettings()
        logger.info("AppSettings loaded.")

        logger.info("Loading DeepResearchConfig...")
        # This will load from environment variables prefixed with DEEP_RESEARCH_
        # Ensure DEEP_RESEARCH_TOGETHER_API_KEY is set!
        deep_research_config = DeepResearchConfig()
        logger.info("DeepResearchConfig loaded.")
        logger.info(f"  Using Reranker Model: {deep_research_config.reranker_model}")

    except ValidationError as e:
        logger.error(f"Pydantic Configuration Validation Error: {e}")
        logger.error("Please ensure all required environment variables (e.g., DEEP_RESEARCH_TOGETHER_API_KEY, "
                     "OPENROUTER_API_KEY, SERPER_API_KEY) are set correctly in your environment or .env file.")
        return
    except ValueError as e: # Catch custom validator errors (like missing API key)
         logger.error(f"Configuration Error: {e}")
         return
    except Exception as e:
        logger.error(f"An unexpected error occurred during configuration loading: {e}", exc_info=True)
        return

    # --- Agent Initialization ---
    try:
        logger.info("Initializing agency agents...")
        # Ensure OPENROUTER_API_KEY is set in the environment
        agency_agents = get_agency_agents()
        logger.info("Agency agents initialized.")
    except ValueError as e: # Catch potential key errors during agent init
        logger.error(f"Agent Initialization Error: {e}")
        logger.error("Please ensure OPENROUTER_API_KEY is set.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred during agent initialization: {e}", exc_info=True)
        return

    # --- Define User Query ---
    user_query = "Compare and contrast the effectiveness of RAG vs Finetuning for adapting LLMs to specific domains, focusing on medical question answering."
    logger.info(f"User Query: '{user_query}'")

    # --- Run Orchestration ---
    try:
        logger.info("Running deep research orchestration...")
        result = await run_deep_research_orchestration(
            user_query=user_query,
            agents_collection=agency_agents,
            config=deep_research_config,
            app_settings=app_settings
        )
        logger.info("Orchestration complete.")

        # --- Display Results ---
        print("\n" + "="*20 + " Final Report " + "="*20 + "\n")
        print(result.report)
        print("\n" + "="*54 + "\n")
        logger.info(f"Usage Statistics: {result.usage_statistics}")

    except Exception as e:
        logger.error(f"An error occurred during orchestration: {e}", exc_info=True)

    logger.info("--- Deep Research Test Finished ---")

if __name__ == "__main__":
    # Install dotenv if you don't have it: pip install python-dotenv
    # Make sure required env vars are set!
    asyncio.run(main())