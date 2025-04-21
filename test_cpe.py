import os
import asyncio
# Add imports for dependencies and configs
from app.core.dependencies import get_settings, get_api_keys
from app.core.config import AppSettings, ApiKeys, SerperConfig
# Load dotenv if you need to load .env for testing
from dotenv import load_dotenv, find_dotenv

from app.agencies.cpe.config import CPEConfig
from app.agencies.cpe.schemas import CPERequest
from app.agencies.cpe.orchestrator import run_cpe

# Load .env file (if needed for local testing)
load_dotenv(find_dotenv())

async def main():
    # Ensure environment variables or override here for testing
    # os.environ['OPENROUTER_API_KEY'] = '<YOUR_OPENROUTER_API_KEY>'
    # os.environ['CPE_PLANNER_MODEL_ID'] = 'gpt-4'
    # os.environ['CPE_EXTRACTOR_MODEL_ID'] = 'gpt-4'

    # --- Dependency Resolution ---
    try:
        settings: AppSettings = get_settings()
        api_keys: ApiKeys = get_api_keys()
    except Exception as e:
        print(f"Error getting settings or API keys: {e}")
        return # Exit if dependencies fail

    # --- Add check for loaded Serper API key ---
    if api_keys.serper_api_key:
        print(f"Serper API Key loaded successfully into ApiKeys object: True (value hidden)")
    else:
        print(f"Serper API Key FAILED to load into ApiKeys object!")
        # Optionally, you might want to check os.getenv('SERPER_API_KEY') directly here too
        # print(f"Direct check os.getenv('SERPER_API_KEY'): {os.getenv('SERPER_API_KEY') is not None}")
        return # Exit if key is missing

    # --- Construct SerperConfig ---
    try:
        serper_config = SerperConfig(
            api_key=api_keys.serper_api_key,
            base_url=settings.serper_base_url,
            default_location=settings.serper_default_location,
            timeout=settings.serper_timeout
        )
    except Exception as e:
        print(f"Error creating SerperConfig: {e}")
        return # Exit if config creation fails

    # Load CPE-specific configuration from environment
    try:
        cpe_config = CPEConfig()
    except Exception as e:
        print(f"Error loading CPEConfig: {e}")
        return # Exit if CPE config fails

    # Define test query and location
    request = CPERequest(
        query="Private Equity firms.",
        location="St. Louis, Missouri",
        max_search_tasks=5
    )

    # Execute CPE workflow, passing the serper_config
    print("Running CPE Orchestrator with constructed SerperConfig...")
    try:
        response = await run_cpe(
            request=request,
            config=cpe_config,
            serper_config=serper_config # Pass the constructed config
        )
        print("CPE Run Completed.")
        # Print extracted profiles and usage stats using the new method
        print(response.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error during run_cpe execution: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 