import os
import asyncio
from app.agencies.cpe.config import CPEConfig
from app.agencies.cpe.schemas import CPERequest
from app.agencies.cpe.orchestrator import run_cpe

async def main():
    # Ensure environment variables or override here for testing
    # os.environ['OPENROUTER_API_KEY'] = '<YOUR_OPENROUTER_API_KEY>'
    # os.environ['CPE_PLANNER_MODEL_ID'] = 'gpt-4'
    # os.environ['CPE_EXTRACTOR_MODEL_ID'] = 'gpt-4'

    # Load configuration from environment
    config = CPEConfig()

    # Define test start URLs
    request = CPERequest(
        start_urls=["https://roundtableadvisory.com/"]
    )

    # Execute CPE workflow
    response = await run_cpe(request, config)

    # Print extracted profiles and usage stats using the new method
    print(response.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 