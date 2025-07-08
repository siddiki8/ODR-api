# Agency Creation Guide & Scaffolding Script

This document outlines the principles and tools for creating new agencies within the ODR-api framework. Adhering to these patterns ensures consistency, maintainability, and interoperability.

## 🏛️ Guiding Principles ("Cursor Rules")

These are the core rules to follow when building a new agency.

1.  **One Agency, One Directory:** Each agency MUST reside in its own subdirectory within `app/agencies/`. (e.g., `app/agencies/my_new_agency/`).

2.  **Standard File Structure:** Every agency MUST contain the following files, even if they are minimal initially:
    *   `__init__.py`
    *   `schemas.py`: Defines all Pydantic data contracts for the agency.
    *   `config.py`: Contains the `...Config` Pydantic model for settings.
    *   `agents.py`: Contains agent factory functions (e.g., `create_..._agent`).
    *   `orchestrator.py`: Contains the main orchestration logic and its wrapper.
    *   `routes.py`: Defines the FastAPI router and WebSocket endpoint.
    *   `callbacks.py`: Defines the WebSocket update handler class.
    *   `helpers.py`: (Optional but recommended) for abstracting complex logic or service calls.

3.  **Configuration is Explicit:** The agency's Pydantic `Config` class (e.g., `MyAgencyConfig`) should be passed as an explicit argument to the orchestrator and other functions that need it. Do not store it as an attribute on the `AgencyAgents` collection.

4.  **Agents are Resilient Factories:**
    *   Agents MUST be created via factory functions (e.g., `create_planner_agent(llm: LLM) -> Agent`).
    *   These factories SHOULD instantiate the `Agent` with `ModelRetry` to handle transient LLM API errors gracefully (e.g., `retry=ModelRetry(tries=3, on_fail="log")`).

5.  **Orchestration is Wrapped:** The main orchestration logic (`run_..._orchestration`) MUST be called from a top-level wrapper function (`run_..._orchestration_wrapper`). This wrapper's sole responsibility is to implement a `try...except` block to catch all errors, log them, and return a standardized error response.

6.  **Services are Abstracted:** External data fetching or complex, non-agent logic SHOULD be abstracted into services. For agency-specific services, place them in `app/agencies/services/`. This keeps the orchestrator clean and focused on workflow.

7.  **Communication is Structured:** All real-time updates MUST be sent via a dedicated `CallbackHandler` class. The handler should have specific methods for each type of update (e.g., `analysis_start()`, `send_news_analysis(data)`), promoting a clear and structured event-based communication protocol with the client.

---

## 🚀 Agency Scaffolding Script

To accelerate development and ensure adherence to these rules, a scaffolding script is invaluable. This script will automatically generate the entire directory structure and stub files for a new agency with placeholder code.

### Proposal

*   **Name:** `create_agency.py`
*   **Location:** `/scripts/` (a new top-level directory)
*   **Technology:** Python, using `pathlib` for OS-agnostic path handling and `argparse` for command-line argument parsing.
*   **Usage:** `python scripts/create_agency.py "My New Agency"`

### Script Implementation

Here is the proposed code for `scripts/create_agency.py`:

```python
import argparse
from pathlib import Path
import re

def to_snake_case(name: str) -> str:
    """Converts a string to snake_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower().replace(" ", "_")

def to_pascal_case(name: str) -> str:
    """Converts a string to PascalCase."""
    return ''.join(word.capitalize() for word in name.replace("_", " ").split())

def create_agency_scaffold(agency_display_name: str):
    """
    Generates the directory structure and stub files for a new agency.
    """
    snake_name = to_snake_case(agency_display_name)
    pascal_name = to_pascal_case(snake_name)
    
    base_path = Path(f"app/agencies/{snake_name}")
    if base_path.exists():
        print(f"Error: Directory '{base_path}' already exists.")
        return

    print(f"Creating new agency '{pascal_name}' at '{base_path}'...")
    base_path.mkdir(parents=True)

    files_and_content = {
        "__init__.py": "",
        "config.py": f"""
from __future__ import annotations
from pydantic_settings import BaseSettings
from app.core.schemas import LLMConfig

class {pascal_name}Config(BaseSettings):
    \"\"\"Configuration for the {pascal_name} Agency.\"\"\"
    # Example: Define LLM config for an agent
    # main_agent_llm_config: LLMConfig = LLMConfig(model="gpt-4-turbo")

    class Config:
        env_file = ".env"
        env_prefix = "{snake_name.upper()}_"
        extra = "ignore"
""",
        "schemas.py": f"""
from __future__ import annotations
from pydantic import BaseModel, Field
from .config import {pascal_name}Config
from app.core.schemas import UsageStatistics

# Example Request Schema
class {pascal_name}Request(BaseModel):
    query: str
    config_overrides: {pascal_name}Config | None = None

# Example Agent Output Schema
class AnalysisResult(BaseModel):
    summary: str = Field(..., description="The summary of the analysis.")

# Example Final Response Schema
class {pascal_name}Response(BaseModel):
    result: AnalysisResult
    usage_statistics: UsageStatistics
""",
        "agents.py": f"""
from __future__ import annotations
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.llm import LLM
from . import schemas
from app.core.dependencies import LLMProvider

# --- Agent Definitions ---
def create_main_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt="You are a helpful assistant.",
        result_type=schemas.AnalysisResult,
        retry=ModelRetry(tries=3, on_fail="log")
    )

# --- Agent Collection ---
class AgencyAgents:
    def __init__(self, llm_provider: LLMProvider, config: schemas.{pascal_name}Config):
        # self.main_agent = create_main_agent(
        #     llm_provider.get_llm(config.main_agent_llm_config)
        # )
        pass
""",
        "helpers.py": f"""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

# Add agency-specific helper functions here
""",
        "orchestrator.py": f"""
from __future__ import annotations
import logging
from . import agents, schemas
from app.core.schemas import RunUsage, UsageStatistics

logger = logging.getLogger(__name__)

async def run_{snake_name}_orchestration(
    request: schemas.{pascal_name}Request,
    agents_collection: agents.AgencyAgents,
    config: schemas.{pascal_name}Config,
    update_callback: "UpdateHandler"
) -> schemas.{pascal_name}Response:
    usage_tracker = RunUsage()
    # Main orchestration logic here
    
    # result = await agents_collection.main_agent.run(request.query)
    # usage_tracker.update_agent_usage("main_agent", result.usage())
    
    # return schemas.{pascal_name}Response(
    #     result=result.data,
    #     usage_statistics=usage_tracker.get_statistics()
    # )
    pass

async def run_{snake_name}_orchestration_wrapper(
    request: schemas.{pascal_name}Request,
    agents_collection: agents.AgencyAgents,
    config: schemas.{pascal_name}Config,
    update_callback: "UpdateHandler"
) -> schemas.{pascal_name}Response:
    try:
        return await run_{snake_name}_orchestration(
            request=request,
            agents_collection=agents_collection,
            config=config,
            update_callback=update_callback
        )
    except Exception as e:
        logger.critical(f"CRITICAL UNHANDLED ERROR in {pascal_name} orchestration: {{e}}", exc_info=True)
        # Handle error response
        raise
""",
        "callbacks.py": f"""
from __future__ import annotations
import logging
from typing import Callable, Coroutine, Any, Dict

logger = logging.getLogger(__name__)

class UpdateHandler:
    def __init__(self, websocket_callback: Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]):
        self._callback = websocket_callback

    async def _send_update(self, event_type: str, data: Dict | None = None):
        if self._callback:
            payload = {{"event": event_type, "data": data or {{}}}}
            await self._callback(payload)
""",
        "routes.py": f"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
from app.core.dependencies import get_llm_provider
from .orchestrator import run_{snake_name}_orchestration_wrapper
from .agents import AgencyAgents
from . import schemas
from .callbacks import UpdateHandler

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/run")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    llm_provider = get_llm_provider()
    
    try:
        request_data = await websocket.receive_json()
        request = schemas.{pascal_name}Request.model_validate(request_data)

        agency_config = schemas.{pascal_name}Config()
        agents_collection = AgencyAgents(llm_provider=llm_provider, config=agency_config)

        async def send_update(payload: dict):
            await websocket.send_json(payload)
        
        update_handler = UpdateHandler(send_update)

        response = await run_{snake_name}_orchestration_wrapper(
            request=request,
            agents_collection=agents_collection,
            config=agency_config,
            update_callback=update_handler
        )
        # Final success message
        await update_handler._send_update("complete", response.model_dump())

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in {pascal_name} WebSocket: {{e}}", exc_info=True)
    finally:
        if websocket.client_state.name != 'DISCONNECTED':
            await websocket.close()
"""
    }

    for filename, content in files_and_content.items():
        file_path = base_path / filename
        file_path.write_text(content.strip())
        print(f"  Created {file_path}")

    print(f"\n✅ Successfully created agency '{pascal_name}'.")
    print(f"Next steps:")
    print(f"1. Update `app/main.py` to include and mount the new router from `app.agencies.{snake_name}.routes`.")
    print(f"2. Customize the placeholder schemas, agents, and orchestrator logic in `{base_path}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new agency scaffold.")
    parser.add_argument("name", type=str, help="The display name of the new agency (e.g., 'Financial Analysis').")
    args = parser.parse_args()
    
    create_agency_scaffold(args.name)

``` 