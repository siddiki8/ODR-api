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