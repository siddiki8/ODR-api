import logging
import os
from pydantic import BaseModel, Field, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import List, Any

# Assuming schemas will be defined in exec_finder/schemas.py later
# For now, let's define a simple placeholder for Planner output
# and assume we reuse SearchTask from core schemas
from app.core.schemas import SearchTask

# Assuming config will be defined in exec_finder/config.py later
from .config import ExecFinderConfig # Placeholder import

logger = logging.getLogger(__name__)

# Add basic config if running standalone for testing
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Planner Prompt Definitions ---

_EXEC_FINDER_PLANNER_SYSTEM_PROMPT = """
You are an expert planning assistant specialized in corporate intelligence and executive searching.
Your goal is to generate a structured search plan to identify companies and potential executive contacts (especially emails) within a specific industry and city provided by the user.

Generate a list of 3 to 7 targeted `search_tasks`. Focus your queries on finding:
1.  Lists or directories of companies in the specified industry and city.
2.  Official company websites (specifically "Contact Us", "About Us", "Team", or "Leadership" pages).
3.  Professional networking profiles (like LinkedIn) for companies and potential executives.
4.  Direct contact information (email patterns, specific executive emails if publicly mentioned).
5.  News articles or press releases mentioning company leadership in the target area.

For each search task, specify:
- `query`: A precise search string designed to maximize relevant results for company/contact finding. Use variations like "[Industry] firms in [City]", "Contact page [Company Name]", "CEO email [Company Name]".
- `endpoint`: Use "/search" for almost all tasks. Only use "/news" if specifically looking for recent leadership change announcements. Do not use "/scholar".
- `num_results`: Typically 10-15 results should be sufficient.
- `reasoning`: Briefly explain why this query helps find companies or contacts in the target industry/city.

Output *only* a single JSON object adhering to the `ExecFinderPlannerOutput` schema below. Do not include any other text before or after the JSON object.

```json
{{
  "search_tasks": [
    {{
      "query": "Specific query string for Serper",
      "endpoint": "/search", // or "/news" rarely
      "num_results": <integer between 10-15>,
      "reasoning": "Why this query helps find companies or contact info in the specified industry/city."
    }}
    // ... (3 to 7 tasks)
  ]
}}
```
"""

_EXEC_FINDER_PLANNER_USER_MESSAGE_TEMPLATE = "Create a search plan to find companies and executive contacts in the '{industry}' industry within '{city}'."

# --- Define Placeholder Schema for Planner Output ---
# (This should ideally live in exec_finder/schemas.py)
class ExecFinderPlannerOutput(BaseModel):
    search_tasks: List[SearchTask] = Field(..., description="List of search tasks to execute.")


# --- Reusable LLM Model Creator (Adapted from deep_research) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY environment variable not set. Agent initialization might fail.")

def create_llm_model(model_id: str) -> OpenAIModel:
    """Helper function to create an OpenAI-compatible model instance for OpenRouter."""
    logger.debug(f"Attempting to create LLM model for ID: {model_id}")
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    logger.debug(f"Using OpenRouter API Key (found: {OPENROUTER_API_KEY is not None})")
    provider = OpenAIProvider(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    # Pass model_id as the first positional argument
    model = OpenAIModel(model_id, provider=provider)
    logger.debug(f"Successfully created OpenAIModel instance for {model_id}")
    return model

# --- Planner Agent Definition ---

def create_exec_finder_planner_agent(config: ExecFinderConfig) -> Agent:
    """Creates the Exec Finder Planner Agent instance."""
    logger.debug(f"Creating Exec Finder Planner Agent with model: {config.planner_model_id}")
    planner_model = create_llm_model(config.planner_model_id)
    agent = Agent[ExecFinderPlannerOutput]( # Use the specific output schema
        model=planner_model,
        system_prompt=_EXEC_FINDER_PLANNER_SYSTEM_PROMPT,
        result_type=ExecFinderPlannerOutput, # Specify the output schema type
        retries=3
    )
    logger.debug("Exec Finder Planner Agent created successfully.")
    return agent

# --- Extractor Agent Placeholder (To be implemented later) ---
def create_exec_finder_extractor_agent(config: ExecFinderConfig) -> Agent:
    """Creates the Exec Finder Extractor Agent instance."""
    # TODO: Implement Extractor Agent (prompts, schema, creation logic)
    logger.warning("Extractor Agent not yet implemented.")
    # Placeholder implementation
    logger.debug(f"Creating Extractor Agent with model: {config.extractor_model_id}")
    extractor_model = create_llm_model(config.extractor_model_id)
    # Define placeholder schema or import from exec_finder/schemas.py later
    class CompanyContactInfo(BaseModel):
        company_name: str | None = None
        company_url: str | None = None
        brief_description: str | None = None
        potential_emails: List[str] = Field(default_factory=list)
        source_url: str | None = None

    agent = Agent[CompanyContactInfo](
        model=extractor_model,
        system_prompt="You are an information extractor. Find company details...", # Placeholder prompt
        result_type=CompanyContactInfo, # Placeholder schema
        retries=3
    )
    logger.debug("Extractor Agent created successfully.")
    return agent


# --- Agency Agent Collection ---
class AgencyAgents(BaseModel):
    planner: Any
    extractor: Any # Add the extractor agent here later

    model_config = ConfigDict(arbitrary_types_allowed=True)

# --- Agent Initialization Function ---
def get_agency_agents(config: ExecFinderConfig) -> AgencyAgents:
    """
    Initializes and returns all PydanticAI Agent instances for the exec_finder agency.
    """
    planner = create_exec_finder_planner_agent(config=config)
    extractor = create_exec_finder_extractor_agent(config=config) # Will be implemented later

    # TODO: Add any result validators if needed later

    logger.info("Initialized Exec Finder Agency Agents (Planner, Extractor [Placeholder]) using OpenRouter.")

    return AgencyAgents(
        planner=planner,
        extractor=extractor # Add the actual extractor agent here
    )
