import os
import logging
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Any

from .config import CPEConfig
from .schemas import ExtractedCompanyData, CompanyProfile
from .prompts import EXTRACTOR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def create_llm_model(model_id: str) -> OpenAIModel:
    """
    Helper to instantiate an OpenAIModel via OpenRouter.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    provider = OpenAIProvider(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    return OpenAIModel(model_id, provider=provider)


def create_extractor_agent(config: CPEConfig) -> Agent[ExtractedCompanyData]:
    """Creates the ExtractedCompanyData extractor Agent."""
    model = create_llm_model(config.extractor_model_id)
    return Agent[ExtractedCompanyData](
        model=model,
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        result_type=ExtractedCompanyData,
        retries=3
    )


class CPEAgents(BaseModel):
    extractor: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)


def get_cpe_agents(config: CPEConfig) -> CPEAgents:
    """Initializes and returns the extractor agent for CPE."""
    extractor = create_extractor_agent(config=config)
    logger.info(f"Initialized extractor agent using model: {config.extractor_model_id}")
    return CPEAgents(
        extractor=extractor
    ) 