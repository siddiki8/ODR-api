import os
from typing import Dict, Any, Optional, List, Literal

from pydantic import BaseModel, Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file explicitly if it exists
load_dotenv()

class ApiKeys(BaseSettings):
    """Loads API keys from environment variables."""
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    serper_api_key: SecretStr = Field(..., alias='SERPER_API_KEY')
    together_api_key: SecretStr = Field(..., alias='TOGETHER_API_KEY')
    openrouter_api_key: Optional[SecretStr] = Field(None, alias='OPENROUTER_API_KEY')
    gemini_api_key: Optional[SecretStr] = Field(None, alias='GEMINI_API_KEY')

class LLMConfig(BaseModel):
    """Model for LLM configuration passed to LiteLLM."""
    model: str = "gemini-pro"
    provider: Optional[Literal['google', 'openrouter']] = None
    api_key: Optional[SecretStr] = None
    api_base: Optional[HttpUrl] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class AppSettings(BaseSettings):
    """Application-level settings."""
    model_config = SettingsConfigDict(env_file='.env', extra='allow')

    llm_provider: Literal['google', 'openrouter'] = Field('openrouter', alias='LLM_PROVIDER')

    # --- Default Model Names (without provider prefix) ---
    # These are for the 'openrouter' provider by default
    default_planner_model_name: str = "google/gemini-2.0-flash-thinking-exp-01-21:free"
    default_summarizer_model_name: str = "google/gemini-2.0-flash-exp:free"
    default_writer_model_name: str = "google/gemini-2.5-pro-exp-03-25:free"
    
    # --- Google Provider Specific Model Names ---
    # Use standard LiteLLM names for Gemini API (Google AI Studio)
    google_planner_model_name: str = "gemini-2.0-flash-thinking-exp-01-21" 
    google_summarizer_model_name: str = "gemini-2.0-flash"
    google_writer_model_name: str = "gemini-2.5-pro-exp-03-25"

    # --- Default LLM Configurations (using model names defined above) ---
    # These will be finalized in the agent __init__ based on the provider
    # Model name is set dynamically using settings.get_model_name(role)
    default_planner_llm: LLMConfig = LLMConfig(
        temperature=0.5
    )
    default_summarizer_llm: LLMConfig = LLMConfig(
        temperature=0.3
    )
    default_writer_llm: LLMConfig = LLMConfig(
        temperature=0.7,
        max_tokens=4000
    )

    reranker_model: str = Field("Salesforce/Llama-Rank-V1", alias='RERANKER_MODEL')
    serper_base_url: HttpUrl = Field("https://google.serper.dev", alias='SERPER_BASE_URL')
    serper_default_location: str = 'us'
    serper_timeout: int = 15
    scraper_strategies: List[str] = ['no_extraction']
    scraper_debug: bool = False
    max_initial_search_tasks: int = 3
    top_m_full_text_sources: int = 3
    max_refinement_iterations: int = 2

    def get_model_name(self, role: Literal['planner', 'summarizer', 'writer']) -> str:
        if self.llm_provider == 'google':
            return getattr(self, f"google_{role}_model_name")
        else:
            return getattr(self, f"default_{role}_model_name")

class ResearchRequest(BaseModel):
    """Request model for the /research endpoint."""
    query: str = Field(..., min_length=10, description="The user's research query.")
    planner_llm_config: Optional[LLMConfig] = None
    summarizer_llm_config: Optional[LLMConfig] = None
    writer_llm_config: Optional[LLMConfig] = None
    scraper_strategies: Optional[List[str]] = None

class ResearchResponse(BaseModel):
    """Response model for the /research endpoint."""
    report: str
    llm_token_usage: Dict[str, int]
    estimated_llm_cost: float
    serper_queries_used: int

# Helper function to get litellm-compatible dict from LLMConfig
# Now takes provider and api_keys as arguments
def get_litellm_params(
    config: LLMConfig, 
    provider: Literal['google', 'openrouter'], 
    api_keys: ApiKeys
) -> Dict[str, Any]:
    params = config.model_dump(exclude_none=True, exclude={'provider'})
    
    model_name = params['model']
    
    if provider == 'google':
        gemini_key_secret = api_keys.gemini_api_key
        if not gemini_key_secret:
            raise ValueError("GEMINI_API_KEY is required when llm_provider is 'google'.")
        params['api_key'] = gemini_key_secret.get_secret_value()
        if not model_name.startswith('gemini/'):
             params['model'] = f"gemini/{model_name}"
             
    elif provider == 'openrouter':
        openrouter_key_secret = api_keys.openrouter_api_key
        if not openrouter_key_secret:
             raise ValueError("OPENROUTER_API_KEY is required when llm_provider is 'openrouter'.")
        params['api_key'] = openrouter_key_secret.get_secret_value()
        if not model_name.startswith('openrouter/'):
             params['model'] = f"openrouter/{model_name}"

    if 'api_base' in params and isinstance(params['api_base'], HttpUrl):
         params['api_base'] = str(params['api_base'])
         
    if 'api_key' in params and params['api_key'] is None:
        del params['api_key']

    return params 