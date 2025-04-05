import os
from typing import Dict, Any, Optional, List, Literal
import logging
from pathlib import Path # Import Path

from pydantic import BaseModel, Field, HttpUrl, SecretStr, field_validator, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv # Import find_dotenv

# Import custom exception
from .exceptions import ConfigurationError, ValidationError

# --- Explicitly load .env from the project root ---
# Find the .env file starting from this file's directory and going up
dotenv_path = find_dotenv(filename='.env', raise_error_if_not_found=False) 
print(f"--- Attempting to load .env file from: {dotenv_path} ---")
if dotenv_path:
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True) # Explicit path and override
    print(f".env file loaded successfully: {loaded}")
else:
    print("!!! .env file not found by find_dotenv() !!!")
    loaded = False # Set loaded to False if file not found
# --- END Load .env ---

# --- DEBUG: Check environment variable directly ---
openrouter_key_env = os.getenv('OPENROUTER_API_KEY')
if openrouter_key_env:
    print(f"Direct os.getenv('OPENROUTER_API_KEY'): {openrouter_key_env[:4]}...{openrouter_key_env[-4:]} (Loaded: {loaded})")
else:
    print(f"Direct os.getenv('OPENROUTER_API_KEY'): Not Found! (.env loaded: {loaded})")
# --- END DEBUG ---

# REMOVE module-level logger definition
# logger = logging.getLogger(__name__)

class ApiKeys(BaseSettings):
    """Loads API keys from environment variables."""
    # Remove env_file, rely on load_dotenv() having populated os.environ
    model_config = SettingsConfigDict(extra='ignore') 

    serper_api_key: SecretStr = Field(..., alias='SERPER_API_KEY')
    together_api_key: SecretStr = Field(..., alias='TOGETHER_API_KEY')
    openrouter_api_key: Optional[SecretStr] = Field(None, alias='OPENROUTER_API_KEY')
    gemini_api_key: Optional[SecretStr] = Field(None, alias='GEMINI_API_KEY')

# --- DEBUG: Check ApiKeys object after instantiation ---
try:
    print("--- Instantiating ApiKeys ---")
    api_keys_instance = ApiKeys()
    print("ApiKeys instantiated successfully.")
    if api_keys_instance.openrouter_api_key:
        key_val = api_keys_instance.openrouter_api_key.get_secret_value()
        print(f"ApiKeys.openrouter_api_key: {key_val[:4]}...{key_val[-4:]}")
    else:
        print("ApiKeys.openrouter_api_key: Not found or None.")
except Exception as e:
    print(f"Error instantiating ApiKeys: {e}")
# --- END DEBUG ---

class LLMConfig(BaseModel):
    """Model for LLM configuration passed to LiteLLM."""
    model: str = "gemini-pro"
    provider: Optional[Literal['google', 'openrouter']] = None
    api_key: Optional[SecretStr] = None
    api_base: Optional[HttpUrl] = None
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)

    # Optional: Add validator for temperature if specific bounds needed beyond ge/le
    # @field_validator('temperature')
    # def check_temperature(cls, v):
    #     if v is not None and not (0.0 <= v <= 2.0):
    #         raise ValueError('temperature must be between 0.0 and 2.0')
    #     return v

    # Optional: Add validator for max_tokens if specific bounds needed beyond gt
    # @field_validator('max_tokens')
    # def check_max_tokens(cls, v):
    #     if v is not None and v <= 0:
    #         raise ValueError('max_tokens must be positive')
    #     return v

class AppSettings(BaseSettings):
    """Application-level settings."""
    model_config = SettingsConfigDict(env_file='.env', extra='allow')

    llm_provider: Literal['google', 'openrouter'] = Field('openrouter', alias='LLM_PROVIDER')

    # --- Default Model Names (without provider prefix) ---
    # These are for the 'openrouter' provider by default
    default_planner_model_name: str = Field(..., min_length=1)
    default_summarizer_model_name: str = Field(..., min_length=1)
    default_writer_model_name: str = Field(..., min_length=1)
    
    # --- Google Provider Specific Model Names ---
    # Use standard LiteLLM names for Gemini API (Google AI Studio)
    google_planner_model_name: str = Field(..., min_length=1)
    google_summarizer_model_name: str = Field(..., min_length=1)
    google_writer_model_name: str = Field(..., min_length=1)

    # --- Default LLM Configurations (using model names defined above) ---
    # These will be finalized in the agent __init__ based on the provider
    # Model name is set dynamically using settings.get_model_name(role)
    default_planner_llm: LLMConfig = LLMConfig(temperature=0.5)
    default_summarizer_llm: LLMConfig = LLMConfig(temperature=0.3)
    default_writer_llm: LLMConfig = LLMConfig(temperature=0.7, max_tokens=100000)

    reranker_model: str = Field(..., alias='RERANKER_MODEL', min_length=1)
    serper_base_url: HttpUrl = Field("https://google.serper.dev", alias='SERPER_BASE_URL')
    serper_default_location: str = 'us'
    serper_timeout: int = Field(15, gt=0)
    scraper_debug: bool = False
    # PDF Handling Settings
    scraper_download_pdfs: bool = Field(False, alias='SCRAPER_DOWNLOAD_PDFS')
    scraper_pdf_save_dir: str = Field("downloaded_pdfs", alias='SCRAPER_PDF_SAVE_DIR')
    scraper_max_pdf_size_mb: int = Field(512, alias='SCRAPER_MAX_PDF_SIZE_MB', gt=0)
    max_initial_search_tasks: int = Field(3, ge=1, le=10)
    top_m_full_text_sources: int = Field(3, ge=1, le=20)
    max_refinement_iterations: int = Field(2, ge=0, le=5)

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
    api_keys: ApiKeys # Expecting an instantiated ApiKeys object
) -> Dict[str, Any]:
    """Prepares LiteLLM parameters, ensuring required keys and API base are set."""
    
    # Start with base config parameters (temp, max_tokens etc.)
    params = config.model_dump(exclude_none=True, exclude={'provider', 'api_base'})
    
    print("\n=== LiteLLM Configuration ===")
    
    # Set model and provider-specific settings explicitly
    if provider == 'google':
        api_key_secret = api_keys.gemini_api_key
        if not api_key_secret:
            raise ConfigurationError("Configuration Error: GEMINI_API_KEY is required when llm_provider is 'google' but was not found in the provided ApiKeys object.")
        
        api_key_value = api_key_secret.get_secret_value()
        params['api_key'] = api_key_value
        
        # Debug print showing part of the key
        key_prefix = api_key_value[:4] if len(api_key_value) > 8 else "****"
        key_suffix = api_key_value[-4:] if len(api_key_value) > 8 else "****"
        print(f"Provider: Google")
        print(f"API Key: {key_prefix}...{key_suffix} (length: {len(api_key_value)})")
        
        model_name = params.get('model', '')
        if not model_name.startswith('gemini/'):
             params['model'] = f"gemini/{model_name}"
        # Google provider typically does not require api_base
        params.pop('api_base', None) # Remove api_base if present for Google
             
    elif provider == 'openrouter':
        openrouter_key_secret = api_keys.openrouter_api_key
        if not openrouter_key_secret:
            raise ConfigurationError("Configuration Error: OPENROUTER_API_KEY is required when llm_provider is 'openrouter' but was not found in ApiKeys.")
        
        api_key_value = openrouter_key_secret.get_secret_value()
        params['api_key'] = api_key_value
        
        # Debug print showing part of the key
        key_prefix = api_key_value[:4] if len(api_key_value) > 8 else "****"
        key_suffix = api_key_value[-4:] if len(api_key_value) > 8 else "****"
        print(f"Provider: OpenRouter")
        print(f"API Key: {key_prefix}...{key_suffix} (length: {len(api_key_value)})")
        
        model_name = params.get('model', '')
        if not model_name.startswith('openrouter/'):
             params['model'] = f"openrouter/{model_name}"
             
        # Explicitly set api_base for OpenRouter
        params['api_base'] = "https://openrouter.ai/api/v1"
    
    # Convert HttpUrl api_base to string (should not be needed now as we set it above)
    # if 'api_base' in params and isinstance(params['api_base'], HttpUrl):
    #      params['api_base'] = str(params['api_base'])
         
    # Clear explicit API key if None (should not be needed with ConfigurationError checks)
    # if 'api_key' in params and params['api_key'] is None:
    #     del params['api_key']
        
    print(f"Model: {params.get('model')}")
    print(f"API Base: {params.get('api_base')}")
    print(f"Temperature: {params.get('temperature')}")
    print(f"Max Tokens: {params.get('max_tokens')}")
    print("============================\n")
    return params 