import os
from typing import Dict, Any, Optional, List, Literal
import logging
from pathlib import Path # Import Path

from pydantic import BaseModel, Field, HttpUrl, SecretStr, field_validator, validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv # Import find_dotenv

# Import custom exception
from .exceptions import ConfigurationError, ValidationError

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Explicitly load .env from the project root ---
# Find the .env file starting from this file's directory and going up
dotenv_path = find_dotenv(filename='.env', raise_error_if_not_found=False) 
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if dotenv_path:
    loaded = load_dotenv(dotenv_path=dotenv_path, override=True) # Explicit path and override
    logger.info(f".env file loaded successfully: {loaded}")
else:
    logger.warning(".env file not found by find_dotenv()!")
    loaded = False # Set loaded to False if file not found
# --- END Load .env ---

# --- DEBUG: Check environment variable directly ---
openrouter_key_env = os.getenv('OPENROUTER_API_KEY')
if openrouter_key_env:
    logger.debug(f"Direct os.getenv('OPENROUTER_API_KEY'): FOUND (Loaded: {loaded})") # Don't log key value
else:
    logger.debug(f"Direct os.getenv('OPENROUTER_API_KEY'): Not Found! (.env loaded: {loaded})")
# --- END DEBUG ---

class ApiKeys(BaseSettings):
    """Loads sensitive API keys from environment variables."""
    # Use SettingsConfigDict for Pydantic v2 settings
    model_config = SettingsConfigDict(extra='ignore') 

    serper_api_key: SecretStr = Field(..., alias='SERPER_API_KEY')
    together_api_key: SecretStr = Field(..., alias='TOGETHER_API_KEY')
    openrouter_api_key: Optional[SecretStr] = Field(None, alias='OPENROUTER_API_KEY')
    gemini_api_key: Optional[SecretStr] = Field(None, alias='GEMINI_API_KEY')

class LLMConfig(BaseModel):
    """Represents configuration parameters for an LLM call via LiteLLM.
    
    Defaults are provided, but typically overridden during agent initialization.
    """
    model_config = ConfigDict(extra='ignore') # Add V2 config

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
    """Loads and validates all application-level settings.
    
    Sources settings from a .env file and environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='allow')

    llm_provider: Literal['google', 'openrouter'] = Field('openrouter', alias='LLM_PROVIDER')

    # --- Model Names per Provider ---
    default_planner_model_name: str = Field(..., min_length=1, description="Model name for planner (OpenRouter default)")
    default_summarizer_model_name: str = Field(..., min_length=1, description="Model name for summarizer (OpenRouter default)")
    default_writer_model_name: str = Field(..., min_length=1, description="Model name for writer (OpenRouter default)")
    
    google_planner_model_name: str = Field(..., min_length=1, description="Model name for planner (Google provider)")
    google_summarizer_model_name: str = Field(..., min_length=1, description="Model name for summarizer (Google provider)")
    google_writer_model_name: str = Field(..., min_length=1, description="Model name for writer (Google provider)")

    # --- Default LLM Parameter Overrides ---
    # Base LLMConfig objects; model name and API details are set later in the agent
    default_planner_llm: LLMConfig = LLMConfig(temperature=0.5)
    default_summarizer_llm: LLMConfig = LLMConfig(temperature=0.3)
    default_writer_llm: LLMConfig = LLMConfig(temperature=0.7, max_tokens=65000) #optionally - max_tokens=100000

    # --- Service Configurations ---
    reranker_model: str = Field(..., alias='RERANKER_MODEL', min_length=1, description="Model name for Together Reranker API")
    openrouter_api_base: HttpUrl = Field("https://openrouter.ai/api/v1", alias='OPENROUTER_API_BASE', description="Base URL for OpenRouter API")
    serper_base_url: HttpUrl = Field("https://google.serper.dev", alias='SERPER_BASE_URL', description="Base URL for Serper API")
    serper_default_location: str = Field('us', description="Default location for Serper searches")
    serper_timeout: int = Field(15, gt=0, description="Timeout in seconds for Serper API calls")
    
    # --- Scraper Configurations ---
    scraper_debug: bool = Field(False, description="Enable debug logging for the scraper")
    scraper_download_pdfs: bool = Field(False, alias='SCRAPER_DOWNLOAD_PDFS', description="Whether to save downloaded PDFs to disk")
    scraper_pdf_save_dir: str = Field("downloaded_pdfs", alias='SCRAPER_PDF_SAVE_DIR', description="Directory to save PDFs if enabled")
    scraper_max_pdf_size_mb: int = Field(512, alias='SCRAPER_MAX_PDF_SIZE_MB', gt=0, description="Maximum size (MB) for PDFs to process (0=unlimited)")
    
    # --- Agent Workflow Parameters ---
    max_initial_search_tasks: int = Field(3, ge=1, le=10, description="Max searches Planner LLM can generate initially")
    max_refinement_iterations: int = Field(2, ge=0, le=5, description="Max refinement loops (search->process->refine)")

    def get_model_name(self, role: Literal['planner', 'summarizer', 'writer']) -> str:
        """Returns the appropriate model name based on the configured llm_provider."""
        if self.llm_provider == 'google':
            return getattr(self, f"google_{role}_model_name")
        else:
            return getattr(self, f"default_{role}_model_name")

# Removed ResearchRequest and ResearchResponse - Moved to schemas.py

# Helper function to get litellm-compatible dict from LLMConfig
# Now takes provider and api_keys as arguments
def get_litellm_params(
    config: LLMConfig, 
    provider: Literal['google', 'openrouter'], 
    api_keys: ApiKeys, # Expecting an instantiated ApiKeys object
    settings: AppSettings # Add settings parameter
) -> Dict[str, Any]:
    """Prepares LiteLLM parameters dictionary from LLMConfig.
    
    Ensures the correct model prefix, API key, and API base are set based on the provider.

    Args:
        config: The base LLMConfig object.
        provider: The LLM provider ('google' or 'openrouter').
        api_keys: The ApiKeys object containing necessary secrets.
        settings: The AppSettings object (needed for OpenRouter base URL).
        
    Returns:
        A dictionary compatible with litellm.completion(**params).
        
    Raises:
        ConfigurationError: If the required API key for the provider is missing.
    """
    
    # Start with base config parameters (temp, max_tokens etc.)
    params = config.model_dump(exclude_none=True, exclude={'provider', 'api_base'})
    
    logger.debug("\n=== Generating LiteLLM Configuration ===")
    
    # Set model and provider-specific settings explicitly
    if provider == 'google':
        api_key_secret = api_keys.gemini_api_key
        if not api_key_secret:
            raise ConfigurationError("Configuration Error: GEMINI_API_KEY is required when llm_provider is 'google' but was not found in the provided ApiKeys object.")
        
        api_key_value = api_key_secret.get_secret_value()
        params['api_key'] = api_key_value
        
        # Log provider and key presence, avoid logging key details
        logger.debug(f"Provider: Google")
        logger.debug(f"API Key: FOUND (length: {len(api_key_value)})")
        
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
        
        # Log provider and key presence, avoid logging key details
        logger.debug(f"Provider: OpenRouter")
        logger.debug(f"API Key: FOUND (length: {len(api_key_value)})")
        
        model_name = params.get('model', '')
        if not model_name.startswith('openrouter/'):
             params['model'] = f"openrouter/{model_name}"
             
        # Explicitly set api_base for OpenRouter
        params['api_base'] = str(settings.openrouter_api_base) # Get from settings & convert HttpUrl to str
    
    # Convert HttpUrl api_base to string (should not be needed now as we set it above)
    # if 'api_base' in params and isinstance(params['api_base'], HttpUrl):
    #      params['api_base'] = str(params['api_base'])
         
    # Clear explicit API key if None (should not be needed with ConfigurationError checks)
    # if 'api_key' in params and params['api_key'] is None:
    #     del params['api_key']
        
    logger.debug(f"  Model: {params.get('model')}")
    logger.debug(f"  API Base: {params.get('api_base')}")
    logger.debug(f"  Temperature: {params.get('temperature')}")
    logger.debug(f"  Max Tokens: {params.get('max_tokens')}")
    logger.debug("======================================\n")
    return params 