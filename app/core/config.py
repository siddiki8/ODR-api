import os
from typing import Optional
import logging

from pydantic import Field, HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv, find_dotenv # Import find_dotenv

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
    firebase_service_account_key_json: SecretStr = Field(..., alias='FIREBASE_SERVICE_ACCOUNT_KEY_JSON')



class AppSettings(BaseSettings):
    """Loads and validates all application-level settings.
    
    Sources settings from a .env file and environment variables.
    """
    model_config = SettingsConfigDict(env_file='.env', extra='allow')

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