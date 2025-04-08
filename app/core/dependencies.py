import logging
from .config import AppSettings, ApiKeys
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# --- Core Config Instances ---
# Load settings and keys once during import
# This assumes AppSettings and ApiKeys can be instantiated from environment/defaults
try:
    app_settings = AppSettings()
    api_keys = ApiKeys()
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to load AppSettings or ApiKeys on import: {e}", exc_info=True)
    # Set to None so dependency functions raise ConfigurationError
    app_settings = None
    api_keys = None

# --- Dependency Injection Functions ---
def get_settings() -> AppSettings:
    """FastAPI dependency function to get application settings."""
    if app_settings is None:
        # This indicates a critical startup failure during import
        raise ConfigurationError("Application settings could not be loaded during startup.")
    return app_settings

def get_api_keys() -> ApiKeys:
    """FastAPI dependency function to get API keys."""
    if api_keys is None:
        # This indicates a critical startup failure during import
        raise ConfigurationError("API keys could not be loaded during startup.")
    return api_keys 