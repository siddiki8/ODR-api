import logging
from .config import AppSettings, ApiKeys
from .exceptions import ConfigurationError
from fastapi import Depends, HTTPException
from typing import Optional
from .state import db, active_tasks, initialize_firebase_sync
from firebase_admin.firestore import Client

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

# Cache instances to avoid reloading on every request
_settings: Optional[AppSettings] = None
_api_keys: Optional[ApiKeys] = None

# --- Dependency Injection Functions ---
def get_settings() -> AppSettings:
    """Dependency function to get application settings."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings

def get_api_keys() -> ApiKeys:
    """Dependency function to get API keys."""
    global _api_keys
    if _api_keys is None:
        _api_keys = ApiKeys()
    return _api_keys

def get_firestore_db() -> Optional[Client]:
    """
    Dependency function to get the Firestore client instance.
    Initializes Firebase if necessary.
    Raises HTTPException if initialization fails and Firestore is required.
    """
    firestore_client = initialize_firebase_sync()
    if firestore_client is None:
        # Decide if this should be a hard error or allow optional usage
        # For now, let's raise an error if the DB is needed but unavailable
        raise HTTPException(
            status_code=503,
            detail="Firestore service is unavailable. Initialization failed."
        )
    return firestore_client 