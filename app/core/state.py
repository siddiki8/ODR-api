import asyncio
from typing import Dict, Any, Optional
import logging
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.firestore import Client # Correct type hint

logger = logging.getLogger(__name__)

# --- Shared State Variables --- #
# Firestore client instance (initialized via lifespan or dependency)
db: Optional[Client] = None # Use Client type hint

# Dictionary to track active background tasks (e.g., research orchestrations)
active_tasks: Dict[str, asyncio.Task] = {}

# --- Firebase Initialization Function --- #
# Renamed to avoid conflict and make it synchronous
def initialize_firebase_sync() -> Optional[Client]:
    """Initializes the Firebase Admin SDK synchronously and returns the client."""
    global db
    # Check if already initialized
    if db is not None:
        return db
    
    try:
        firebase_cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON")
        if not firebase_cred_path:
            logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON env var not set. Firestore integration disabled.")
            return None

        if not os.path.exists(firebase_cred_path):
            logger.warning(f"Firebase credentials file not found at: {firebase_cred_path}. Firestore integration disabled.")
            return None

        # Avoid double initialization if called multiple times before db is set
        if not firebase_admin._apps:
            cred = credentials.Certificate(firebase_cred_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized successfully.")
        else:
            logger.info("Firebase Admin SDK already initialized.")

        db = firestore.client() # Set the global db state
        return db
    except ValueError as e:
        logger.error(f"Error initializing Firebase Admin SDK (likely invalid creds path/format): {e}", exc_info=False)
        db = None # Ensure db is None on failure
        return None
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
        db = None # Ensure db is None on failure
        return None 