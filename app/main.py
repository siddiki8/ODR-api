from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
import logging
import traceback
import uuid
import json
import asyncio
from typing import Dict, Any, Optional
import os

# Import configuration
from .core.config import AppSettings, ApiKeys
# Import custom exceptions (Keep all as they are used by handlers)
from .core.exceptions import (
    DeepResearchError, ConfigurationError, ValidationError, LLMError,
    ExternalServiceError, SearchAPIError, RankingAPIError, ScrapingError,
    AgentExecutionError, LLMCommunicationError, LLMRateLimitError, LLMOutputValidationError
)
from fastapi.middleware.cors import CORSMiddleware

# Firebase Admin SDK Imports
import firebase_admin
from firebase_admin import credentials, firestore

# --- Import Agency Routers ---
from app.agencies.deep_research.routes import router as deep_research_router

# --- Import Shared State ---
from .core import state

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Deep Research API",
    description="API service providing access to various research agencies.",
    version="0.1.0"
)

# --- Global Task Tracking (Accessed by agency routers) ---
# TODO: Move to app.core.state
# active_tasks: Dict[str, asyncio.Task] = {} # Now defined in core.state

# --- Initialize Firebase Admin SDK (Accessed by agency routers) ---
# TODO: Move to app.core.state and initialize here
# db = None # Now defined in core.state
try:
    firebase_cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY_JSON")
    if not firebase_cred_path:
        logger.warning("FIREBASE_SERVICE_ACCOUNT_KEY_JSON env var not set. Firestore integration disabled.")
    else:
        if not os.path.exists(firebase_cred_path):
             logger.warning(f"Firebase credentials file not found at: {firebase_cred_path}. Firestore integration disabled.")
        else:
            cred = credentials.Certificate(firebase_cred_path)
            firebase_admin.initialize_app(cred)
            # db = firestore.client() # Get Firestore client
            state.db = firestore.client() # Initialize the db instance in core.state
            logger.info("Firebase Admin SDK initialized successfully.")
except ValueError as e:
     logger.error(f"Error initializing Firebase Admin SDK (likely invalid creds path/format): {e}", exc_info=False)
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
    # db remains None, subsequent checks will handle this

# --- Import Core Dependencies ---
from .core.dependencies import get_settings, get_api_keys

# --- Global Exception Handlers (Apply to all routes) --- #

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Configuration Error (ID: {error_id}): {error_type} - {exc}", exc_info=False)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server configuration error. Please contact administrator. Error ID: {error_id}"},
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.warning(f"Validation Error (ID: {error_id}): {exc}", exc_info=False)
    return JSONResponse(
        status_code=422,
        content={"detail": f"Validation Error: {str(exc)}. Error ID: {error_id}"},
    )

@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_error_handler(request: Request, exc: LLMRateLimitError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.warning(f"LLM Rate Limit Error (ID: {error_id}): {error_type}", exc_info=False)
    return JSONResponse(
        status_code=429,
        content={"detail": f"LLM service rate limit exceeded. Please try again later. Error ID: {error_id}"},
    )

@app.exception_handler(LLMCommunicationError)
async def llm_communication_error_handler(request: Request, exc: LLMCommunicationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"LLM Communication Error (ID: {error_id}): {error_type}", exc_info=False)
    return JSONResponse(
        status_code=503,
        content={"detail": f"Error communicating with LLM service. Please try again later. Error ID: {error_id}"},
    )

@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"External Service Error (ID: {error_id}) - Type: {error_type}: {exc}", exc_info=False)
    return JSONResponse(
        status_code=503,
        content={"detail": f"Error communicating with an external service ({error_type}). Please try again later. Error ID: {error_id}"},
    )

@app.exception_handler(AgentExecutionError)
async def agent_error_handler(request: Request, exc: AgentExecutionError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Agent Execution Error (ID: {error_id}): {error_type} - {exc}", exc_info=False)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred during the agent execution process. Error ID: {error_id}"},
    )

@app.exception_handler(DeepResearchError)
async def deep_research_error_handler(request: Request, exc: DeepResearchError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Deep Research Error (ID: {error_id}) - Type: {error_type}", exc_info=False)
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred during deep research. Error ID: {error_id}"},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.critical(f"Unhandled Exception (ID: {error_id}): {error_type}", exc_info=True)
    # traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"An internal server error occurred. Error ID: {error_id}"},
    )

# --- Root Endpoint --- #
@app.get("/")
async def root():
    _ = get_settings() # Depends() will resolve this using the imported function
    return {"message": "ODR Multi-Agency API is running."}

# --- Include Agency Routers ---
# Add routers from different agencies here
app.include_router(
    deep_research_router, 
    prefix="/deep_research",
    tags=["Deep Research Agency"]
)
# Add other agency routers here in the future
# app.include_router(another_agency_router, prefix="/another_agency", tags=["Another Agency"])

# --- CORS Middleware --- #
# TODO: Make origins configurable via settings
allowed_origins = [
    "http://https://odr-frontend.vercel.app",
    # Add other allowed origins like localhost for development
    "http://localhost:3000", 
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 