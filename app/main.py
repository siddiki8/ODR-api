from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState # Import WebSocketState for checking connection state
import logging
import traceback
import uuid
import json
import asyncio # Keep asyncio for potential use elsewhere, though direct sleeps removed
from typing import Dict, Any, Optional # <-- Import Dict, Any, Optional

# Import configuration, models, and the agent
from .core.config import AppSettings, ApiKeys # Import only AppSettings and ApiKeys from config
from .core.schemas import ResearchRequest, ResearchResponse # Import schemas from the correct file
# Dependencies removed from config import to avoid circularity if handlers need them
from .core.agent import DeepResearchAgent
# Import custom exceptions
from .core.exceptions import (
    DeepResearchError, ConfigurationError, ValidationError, LLMError,
    ExternalServiceError, SearchAPIError, RankingAPIError, ScrapingError,
    AgentExecutionError, LLMCommunicationError, LLMRateLimitError, LLMOutputValidationError
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Deep Research API",
    description="API service for performing deep research on a given query.",
    version="0.1.0"
)

# --- Dependency Injection (Keep separate, potentially move to a deps.py later) ---
# Load settings and keys once
try:
    app_settings = AppSettings()
    api_keys = ApiKeys()
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to load AppSettings or ApiKeys: {e}", exc_info=True)
    # If config fails to load, the app cannot function. We might let it crash,
    # or handle it depending on deployment strategy. For now, log critical error.
    # Raising here might prevent app startup. Consider a check within endpoints/dependencies.
    app_settings = None # Or some default safe object
    api_keys = None # Or some default safe object

def get_settings() -> AppSettings:
    if app_settings is None:
        # This indicates a critical startup failure
        raise ConfigurationError("Application settings could not be loaded.")
    return app_settings

def get_api_keys() -> ApiKeys:
    if api_keys is None:
        # This indicates a critical startup failure
        raise ConfigurationError("API keys could not be loaded.")
    return api_keys

# --- Exception Handlers --- #

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Configuration Error (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=500, # Configuration errors are server-side issues
        content={"detail": f"Server configuration error. Please contact administrator. Error ID: {error_id}"},
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.warning(f"Validation Error (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=422, # Unprocessable Entity for validation errors
        content={"detail": f"Validation Error: {error_type}. Error ID: {error_id}"},
    )

@app.exception_handler(LLMRateLimitError)
async def llm_rate_limit_error_handler(request: Request, exc: LLMRateLimitError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.warning(f"LLM Rate Limit Error (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=429, # Too Many Requests
        content={"detail": f"LLM service rate limit exceeded. Please try again later. Error ID: {error_id}"},
    )

@app.exception_handler(LLMCommunicationError)
async def llm_communication_error_handler(request: Request, exc: LLMCommunicationError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"LLM Communication Error (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=503, # Service Unavailable
        content={"detail": f"Error communicating with LLM service. Please try again later. Error ID: {error_id}"},
    )

# Catch LLMOutputValidationError specifically if needed, but ValidationError might cover it
# @app.exception_handler(LLMOutputValidationError)
# async def llm_output_validation_error_handler(request: Request, exc: LLMOutputValidationError):
#     ... defaults to ValidationError handler ...

@app.exception_handler(ExternalServiceError)
async def external_service_error_handler(request: Request, exc: ExternalServiceError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"External Service Error (ID: {error_id}) - Type: {error_type}: {exc}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=503, # Service Unavailable
        content={"detail": f"Error communicating with an external service ({error_type}). Please try again later. Error ID: {error_id}"},
    )

@app.exception_handler(AgentExecutionError)
async def agent_error_handler(request: Request, exc: AgentExecutionError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Agent Execution Error (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=500, # Internal Server Error for agent logic failures
        content={"detail": f"An error occurred during the research process. Error ID: {error_id}"},
    )

# Fallback for our custom errors if not caught above
@app.exception_handler(DeepResearchError)
async def deep_research_error_handler(request: Request, exc: DeepResearchError):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.error(f"Unhandled Deep Research Error (ID: {error_id}) - Type: {error_type}", exc_info=False) # Log type only
    return JSONResponse(
        status_code=500,
        content={"detail": f"An unexpected error occurred. Error ID: {error_id}"},
    )

# Generic fallback for any other unhandled exceptions
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_id = uuid.uuid4()
    error_type = type(exc).__name__
    logger.critical(f"Unhandled Exception (ID: {error_id}): {error_type}", exc_info=False) # Log type only
    # traceback.print_exc() # Avoid printing full traceback to console by default
    return JSONResponse(
        status_code=500,
        content={"detail": f"An internal server error occurred. Error ID: {error_id}"},
    )

# --- API Endpoint --- #
# Deprecate the old endpoint, add description pointing to WebSocket
@app.post("/research", response_model=ResearchResponse, deprecated=True, description="Synchronous research endpoint. Prefer /ws/research for streaming updates.")
async def run_research(
    request: ResearchRequest,
    settings: AppSettings = Depends(get_settings),
    keys: ApiKeys = Depends(get_api_keys)
):
    """
    Endpoint to initiate a deep research task.
    
    Takes a user query and optional configuration overrides.
    Returns the generated report and usage statistics.
    ---
    DEPRECATED: This synchronous endpoint is no longer functional. 
    Please use the WebSocket endpoint at /ws/research.
    """
    # Immediately raise an error indicating deprecation and pointing to WebSocket
    raise HTTPException(
        status_code=404, # Or 400 Bad Request, 404 seems reasonable for 'not found anymore'
        detail="This synchronous endpoint is deprecated and no longer functional. Please use the WebSocket endpoint at /ws/research."
    )
    
    # --- Removed Old Logic --- 
    # request_id = uuid.uuid4()
    # logger.info(f"[Request ID: {request_id}] Received research request for query: '{request.query[:50]}...')
    # agent = DeepResearchAgent(
    #     settings=settings,
    #     api_keys=keys,
    #     planner_llm_override=request.planner_llm_config,
    #     summarizer_llm_override=request.summarizer_llm_config,
    #     writer_llm_override=request.writer_llm_config,
    #     # These parameters are incorrect/missing based on current agent __init__
    #     # scraper_strategies_override=request.scraper_strategies, 
    #     # logger_callback=lambda msg, level=logging.INFO: logger.info(f"[Agent - Request {request_id}] {msg}") 
    # )
    # logger.info(f"[Request ID: {request_id}] Starting deep research process...")
    # result_dict = await agent.run_deep_research(request.query)
    # logger.info(f"[Request ID: {request_id}] Deep research process completed successfully.")
    # return ResearchResponse(**result_dict)
    # --- End Removed Old Logic ---

# WebSocket Endpoint (Updated)
@app.websocket("/ws/research")
async def websocket_research(
    websocket: WebSocket,
    settings: AppSettings = Depends(get_settings),
    keys: ApiKeys = Depends(get_api_keys)
):
    request_id = uuid.uuid4()
    connection_active = True # Flag to manage sending messages only if active
    await websocket.accept()
    logger.info(f"[WS Request ID: {request_id}] WebSocket connection accepted.")

    # --- Define the callback function for the agent ---
    async def send_status_update(step: str, status: str, message: str, details: dict | None = None):
        """Formats and sends status updates over the WebSocket, checking connection state."""
        # Ensure we don't try to send if connection is closed
        nonlocal connection_active
        if not connection_active:
             logger.warning(f"[WS Request ID: {request_id}] Skipping send_status_update; connection inactive.")
             return

        payload = {"step": step, "status": status, "message": message, "details": details or {}}
        try:
            await websocket.send_json(payload)
            logger.debug(f"[WS Request ID: {request_id}] Sent update: {step}/{status} - {message}")
        except WebSocketDisconnect:
            logger.warning(f"[WS Request ID: {request_id}] Client disconnected while trying to send update {step}/{status}.")
            connection_active = False # Mark connection as inactive
        except RuntimeError as e: # Catch errors if sending on a closed/closing connection
             logger.warning(f"[WS Request ID: {request_id}] Runtime error sending update {step}/{status} (likely closing): {e}")
             connection_active = False
        except Exception as e:
            logger.error(f"[WS Request ID: {request_id}] Failed to send WebSocket message for {step}/{status}: {e}", exc_info=True)
            connection_active = False # Assume connection is broken

    # Define agent type hint
    agent: Optional[DeepResearchAgent] = None # Define agent variable

    try:
        # 1. Receive and validate initial research request parameters
        raw_data = await websocket.receive_text() # Use receive_text first
        request: Optional[ResearchRequest] = None
        try:
            data = json.loads(raw_data)
            request = ResearchRequest.model_validate(data)
            logger.info(f"[WS Request ID: {request_id}] Received research request for query: '{request.query[:50]}...'")
        except json.JSONDecodeError:
            logger.error(f"[WS Request ID: {request_id}] Invalid JSON received.")
            await send_status_update("ERROR", "ERROR", "Invalid JSON format received.", {"error": "Bad JSON"})
            await websocket.close(code=1003) # 1003: Cannot accept data type
            return
        except ValidationError as e: # Catch Pydantic validation errors etc.
             logger.error(f"[WS Request ID: {request_id}] Invalid request data: {e}", exc_info=False) # Less noisy log for validation
             # Send specific validation error details if possible
             error_details = {"error": "Invalid request structure or content.", "details": str(e)}
             await send_status_update("ERROR", "ERROR", "Invalid research request parameters.", error_details)
             await websocket.close(code=1003)
             return
        except Exception as e: # Catch other unexpected parsing errors
             logger.error(f"[WS Request ID: {request_id}] Error processing request data: {e}", exc_info=True)
             error_details = {"error": "Failed to process request data.", "details": str(e)}
             await send_status_update("ERROR", "ERROR", "Error processing request.", error_details)
             await websocket.close(code=1011) # Internal error
             return

        # Ensure request is valid before proceeding
        if not request:
             # Should be caught above, but as safeguard
             await send_status_update("ERROR", "ERROR", "Failed to validate request.", {})
             await websocket.close(code=1011)
             return

        # 2. Instantiate the Agent
        await send_status_update("INITIALIZING", "START", "Initializing research agent...")
        logger.info(f"[WS Request ID: {request_id}] Instantiating research agent...")
        try:
            agent = DeepResearchAgent(
                settings=settings,
                api_keys=keys,
                planner_llm_override=request.planner_llm_config,
                summarizer_llm_override=request.summarizer_llm_config,
                writer_llm_override=request.writer_llm_config,
                max_search_tasks_override=request.max_search_tasks,
                llm_provider_override=request.llm_provider,
                websocket_callback=send_status_update
            )
            await send_status_update("INITIALIZING", "END", "Agent initialized successfully.")
            logger.info(f"[WS Request ID: {request_id}] Research agent instantiated.")
        except (ConfigurationError, ValueError) as e: # Catch specific init errors
            logger.error(f"[WS Request ID: {request_id}] Agent Initialization Failed: {e}", exc_info=True)
            await send_status_update("INITIALIZING", "ERROR", f"Agent initialization failed: {e}", {"error": str(e)})
            await websocket.close(code=1011) # Config error is internal
            return
        except Exception as e: # Catch unexpected init errors
            logger.error(f"[WS Request ID: {request_id}] Unexpected error during agent initialization: {e}", exc_info=True)
            await send_status_update("INITIALIZING", "ERROR", "Unexpected error during agent setup.", {"error": str(e)})
            await websocket.close(code=1011)
            return

        # 3. Start the research process
        logger.info(f"[WS Request ID: {request_id}] Starting deep research process via WebSocket...")
        # Agent will send STARTING/START itself now
        # await send_status_update("RUNNING", "START", "Starting deep research...")

        # The agent's run_deep_research handles sending all intermediate updates
        # and the final COMPLETE or ERROR message via the callback.
        # We just need to await its completion and handle potential exceptions it raises.
        research_result: Optional[Dict[str, Any]] = None
        try:
            research_result = await agent.run_deep_research(request.query)
            # If successful, the agent should have sent the "COMPLETE" message already.
            logger.info(f"[WS Request ID: {request_id}] Agent research process completed.")
            # Optionally log result summary (careful with large reports)
            if research_result:
                 logger.info(f"[WS Request ID: {request_id}] Final report length: {len(research_result.get('final_report',''))}")
                 logger.debug(f"[WS Request ID: {request_id}] Usage stats: {research_result.get('usage_statistics')}")

        except AgentExecutionError as e: # Catch agent execution errors specifically
            error_id = uuid.uuid4()
            logger.error(f"[WS Request ID: {request_id}] Agent Execution Error (ID: {error_id}): {e}", exc_info=True)
            # Agent might have already sent an error via callback, but send a final one from here too
            await send_status_update("ERROR", "ERROR", f"Research process failed during execution. Error ID: {error_id}", {"error_type": type(e).__name__, "error_id": str(error_id)})
            # Close with error code 1011 (Internal Server Error)
            await websocket.close(code=1011)
            return # Exit after handling error

        except DeepResearchError as e: # Catch other custom errors from the agent/subsystems
            error_id = uuid.uuid4()
            logger.error(f"[WS Request ID: {request_id}] Deep Research Error (ID: {error_id}): {e}", exc_info=True)
            await send_status_update("ERROR", "ERROR", f"An error occurred during research. Error ID: {error_id}", {"error_type": type(e).__name__, "error_id": str(error_id)})
            await websocket.close(code=1011)
            return

        # Catch other unexpected errors during agent execution
        except Exception as e:
            error_id = uuid.uuid4()
            logger.critical(f"[WS Request ID: {request_id}] Unexpected critical error during agent execution (Error ID: {error_id}): {e}", exc_info=True)
            traceback.print_exc()
            await send_status_update("ERROR", "ERROR", f"An unexpected server error occurred. Error ID: {error_id}", {"error_type": type(e).__name__, "error_id": str(error_id)})
            await websocket.close(code=1011)
            return

        # 4. Wait for client disconnect or close from server side after completion/error
        # FastAPI handles keeping the connection open while the await agent.run...() is running.
        # If the agent completes successfully, it sends the COMPLETE message.
        # We can now close the connection gracefully from the server side.

    except WebSocketDisconnect:
        connection_active = False # Mark as inactive
        logger.info(f"[WS Request ID: {request_id}] WebSocket disconnected by client.")
        # Handle cleanup if necessary (e.g., signal the agent to stop if it were running longer/cancellable)
        # If agent is still running somehow, this disconnect might be logged by the send_status_update callback.

    except Exception as e: # Catch errors in the WebSocket handler itself (e.g., initial receive_text)
        connection_active = False
        error_id = uuid.uuid4()
        logger.critical(f"[WS Request ID: {request_id}] Unhandled exception in WebSocket handler (Error ID: {error_id}): {e}", exc_info=True)
        traceback.print_exc()
        # Try to inform client if possible (might fail if connection already broken)
        try:
            # Avoid calling send_status_update if connection is already known inactive
            # Check websocket state directly for robustness
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_json({
                     "step": "ERROR", "status": "ERROR", 
                     "message": f"An unexpected server error occurred in the connection handler. Error ID: {error_id}",
                     "details": {"error_type": type(e).__name__, "error_id": str(error_id)}
                 })
        except Exception:
             logger.error(f"[WS Request ID: {request_id}] Failed to send final error message during handler exception.")

        # Ensure connection is closed on server error
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011) # Internal Server Error

    finally:
        # Final check to ensure connection is closed from server-side if it wasn't already.
        connection_active = False # Explicitly mark inactive
        if websocket.client_state == WebSocketState.CONNECTED:
             try:
                 await websocket.close()
                 logger.info(f"[WS Request ID: {request_id}] WebSocket connection explicitly closed by server in finally block.")
             except RuntimeError as e:
                 logger.warning(f"[WS Request ID: {request_id}] Error closing WebSocket in finally block (might be closing concurrently): {e}")
        else:
            logger.info(f"[WS Request ID: {request_id}] WebSocket connection was already closed before finally block finished.")


# Optional: Add a root endpoint for health checks or basic info
@app.get("/")
async def root():
    # Basic check to see if settings loaded, otherwise ConfigurationError handled globally
    _ = get_settings()
    return {"message": "Deep Research API is running."}

# Optional: Configure CORS if needed
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     # Make sure to load origins from settings correctly if uncommenting
#     allow_origins=["*"] if not app_settings else app_settings.cors_origins, # Example
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# ) 