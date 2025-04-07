from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect, Path
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketState # Import WebSocketState for checking connection state
import logging
import traceback
import uuid
import json
import asyncio # Keep asyncio for potential use elsewhere, though direct sleeps removed
from typing import Dict, Any, Optional # <-- Import Dict, Any, Optional
import os # <-- Import os

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
from fastapi.middleware.cors import CORSMiddleware

# Firebase Admin SDK Imports
import firebase_admin
from firebase_admin import credentials, firestore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Deep Research API",
    description="API service for performing deep research on a given query.",
    version="0.1.0"
)

# --- Global Task Tracking --- #
# Stores references to active agent tasks, keyed by task_id_str
active_tasks: Dict[str, asyncio.Task] = {}

# --- Initialize Firebase Admin SDK --- #
db = None
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
            db = firestore.client() # Get Firestore client
            logger.info("Firebase Admin SDK initialized successfully.")
except ValueError as e:
     logger.error(f"Error initializing Firebase Admin SDK (likely invalid creds path/format): {e}", exc_info=False)
except Exception as e:
    logger.critical(f"CRITICAL ERROR: Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
    # db remains None, subsequent checks will handle this

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
    task_id = uuid.uuid4()
    task_id_str = str(task_id)
    connection_active = True # Flag to manage sending messages only if active
    agent_task: Optional[asyncio.Task] = None # Reference to the agent task

    await websocket.accept()
    logger.info(f"[WS Task ID: {task_id_str}] WebSocket connection accepted.")

    # --- Define the callback function for the agent ---
    async def send_status_update(step: str, status: str, message: str, details: dict | None = None):
        """Formats and sends status updates over the WebSocket, checking connection state."""
        # Ensure we don't try to send if connection is closed
        nonlocal connection_active
        if not connection_active:
             logger.warning(f"[WS Task ID: {task_id_str}] Skipping send_status_update; connection inactive.")
             return

        payload = {"step": step, "status": status, "message": message, "details": details or {}}
        try:
            await websocket.send_json(payload)
            logger.debug(f"[WS Task ID: {task_id_str}] Sent update: {step}/{status} - {message}")
        except WebSocketDisconnect:
            logger.warning(f"[WS Task ID: {task_id_str}] Client disconnected while trying to send update {step}/{status}.")
            connection_active = False # Mark connection as inactive
        except RuntimeError as e: # Catch errors if sending on a closed/closing connection
             logger.warning(f"[WS Task ID: {task_id_str}] Runtime error sending update {step}/{status} (likely closing): {e}")
             connection_active = False
        except Exception as e:
            logger.error(f"[WS Task ID: {task_id_str}] Failed to send WebSocket message for {step}/{status}: {e}", exc_info=True)
            connection_active = False # Assume connection is broken

    # Define agent type hint
    agent: Optional[DeepResearchAgent] = None # Define agent variable
    task_doc_ref = None
    firestore_available_this_request = False # Local flag for this request
    # Check if global db is initialized and try to get doc ref
    if db:
        try:
            task_doc_ref = db.collection("research_tasks").document(task_id_str)
            firestore_available_this_request = True # Mark Firestore as usable for this request
            logger.debug(f"[WS Task ID: {task_id_str}] Firestore document reference obtained.")
        except Exception as e:
             logger.error(f"[WS Task ID: {task_id_str}] Failed to get Firestore document reference: {e}. Firestore disabled for this request.")
             # Keep firestore_available_this_request = False
             task_doc_ref = None
    else:
        logger.warning(f"[WS Task ID: {task_id_str}] Global Firestore client (db) is not initialized. Firestore disabled for this request.")

    try:
        # 1. Receive and validate initial research request parameters
        raw_data = await websocket.receive_text() # Use receive_text first
        request: Optional[ResearchRequest] = None
        try:
            data = json.loads(raw_data)
            request = ResearchRequest.model_validate(data)
            logger.info(f"[WS Task ID: {task_id_str}] Received research request for query: '{request.query[:50]}...'")
        except json.JSONDecodeError:
            logger.error(f"[WS Task ID: {task_id_str}] Invalid JSON received.")
            await send_status_update("ERROR", "ERROR", "Invalid JSON format received.", {"error": "Bad JSON"})
            await websocket.close(code=1003) # 1003: Cannot accept data type
            return
        except ValidationError as e: # Catch Pydantic validation errors etc.
             logger.error(f"[WS Task ID: {task_id_str}] Invalid request data: {e}", exc_info=False) # Less noisy log for validation
             # Send specific validation error details if possible
             error_details = {"error": "Invalid request structure or content.", "details": str(e)}
             await send_status_update("ERROR", "ERROR", "Invalid research request parameters.", error_details)
             await websocket.close(code=1003)
             return
        except Exception as e: # Catch other unexpected parsing errors
             logger.error(f"[WS Task ID: {task_id_str}] Error processing request data: {e}", exc_info=True)
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

        # *** Send Task ID to Client ***
        await send_status_update("INITIALIZING", "TASK_ID", "Task ID assigned.", {"task_id": task_id_str})

        # *** Create Initial Firestore Document ***
        if firestore_available_this_request: # Use the local flag
            try:
                initial_data = {
                    "taskId": task_id_str,
                    "query": request.query,
                    "status": "PENDING",
                    "createdAt": firestore.SERVER_TIMESTAMP,
                    "llmProvider": request.llm_provider or settings.llm_provider, # Log provider used
                    # Log overrides if provided
                    "plannerConfigOverride": request.planner_llm_config.model_dump() if request.planner_llm_config else None,
                    "summarizerConfigOverride": request.summarizer_llm_config.model_dump() if request.summarizer_llm_config else None,
                    "writerConfigOverride": request.writer_llm_config.model_dump() if request.writer_llm_config else None,
                    "maxSearchTasksOverride": request.max_search_tasks
                }
                # Filter out None values before setting
                filtered_initial_data = {k: v for k, v in initial_data.items() if v is not None}
                task_doc_ref.set(filtered_initial_data)
                logger.info(f"[WS Task ID: {task_id_str}] Initial Firestore document created.")
            except Exception as e:
                logger.error(f"[WS Task ID: {task_id_str}] Failed to create initial Firestore document: {e}")
                # Disable Firestore for subsequent ops in this request if creation fails
                firestore_available_this_request = False

        # 2. Instantiate the Agent
        await send_status_update("INITIALIZING", "START", "Initializing research agent...")
        logger.info(f"[WS Task ID: {task_id_str}] Instantiating research agent...")
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
            logger.info(f"[WS Task ID: {task_id_str}] Research agent instantiated.")
        except (ConfigurationError, ValueError) as e: # Catch specific init errors
            error_msg = f"Agent Initialization Failed: {e}"
            logger.error(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            await send_status_update("INITIALIZING", "ERROR", f"Agent initialization failed: {e}", {"error": str(e)})
            # Update Firestore status if possible
            if firestore_available_this_request: # Use the local flag
                try: task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Agent Init Fail: {fs_e}")
            await websocket.close(code=1011) # Config error is internal
            return
        except Exception as e: # Catch unexpected init errors
            error_msg = f"Unexpected error during agent initialization: {e}"
            logger.error(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            await send_status_update("INITIALIZING", "ERROR", "Unexpected error during agent setup.", {"error": str(e)})
            # Update Firestore status if possible
            if firestore_available_this_request: # Use the local flag
                try: task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Agent Init Fail: {fs_e}")
            await websocket.close(code=1011)
            return

        # 3. Start the research process
        logger.info(f"[WS Task ID: {task_id_str}] Starting deep research process via WebSocket...")
        # Agent will send STARTING/START itself now
        # await send_status_update("RUNNING", "START", "Starting deep research...")

        # *** Update Firestore Status before run ***
        if firestore_available_this_request: # Use the local flag
            try:
                task_doc_ref.update({"status": "PROCESSING", "updatedAt": firestore.SERVER_TIMESTAMP})
                logger.info(f"[WS Task ID: {task_id_str}] Updated Firestore status to PROCESSING.")
            except Exception as e:
                logger.error(f"[WS Task ID: {task_id_str}] Failed to update Firestore status to PROCESSING: {e}")
                # Continue, but persistence might be inconsistent

        # The agent's run_deep_research handles sending all intermediate updates
        # and the final COMPLETE or ERROR message via the callback.
        # We just need to await its completion and handle potential exceptions it raises.
        research_result: Optional[Dict[str, Any]] = None
        try:
            # *** Create and track the agent task ***
            agent_task = asyncio.create_task(agent.run_deep_research(request.query))
            active_tasks[task_id_str] = agent_task
            logger.info(f"[WS Task ID: {task_id_str}] Agent task created and tracked.")

            # *** Await the task completion ***
            research_result = await agent_task

            # If successful, the agent should have sent the "COMPLETE" message already.
            logger.info(f"[WS Task ID: {task_id_str}] Agent task completed successfully.")
            # Optionally log result summary (careful with large reports)
            if research_result:
                 logger.info(f"[WS Task ID: {task_id_str}] Final report length: {len(research_result.get('final_report',''))}")
                 logger.debug(f"[WS Task ID: {task_id_str}] Usage stats: {research_result.get('usage_statistics')}")

                 # *** Store Successful Result in Firestore ***
                 if firestore_available_this_request: # Use the local flag
                     try:
                         final_report = research_result.get('final_report')
                         usage_statistics = research_result.get('usage_statistics')
                         # Ensure sources are JSON serializable (should be if HttpUrl was handled)
                         sources = research_result.get('final_context')

                         update_payload = {
                             "status": "COMPLETE",
                             "result": {
                                 "finalReport": final_report,
                                 "sources": sources,
                                 "usageStatistics": usage_statistics
                             },
                             "updatedAt": firestore.SERVER_TIMESTAMP
                         }
                         task_doc_ref.update(update_payload)
                         logger.info(f"[WS Task ID: {task_id_str}] Stored successful result in Firestore.")
                     except Exception as e:
                          logger.error(f"[WS Task ID: {task_id_str}] Failed to store successful result in Firestore: {e}")

        # --- Cancellation Handling --- #
        except asyncio.CancelledError:
            connection_active = False # Stop trying to send WS messages
            logger.info(f"[WS Task ID: {task_id_str}] Agent task was cancelled.")
            # Update Firestore status if possible (might already be set by stop endpoint)
            if firestore_available_this_request:
                 try:
                     # Check current status before overwriting
                     doc_snapshot_cancel = task_doc_ref.get()
                     if doc_snapshot_cancel.exists and doc_snapshot_cancel.to_dict().get("status") != "CANCELLED":
                         task_doc_ref.update({"status": "CANCELLED", "stoppedReason": "Cancelled during execution.", "updatedAt": firestore.SERVER_TIMESTAMP})
                         logger.info(f"[WS Task ID: {task_id_str}] Updated Firestore status to CANCELLED on task cancellation.")
                     else:
                          logger.info(f"[WS Task ID: {task_id_str}] Firestore status already CANCELLED or doc missing on task cancellation.")
                 except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on CancelledError: {fs_e}")
            # No need to close websocket here, finally block handles it.
            # Do not re-raise CancelledError, let it exit the try block cleanly.

        except AgentExecutionError as e: # Catch agent execution errors specifically
            connection_active = False
            error_id_agent = uuid.uuid4()
            error_msg = f"Agent Execution Error (ID: {error_id_agent}): {e}"
            logger.error(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            # Agent might have already sent an error via callback, but send a final one from here too
            await send_status_update("ERROR", "ERROR", f"Research process failed during execution. Error ID: {error_id_agent}", {"error_type": type(e).__name__, "error_id": str(error_id_agent)})
            # *** Store Error in Firestore ***
            if firestore_available_this_request: # Use the local flag
                 try:
                     task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                     logger.info(f"[WS Task ID: {task_id_str}] Stored agent execution error in Firestore.")
                 except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Agent Exec Fail: {fs_e}")
            # Close with error code 1011 (Internal Server Error)
            # Check state before closing
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return # Exit after handling error

        except DeepResearchError as e: # Catch other custom errors from the agent/subsystems
            error_id_deep = uuid.uuid4()
            error_msg = f"Deep Research Error (ID: {error_id_deep}): {e}"
            logger.error(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            await send_status_update("ERROR", "ERROR", f"An error occurred during research. Error ID: {error_id_deep}", {"error_type": type(e).__name__, "error_id": str(error_id_deep)})
            # *** Store Error in Firestore ***
            if firestore_available_this_request: # Use the local flag
                 try:
                     task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                     logger.info(f"[WS Task ID: {task_id_str}] Stored deep research error in Firestore.")
                 except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on DeepResearchError: {fs_e}")
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return

        # Catch other unexpected errors during agent execution
        except Exception as e:
            connection_active = False
            error_id_unexp = uuid.uuid4()
            error_msg = f"Unexpected critical error during agent execution (Error ID: {error_id_unexp}): {e}"
            logger.critical(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            traceback.print_exc()
            await send_status_update("ERROR", "ERROR", f"An unexpected server error occurred. Error ID: {error_id_unexp}", {"error_type": type(e).__name__, "error_id": str(error_id_unexp)})
            # *** Store Error in Firestore ***
            if firestore_available_this_request: # Use the local flag
                 try:
                     task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                     logger.info(f"[WS Task ID: {task_id_str}] Stored unexpected critical error in Firestore.")
                 except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Unexpected Critical: {fs_e}")
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return

        # 4. Wait for client disconnect or close from server side after completion/error
        # FastAPI handles keeping the connection open while the await agent.run...() is running.
        # If the agent completes successfully, it sends the COMPLETE message.
        # We can now close the connection gracefully from the server side.

    except WebSocketDisconnect:
        connection_active = False # Mark as inactive
        logger.info(f"[WS Task ID: {task_id_str}] WebSocket disconnected by client.")
        # If the agent task is still running, attempt to cancel it
        if agent_task and not agent_task.done():
             agent_task.cancel()
             logger.info(f"[WS Task ID: {task_id_str}] Sent cancellation signal to agent task due to client disconnect.")

    except Exception as e: # Catch errors in the WebSocket handler itself (e.g., initial receive_text)
        connection_active = False
        error_id_handler = uuid.uuid4()
        error_msg = f"Unhandled exception in WebSocket handler (Error ID: {error_id_handler}): {e}"
        logger.critical(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
        traceback.print_exc()
        # *** Store Error in Firestore *** (If task_doc_ref exists and FS enabled for request)
        if firestore_available_this_request: # Use the local flag
             try:
                 task_doc_ref.update({"status": "ERROR", "error": error_msg, "updatedAt": firestore.SERVER_TIMESTAMP})
                 logger.info(f"[WS Task ID: {task_id_str}] Stored WS handler error in Firestore.")
             except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on WS Handler Fail: {fs_e}")
        # Try to inform client if possible (might fail if connection already broken)
        try:
            # Avoid calling send_status_update if connection is already known inactive
            # Check websocket state directly for robustness
            if websocket.client_state == WebSocketState.CONNECTED:
                 await websocket.send_json({
                     "step": "ERROR", "status": "ERROR", 
                     "message": f"An unexpected server error occurred in the connection handler. Error ID: {error_id_handler}",
                     "details": {"error_type": type(e).__name__, "error_id": str(error_id_handler)}
                 })
        except Exception:
             logger.error(f"[WS Task ID: {task_id_str}] Failed to send final error message during handler exception.")

        # Ensure connection is closed on server error
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011) # Internal Server Error

    finally:
        # Final check to ensure connection is closed from server-side if it wasn't already.
        connection_active = False # Explicitly mark inactive

        # --- Remove task from tracking --- #
        removed_task = active_tasks.pop(task_id_str, None)
        if removed_task:
             logger.info(f"[WS Task ID: {task_id_str}] Removed task from active tracking.")

        # Update Firestore status to indicate potential interruption if still PROCESSING?
        # This is tricky, as a clean exit (COMPLETE/ERROR) would have already updated.
        # Maybe check status? If status is PROCESSING, update to INTERRUPTED?
        # For simplicity, let's skip this for now. Assume COMPLETE/ERROR covers most cases.

        if websocket.client_state == WebSocketState.CONNECTED:
             try:
                 await websocket.close()
                 logger.info(f"[WS Task ID: {task_id_str}] WebSocket connection explicitly closed by server in finally block.")
             except RuntimeError as e:
                 logger.warning(f"[WS Task ID: {task_id_str}] Error closing WebSocket in finally block (might be closing concurrently): {e}")
        else:
            logger.info(f"[WS Task ID: {task_id_str}] WebSocket connection was already closed before finally block finished.")


# --- New HTTP GET Endpoint for Results ---
@app.get("/research/result/{task_id}")
async def get_research_result(task_id: str = Path(..., title="The ID of the research task to retrieve")):
    """Gets the status and result of a research task by its ID from Firestore."""
    # Check global db initialization status first
    if not db:
        # Log the attempt
        logger.warning(f"Attempted to access /research/result/{task_id} but Firestore is not initialized.")
        raise HTTPException(status_code=503, detail="Result storage is currently unavailable.")
    
    try:
        task_doc_ref = db.collection("research_tasks").document(task_id)
        doc_snapshot = task_doc_ref.get()
    except Exception as e:
        logger.error(f"Firestore error fetching task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving research task details.")

    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="Research task not found.")
    
    # Return the full document data
    # Timestamps will be serialized automatically by FastAPI based on Firestore types
    return doc_snapshot.to_dict()


# --- New HTTP POST Endpoint to Stop/Cancel Task ---
# from fastapi import Body # Import Body if needed for future payload, although not used now

@app.post("/research/stop/{task_id}", status_code=200)
async def stop_research_task(task_id: str = Path(..., title="The ID of the research task to request cancellation for")):
    """
    Requests cancellation of a research task by updating its status in Firestore
    and attempting to cancel the background task.
    Note: Cancellation might not be immediate if the task is in a non-awaiting state.
    """
    logger.info(f"Received stop request for task ID: {task_id}")

    # Check global db initialization status first
    if not db:
        logger.error(f"Stop request failed for task {task_id}: Firestore is not initialized.")
        raise HTTPException(status_code=503, detail="Task persistence service is currently unavailable.")

    try:
        task_doc_ref = db.collection("research_tasks").document(task_id)
        doc_snapshot = task_doc_ref.get()
    except Exception as e:
        logger.error(f"Firestore error fetching task {task_id} for stop request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving task details to process stop request.")

    if not doc_snapshot.exists:
        logger.warning(f"Stop request failed: Task {task_id} not found.")
        raise HTTPException(status_code=404, detail="Research task not found.")

    current_data = doc_snapshot.to_dict()
    current_status = current_data.get("status")

    stoppable_states = ["PENDING", "PROCESSING"]
    if current_status not in stoppable_states:
        logger.warning(f"Stop request ignored: Task {task_id} has status '{current_status}', which cannot be stopped.")
        # Return 200 OK but indicate why it wasn't stopped, or use 400? Let's use 400 for invalid state action.
        raise HTTPException(status_code=400, detail=f"Task is already in status '{current_status}'. Cannot request cancellation.")

    # --- Attempt to cancel the background task --- #
    task_to_cancel = active_tasks.get(task_id)
    cancelled_in_memory = False
    if task_to_cancel:
        if not task_to_cancel.done():
            task_to_cancel.cancel()
            cancelled_in_memory = True
            logger.info(f"Cancellation signal sent to running task {task_id}.")
        else:
            logger.info(f"Task {task_id} found in memory but was already done. Proceeding with Firestore update.")
            # Clean up if somehow missed?
            active_tasks.pop(task_id, None)
    else:
        logger.warning(f"Task {task_id} not found in active tasks memory. It might have already finished, failed, or the server restarted. Proceeding with Firestore update.")

    # Update Firestore status to CANCELLED
    try:
        update_payload = {
            "status": "CANCELLED",
            "stoppedReason": "Cancelled by user request via API.",
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        task_doc_ref.update(update_payload)
        logger.info(f"Successfully updated status to CANCELLED for task {task_id}.")
        return {"message": f"Cancellation requested for task {task_id}. Status updated to CANCELLED."}
    except Exception as e:
        logger.error(f"Firestore error updating task {task_id} status to CANCELLED: {e}", exc_info=True)
        # If update fails, the task state is uncertain from client's perspective
        raise HTTPException(status_code=500, detail="Failed to update task status to cancelled.")


# Optional: Add a root endpoint for health checks or basic info
@app.get("/")
async def root():
    # Basic check to see if settings loaded, otherwise ConfigurationError handled globally
    _ = get_settings()
    return {"message": "Deep Research API is running."}

# Optional: Configure CORS if needed
#app.add_middleware(
#    CORSMiddleware,
#    allow_origins=["http://localhost:3000"], # Allow all origins for now
#    allow_credentials=True,
#    allow_methods=["*"], # Allow all methods
#    allow_headers=["*"], # Allow all headers
#) 