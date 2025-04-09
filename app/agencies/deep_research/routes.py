import asyncio
import json
import logging
import traceback
import uuid
from typing import Dict, Any, Optional

# --- Add Firestore Client type --- #
from google.cloud.firestore_v1.client import Client
from fastapi import APIRouter, Depends, HTTPException, Path, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from firebase_admin import firestore
from pydantic import ValidationError

# --- Core Imports ---
from app.core.config import AppSettings, ApiKeys
# Adapt schema imports - need ResearchRequest from core, ResearchResponse from agency schema
# Import ResearchRequest from the agency schema file where it's defined
from .schemas import ResearchRequest as CoreResearchRequest
from app.agencies.deep_research.schemas import ResearchResponse # Agency for output
from app.core.exceptions import (
    ConfigurationError, DeepResearchError, AgentExecutionError
)

# --- Agency Specific Imports ---
from .orchestrator import run_deep_research_orchestration_wrapper # Import the wrapper
from .agents import (
    AgencyAgents, # Import the container
    create_planner_agent, # Import agent creation functions
    create_summarizer_agent,
    create_writer_agent,
    create_refiner_agent
)
from .config import DeepResearchConfig # Import the agency-specific config
from .callbacks import WebSocketUpdateHandler # Import the handler

# --- Main App Imports (for shared state - temporary) ---
# TODO: Refactor db and active_tasks to a shared module (e.g., app.core.state)
# For now, import directly from main to get things working
# from app.main import get_settings, get_api_keys # No longer import from main
from app.core.state import active_tasks # Import ONLY active_tasks from the new state module
# Import dependencies from the new core dependencies file
from app.core.dependencies import get_settings, get_api_keys, get_firestore_db # Add get_firestore_db

# Configure logger for this module
logger = logging.getLogger(__name__)

# --- Router Definition ---
router = APIRouter()

# --- Dependency for Agency Components ---
# TODO: Determine the best way to manage/instantiate these per request or globally
# For now, create them here. Consider a more robust dependency injection setup.
def get_deep_research_config() -> DeepResearchConfig:
    try:
        # Assuming DeepResearchConfig can be loaded from env/defaults
        return DeepResearchConfig()
    except Exception as e:
        logger.error(f"Failed to load DeepResearchConfig: {e}", exc_info=True)
        raise ConfigurationError(f"Could not load deep research agency configuration: {e}")

def get_agency_agents(
    settings: AppSettings = Depends(get_settings),
    api_keys: ApiKeys = Depends(get_api_keys),
    agency_config: DeepResearchConfig = Depends(get_deep_research_config)
) -> AgencyAgents:
    try:
        # 1. Instantiate individual agents using their creation functions
        planner = create_planner_agent(config=agency_config)
        summarizer = create_summarizer_agent(config=agency_config)
        writer = create_writer_agent(config=agency_config)
        refiner = create_refiner_agent(config=agency_config)

        # 2. Create and return the AgencyAgents container with the instances
        return AgencyAgents(
            planner=planner,
            summarizer=summarizer,
            writer=writer,
            refiner=refiner
        )
    except Exception as e:
        logger.error(f"Failed to initialize AgencyAgents: {e}", exc_info=True)
        raise ConfigurationError(f"Could not initialize agency agents: {e}")


# --- Migrated WebSocket Endpoint ---
@router.websocket("/ws/research")
async def websocket_research_route(
    websocket: WebSocket,
    settings: AppSettings = Depends(get_settings),
    # api_keys: ApiKeys = Depends(get_api_keys), # Needed by get_agency_agents
    agency_config: DeepResearchConfig = Depends(get_deep_research_config),
    agents_collection: AgencyAgents = Depends(get_agency_agents),
    db: Optional[Client] = Depends(get_firestore_db) # Inject Firestore DB
):
    """
    WebSocket endpoint to initiate and monitor a deep research task
    using the multi-agent orchestration flow.
    """
    # --- Remove Debug Log ---
    # logger.info(f"WebSocket handler db object exists: {db is not None}")
    # --- End Debug Log ---

    task_id = uuid.uuid4()
    task_id_str = str(task_id)
    connection_active = True # Flag to manage sending messages only if active
    orchestration_task: Optional[asyncio.Task] = None # Reference to the orchestration task

    await websocket.accept()
    logger.info(f"[WS Task ID: {task_id_str}] Deep Research WebSocket connection accepted.")

    # --- Define the raw WebSocket send function ---
    async def send_status_update(step: str, status: str, message: str, details: dict | None = None):
        """Lowest level function to send JSON payload over the WebSocket."""
        nonlocal connection_active
        if not connection_active:
            logger.warning(f"[WS Task ID: {task_id_str}] Skipping send_status_update; connection inactive.")
            return

        payload = {"step": step, "status": status, "message": message, "details": details or {}}
        try:
            # Serialize details containing Pydantic models if necessary
            # json_payload = json.dumps(payload, default=lambda o: o.model_dump() if hasattr(o, 'model_dump') else str(o))
            # await websocket.send_text(json_payload)
            await websocket.send_json(payload) # FastAPI handles Pydantic serialization
            logger.debug(f"[WS Task ID: {task_id_str}] Sent update: {step}/{status} - {message}")
        except WebSocketDisconnect:
            logger.warning(f"[WS Task ID: {task_id_str}] Client disconnected while trying to send update {step}/{status}.")
            connection_active = False
        except RuntimeError as e:
            logger.warning(f"[WS Task ID: {task_id_str}] Runtime error sending update {step}/{status} (likely closing): {e}")
            connection_active = False
        except Exception as e:
            logger.error(f"[WS Task ID: {task_id_str}] Failed to send WebSocket message for {step}/{status}: {e}", exc_info=True)
            connection_active = False # Assume connection is broken

    # --- Instantiate the Callback Handler --- #
    # Pass the raw send function to the handler
    callback_handler = WebSocketUpdateHandler(send_status_update)

    # --- Firestore Setup --- #
    # DB is now injected via Depends(get_firestore_db)
    task_doc_ref = None
    firestore_available_this_request = False # Flag still useful for logic
    if db is not None: # Check if the dependency returned a client
        try:
            # Collection name should probably be configurable or standardized
            task_doc_ref = db.collection("research_tasks").document(task_id_str)
            firestore_available_this_request = True
            logger.debug(f"[WS Task ID: {task_id_str}] Firestore document reference obtained via dependency.")
        except Exception as e:
            # This block might be less likely if get_firestore_db handles init errors
            logger.error(f"[WS Task ID: {task_id_str}] Failed to get Firestore document reference from injected db: {e}. Firestore may be disabled.")
            task_doc_ref = None
            firestore_available_this_request = False
    else:
        # This means get_firestore_db returned None (initialization failed)
        logger.warning(f"[WS Task ID: {task_id_str}] Firestore client (db) could not be initialized via dependency. Firestore disabled for this request.")

    try:
        # 1. Receive and validate initial research request parameters
        raw_data = await websocket.receive_text()
        request: Optional[CoreResearchRequest] = None # Use core schema for request validation
        try:
            data = json.loads(raw_data)
            # Validate against the CoreResearchRequest which defines the expected input
            request = CoreResearchRequest.model_validate(data)
            logger.info(f"[WS Task ID: {task_id_str}] Received research request for query: '{request.query[:50]}...'")
        except json.JSONDecodeError:
            logger.error(f"[WS Task ID: {task_id_str}] Invalid JSON received.")
            await send_status_update("ERROR", "VALIDATION_ERROR", "Invalid JSON format received.", {"error": "Bad JSON"})
            await websocket.close(code=1003)
            return
        except ValidationError as e:
            logger.error(f"[WS Task ID: {task_id_str}] Invalid request data: {e}", exc_info=False)
            error_details = {"error": "Invalid request structure or content.", "details": str(e)}
            await send_status_update("ERROR", "VALIDATION_ERROR", "Invalid research request parameters.", error_details)
            await websocket.close(code=1003)
            return
        except Exception as e:
            logger.error(f"[WS Task ID: {task_id_str}] Error processing request data: {e}", exc_info=True)
            error_details = {"error": "Failed to process request data.", "details": str(e)}
            await send_status_update("ERROR", "PROCESSING_ERROR", "Error processing request.", error_details)
            await websocket.close(code=1011)
            return

        if not request:
            await send_status_update("ERROR", "INTERNAL_ERROR", "Failed to validate request.", {})
            await websocket.close(code=1011)
            return

        # *** Send Task ID to Client (using handler for consistency) ***
        await callback_handler._send_update("INITIALIZING", "TASK_ID", "Task ID assigned.", {"task_id": task_id_str})

        # *** Create Initial Firestore Document ***
        if firestore_available_this_request and task_doc_ref:
            try:
                # Note: Overrides from CoreResearchRequest might not directly map to AgencyConfig
                # We might need a translation layer or adjust AgencyConfig/Orchestrator initialization
                # For now, log the basic info.
                initial_data = {
                    "taskId": task_id_str,
                    "query": request.query,
                    "status": "PENDING",
                    "createdAt": firestore.SERVER_TIMESTAMP,
                    # Add any relevant config/overrides from 'request' if needed,
                    # ensuring they are Firestore-compatible. Example:
                    # "llmProviderOverride": request.llm_provider,
                    # "raw_overrides": request.model_dump(exclude_unset=True) # Store raw overrides if needed
                }
                filtered_initial_data = {k: v for k, v in initial_data.items() if v is not None}
                task_doc_ref.set(filtered_initial_data)
                logger.info(f"[WS Task ID: {task_id_str}] Initial Firestore document created.")
            except Exception as e:
                logger.error(f"[WS Task ID: {task_id_str}] Failed to create initial Firestore document: {e}")
                # Do not disable Firestore for the rest of the request if initial set fails
                # firestore_available_this_request = False

        # 2. Dependencies are already injected (agents_collection, agency_config, settings)
        # Orchestrator will send STARTING/START via callback
        # await callback_handler._send_update("INITIALIZING", "START", "Research task accepted and preparing...") # Optional: keep if needed before orchestrator start message
        logger.info(f"[WS Task ID: {task_id_str}] Dependencies resolved. Preparing orchestration...")

        # 3. Start the research orchestration process
        logger.info(f"[WS Task ID: {task_id_str}] Starting deep research orchestration...")

        # *** Update Firestore Status before run ***
        if firestore_available_this_request and task_doc_ref:
            try:
                task_doc_ref.update({"status": "PROCESSING", "startedAt": firestore.SERVER_TIMESTAMP})
                logger.info(f"[WS Task ID: {task_id_str}] Updated Firestore status to PROCESSING.")
                # Orchestrator callback will handle sending RUNNING/START or specific phase starts
            except Exception as e:
                logger.error(f"[WS Task ID: {task_id_str}] Failed to update Firestore status to PROCESSING: {e}")
                # Continue, but persistence might be inconsistent

        # The orchestrator runs and returns the final result.
        # We await its completion and handle potential exceptions it raises.
        research_result_response: Optional[ResearchResponse] = None # Use agency schema for result
        try:
            # *** Create and track the orchestration task ***
            # Pass the required arguments to the orchestrator function
            orchestration_task = asyncio.create_task(
                # Use the wrapper function to ensure critical errors are caught and sent via callback
                run_deep_research_orchestration_wrapper(
                    user_query=request.query,
                    agents_collection=agents_collection,
                    config=agency_config,
                    app_settings=settings, # Pass AppSettings from dependencies
                    update_callback=callback_handler, # Pass the handler instance
                    # --- Pass Firestore details --- #
                    task_doc_ref=task_doc_ref,
                    firestore_available=firestore_available_this_request
                )
            )
            # Use the global active_tasks from main.py (temporary)
            active_tasks[task_id_str] = orchestration_task
            logger.info(f"[WS Task ID: {task_id_str}] Orchestration task created and tracked.")

            # *** Await the task completion ***
            # This will block until the orchestrator finishes or raises an exception
            research_result_response = await orchestration_task

            # If successful, the orchestrator returns the ResearchResponse
            logger.info(f"[WS Task ID: {task_id_str}] Orchestration task completed successfully.")

            # *** Add explicit check log ***
            logger.info(f"[WS Task ID: {task_id_str}] Checking Firestore availability flag before final save: {firestore_available_this_request}")

            # *** Update Firestore on Success ***
            if firestore_available_this_request and research_result_response and task_doc_ref:
                try:
                    update_data = {
                        "status": "COMPLETED",
                        "report": research_result_response.report, # Store the final report
                        "usageStatistics": research_result_response.usage_statistics.model_dump(mode='json') if research_result_response.usage_statistics else None, # Store usage stats
                        "completedAt": firestore.SERVER_TIMESTAMP
                    }
                    task_doc_ref.update(update_data)
                    logger.info(f"[WS Task ID: {task_id_str}] Updated Firestore with COMPLETED status, report, and usage stats.")
                except Exception as e:
                    logger.error(f"[WS Task ID: {task_id_str}] Failed to update Firestore with final results: {e}")
                    # Consider how to handle this - maybe send an error message back?
            elif not firestore_available_this_request:
                logger.warning(f"[WS Task ID: {task_id_str}] Firestore not available, cannot save final results.")

        # --- Cancellation Handling --- #
        except asyncio.CancelledError:
            connection_active = False
            logger.warning(f"[WS Task ID: {task_id_str}] Orchestration task was cancelled (likely WebSocket closure).")
            if firestore_available_this_request and task_doc_ref:
                try:
                    doc_snapshot_cancel = task_doc_ref.get()
                    if doc_snapshot_cancel.exists and doc_snapshot_cancel.to_dict().get("status") not in ["CANCELLED", "COMPLETE", "ERROR"]:
                        task_doc_ref.update({"status": "CANCELLED", "stoppedReason": "Cancelled during execution.", "updatedAt": firestore.SERVER_TIMESTAMP})
                        logger.info(f"[WS Task ID: {task_id_str}] Updated Firestore status to CANCELLED on task cancellation.")
                    else:
                         logger.info(f"[WS Task ID: {task_id_str}] Firestore status already final or doc missing on task cancellation.")
                except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on CancelledError: {fs_e}")
            # Do not re-raise, let finally block handle WS close

        # --- Error Handling (Catch errors raised BY the orchestrator) --- #
        # Specific exceptions should ideally be defined in a shared location if raised by orchestrator
        except (AgentExecutionError, DeepResearchError) as e: # Example specific errors
            connection_active = False
            error_id = uuid.uuid4()
            error_msg = f"Orchestration failed (ID: {error_id}): {type(e).__name__} - {e}"
            logger.error(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            # Orchestrator (or wrapper) should send error via callback before raising/returning
            # If we get here, it means the wrapper failed or didn't send.
            # Send a generic fallback error if possible.
            await callback_handler._send_update("ERROR", "ORCHESTRATION_ERROR", f"Orchestration failed unexpectedly. Error ID: {error_id}", {"error_type": type(e).__name__, "error_id": str(error_id)})
            if firestore_available_this_request and task_doc_ref:
                try:
                    task_doc_ref.update({"status": "ERROR", "error": error_msg, "completedAt": firestore.SERVER_TIMESTAMP})
                    logger.info(f"[WS Task ID: {task_id_str}] Stored orchestration error in Firestore.")
                except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Orchestration Fail: {fs_e}")
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return # Exit after handling error

        except Exception as e: # Catch unexpected errors from orchestrator or await
            connection_active = False
            error_id_unexp = uuid.uuid4()
            error_msg = f"Unexpected critical error during orchestration (Error ID: {error_id_unexp}): {e}"
            logger.critical(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            traceback.print_exc()
            # Orchestrator (or wrapper) should send error via callback before raising/returning
            # Send a generic fallback error if possible.
            await callback_handler._send_update("ERROR", "CRITICAL_ERROR", f"Critical orchestration error occurred. Error ID: {error_id_unexp}", {"error_type": type(e).__name__, "error_id": str(error_id_unexp)})
            if firestore_available_this_request and task_doc_ref:
                try:
                    task_doc_ref.update({"status": "ERROR", "error": error_msg, "completedAt": firestore.SERVER_TIMESTAMP})
                    logger.info(f"[WS Task ID: {task_id_str}] Stored unexpected critical error in Firestore.")
                except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on Unexpected Critical: {fs_e}")
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=1011)
            return

    # --- WebSocket Disconnect Handling ---
    except WebSocketDisconnect:
        connection_active = False
        logger.info(f"[WS Task ID: {task_id_str}] WebSocket disconnected by client.")
        logger.info(f"[WS Task ID: {task_id_str}] Orchestration task will continue running in the background.")
        # Firestore status update for cancellation will happen in the CancelledError handler above IF explicitly stopped via API

    # --- General Exception Handling (WebSocket Handler Level) ---
    except Exception as e:
        connection_active = False
        error_id_handler = uuid.uuid4()
        error_msg = f"Unhandled exception in WebSocket handler (Error ID: {error_id_handler}): {e}"
        logger.critical(f"[WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
        traceback.print_exc()
        if firestore_available_this_request and task_doc_ref: # Check task_doc_ref exists
            try:
                task_doc_ref.update({"status": "ERROR", "error": error_msg, "completedAt": firestore.SERVER_TIMESTAMP})
                logger.info(f"[WS Task ID: {task_id_str}] Stored WS handler error in Firestore.")
            except Exception as fs_e: logger.error(f"[WS Task ID: {task_id_str}] FS Update Error on WS Handler Fail: {fs_e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "step": "ERROR", "status": "HANDLER_ERROR",
                    "message": f"Internal server error handling connection. Error ID: {error_id_handler}",
                    "details": {"error_type": type(e).__name__, "error_id": str(error_id_handler)}
                })
        except Exception:
            logger.error(f"[WS Task ID: {task_id_str}] Failed to send final error message during handler exception.")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011)

    # --- Finally Block ---
    finally:
        connection_active = False
        # Use the global active_tasks from main.py (temporary)
        removed_task = active_tasks.pop(task_id_str, None)
        if removed_task:
            logger.info(f"[WS Task ID: {task_id_str}] Removed task from active tracking.")
        else:
             logger.debug(f"[WS Task ID: {task_id_str}] Task already removed or not found in tracking upon final exit.")

        # Check status before closing - avoid closing if already closed
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
                logger.info(f"[WS Task ID: {task_id_str}] WebSocket connection explicitly closed by server in finally block.")
            except RuntimeError as e:
                 # This can happen if the client disconnects concurrently while the server is closing
                 if "WebSocket is not connected" in str(e):
                     logger.info(f"[WS Task ID: {task_id_str}] WebSocket already closed when finally block tried to close it.")
                 else:
                      logger.warning(f"[WS Task ID: {task_id_str}] Error closing WebSocket in finally block: {e}")
        else:
            logger.info(f"[WS Task ID: {task_id_str}] WebSocket connection was already closed before finally block reached close attempt.")


# --- Migrated GET Endpoint for Results ---
@router.get("/result/{task_id}")
async def get_research_result_route(
    task_id: str = Path(..., title="The ID of the research task to retrieve"),
    db: Optional[Client] = Depends(get_firestore_db) # Inject Firestore DB
):
    """Gets the status and result of a research task by its ID from Firestore."""
    # Use the injected db dependency
    if not db:
        # This case should now be handled by the dependency raising HTTPException 503
        # If we reach here with db=None, it means the dependency allowed it (which it currently doesn't)
        logger.error(f"Accessing /result/{task_id} but Firestore DB dependency returned None unexpectedly.")
        raise HTTPException(status_code=503, detail="Result storage is unavailable (DB dependency failed).")

    try:
        # Collection name should match the one used in the WebSocket endpoint
        task_doc_ref = db.collection("research_tasks").document(task_id)
        doc_snapshot = task_doc_ref.get()
    except Exception as e:
        logger.error(f"Firestore error fetching task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving research task details.")

    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="Research task not found.")

    # Return the full document data
    # Timestamps will be serialized automatically by FastAPI
    return doc_snapshot.to_dict()


# --- Migrated POST Endpoint to Stop/Cancel Task ---
@router.post("/stop/{task_id}", status_code=200)
async def stop_research_task_route(
    task_id: str = Path(..., title="The ID of the research task to request cancellation for"),
    db: Optional[Client] = Depends(get_firestore_db) # Inject Firestore DB
):
    """
    Requests cancellation of a research task by updating its status in Firestore
    and attempting to cancel the background task.
    Note: Cancellation might not be immediate.
    """
    logger.info(f"Received stop request for task ID: {task_id}")

    # Use injected db dependency
    if not db:
        # Handled by dependency
        logger.error(f"Stop request failed for task {task_id}: Firestore DB dependency returned None.")
        raise HTTPException(status_code=503, detail="Task persistence service is currently unavailable (DB dependency failed).")

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

    # Stoppable states can be PENDING or PROCESSING
    stoppable_states = ["PENDING", "PROCESSING"]
    if current_status not in stoppable_states:
        logger.warning(f"Stop request ignored: Task {task_id} has status '{current_status}', which cannot be stopped.")
        raise HTTPException(status_code=400, detail=f"Task is already in status '{current_status}'. Cannot request cancellation.")

    # --- Attempt to cancel the background task --- #
    # Use the global active_tasks dictionary
    task_to_cancel = active_tasks.get(task_id)
    cancelled_in_memory = False
    if task_to_cancel:
        if not task_to_cancel.done():
            task_to_cancel.cancel()
            cancelled_in_memory = True
            logger.info(f"Cancellation signal sent to running task {task_id}.")
            # The actual Firestore update upon cancellation is handled within the CancelledError
            # exception handler in the websocket_research_route
        else:
            logger.info(f"Task {task_id} found in memory but was already done. Proceeding with Firestore update only.")
            # Clean up just in case it wasn't removed by the finally block
            active_tasks.pop(task_id, None)
    else:
        logger.warning(f"Task {task_id} not found in active tasks memory. It might have already finished, failed, or the server restarted. Proceeding with Firestore update.")

    # Update Firestore status to CANCELLED regardless of task cancellation signal success
    try:
        update_payload = {
            "status": "CANCELLED",
            "stoppedReason": "Cancelled by user request via API.",
            "updatedAt": firestore.SERVER_TIMESTAMP # Use server timestamp
        }
        task_doc_ref.update(update_payload)
        logger.info(f"Successfully updated status to CANCELLED for task {task_id} in Firestore.")
        return {"message": f"Cancellation requested for task {task_id}. Status updated to CANCELLED."}
    except Exception as e:
        logger.error(f"Firestore error updating task {task_id} status to CANCELLED: {e}", exc_info=True)
        # Even if cancellation signal was sent, the final status update failed
        raise HTTPException(status_code=500, detail="Failed to update task status to cancelled in storage.")
