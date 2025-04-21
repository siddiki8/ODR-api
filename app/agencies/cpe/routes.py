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

# --- Core Imports --- #
from app.core.config import AppSettings, ApiKeys
from app.core.exceptions import ConfigurationError, DeepResearchError # Correct base exception
# --- CPE Specific Imports --- #
from .schemas import CPERequest, CPEResponse # Use CPE schemas
from .orchestrator import run_cpe_wrapper # Import the CPE wrapper
from .agents import AgencyAgents, get_cpe_agents # Import CPE agent getter
from .config import CPEConfig # Import CPE config
from .callbacks import CpeWebSocketUpdateHandler # Import CPE handler

# --- Shared State & Dependencies --- #
from app.core.state import active_tasks # Import active_tasks
from app.core.dependencies import get_settings, get_api_keys, get_firestore_db # Core dependencies (api_keys is used indirectly)

# Configure logger
logger = logging.getLogger(__name__)

# --- Router Definition --- #
router = APIRouter()

# --- Dependency for CPE Config --- #
def get_cpe_config() -> CPEConfig:
    try:
        return CPEConfig()
    except Exception as e:
        logger.error(f"Failed to load CPEConfig: {e}", exc_info=True)
        raise ConfigurationError(f"Could not load CPE agency configuration: {e}")

# Dependency for CPE Agents is already defined in .agents module
# We'll use get_cpe_agents directly in the route dependency list.

# --- WebSocket Endpoint --- #
@router.websocket("/ws/cpe")
async def websocket_cpe_route(
    websocket: WebSocket,
    settings: AppSettings = Depends(get_settings),
    # api_keys: ApiKeys = Depends(get_api_keys),  # Not required explicitly; get_cpe_agents uses OPENROUTER_API_KEY
    agency_config: CPEConfig = Depends(get_cpe_config),
    agents_collection: AgencyAgents = Depends(get_cpe_agents),
    db: Optional[Client] = Depends(get_firestore_db)
):
    """
    WebSocket endpoint for the Company Profile Extractor (CPE) workflow.
    """
    task_id = uuid.uuid4()
    task_id_str = str(task_id)
    connection_active = True
    orchestration_task: Optional[asyncio.Task] = None

    await websocket.accept()
    logger.info(f"[CPE WS Task ID: {task_id_str}] CPE WebSocket connection accepted.")

    # --- Raw WebSocket Send Function --- #
    async def send_status_update(step: str, status: str, message: str, details: dict | None = None):
        nonlocal connection_active
        if not connection_active:
            logger.warning(f"[CPE WS Task ID: {task_id_str}] Skipping send_status_update; connection inactive.")
            return
        payload = {"step": step, "status": status, "message": message, "details": details or {}}
        try:
            await websocket.send_json(payload)
            logger.debug(f"[CPE WS Task ID: {task_id_str}] Sent update: {step}/{status} - {message}")
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.warning(f"[CPE WS Task ID: {task_id_str}] Client disconnected or runtime error during send for {step}/{status}: {e}")
            connection_active = False
        except Exception as e:
            logger.error(f"[CPE WS Task ID: {task_id_str}] Failed to send WebSocket message for {step}/{status}: {e}", exc_info=True)
            connection_active = False

    # --- Instantiate Callback Handler --- #
    callback_handler = CpeWebSocketUpdateHandler(send_status_update)

    # --- Firestore Setup --- #
    task_doc_ref = None
    firestore_available_this_request = False
    if db is not None:
        try:
            task_doc_ref = db.collection("cpe_tasks").document(task_id_str) # Use specific collection
            firestore_available_this_request = True
            logger.debug(f"[CPE WS Task ID: {task_id_str}] Firestore doc reference obtained.")
        except Exception as e:
            logger.error(f"[CPE WS Task ID: {task_id_str}] Failed to get Firestore doc reference: {e}.")
            task_doc_ref = None
    else:
        logger.warning(f"[CPE WS Task ID: {task_id_str}] Firestore client unavailable.")

    try:
        # 1. Receive and Validate Request
        raw_data = await websocket.receive_text()
        request: Optional[CPERequest] = None
        try:
            data = json.loads(raw_data)
            request = CPERequest.model_validate(data)
            logger.info(f"[CPE WS Task ID: {task_id_str}] Received CPE request for query: '{request.query[:50]}...'")
        except (json.JSONDecodeError, ValidationError) as e:
            error_type = "VALIDATION_ERROR" if isinstance(e, ValidationError) else "JSON_ERROR"
            error_msg = f"Invalid request data: {e}"
            logger.error(f"[CPE WS Task ID: {task_id_str}] {error_msg}", exc_info=False)
            await send_status_update("ERROR", error_type, "Invalid request.", {"error": str(e)})
            await websocket.close(code=1003)
            return
        except Exception as e:
            logger.error(f"[CPE WS Task ID: {task_id_str}] Error processing request data: {e}", exc_info=True)
            await send_status_update("ERROR", "PROCESSING_ERROR", "Error processing request.", {"error": str(e)})
            await websocket.close(code=1011)
            return

        if not request:
            await send_status_update("ERROR", "INTERNAL_ERROR", "Failed validation.", {})
            await websocket.close(code=1011)
            return

        # *** Send Task ID ***
        await callback_handler._send_update("INITIALIZING", "TASK_ID", "Task ID assigned.", {"task_id": task_id_str})

        # *** Create Initial Firestore Document ***
        if firestore_available_this_request and task_doc_ref:
            try:
                initial_data = {
                    "taskId": task_id_str,
                    "query": request.query,
                    "location": request.location,
                    "maxSearchTasks": request.max_search_tasks,
                    "status": "PENDING",
                    "createdAt": firestore.SERVER_TIMESTAMP,
                }
                task_doc_ref.set({k: v for k, v in initial_data.items() if v is not None})
                logger.info(f"[CPE WS Task ID: {task_id_str}] Initial Firestore document created.")
            except Exception as e:
                logger.error(f"[CPE WS Task ID: {task_id_str}] Failed to create initial Firestore doc: {e}")

        # 2. Dependencies Resolved (agents, config, settings)
        logger.info(f"[CPE WS Task ID: {task_id_str}] Dependencies resolved. Preparing orchestration...")
        
        # 3. Start Orchestration
        logger.info(f"[CPE WS Task ID: {task_id_str}] Starting CPE orchestration...")

        # *** Update Firestore Status ***
        if firestore_available_this_request and task_doc_ref:
            try:
                task_doc_ref.update({"status": "PROCESSING", "startedAt": firestore.SERVER_TIMESTAMP})
                logger.info(f"[CPE WS Task ID: {task_id_str}] Updated Firestore status to PROCESSING.")
            except Exception as e:
                logger.error(f"[CPE WS Task ID: {task_id_str}] Failed to update Firestore status to PROCESSING: {e}")

        cpe_result_response: Optional[CPEResponse] = None
        try:
            # *** Create and track the task ***
            orchestration_task = asyncio.create_task(
                run_cpe_wrapper(
                    request=request,
                    agents_collection=agents_collection,
                    config=agency_config,
                    app_settings=settings,
                    update_callback=callback_handler,
                    task_doc_ref=task_doc_ref,
                    firestore_available=firestore_available_this_request
                )
            )
            active_tasks[task_id_str] = orchestration_task
            logger.info(f"[CPE WS Task ID: {task_id_str}] CPE Orchestration task created and tracked.")

            # *** Await completion ***
            cpe_result_response = await orchestration_task
            logger.info(f"[CPE WS Task ID: {task_id_str}] CPE Orchestration task completed successfully.")

            # *** Update Firestore on Success ***
            # Note: Final stats are already saved by the wrapper/orchestrator
            if firestore_available_this_request and task_doc_ref:
                try:
                    # Only update final status and timestamp here
                    final_success_update = {
                        "status": "COMPLETED", # Final success status
                        "completedAt": firestore.SERVER_TIMESTAMP 
                    }
                    task_doc_ref.update(final_success_update)
                    logger.info(f"[CPE WS Task ID: {task_id_str}] Updated Firestore with final COMPLETED status.")
                except Exception as e:
                    logger.error(f"[CPE WS Task ID: {task_id_str}] Failed to update Firestore with final COMPLETED status: {e}")
            
            # *** Send Final WS Completion Message ***
            if cpe_result_response and cpe_result_response.usage_statistics:
                 await callback_handler.orchestration_complete(
                      profile_count=len(cpe_result_response.profiles),
                      usage_stats=cpe_result_response.usage_statistics.model_dump(mode='json')
                 )
            else:
                # Handle case where response might be minimal on error inside wrapper
                await callback_handler.orchestration_complete(profile_count=0, usage_stats={})

        # --- Cancellation Handling --- #
        except asyncio.CancelledError:
            connection_active = False
            logger.warning(f"[CPE WS Task ID: {task_id_str}] CPE Orchestration task cancelled.")
            if firestore_available_this_request and task_doc_ref:
                try:
                    doc_snapshot_cancel = task_doc_ref.get()
                    if doc_snapshot_cancel.exists and doc_snapshot_cancel.to_dict().get("status") not in ["CANCELLED", "COMPLETED", "ERROR"]:
                        task_doc_ref.update({"status": "CANCELLED", "stoppedReason": "Cancelled during execution.", "updatedAt": firestore.SERVER_TIMESTAMP})
                        logger.info(f"[CPE WS Task ID: {task_id_str}] Updated Firestore status to CANCELLED.")
                except Exception as fs_e: logger.error(f"[CPE WS Task ID: {task_id_str}] FS Update Error on CancelledError: {fs_e}")

        # --- Error Handling (from Orchestrator/Wrapper) --- #
        # BaseAgencyError catches Config/Agent/etc errors
        except DeepResearchError as e: # Catch correct base exception
            connection_active = False
            error_id = uuid.uuid4()
            error_msg = f"CPE Orchestration failed (ID: {error_id}): {type(e).__name__} - {e}"
            logger.error(f"[CPE WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            # Wrapper should have sent orchestration_error callback
            # Firestore status should be updated by wrapper
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return

        except Exception as e: # Catch unexpected errors
            connection_active = False
            error_id_unexp = uuid.uuid4()
            error_msg = f"Unexpected critical error during CPE orchestration (Error ID: {error_id_unexp}): {e}"
            logger.critical(f"[CPE WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
            traceback.print_exc()
            # Wrapper should have sent orchestration_error callback & updated Firestore
            # Send a fallback WS message if connection is still alive
            await callback_handler._send_update("ERROR", "CRITICAL_ERROR", f"Critical error. ID: {error_id_unexp}", {"error_type": type(e).__name__, "error_id": str(error_id_unexp)})
            if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)
            return

    # --- WebSocket Disconnect Handling --- #
    except WebSocketDisconnect:
        connection_active = False
        logger.info(f"[CPE WS Task ID: {task_id_str}] WebSocket disconnected by client.")
        # Orchestration continues; cancellation handled via API or task end

    # --- General Exception Handling (Handler Level) --- #
    except Exception as e:
        connection_active = False
        error_id_handler = uuid.uuid4()
        error_msg = f"Unhandled exception in CPE WebSocket handler (Error ID: {error_id_handler}): {e}"
        logger.critical(f"[CPE WS Task ID: {task_id_str}] {error_msg}", exc_info=True)
        traceback.print_exc()
        if firestore_available_this_request and task_doc_ref:
            try: task_doc_ref.update({"status": "ERROR", "error": error_msg, "completedAt": firestore.SERVER_TIMESTAMP})
            except Exception as fs_e: logger.error(f"[CPE WS Task ID: {task_id_str}] FS Update Error on WS Handler Fail: {fs_e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({"step": "ERROR", "status": "HANDLER_ERROR", "message": f"Internal error. ID: {error_id_handler}", "details": {"error_type": type(e).__name__, "error_id": str(error_id_handler)}})
        except Exception: pass
        if websocket.client_state == WebSocketState.CONNECTED: await websocket.close(code=1011)

    # --- Finally Block --- #
    finally:
        connection_active = False
        removed_task = active_tasks.pop(task_id_str, None)
        if removed_task: logger.info(f"[CPE WS Task ID: {task_id_str}] Removed task from active tracking.")
        if websocket.client_state == WebSocketState.CONNECTED:
            try: 
                await websocket.close()
                logger.info(f"[CPE WS Task ID: {task_id_str}] WebSocket closed by server.")
            except RuntimeError as e:
                 if "WebSocket is not connected" in str(e): logger.info(f"[CPE WS Task ID: {task_id_str}] WS already closed.")
                 else: logger.warning(f"[CPE WS Task ID: {task_id_str}] Error closing WS: {e}")
        else:
            logger.info(f"[CPE WS Task ID: {task_id_str}] WS already closed before finally.")


# --- GET Endpoint for Results --- #
@router.get("/cpe/result/{task_id}")
async def get_cpe_result_route(
    task_id: str = Path(..., title="The ID of the CPE task to retrieve"),
    db: Optional[Client] = Depends(get_firestore_db)
):
    """Gets the status and result of a CPE task by its ID from Firestore."""
    if not db: 
        raise HTTPException(status_code=503, detail="Result storage unavailable.")
    try:
        task_doc_ref = db.collection("cpe_tasks").document(task_id)
        doc_snapshot = task_doc_ref.get()
    except Exception as e:
        logger.error(f"Firestore error fetching CPE task {task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving CPE task details.")
    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="CPE task not found.")
    return doc_snapshot.to_dict()


# --- POST Endpoint to Stop/Cancel Task --- #
@router.post("/cpe/stop/{task_id}", status_code=200)
async def stop_cpe_task_route(
    task_id: str = Path(..., title="The ID of the CPE task to request cancellation for"),
    db: Optional[Client] = Depends(get_firestore_db)
):
    """Requests cancellation of a CPE task."""
    logger.info(f"Received stop request for CPE task ID: {task_id}")
    if not db:
        raise HTTPException(status_code=503, detail="Task persistence service unavailable.")
    try:
        task_doc_ref = db.collection("cpe_tasks").document(task_id)
        doc_snapshot = task_doc_ref.get()
    except Exception as e:
        logger.error(f"Firestore error fetching CPE task {task_id} for stop: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error retrieving task details for stop request.")
    if not doc_snapshot.exists:
        raise HTTPException(status_code=404, detail="CPE task not found.")

    current_data = doc_snapshot.to_dict()
    current_status = current_data.get("status")
    stoppable_states = ["PENDING", "PROCESSING", "PLANNING_COMPLETE", "SEARCH_COMPLETE", "EXTRACTING"]
    if current_status not in stoppable_states:
        raise HTTPException(status_code=400, detail=f"Task status '{current_status}' cannot be stopped.")

    # --- Attempt to cancel background task --- #
    task_to_cancel = active_tasks.get(task_id)
    cancelled_in_memory = False
    if task_to_cancel:
        if not task_to_cancel.done():
            task_to_cancel.cancel()
            cancelled_in_memory = True
            logger.info(f"Cancellation signal sent to running CPE task {task_id}.")
        else:
            logger.info(f"CPE task {task_id} already done. Updating Firestore only.")
            active_tasks.pop(task_id, None)
    else:
        logger.warning(f"CPE task {task_id} not in active memory. Updating Firestore only.")

    # Update Firestore status
    try:
        update_payload = {
            "status": "CANCELLED",
            "stoppedReason": "Cancelled by user request via API.",
            "updatedAt": firestore.SERVER_TIMESTAMP
        }
        task_doc_ref.update(update_payload)
        logger.info(f"Updated status to CANCELLED for CPE task {task_id} in Firestore.")
        return {"message": f"Cancellation requested for CPE task {task_id}. Status updated."}
    except Exception as e:
        logger.error(f"Firestore error updating CPE task {task_id} status to CANCELLED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update task status in storage.") 