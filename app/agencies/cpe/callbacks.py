import logging
import asyncio
from typing import Optional, Callable, Coroutine, Any, Dict, List

# Import relevant schemas if needed for type hinting details
from .schemas import UsageStatistics

logger = logging.getLogger(__name__)

class CpeWebSocketUpdateHandler:
    """Handles formatting and sending status updates via a WebSocket callback for CPE."""

    def __init__(self, websocket_callback: Optional[Callable[..., Coroutine]]):
        """
        Initializes the handler with the actual async function used to send messages.

        Args:
            websocket_callback: The async function (e.g., from routes.py) that sends a JSON payload.
                                Should accept step, status, message, details arguments.
        """
        self._callback = websocket_callback

    async def _send_update(self, step: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Internal helper to safely call the stored callback function."""
        if self._callback:
            try:
                details_payload = details if details is not None else {}
                await self._callback(step, status, message, details_payload)
                logger.debug(f"CPE WS Update Sent: {step}/{status} - {message}")
            except Exception as e:
                logger.error(f"CpeWebSocketUpdateHandler failed to send update ({step}/{status}): {e}", exc_info=False)
        else:
             logger.debug(f"CPE WS Update Skipped (no callback): {step}/{status} - {message}")

    # --- Orchestration Lifecycle Updates ---
    async def orchestration_start(self):
        await self._send_update("STARTING", "START", "CPE process initiated.")

    async def orchestration_complete(self, profile_count: int, usage_stats: Dict): # Changed detail name
        await self._send_update(
            "COMPLETE",
            "END",
            "CPE process completed successfully.",
            {"profiles_extracted": profile_count, "usage": usage_stats} # Changed detail key
        )

    async def orchestration_error(self, error: Exception):
        await self._send_update(
            "ERROR",
            "FATAL",
            f"CPE orchestration failed critically: {error}",
            {"error": str(error), "error_type": type(error).__name__}
        )

    # --- Planning Phase Updates ---
    async def planning_start(self):
        await self._send_update("PLANNING", "START", "Generating search plan...")

    async def planning_end(self, search_task_count: int): # Simplified details
        await self._send_update(
            "PLANNING", 
            "END", 
            f"Search plan generated with {search_task_count} tasks.", 
            {"search_task_count": search_task_count}
            )

    async def planning_error(self, error: Exception):
        await self._send_update(
            "PLANNING",
            "ERROR",
            f"Failed to generate search plan: {error}",
            {"error": str(error)}
        )

    # --- Search Phase Updates ---
    async def search_start(self, task_count: int):
        await self._send_update(
            "SEARCHING",
            "START",
            f"Performing search using {task_count} tasks...",
            {"task_count": task_count}
        )

    async def search_end(self, unique_result_count: int):
        await self._send_update(
            "SEARCHING",
            "END",
            f"Search complete. Found {unique_result_count} unique URLs.",
            {"unique_result_count": unique_result_count}
        )

    async def search_error(self, error: Exception):
        await self._send_update(
            "SEARCHING",
            "ERROR",
            f"Search failed: {error}",
            {"error": str(error)}
        )
        
    # --- Extraction Phase Updates (Covers Crawling, Aggregating, LLM Extract) ---
    async def extraction_start(self, url_count: int):
        await self._send_update(
            "EXTRACTING",
            "START",
            f"Starting extraction process for {url_count} unique URLs...",
            {"url_count": url_count}
        )
        
    async def extraction_domain_start(self, domain: str, start_url: str):
         await self._send_update(
            "EXTRACTING",
            "IN_PROGRESS",
            f"Processing domain: {domain}...",
            {"domain": domain, "start_url": start_url, "action": "Starting Domain"}
        )
        
    async def extraction_crawling(self, domain: str):
        await self._send_update(
            "EXTRACTING",
            "IN_PROGRESS",
            f"Crawling for emails on {domain}...",
            {"domain": domain, "action": "Crawling"}
        )

    async def extraction_aggregating(self, domain: str, page_count: int):
        await self._send_update(
            "EXTRACTING",
            "IN_PROGRESS",
            f"Aggregating HTML from {page_count} pages for {domain}...",
            {"domain": domain, "action": "Aggregating", "page_count": page_count}
        )

    async def extraction_calling_llm(self, domain: str):
        await self._send_update(
            "EXTRACTING",
            "IN_PROGRESS",
            f"Calling extractor LLM for {domain}...",
            {"domain": domain, "action": "LLM Extraction"}
        )
        
    async def extraction_profile_success(self, domain: str):
        await self._send_update(
            "EXTRACTING",
            "SUCCESS",
            f"Successfully extracted profile for domain: {domain}.",
            {"domain": domain}
        )

    async def extraction_domain_skipped(self, domain: str, reason: str):
         await self._send_update(
            "EXTRACTING",
            "INFO", # Use INFO for skips
            f"Skipped processing domain {domain}. Reason: {reason}",
            {"domain": domain, "reason": reason}
        )

    async def extraction_domain_error(self, domain: str, error: Exception):
        await self._send_update(
            "EXTRACTING",
            "WARNING", # Use WARNING for non-fatal domain errors
            f"Error processing domain {domain}: {error}",
            {"domain": domain, "error": str(error)}
        )
        
    async def extraction_end(self, processed_domain_count: int, extracted_profile_count: int):
        await self._send_update(
            "EXTRACTING",
            "END",
            f"Extraction phase complete. Processed {processed_domain_count} domains, extracted {extracted_profile_count} profiles.",
            {"processed_domain_count": processed_domain_count, "extracted_profile_count": extracted_profile_count}
        )

    # --- Finalizing Phase Updates (Minimal for CPE) ---
    async def finalizing_start(self):
        await self._send_update("FINALIZING", "START", "Finalizing CPE results...")

    async def finalizing_end(self):
        await self._send_update(
            "FINALIZING",
            "END",
            "CPE results finalized."
        ) 