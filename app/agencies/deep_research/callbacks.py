import logging
import asyncio
from typing import Optional, Callable, Coroutine, Any, Dict, List

# Import relevant schemas if needed for type hinting details
# from . import schemas
# from app.core.schemas import UsageStatistics

logger = logging.getLogger(__name__)

class WebSocketUpdateHandler:
    """Handles formatting and sending status updates via a WebSocket callback."""

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
                # Ensure details is a dict or None before sending
                details_payload = details if details is not None else {}
                await self._callback(step, status, message, details_payload)
                logger.debug(f"WS Update Sent: {step}/{status} - {message}")
            except Exception as e:
                # Log error but don't crash the orchestrator if WS send fails
                logger.error(f"WebSocketUpdateHandler failed to send update ({step}/{status}): {e}", exc_info=False)
        else:
             logger.debug(f"WS Update Skipped (no callback): {step}/{status} - {message}")

    # --- Orchestration Lifecycle Updates ---
    async def orchestration_start(self):
        await self._send_update("STARTING", "START", "Research process initiated.")

    async def orchestration_complete(self, final_report_length: int, usage_stats: Dict):
        await self._send_update(
            "COMPLETE",
            "END",
            "Research process completed successfully.",
            {"final_report_length": final_report_length, "usage": usage_stats}
        )

    async def orchestration_error(self, error: Exception):
        await self._send_update(
            "ERROR",
            "FATAL",
            f"Orchestration failed critically: {error}",
            {"error": str(error), "error_type": type(error).__name__}
        )

    # --- Planning Phase Updates ---
    async def planning_start(self):
        await self._send_update("PLANNING", "START", "Generating initial research plan...")

    async def planning_end(self, plan_details: Dict):
        await self._send_update("PLANNING", "END", "Research plan generated.", plan_details)

    async def planning_error(self, error: Exception):
        await self._send_update(
            "PLANNING",
            "ERROR",
            f"Failed to generate research plan: {error}",
            {"error": str(error)}
        )

    # --- Initial Search Phase Updates ---
    async def initial_search_start(self, initial_task_count: int):
        await self._send_update(
            "SEARCHING",
            "START",
            f"Performing initial search based on {initial_task_count} tasks...",
            {"initial_task_count": initial_task_count}
        )

    async def initial_search_end(self, unique_result_count: int, queries_executed: int):
        await self._send_update(
            "SEARCHING",
            "END",
            f"Initial search complete. Found {unique_result_count} unique results.",
            {"unique_result_count": unique_result_count, "queries_executed": queries_executed}
        )

    async def initial_search_error(self, error: Exception):
        await self._send_update(
            "SEARCHING",
            "ERROR",
            f"Initial search failed: {error}",
            {"error": str(error)}
        )

    # --- Initial Reranking Phase Updates ---
    async def initial_rerank_start(self, results_to_rank: int):
        await self._send_update(
            "RANKING",
            "START",
            f"Reranking {results_to_rank} initial search results...",
            {"results_to_rank": results_to_rank}
         )

    async def initial_rerank_end(self, results_for_summary: int, results_for_chunking: int):
        await self._send_update(
            "RANKING",
            "END",
            f"Initial results reranked. Split: {results_for_summary} for summary, {results_for_chunking} for chunking.",
            {"results_for_summary": results_for_summary, "results_for_chunking": results_for_chunking}
        )

    async def initial_rerank_error(self, error: Exception):
        await self._send_update(
            "RANKING",
            "ERROR",
            f"Initial reranking failed: {error}",
            {"error": str(error)}
        )

    # --- Source Processing Phase Updates (Overall) ---
    async def processing_start(self, urls_to_process: int):
        await self._send_update(
            "PROCESSING",
            "START",
            f"Starting source processing for {urls_to_process} URLs (summaries & chunks)...",
            {"urls_to_process": urls_to_process}
        )

    async def processing_end(self, processed_source_count: int, total_context_items: int):
        await self._send_update(
            "PROCESSING",
            "END",
            f"Finished processing {processed_source_count} sources. Generated {total_context_items} context items.",
            {"processed_source_count": processed_source_count, "total_context_items": total_context_items}
        )

    # --- Source Processing Phase Updates (Per Source) ---
    async def processing_source_fetching(self, source_url: str, is_refinement: bool = False):
        prefix = "[Refinement] " if is_refinement else ""
        await self._send_update(
            "PROCESSING",
            "IN_PROGRESS",
            f"{prefix}Fetching content...",
            {"source_url": source_url, "action": "Fetching"}
        )

    async def processing_source_summarizing(self, source_url: str):
        await self._send_update(
            "PROCESSING",
            "IN_PROGRESS",
            "Summarizing content...",
            {"source_url": source_url, "action": "Summarizing"}
        )

    async def processing_source_chunking(self, source_url: str, is_refinement: bool = False):
        prefix = "[Refinement] " if is_refinement else ""
        await self._send_update(
            "PROCESSING",
            "IN_PROGRESS",
            f"{prefix}Chunking content...",
            {"source_url": source_url, "action": "Chunking"}
        )

    async def processing_source_reranking_chunks(self, source_url: str, chunk_count: int, is_refinement: bool = False):
        prefix = "[Refinement] " if is_refinement else ""
        await self._send_update(
            "PROCESSING",
            "IN_PROGRESS",
            f"{prefix}Reranking {chunk_count} chunks...",
            {"source_url": source_url, "action": "Reranking", "chunk_count": chunk_count}
        )

    async def processing_source_summary_success(self, source_url: str):
        await self._send_update(
            "PROCESSING",
            "SUCCESS",
            "Successfully summarized source.",
            {"source_url": source_url, "type": "summary"}
        )

    async def processing_source_chunks_success(self, source_url: str, relevant_chunk_count: int, is_refinement: bool = False):
        prefix = "[Refinement] " if is_refinement else ""
        await self._send_update(
            "PROCESSING",
            "SUCCESS",
            f"{prefix}Processed {relevant_chunk_count} relevant chunks from source.",
            {"source_url": source_url, "type": "chunks", "relevant_chunk_count": relevant_chunk_count}
        )

    async def processing_source_warning(self, source_url: str, reason: str, is_refinement: bool = False):
        prefix = "[Refinement] " if is_refinement else ""
        await self._send_update(
            "PROCESSING",
            "WARNING",
            f"{prefix}Skipping source due to processing issue.",
            {"source_url": source_url, "reason": reason}
        )

    # --- Initial Writing Phase Updates ---
    async def writing_start(self, context_item_count: int):
        await self._send_update(
            "WRITING",
            "START",
            f"Generating initial report draft using {context_item_count} context items...",
            {"context_item_count": context_item_count}
        )

    async def writing_end(self, requested_searches_count: int):
        msg = "Initial report draft generated."
        if requested_searches_count > 0:
            msg += f" Requested {requested_searches_count} additional searches."
        await self._send_update(
            "WRITING",
            "END",
            msg,
            {"requested_searches_count": requested_searches_count}
        )

    async def writing_error(self, error: Exception):
        await self._send_update(
            "WRITING",
            "ERROR",
            f"Failed to generate initial draft: {error}",
            {"error": str(error)}
        )

    # --- Refinement Loop Phase Updates ---
    async def refinement_loop_start(self, iteration: int, max_loops: int):
        await self._send_update(
            "REFINING",
            "START",
            f"Starting refinement iteration {iteration}/{max_loops}...",
            {"iteration": iteration}
        )

    async def refinement_search_start(self, iteration: int, query: str):
        await self._send_update(
            "SEARCHING",
            "START",
            f"[Refinement {iteration}] Performing search for: \"{query}\"",
            {"iteration": iteration, "query": query}
        )

    async def refinement_search_end(self, iteration: int, new_result_count: int):
        await self._send_update(
            "SEARCHING",
            "END",
            f"[Refinement {iteration}] Search complete. Found {new_result_count} new results.",
            {"iteration": iteration, "new_result_count": new_result_count}
        )

    async def refinement_processing_start(self, iteration: int, sources_to_process: int):
        await self._send_update(
            "PROCESSING",
            "START",
            f"[Refinement {iteration}] Processing {sources_to_process} new sources...",
            {"iteration": iteration, "sources_to_process": sources_to_process}
        )

    async def refinement_processing_end(self, iteration: int, new_context_items: int):
        await self._send_update(
            "PROCESSING",
            "END",
            f"[Refinement {iteration}] Finished processing sources. Added {new_context_items} new context items.",
            {"iteration": iteration, "new_context_items": new_context_items}
        )

    async def refinement_refiner_start(self, iteration: int):
        await self._send_update(
            "REFINING",
            "IN_PROGRESS",
            f"[Refinement {iteration}] Calling LLM to refine draft...",
            {"iteration": iteration}
        )

    async def refinement_refiner_end(self, iteration: int, requested_searches_count: int):
        msg = f"[Refinement {iteration}] Draft refined."
        if requested_searches_count > 0:
             msg += f" Requested {requested_searches_count} further searches."
        await self._send_update(
            "REFINING",
            "IN_PROGRESS", # Still IN_PROGRESS as loop might continue
            msg,
            {"iteration": iteration, "requested_searches_count": requested_searches_count}
        )

    async def refinement_refiner_error(self, iteration: int, error: Exception):
        await self._send_update(
            "REFINING",
            "ERROR",
            f"[Refinement {iteration}] Refiner LLM failed: {error}",
            {"iteration": iteration, "error": str(error)}
        )

    async def refinement_loop_end(self, iteration: int, reason: str):
        # Reason could be 'Completed normally', 'Max iterations reached', 'No new info found'
        status = "END" if reason == "Completed normally" else "INFO"
        await self._send_update(
            "REFINING",
            status,
            f"Refinement process completed after {iteration} iterations. Reason: {reason}",
            {"iteration": iteration, "reason": reason}
        )

    # --- Finalizing Phase Updates ---
    async def finalizing_start(self):
        await self._send_update("FINALIZING", "START", "Assembling final report and references...")

    async def finalizing_end(self, final_report: str):
        await self._send_update(
            "FINALIZING",
            "END",
            "Final report assembled.",
            {"final_report": final_report} # Send final report content here
        ) 