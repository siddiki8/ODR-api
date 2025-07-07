from __future__ import annotations
import logging
from typing import Optional, Callable, Coroutine, Any, Dict
from . import schemas

logger = logging.getLogger(__name__)

class FinancialsWebSocketUpdateHandler:
    """Handles sending structured status updates for the Financial Research agency."""

    def __init__(self, websocket_callback: Optional[Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]]):
        self._callback = websocket_callback

    async def _send_update(self, event_type: str, data: Optional[Dict] = None):
        if self._callback:
            payload = {"event": event_type, "data": data or {}}
            try:
                await self._callback(payload)
                logger.debug(f"Sent WebSocket update: {event_type}")
            except Exception as e:
                logger.error(f"Failed to send WebSocket update for event {event_type}: {e}", exc_info=True)

    async def data_fetching_start(self):
        await self._send_update("status", {"message": "Fetching financial data sources..."})

    async def analysis_start(self):
        await self._send_update("status", {"message": "Analyzing data..."})

    async def send_analysis_update(self, analysis_data: Any):
        """Sends a specific analysis result to the dashboard."""
        if isinstance(analysis_data, schemas.FundamentalAnalysis):
            await self._send_update("fundamental_analysis", analysis_data.model_dump())
        elif isinstance(analysis_data, schemas.SentimentAnalysis):
            await self._send_update("sentiment_analysis", analysis_data.model_dump())
        elif isinstance(analysis_data, schemas.NewsAnalysis):
            await self._send_update("news_analysis", analysis_data.model_dump())
        elif isinstance(analysis_data, schemas.EarningsCallAnalysis):
            await self._send_update("earnings_call_analysis", analysis_data.model_dump())
        else:
            logger.warning(f"Attempted to send unknown analysis data type: {type(analysis_data)}")

    async def report_generation_start(self):
        await self._send_update("status", {"message": "Generating final report..."})

    async def send_final_report(self, report: schemas.FinancialReport):
        await self._send_update("final_report", report.model_dump())

    async def send_error(self, error: Exception):
        await self._send_update("error", {"message": str(error)}) 