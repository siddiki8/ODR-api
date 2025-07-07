from __future__ import annotations
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class EarningsCallTranscript(BaseModel):
    """Represents the transcript of an earnings call."""
    ticker: str
    quarter: str
    year: int
    transcript: str

class MockEarningsCallService:
    """A mock service for fetching earnings call transcripts."""

    def get_latest_earnings_call_transcript(self, ticker_symbol: str) -> EarningsCallTranscript:
        """
        Returns a mock earnings call transcript for a given ticker symbol.
        """
        logger.warning(f"Using mock earnings call service for '{ticker_symbol}'. Returning placeholder data.")
        
        mock_transcript_text = f"""
Operator: Good day, and welcome to the {ticker_symbol} Q3 2025 Earnings Conference Call.

CFO: Thank you. Good morning, everyone. We are pleased to report another quarter of solid growth. Revenue was up 15% year-over-year, beating analyst expectations. Our new initiatives in the AI space are showing tremendous promise and early adoption rates are exceeding our forecasts.

Analyst 1: Can you provide more color on the margin pressures you alluded to in the press release?

CFO: Certainly. We are seeing some supply chain headwinds, which have slightly impacted our gross margin. However, we are confident in our ability to manage these costs effectively through the end of the fiscal year. Our outlook remains strong.

CEO: I'd like to add that our team has executed flawlessly. The strategic investments we've made are paying off, and we are incredibly optimistic about our position in the market. We believe we are well-positioned for sustained, long-term growth.
"""
        return EarningsCallTranscript(
            ticker=ticker_symbol,
            quarter="Q3",
            year=2025,
            transcript=mock_transcript_text.strip()
        ) 