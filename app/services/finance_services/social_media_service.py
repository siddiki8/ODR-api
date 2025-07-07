from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List
import logging

logger = logging.getLogger(__name__)

class SocialMediaPost(BaseModel):
    """Represents a single social media post."""
    username: str
    text: str
    timestamp: str

class MockSocialMediaService:
    """A mock service for fetching social media data."""

    def get_social_media_mentions(self, ticker_symbol: str, limit: int = 100) -> List[SocialMediaPost]:
        """
        Returns a list of mock social media posts for a given ticker symbol.
        """
        logger.warning(f"Using mock social media service for '{ticker_symbol}'. Returning placeholder data.")
        mock_posts = [
            SocialMediaPost(username="bullish_trader_123", text=f"I'm so bullish on ${ticker_symbol}! The new product line looks amazing. To the moon! 🚀", timestamp="2025-07-08T10:00:00Z"),
            SocialMediaPost(username="bearish_investor_456", text=f"Not sure about ${ticker_symbol}. Competition is heating up and their margins look squeezed.", timestamp="2025-07-08T10:05:00Z"),
            SocialMediaPost(username="value_finder", text=f"Considering a long-term position in ${ticker_symbol}. Their P/E ratio seems reasonable compared to peers.", timestamp="2025-07-08T10:15:00Z"),
            SocialMediaPost(username="meme_stock_king", text=f"${ticker_symbol} has diamond hands potential! 💎🙌", timestamp="2025-07-08T10:20:00Z"),
        ]
        return mock_posts[:limit] 