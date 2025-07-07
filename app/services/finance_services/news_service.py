from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NewsArticle(BaseModel):
    """Represents a single news article."""
    headline: str
    summary: str
    source: str
    url: str
    published_date: str

class MockNewsService:
    """A mock service for fetching news articles."""

    def get_news(self, ticker_symbol: str, lookback_days: int) -> List[NewsArticle]:
        """
        Returns a list of mock news articles for a given ticker symbol.
        """
        logger.warning(f"Using mock news service for '{ticker_symbol}'. Returning placeholder data.")
        today = datetime.now()
        mock_articles = [
            NewsArticle(
                headline=f"{ticker_symbol} Announces Record Profits in Q3",
                summary=f"The company reported its strongest quarter ever, driven by high demand for its new flagship product. CEO expressed strong optimism for the upcoming fiscal year.",
                source="Major Financial News Outlet",
                url=f"http://example.com/news/{ticker_symbol}-q3-profits",
                published_date=(today - timedelta(days=2)).isoformat()
            ),
            NewsArticle(
                headline=f"Regulatory Scrutiny Mounts for {ticker_symbol} Over Market Dominance",
                summary=f"Government regulators are launching an investigation into potential anti-competitive practices, casting a shadow over the company's recent successes.",
                source="Regulatory Affairs Chronicle",
                url=f"http://example.com/news/{ticker_symbol}-regulatory-scrutiny",
                published_date=(today - timedelta(days=5)).isoformat()
            ),
            NewsArticle(
                headline=f"Analyst Upgrades {ticker_symbol} to 'Strong Buy'",
                summary=f"A prominent market analyst has upgraded their rating for {ticker_symbol}, citing strong fundamentals and a positive long-term outlook.",
                source="Market Insights Today",
                url=f"http://example.com/news/{ticker_symbol}-analyst-upgrade",
                published_date=(today - timedelta(days=10)).isoformat()
            ),
        ]
        return mock_articles 