from __future__ import annotations
import asyncio
import logging
from .schemas import FinancialResearchConfig
from app.services.finance_services.yfinance_service import YFinanceService, StockInfo, Financials
from app.services.finance_services.social_media_service import MockSocialMediaService, SocialMediaPost
from app.services.finance_services.news_service import MockNewsService, NewsArticle
from app.services.finance_services.earnings_call_service import MockEarningsCallService, EarningsCallTranscript

logger = logging.getLogger(__name__)

async def fetch_all_data_sources(
    symbol: str,
    config: FinancialResearchConfig,
    yfinance_service: YFinanceService,
    social_service: MockSocialMediaService,
    news_service: MockNewsService,
    earnings_service: MockEarningsCallService,
) -> dict:
    """
    Asynchronously fetches all necessary data from the various financial services.
    Uses asyncio.gather to run all fetching operations concurrently.
    """
    logger.info(f"Starting concurrent data fetching for {symbol}...")

    async def _fetch_financials():
        # yfinance is not async, so we run it in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, yfinance_service.get_financials, symbol)

    async def _fetch_social_media():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, social_service.get_social_media_mentions, symbol, config.max_social_posts_to_analyze)

    async def _fetch_news():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, news_service.get_news, symbol, config.news_lookback_days)

    async def _fetch_earnings_call():
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, earnings_service.get_latest_earnings_call_transcript, symbol)

    results = await asyncio.gather(
        _fetch_financials(),
        _fetch_social_media(),
        _fetch_news(),
        _fetch_earnings_call(),
        return_exceptions=True
    )

    financials, social_media, news, earnings_call = results

    # Basic error handling for fetched data
    if isinstance(financials, Exception):
        logger.error(f"Failed to fetch financials for {symbol}: {financials}")
        financials = None # or some default/error state
    if isinstance(social_media, Exception):
        logger.error(f"Failed to fetch social media data for {symbol}: {social_media}")
        social_media = []
    if isinstance(news, Exception):
        logger.error(f"Failed to fetch news for {symbol}: {news}")
        news = []
    if isinstance(earnings_call, Exception):
        logger.error(f"Failed to fetch earnings call for {symbol}: {earnings_call}")
        earnings_call = None

    logger.info(f"Finished concurrent data fetching for {symbol}.")

    return {
        "financials": financials,
        "social_media_posts": social_media,
        "news_articles": news,
        "earnings_call_transcript": earnings_call,
    }

