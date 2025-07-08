from __future__ import annotations
import asyncio
import logging
from .schemas import FinancialResearchConfig
from app.agencies.services.finance_services.yfinance_service import YFinanceService, StockInfo, Financials
from app.agencies.services.finance_services.social_media_service import MockSocialMediaService, SocialMediaPost
from app.agencies.services.finance_services.news_service import MockNewsService, NewsArticle
from app.agencies.services.finance_services.earnings_call_service import MockEarningsCallService, EarningsCallTranscript

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

def calculate_key_metrics(financials: Financials) -> dict:
    """
    Calculates key financial metrics from the financial statements.
    This function processes the raw data to ensure calculations are done programmatically for accuracy.
    
    Args:
        financials: A Financials object containing income statement, balance sheet, and cash flow data.

    Returns:
        A dictionary of calculated key metrics.
    """
    metrics = {}
    
    try:
        # yfinance data is structured with dates as keys. We get the most recent data column.
        balance_sheet = next(iter(financials.balance_sheet.values()))
        income_statement = next(iter(financials.income_statement.values()))
        
        # --- Safely extract values using .get() to avoid KeyErrors ---
        total_liabilities = balance_sheet.get("Total Liab")
        total_stockholder_equity = balance_sheet.get("Total Stockholder Equity")
        total_current_assets = balance_sheet.get("Total Current Assets")
        total_current_liabilities = balance_sheet.get("Total Current Liabilities")
        net_income = income_statement.get("Net Income")
        total_revenue = income_statement.get("Total Revenue")

        # --- Calculate metrics if data is available and denominators are not zero ---
        if total_liabilities is not None and total_stockholder_equity and total_stockholder_equity != 0:
            metrics["Debt-to-Equity Ratio"] = f"{total_liabilities / total_stockholder_equity:.2f}"

        if total_current_assets is not None and total_current_liabilities and total_current_liabilities != 0:
            metrics["Current Ratio"] = f"{total_current_assets / total_current_liabilities:.2f}"

        if net_income is not None and total_stockholder_equity and total_stockholder_equity != 0:
            metrics["Return on Equity (ROE)"] = f"{net_income / total_stockholder_equity:.2%}"

        if net_income is not None and total_revenue and total_revenue != 0:
            metrics["Net Profit Margin"] = f"{net_income / total_revenue:.2%}"

        logger.info(f"Successfully calculated key metrics for symbol: {list(metrics.keys())}")
        
    except (StopIteration, TypeError, AttributeError) as e:
        logger.error(f"Could not calculate metrics due to a data structure issue: {e}")
        return {"error": "Could not process financial statements for metric calculation."}
    except Exception as e:
        logger.error(f"An unexpected error occurred during metric calculation: {e}", exc_info=True)
        return {"error": "An unexpected error occurred during metric calculation."}

    return metrics

