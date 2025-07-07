from __future__ import annotations
import yfinance as yf
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class Ticker(BaseModel):
    """Represents a stock ticker."""
    symbol: str = Field(..., description="The stock symbol, e.g., 'AAPL'.")

class StockInfo(BaseModel):
    """Pydantic model for holding stock information."""
    info: dict

class Financials(BaseModel):
    """Pydantic model for holding financial statements."""
    income_statement: dict
    balance_sheet: dict
    cash_flow: dict

class YFinanceService:
    """A service to interact with the yfinance library."""

    def get_stock_info(self, ticker_symbol: str) -> StockInfo:
        """Gets general information about a stock."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            if not info or info.get('trailingEps') is None: # Basic check for a valid ticker
                raise ValueError(f"Invalid or delisted ticker symbol: {ticker_symbol}")
            return StockInfo(info=info)
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker_symbol} from yfinance: {e}", exc_info=True)
            raise

    def get_financials(self, ticker_symbol: str) -> Financials:
        """Gets financial statements for a stock."""
        try:
            ticker = yf.Ticker(ticker_symbol)
            income_statement = ticker.income_stmt
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            return Financials(
                income_statement=income_statement.to_dict(),
                balance_sheet=balance_sheet.to_dict(),
                cash_flow=cash_flow.to_dict()
            )
        except Exception as e:
            logger.error(f"Error fetching financials for {ticker_symbol} from yfinance: {e}", exc_info=True)
            raise

# Example Usage:
# if __name__ == '__main__':
#     service = YFinanceService()
#     aapl_ticker = Ticker(symbol="AAPL")
#     try:
#         stock_info = service.get_stock_info(aapl_ticker.symbol)
#         print("--- Stock Info ---")
#         print(stock_info.info.get('longBusinessSummary'))
#
#         financials = service.get_financials(aapl_ticker.symbol)
#         print("\n--- Income Statement (sample) ---")
#         # Print a small sample of the income statement
#         for date, data in list(financials.income_statement.items())[:1]:
#             print(f"Date: {date}")
#             for key, value in list(data.items())[:5]:
#                 print(f"  {key}: {value}")
#
#     except Exception as e:
#         print(f"An error occurred: {e}") 