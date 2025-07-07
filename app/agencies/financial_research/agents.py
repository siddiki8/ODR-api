from __future__ import annotations
import json
from pydantic_ai import pydantic_ai_processor
from pydantic_ai.llm import LLM
from . import schemas
from app.core.dependencies import LLMProvider

# --- Prompt Templates ---

FINANCIALS_AGENT_SYSTEM_PROMPT = """
You are a senior financial analyst. Your task is to analyze the provided financial statements (income statement, balance sheet, cash flow) for a company.
Provide a concise summary for each statement and an overall conclusion about the company's financial health. Extract key performance metrics.
"""
FINANCIALS_AGENT_USER_TEMPLATE = """
Please analyze the following financial data for {ticker}.

Income Statement:
{income_statement}

Balance Sheet:
{balance_sheet}

Cash Flow:
{cash_flow}
"""

SENTIMENT_AGENT_SYSTEM_PROMPT = """
You are a market sentiment analyst. Your task is to analyze a collection of social media posts related to a specific stock.
Determine the overall sentiment (Positive, Negative, or Neutral), provide a sentiment score between -1 (very negative) and 1 (very positive), identify key discussion themes, and write a summary.
"""
SENTIMENT_AGENT_USER_TEMPLATE = """
Please analyze the sentiment of these social media posts regarding {ticker}:

{posts}
"""

NEWS_AGENT_SYSTEM_PROMPT = """
You are a financial news analyst. Your task is to review a list of recent news articles about a company.
Summarize the most impactful news and highlight the key stories that an investor should be aware of.
"""
NEWS_AGENT_USER_TEMPLATE = """
Please analyze the following news articles for {ticker}:

{articles}
"""

EARNINGS_CALL_AGENT_SYSTEM_PROMPT = """
You are an expert financial analyst specializing in corporate communications. Your task is to analyze an earnings call transcript.
Summarize the key discussion points, and determine the overall tone of the management (Optimistic, Pessimistic, or Neutral).
"""
EARNINGS_CALL_AGENT_USER_TEMPLATE = """
Please analyze the following earnings call transcript for {ticker}:

{transcript}
"""

REPORT_GENERATOR_SYSTEM_PROMPT = """
You are a chief investment strategist. Your task is to synthesize analyses from multiple domains (fundamentals, sentiment, news, earnings calls) into a single, cohesive investment report.
The user will provide a specific "flavor" or focus for the report. You must adhere to this instruction while structuring your report.
"""
REPORT_GENERATOR_USER_TEMPLATE = """
Please generate a comprehensive investment report for {ticker}.

User's requested focus: "{report_flavor}"

Use the following structured analysis data to construct your report. Do not make up information; base your report entirely on the data provided.

Fundamental Analysis:
{fundamental_analysis}

Sentiment Analysis:
{sentiment_analysis}

News Analysis:
{news_analysis}

Earnings Call Analysis:
{earnings_call_analysis}
"""


# --- Agent Definitions ---

@pydantic_ai_processor
def create_financials_agent(llm: LLM) -> schemas.FundamentalAnalysis:
    """An agent that analyzes financial statements."""
    pass

@pydantic_ai_processor
def create_sentiment_agent(llm: LLM) -> schemas.SentimentAnalysis:
    """An agent that analyzes social media sentiment."""
    pass

@pydantic_ai_processor
def create_news_agent(llm: LLM) -> schemas.NewsAnalysis:
    """An agent that analyzes news articles."""
    pass

@pydantic_ai_processor
def create_earnings_agent(llm: LLM) -> schemas.EarningsCallAnalysis:
    """An agent that analyzes earnings call transcripts."""
    pass

@pydantic_ai_processor
def create_report_generator_agent(llm: LLM) -> schemas.FinancialReport:
    """An agent that generates a final report from multiple analyses."""
    pass


# --- Agent Collection ---

class AgencyAgents:
    """A collection of all agents used by the Financial Research agency."""

    def __init__(self, llm_provider: LLMProvider, config: schemas.FinancialResearchConfig):
        self.financials_agent = create_financials_agent(
            llm_provider.get_llm(config.financials_agent_llm_config)
        )
        self.sentiment_agent = create_sentiment_agent(
            llm_provider.get_llm(config.sentiment_agent_llm_config)
        )
        self.news_agent = create_news_agent(
            llm_provider.get_llm(config.news_agent_llm_config)
        )
        self.earnings_agent = create_earnings_agent(
            llm_provider.get_llm(config.earnings_agent_llm_config)
        )
        self.report_generator_agent = create_report_generator_agent(
            llm_provider.get_llm(config.report_generator_llm_config)
        )

def format_data_for_prompt(data: object) -> str:
    """Formats Pydantic models or other data into a JSON string for LLM prompts."""
    if isinstance(data, list) and all(hasattr(item, 'model_dump_json') for item in data):
        return json.dumps([item.model_dump() for item in data], indent=2)
    if hasattr(data, 'model_dump_json'):
        return data.model_dump_json(indent=2)
    if isinstance(data, dict) or isinstance(data, list):
        return json.dumps(data, indent=2)
    return str(data)

