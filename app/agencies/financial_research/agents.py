from __future__ import annotations
import json
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.llm import LLM
from . import schemas
from app.core.dependencies import LLMProvider

# --- Prompt Templates ---

FINANCIALS_AGENT_SYSTEM_PROMPT = """
You are a senior financial analyst. Your task is to interpret provided financial data for a company.
You will receive pre-calculated key metrics and summaries of the financial statements.
Your goal is to provide a concise summary for each statement and an overall conclusion about the company's financial health based on all the information provided.
Do not perform calculations. Your role is analysis and interpretation.
"""
FINANCIALS_AGENT_USER_TEMPLATE = """
Please analyze the following financial data for {ticker}.

Key Metrics:
{key_metrics}

Income Statement Summary:
{income_statement}

Balance Sheet Summary:
{balance_sheet}

Cash Flow Summary:
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

def create_financials_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt=FINANCIALS_AGENT_SYSTEM_PROMPT,
        result_type=schemas.FundamentalAnalysis,
        retry=ModelRetry(tries=3, on_fail="log")
    )

def create_sentiment_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt=SENTIMENT_AGENT_SYSTEM_PROMPT,
        result_type=schemas.SentimentAnalysis,
        retry=ModelRetry(tries=3, on_fail="log")
    )

def create_news_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt=NEWS_AGENT_SYSTEM_PROMPT,
        result_type=schemas.NewsAnalysis,
        retry=ModelRetry(tries=3, on_fail="log")
    )

def create_earnings_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt=EARNINGS_CALL_AGENT_SYSTEM_PROMPT,
        result_type=schemas.EarningsCallAnalysis,
        retry=ModelRetry(tries=3, on_fail="log")
    )

def create_report_generator_agent(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        system_prompt=REPORT_GENERATOR_SYSTEM_PROMPT,
        result_type=schemas.FinancialReport,
        retry=ModelRetry(tries=3, on_fail="log")
    )


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

