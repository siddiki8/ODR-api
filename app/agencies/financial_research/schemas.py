from __future__ import annotations
from pydantic import BaseModel, Field
from app.core.schemas import UsageStatistics, LLMConfig
from typing import List, Dict, Any, Literal, Optional

class FinancialAnalysisRequest(BaseModel):
    """The user's initial request for a financial analysis."""
    stock_symbol: str = Field(..., description="The stock symbol to analyze, e.g., 'AAPL'.")
    report_flavor: str = Field(
        "Focus on risks and opportunities for a long-term investor.",
        description="User's instruction for the final report."
    )
    config_overrides: Optional[FinancialResearchConfig] = Field(None, description="Optional overrides for the agency's configuration.")

class FundamentalAnalysis(BaseModel):
    """Structured output for fundamental financial analysis."""
    key_metrics: Dict[str, Any] = Field(..., description="Key financial metrics like P/E, Debt-to-Equity, etc.")
    income_statement_summary: str = Field(..., description="A summary of the income statement.")
    balance_sheet_summary: str = Field(..., description="A summary of the balance sheet.")
    cash_flow_summary: str = Field(..., description="A summary of the cash flow statement.")
    overall_conclusion: str = Field(..., description="The agent's overall conclusion based on fundamental analysis.")

class SentimentAnalysis(BaseModel):
    """Structured output for social media sentiment analysis."""
    overall_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(..., description="The overall sentiment score.")
    sentiment_score: float = Field(..., description="A numerical representation of the sentiment, e.g., 0.8.")
    key_themes: List[str] = Field(..., description="Key themes identified in social media discussions.")
    summary: str = Field(..., description="A summary of the sentiment analysis.")

class NewsAnalysis(BaseModel):
    """Structured output for news analysis."""
    summary: str = Field(..., description="A summary of the most important news.")
    key_stories: List[Dict[str, str]] = Field(..., description="A list of key stories with headlines and summaries.")

class EarningsCallAnalysis(BaseModel):
    """Structured output for earnings call analysis."""
    summary: str = Field(..., description="A summary of the earnings call.")
    management_tone: Literal["Optimistic", "Pessimistic", "Neutral"] = Field(..., description="The tone of the management during the call.")
    key_discussion_points: List[str] = Field(..., description="Key points discussed during the earnings call.")

class FinancialReport(BaseModel):
    """The final, comprehensive financial report."""
    title: str = Field(..., description="The title of the report.")
    executive_summary: str = Field(..., description="A high-level executive summary.")
    fundamental_analysis_section: str = Field(..., description="The section of the report detailing fundamental analysis.")
    sentiment_analysis_section: str = Field(..., description="The section of the report detailing sentiment analysis.")
    news_analysis_section: str = Field(..., description="The section of the report detailing news analysis.")
    earnings_call_section: str = Field(..., description="The section of the report detailing earnings call analysis.")
    final_conclusion: str = Field(..., description="The final conclusion of the report, incorporating all analyses.")

class FinancialResearchResponse(BaseModel):
    """The final API response containing the report and usage statistics."""
    final_report: FinancialReport
    usage_statistics: UsageStatistics

class FinancialResearchConfig(BaseModel):
    """Configuration for the Financial Research Agency."""
    financials_agent_llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4-turbo"))
    sentiment_agent_llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-3.5-turbo"))
    news_agent_llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-3.5-turbo"))
    earnings_agent_llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4-turbo"))
    report_generator_llm_config: LLMConfig = Field(default_factory=lambda: LLMConfig(model="gpt-4-turbo"))
    news_lookback_days: int = 30
    max_social_posts_to_analyze: int = 100

    class Config:
        env_file = ".env"
        env_prefix = "FINANCIAL_RESEARCH_"
        extra = "ignore"

