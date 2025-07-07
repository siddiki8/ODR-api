Of course. Based on the architecture and the documentation we've established, here is a comprehensive plan for creating the Financial Research Agency.

This plan outlines the necessary components, workflows, and data structures required to build a robust agency capable of performing multi-faceted stock analysis.

### **Project Plan: Financial Research Agency**

The goal is to create an agency at `app/agencies/financial_research` that accepts a stock symbol, performs various analyses, and generates both a real-time dashboard feed and a final, user-customized report.

---

### **Phase 1: Scaffolding & Configuration**

1.  **Create File Structure:**
    *   Create the directory `app/agencies/financial_research/`.
    *   Inside, create the standard agency files:
        *   `__init__.py`
        *   `agents.py`
        *   `callbacks.py`
        *   `config.py`
        *   `helpers.py`
        *   `orchestrator.py`
        *   `routes.py`
        *   `schemas.py`

2.  **Define Configuration (`config.py`):**
    *   Create a `FinancialResearchConfig` class.
    *   It will define `LLMConfig` for each of the five agents:
        *   `financials_agent_llm_config`
        *   `sentiment_agent_llm_config`
        *   `news_agent_llm_config`
        *   `earnings_agent_llm_config`
        *   `report_generator_llm_config`
    *   It will also include parameters like `news_lookback_days` or `max_social_posts_to_analyze`.

3.  **Define External Service Dependencies:**
    *   This agency will require external financial data APIs. We will need to add new services to handle this.
    *   **Proposal:** Create a new service directory `app/services/financial_data/`.
    *   Inside, we could have a `provider.py` that abstracts fetching data from a source like **Financial Modeling Prep** or **Alpha Vantage**. This keeps the agency's helper functions clean and focused on orchestration rather than API specifics.
    *   The necessary API keys (e.g., `FMP_API_KEY`) will be added to the global `.env` file and the `AppSettings` in `app/core/config.py`.

---

### **Phase 2: Schema and Data Contract Design**

In `schemas.py`, we will define the data structures that act as the contracts between our components.

1.  **Input Schema:**
    *   `FinancialAnalysisRequest(BaseModel)`:
        *   `stock_symbol: str` (e.g., "AAPL")
        *   `report_flavor: str` (User's instruction for the final report, e.g., "Focus on risks and opportunities for a long-term investor.")
        *   `config_overrides: Optional[FinancialResearchConfig]`

2.  **Individual Analysis Schemas (for Dashboard):**
    *   `FundamentalAnalysis(BaseModel)`:
        *   `key_metrics: Dict[str, Any]` (P/E, Debt-to-Equity, etc.)
        *   `income_statement_summary: str`
        *   `balance_sheet_summary: str`
        *   `cash_flow_summary: str`
        *   `overall_conclusion: str`
    *   `SentimentAnalysis(BaseModel)`:
        *   `overall_sentiment: Literal["Positive", "Negative", "Neutral"]`
        *   `sentiment_score: float` (e.g., 0.8)
        *   `key_themes: List[str]` (e.g., "iPhone 17 excitement", "Concerns about EU regulation")
        *   `summary: str`
    *   `NewsAnalysis(BaseModel)`:
        *   `summary: str`
        *   `key_stories: List[Dict[str, str]]` (e.g., `[{"headline": ..., "summary": ...}]`)
    *   `EarningsCallAnalysis(BaseModel)`:
        *   `summary: str`
        *   `management_tone: Literal["Optimistic", "Pessimistic", "Neutral"]`
        *   `key_discussion_points: List[str]`

3.  **Final Output Schema:**
    *   `FinancialReport(BaseModel)`:
        *   `title: str`
        *   `executive_summary: str`
        *   `fundamental_analysis_section: str`
        *   `sentiment_analysis_section: str`
        *   `news_analysis_section: str`
        *   `earnings_call_section: str`
        *   `final_conclusion: str`
    *   `FinancialResearchResponse(BaseModel)`:
        *   `final_report: FinancialReport`
        *   `usage_statistics: UsageStatistics`

---

### **Phase 3: Agent & Orchestration Logic**

This phase involves implementing the core logic of the agency.

1.  **Agent Implementation (`agents.py`):**
    *   Create five distinct Pydantic-AI agents, each returning its corresponding Pydantic schema from Phase 2.
    *   **`FundamentalFinancialsAgent`**: Takes raw financial statements (JSON/text) and generates a `FundamentalAnalysis`.
    *   **`SocialMediaSentimentAgent`**: Takes a list of social media posts and generates a `SentimentAnalysis`.
    *   **`NewsAnalysisAgent`**: Takes a list of news articles and generates a `NewsAnalysis`.
    *   **`EarningsCallAnalysisAgent`**: Takes an earnings call transcript and generates an `EarningsCallAnalysis`.
    *   **`ReportGeneratorAgent`**: This is the final agent. It takes the structured output from all four previous agents *and* the user's `report_flavor` to generate the comprehensive `FinancialReport`.

2.  **Helper Functions (`helpers.py`):**
    *   Create async helper functions to abstract the data fetching:
        *   `fetch_financials(symbol)`
        *   `fetch_news(symbol)`
        *   `fetch_social_media_mentions(symbol)`
        *   `fetch_earnings_transcript(symbol)`
    *   These helpers will call the new services defined in `app/services/financial_data/`.

3.  **Orchestrator Workflow (`orchestrator.py`):**
    *   The `run_financial_analysis_orchestration` function will be the centerpiece.
    *   **Step 1: Parallel Data Fetching.** Use `asyncio.gather` to fetch all data sources (financials, news, social, earnings) concurrently. Send WebSocket updates as each source is acquired.
    *   **Step 2: Parallel Analysis.** Once data is fetched, use `asyncio.gather` again to run the first four analysis agents concurrently on their respective data.
    *   **Step 3: Real-time Dashboard Updates.** As each analysis agent completes, its structured Pydantic output (`FundamentalAnalysis`, `SentimentAnalysis`, etc.) is immediately sent over the WebSocket to the client dashboard. A new `callbacks.py` handler will manage this.
    *   **Step 4: Final Report Generation.** After all four analyses are complete, the orchestrator will trigger the `ReportGeneratorAgent`. It will provide the agent with all four analysis results and the user's `report_flavor`.
    *   **Step 5: Finalize.** The generated `FinancialReport` is sent as the final message, and the `FinancialResearchResponse` is returned.

### **Phase 4: API and Communication**

1.  **Callback Handler (`callbacks.py`):**
    *   Create a `FinancialsWebSocketUpdateHandler`.
    *   It will have specific methods to push structured data for the dashboard:
        *   `send_fundamental_analysis(data: FundamentalAnalysis)`
        *   `send_sentiment_analysis(data: SentimentAnalysis)`
        *   `send_news_analysis(data: NewsAnalysis)`
        *   `send_earnings_analysis(data: EarningsCallAnalysis)`
        *   `send_final_report(data: FinancialReport)`

2.  **API Endpoint (`routes.py`):**
    *   Create a FastAPI router with a WebSocket endpoint at `/financial_research/ws/analyze`.
    *   This endpoint will accept the `FinancialAnalysisRequest`, initialize all the necessary components (configs, agents, callback handler), and start the orchestration.

3.  **Integrate into Main App:**
    *   Mount the new router in `app/main.py` under the `/financial_research` prefix tag.

This plan establishes a clear separation of concerns, leverages asynchronous operations for performance, and creates a highly modular and extensible Financial Research Agency.