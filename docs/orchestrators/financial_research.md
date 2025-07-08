# Financial Research Orchestrator

This document provides a high-level overview of the `financial_research` agency's orchestration process.

## Workflow

The orchestrator is designed to perform a multi-faceted analysis of a given stock symbol, providing both real-time dashboard updates and a final, comprehensive report.

1.  **Parallel Data Fetching:** The orchestrator concurrently fetches data from multiple financial sources.
2.  **Parallel Analysis:** Once the data is acquired, it concurrently runs four specialized analysis agents:
    *   `FundamentalFinancialsAgent`
    *   `SocialMediaSentimentAgent`
    *   `NewsAnalysisAgent`
    *   `EarningsCallAnalysisAgent`
3.  **Real-time Updates:** As each analysis agent completes, its structured output is immediately sent to the client via a WebSocket to populate a dashboard.
4.  **Final Report Generation:** After all analyses are complete, a final `ReportGeneratorAgent` synthesizes all the results into a cohesive report, tailored by the user's initial request.

## Services Used

-   **`yfinance_service`:** For fetching stock information and financial statements from Yahoo! Finance.
-   **`MockSocialMediaService`:** (Mock) For fetching social media mentions.
-   **`MockNewsService`:** (Mock) For fetching news articles.
-   **`MockEarningsCallService`:** (Mock) For fetching earnings call transcripts. 