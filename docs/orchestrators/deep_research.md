# Deep Research Orchestrator

This document provides a high-level overview of the `deep_research` agency's orchestration process.

## Workflow

The orchestrator executes a multi-step research process designed for in-depth analysis of a user's query.

1.  **Planning:** A `Planner` agent generates a detailed research plan and a list of specific search queries.
2.  **Initial Search & Reranking:** The orchestrator executes the search queries and uses a reranking model to prioritize the most relevant results.
3.  **Content Processing:** It scrapes the content from the top-ranked URLs. This content is then either summarized by a `Summarizer` agent or broken into chunks and reranked again for relevance.
4.  **Iterative Writing & Refinement:** A `Writer` agent creates an initial draft. If the agent identifies information gaps, it can request additional searches. This triggers a refinement loop where new information is fetched and a `Refiner` agent integrates it into the draft.
5.  **Final Assembly:** The final report is assembled with proper citations and a reference list.

## Services Used

-   **Search Service (`Serper`):** For executing web, academic, and news searches.
-   **Web Scraper Service (`Crawl4AI`):** For extracting content from web pages.
-   **Ranking Service (`Together AI`):** For reranking both search results and text chunks to find the most relevant information.
-   **Chunking Service:** For breaking down large pieces of text into smaller, manageable chunks for analysis.
-   **Firestore Service (Optional):** For persisting task state and results. 