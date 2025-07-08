# Corporate Profile Extractor (CPE) Orchestrator

This document provides a high-level overview of the `cpe` agency's orchestration process.

## Workflow

The orchestrator is designed to find and extract key information about a company, such as contact emails and a corporate profile, based on a domain name.

1.  **Planning:** A `Planner` agent generates a search plan to find relevant URLs.
2.  **Searching:** The orchestrator executes web searches to find potential company websites.
3.  **Extraction:** For each found domain, the orchestrator performs a series of steps:
    *   Crawls the website for email addresses.
    *   Aggregates the HTML content.
    *   Uses an `Extractor` LLM to process the content and extract a structured company profile.
4.  **Finalizing:** The results are compiled and finalized.

## Services Used

-   **Search Service (`Serper`):** For executing web searches to find company domains.
-   **Web Scraper Service (`Crawl4AI`):** For crawling websites and extracting content and emails. 