# Exec Finder Agency - Orchestrator Workflow

This document outlines the proposed workflow for the Exec Finder agency's orchestrator. The goal is to identify companies and potential executive contact information (primarily emails) within a specific industry and city.

**Input:**
*   `industry`: The target industry (e.g., "Marketing", "Software Development").
*   `city`: The target city (e.g., "Austin, TX", "London").

**Workflow Steps:**

1.  **Initialization:**
    *   Receive `industry` and `city`.
    *   Initialize logging, usage tracking (if applicable), and data structures to hold results.
    *   (Optional) Send `orchestration_start` notification via WebSocket if real-time updates are needed.

2.  **Planning:**
    *   **(Agent: Planner)**
    *   **Input:** `industry`, `city`.
    *   **Action:** Call a specialized `Planner` LLM agent. The prompt will guide the LLM to generate targeted search queries relevant to finding companies and executives in the given industry/city. Examples:
        *   "Top [industry] companies in [city]"
        *   "List of [industry] firms located in [city]"
        *   "[Specific Company Name found previously] contact page"
        *   "CEO email address [Specific Company Name]"
        *   "LinkedIn page for [Specific Company Name]"
    *   **Output:** A structured `SearchPlan` object containing a list of `SearchTask` items (similar to `deep_research`).
    *   (Optional) Send `planning_complete` notification.

3.  **Search Execution:**
    *   **(Service: Search Service - e.g., Serper)**
    *   **Input:** `SearchPlan` object from the Planner.
    *   **Action:** Execute the search queries defined in the `SearchPlan` using the configured search service.
    *   **Output:** A list of `SearchResult` objects (containing URLs, titles, snippets).
    *   (Optional) Send `search_complete` notification.

4.  **Content Processing (Scraping & Extraction Loop):**
    *   Initialize an empty list to store extracted `CompanyContactInfo` objects.
    *   Iterate through the unique URLs obtained from the search results:
        *   **(Service: Scraper Service - e.g., Crawl4AI)**
        *   **Input:** URL.
        *   **Action:** Scrape the content of the URL. Prioritize fetching main text, contact details sections, "About Us" pages, etc.
        *   **Output:** Scraped content (e.g., Markdown or clean text).
        *   **(Agent: Extractor)**
        *   **Input:** Scraped content, original `industry`, `city` (for context).
        *   **Action:** Call a specialized `Extractor` LLM agent. The prompt will instruct the LLM to identify if the content relates to a company within the target industry/city and extract the following information into a predefined Pydantic schema (e.g., `CompanyContactInfo`).
        *   **Output Schema (`CompanyContactInfo`):**
            *   `company_name`: str | None
            *   `company_url`: str | None (Primary domain if identifiable)
            *   `brief_description`: str | None (Short summary relevant to the industry)
            *   `potential_emails`: List[str] (List of unique email addresses found)
            *   `source_url`: str (The URL the info was extracted from)
        *   **Action:** If the `Extractor` successfully returns a valid `CompanyContactInfo` object (especially with a `company_name`), add it to the results list. Handle potential LLM errors or validation failures gracefully (log and continue).
        *   (Optional) Send `processing_url` / `extraction_complete` notifications for each URL.

5.  **Programmatic Aggregation & Deduplication:**
    *   **Input:** List of `CompanyContactInfo` objects collected from the Extractor agent.
    *   **Action:**
        *   Process the list to consolidate information.
        *   Identify unique companies (e.g., based on `company_name` or `company_url`).
        *   For each unique company, aggregate all found `potential_emails` from different source URLs into a single list.
        *   Potentially refine descriptions or select the best one.
    *   **Output:** A final, deduplicated list of `AggregatedCompanyContactInfo` objects.
    *   **Output Schema (`AggregatedCompanyContactInfo`):**
        *   `company_name`: str
        *   `company_url`: str | None
        *   `brief_description`: str | None
        *   `aggregated_emails`: List[str]
        *   `source_urls`: List[str] (List of URLs where info for this company was found)

6.  **Final Response:**
    *   **Input:** Aggregated list of company contact info.
    *   **Action:** Format the final response, potentially including usage statistics.
    *   (Optional) Send `orchestration_complete` notification.
    *   **Output:** Return the final list of `AggregatedCompanyContactInfo`. 