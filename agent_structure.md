# DeepResearchAgent Structure and Workflow

This document outlines the structure and workflow of the `DeepResearchAgent` within the FastAPI application.

## Agent Purpose

The `DeepResearchAgent` is designed to perform in-depth research on a given user query. It orchestrates multiple steps, including planning searches, retrieving web content, ranking sources, processing content (summarization/chunking), synthesizing a report, and iteratively refining the report based on LLM feedback.

## Core Class

-   **Class:** `DeepResearchAgent`
-   **File:** `deep_research_api/app/core/agent.py`

This class contains the main logic for the research process. It initializes various service components (LLMs, search, scraper, ranker, chunker) and manages the flow of data between them.

## Configuration and Validation

Configuration and validation are managed centrally:

-   **File:** `deep_research_api/app/core/config.py`
    -   Uses Pydantic models (`AppSettings`, `ApiKeys`, `LLMConfig`) to load settings from environment variables.
    -   Defines default settings for LLMs, reranker, scraper, and workflow parameters (e.g., `max_initial_search_tasks`, `max_refinement_iterations`).
-   **File:** `deep_research_api/app/core/schemas.py`
    -   Contains Pydantic models (`PlannerOutput`, `SearchTask`, `WritingPlan`, `SourceSummary`, `SearchRequest`) for validating LLM outputs and intermediate data structures.
    -   Provides field validation (e.g., non-empty queries) for critical data structures.
-   **File:** `.env` (or `.env.example` for template)
    -   Stores sensitive API keys and allows overriding default settings.

## Workflow Overview

The agent follows these general steps, implemented within the `run_deep_research` method in `agent.py`:

1.  **Initialization:** Resets token/cost counters.
2.  **Planning (Step 1):**
    -   An LLM (Planner) generates search tasks (1 to `max_initial_search_tasks`) and a writing plan based on the user query and configured task limit.
    -   Output is validated against `PlannerOutput` schema (with LiteLLM structural output parsing and fallback validation).
    -   Uses `get_planner_prompt` from `prompts.py`.
3.  **Initial Search (Step 2):**
    -   Executes the planned search tasks using the Serper API via `execute_batch_serper_search` (`services/search.py`).
4.  **Source Reranking (Step 3):**
    -   Consolidates unique sources from search results.
    -   Reranks unique sources based on relevance to the query using the Together Rerank API via `rerank_with_together_api` (`services/ranking.py`). Filters results below a relevance threshold (0.2).
5.  **Content Fetching & Processing (Steps 4 & 5):**
    -   Iterates through the reranked sources sequentially.
    -   **Fetching (4a):** Retrieves full content using `WebScraper` (`services/scraping.py`).
    -   **Processing (5a):**
        -   **Top N (Max 10):** Generates full summaries using the Summarizer LLM (`get_summarizer_prompt`, `call_litellm_acompletion`). A dynamic `max_tokens` limit per summary is calculated based on the number being summarized and a total budget.
        -   **Remaining:** Chunks the content using `Chunker` (`services/chunking.py`), reranks chunks using Together API (threshold 0.5), and keeps the content of relevant chunks.
    -   Stores summaries and selected chunks in `source_summaries` list.
6.  **Input Context Management (Before Writer):**
    -   Estimates the character count of the prompt input for the Writer LLM.
    -   If the estimate exceeds `WRITER_INPUT_CHAR_LIMIT`, it iteratively removes the least relevant *chunks* (identified by title) from the end of the `source_summaries` list until the limit is met.
7.  **Initial Report Generation (Step 6):**
    -   An LLM (Writer) synthesizes the (potentially filtered) summaries/chunks into an initial draft report based on the writing plan.
    -   Uses `get_writer_initial_prompt` (which groups sources and instructs on citation) and `call_litellm_acompletion`.
    -   Logs raw writer output for debugging.
8.  **Refinement Loop (Step 7, Max `max_refinement_iterations`):**
    -   Detects if the current `report_draft` contains a `<search_request query="...">` tag using `_extract_search_request`.
    -   If a request is found:
        -   **Direct Search (7a):** Executes *only* the requested query using `execute_batch_serper_search`.
        -   **Process Results (7b):** Fetches content for search results and processes them *only* using the Chunk & Rerank strategy.
        -   **Call Refiner (7c):** Invokes the `_call_refiner_llm` method. This uses the Summarizer LLM config and `get_refiner_prompt` to revise the draft using the previous draft and the newly generated chunks.
        -   Updates `report_draft` if the Refiner returns a valid revision.
    -   The loop continues until max iterations or no search request is found.
9.  **Final Output Assembly (Step 8):**
    -   Uses the helper `_assemble_final_report`.
    -   Appends a formatted list of references to the final `report_draft`. The reference list is built from the *unfiltered* list of all summaries and chunks gathered throughout the process (`all_summaries_unfiltered`), ensuring all original sources are potentially listable.
    -   Returns the final report, total usage stats, and a detailed per-role usage/cost breakdown in the `llm_usage_breakdown` field.

## LLM Interaction & Validation

-   All LLM calls are made via the `call_litellm_acompletion` service function (`services/llm.py`), which handles retries and basic error propagation.
-   Planner output uses LiteLLM's Pydantic model integration for structured output and validation.
-   Writer/Refiner outputs are checked for emptiness.
-   Search requests from LLMs are extracted using regex and validated via the `SearchRequest` Pydantic model (`_extract_search_request`).

## Citation Handling

-   The `format_summaries_for_prompt` function groups source materials (summaries and chunks) by their original source link.
-   It assigns a single citation number (e.g., `[1]`) to each unique original source.
-   The Writer prompt instructs the LLM to use these original source numbers for citation and to avoid over-citing.
-   The final reference list is generated based on all unique sources gathered, matching citations found in the final draft.

## Validation System

The agent includes a robust validation system to ensure reliable LLM outputs:

1. **Schema Validation:**
   - Uses Pydantic models to define and validate data structures
   - Includes field-level validation rules and custom validators
   - Handles empty/invalid values gracefully

2. **LLM Output Correction:**
   - Implements `_validate_llm_json_output` method for JSON validation and correction
   - Attempts to fix malformed JSON using a repair prompt
   - Retries with schema information when validation fails
   - Maximum retry attempts configurable

3. **Refinement Request Parsing:**
   - Validates refinement requests using regex pattern matching
   - Extracts topic and optional context
   - Returns structured `RefinementRequest` objects

## FastAPI Integration

-   **File:** `deep_research_api/app/main.py`
    -   Defines the FastAPI application and the `/research` endpoint.
    -   Handles incoming requests (`ResearchRequest`), initializes the `DeepResearchAgent` with appropriate settings and API keys, runs the research process, and returns the final report (`ResearchResponse`). 