# Deep Research API

A FastAPI service providing deep research capabilities based on a user query.

This service takes a user query, plans a research strategy, executes web searches, fetches and summarizes relevant content, and synthesizes the findings into a comprehensive report.

## Setup

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd deep_research_api
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and add your actual API keys:
        *   `SERPER_API_KEY`: Your Google Serper API key.
        *   `JINA_API_KEY`: Your Jina AI API key (for reranking).
        *   `OPENROUTER_API_KEY`: Your OpenRouter API key (if using default OpenRouter models for LLMs).

## Running the API

Use Uvicorn to run the FastAPI application:

```bash
uvicorn app.main:app --reload
```

*   `app.main:app`: Points to the `app` instance in the `app/main.py` file.
*   `--reload`: Enables auto-reloading during development.

The API will typically be available at `http://127.0.0.1:8000`.
You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## API Usage

Send a POST request to the `/research` endpoint with a JSON body containing the user query:

**Request:**

```json
{
  "query": "What are the latest advancements in quantum computing resistant cryptography?",
  "planner_llm_config": null,      // Optional: Override default LLM configs
  "summarizer_llm_config": null,
  "writer_llm_config": null,
  "scraper_strategies": null     // Optional: Override default scraper strategies
}
```

**Response (Success - 200 OK):**

```json
{
  "report": "[Generated research report content...]\n\nReferences:\n1. [Source Title 1](http://example.com/source1)\n2. [Source Title 2](http://example.com/source2)\n...",
  "llm_token_usage": {
    "completion_tokens": 1500,
    "prompt_tokens": 2500,
    "total_tokens": 4000
  },
  "estimated_llm_cost": 0.0025,
  "serper_queries_used": 4
}
```

**Response (Error - 500 Internal Server Error):**

```json
{
  "detail": "An internal server error occurred: [Error details]"
}
```

## TODO / Next Steps

*   Implement the actual `Chunker` logic in `app/services/chunking.py`.
*   Refine the `WebScraper` in `app/services/scraping.py`, particularly the `extract` method if DOM-based strategies (CSS/XPath) or advanced LLM strategies are required.
*   Add comprehensive error handling and validation.
*   Implement unit and integration tests.
*   Consider splitting the large `scraping.py` file back into logical components (utils, factory, scraper class) within the `services` directory if it becomes unmanageable.
*   Add more robust configuration management (e.g., using environment-specific settings). 