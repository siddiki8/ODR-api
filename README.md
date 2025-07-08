# Open Deep Research API Framework

<a href="https://github.com/unclecode/crawl4ai">
  <img src="https://raw.githubusercontent.com/unclecode/crawl4ai/main/docs/assets/powered-by-dark.svg" alt="Powered by Crawl4AI" width="200"/>
</a>
<!-- Add other badges here later, e.g., Build Status, License, PyPI version -->

<p align="center">
  <img src="public/odr-api.png" alt="Open Deep Research API Logo" width="200"/>
</p>

**Developed by [Luminary AI Solutions LLC](https://luminarysolutions.ai)**

**The Open Deep Research API Framework provides a powerful, modular, multi-agency foundation for building sophisticated AI-powered research systems.** Instead of a single monolithic application, this framework allows you to create distinct `Agencies` (e.g., Deep Research, Financial Analysis, Corporate Profile Extraction), each orchestrating multiple specialized Large Language Model (LLM) `Agents`. Agencies leverage shared core `tools` for LLMs to invoke (currently no tools are implemented) and `Services` to be invoked programatically. Currently implemented are web search, advanced content scraping, chunking, and ranking to generate comprehensive, cited reports, often streamed over WebSockets.

**Build Your Own Research AI!** This framework is designed for extension. Easily add new agencies, agents, or services to tackle diverse research domains.

## ✨ Features

*   **Highly Modular Design:** Built for extension! Easily add new specialized research `Agencies` (e.g., for finance, legal, corporate profiling) in `app/agencies/` or enhance existing ones with new `Agents`.
*   **Multi-Agency Architecture:** Organizes research tasks using specialized `Agencies`. Each agency runs independently with its own set of agents and orchestration logic.
*   **Shareable Core Tools and Services:** Common tasks like web search (`app/services/search`), content scraping (`app/services/scraper`), text chunking (`app/services/chunking`), PDF handling (`app/services/scraping_utils`), and ranking (`app/agencies/services/ranking.py`) are isolated in `app/services/` or `app/agencies/services/`, ready to be reused by any agency. an LLM tool directory is also implemented but currently no tools are implemented.
*   **Agency-Specific Orchestration:** Each agency defines its own workflow logic in its `orchestrator.py` file, allowing for diverse and complex research processes tailored to the domain.
*   **Clear Agent Roles:** Agents within an agency typically have defined responsibilities (e.g., Planning, Summarizing, Writing, Refining), simplifying development, testing, and maintenance (`app/agencies/<agency_name>/agents.py`).
*   **Structured LLM Interaction:** Leverages Pydantic (`app/core/schemas.py`, `app/agencies/<agency_name>/schemas.py`) to define clear input/output schemas for LLM agents, ensuring reliable and validated data flow.
*   **Integrated Web Search:** Built-in support for [Serper](https://serper.dev/) (`app/services/search/serper_service.py`), easily adaptable for other providers.
*   **Advanced Content Scraping:** Uses [Crawl4AI](https://github.com/extractus/crawl4ai) (`app/services/scraper/crawl4ai_scraper.py`) for robust web content extraction, including utilities for handling complex sites and formats like PDFs.
*   **Content Reranking:** Employs reranking models (e.g., via Together AI API) to prioritize the most relevant search results and text chunks for LLM context (`app/agencies/services/ranking.py`, used by `helpers.py`).
*   **Asynchronous Streaming:** Provides real-time progress updates via WebSockets (see `websocket_guide.md`).
*   **Configurable:** Easily override default LLM models, API keys, and workflow parameters via environment variables (`app/core/config.py`) and API request payloads.
*   **Optional State Persistence:** Can track task status and store final results using Firestore (`firestore_schema.md`).

## 🏛️ Core Architecture Principles

The framework promotes modularity through a clear separation of concerns:

1.  **Agencies (`app/agencies/<agency_name>/`)**: Self-contained units focused on a specific research domain. Each agency typically contains:
    *   `orchestrator.py`: Defines the main workflow and sequence of steps for the agency.
    *   `agents.py`: Implements the specialized LLM-powered agents (e.g., Planner, Writer) used in the orchestration.
    *   `schemas.py`: Defines Pydantic models for the agency's specific data structures and agent outputs.
    *   `prompts.py` (Optional): Stores prompts used by the agents.
    *   `helpers.py` (Optional): Contains utility functions specific to the agency's workflow, often combining calls to shared services.
2.  **Services (`app/services/`, `app/agencies/services/`)**: Reusable, often non-LLM components providing core functionalities like search, scraping, chunking, ranking, etc. These are designed to be stateless and callable by any agency's orchestrator or helpers.
3.  **Core (`app/core/`)**: Contains application-wide configurations (`config.py`), common Pydantic schemas (`schemas.py`), exception handling (`exceptions.py`), and the FastAPI application setup (`main.py`).
4.  **Pydantic Schemas**: Act as the "glue" defining the data contracts between agents, services, and the API layer, ensuring consistency and enabling validation.

## ⚙️ Example Workflow: Deep Research Agency

The included `deep_research` agency (`app/agencies/deep_research/`) serves as an example implementation, orchestrating the following steps:

1.  **Planning:** The `Planner` agent generates a `WritingPlan` and `SearchTasks`.
2.  **Initial Search:** Calls the `SearchService`.
3.  **Initial Reranking:** Uses the `RankingService` (via `helpers.py`) to prioritize results.
4.  **Content Processing:** Calls `ScraperService` (via `helpers.py`), then uses `Summarizer` agent or `ChunkingService` + `RankingService` (via `helpers.py`).
5.  **Initial Writing:** The `Writer` agent creates a draft using processed content, potentially requesting more info via `SearchRequest` tags.
6.  **Refinement Loop:** Executes further searches if requested, processes new content, and uses the `Refiner` agent with *only new information* to update the draft iteratively.
7.  **Final Assembly:** Formats citations and adds a reference list using helper functions.
8.  **Response & Persistence:** Sends the final report and usage stats via WebSocket and saves to Firestore (if configured).

This detailed workflow is specific to the `deep_research` agency; other agencies can implement entirely different processes while reusing the core services.

## 🛠️ Technology Stack

*   **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
*   **Data Validation & Settings:** [Pydantic](https://docs.pydantic.dev/) V2
*   **LLM Interaction:** Primarily OpenAI API client (via libraries like `openai` or potentially routing services like OpenRouter), adaptable for others supporting structured output (JSON mode/Tool Calling). Pydantic enforces output structure.
*   **Multi-Agency Orchestration:** Custom logic or Pydantic-AI within `app/agencies/`
*   **LLM tools** To be implemented with Pydantic-AI in future agencies.
*   **Web Scraping:** [Crawl4AI](https://github.com/extractus/crawl4ai) (via `app/services/scraper/`)
*   **PDF Parsing:** [MarkItDown](https://github.com/microsoft/markitdown) (via Crawl4AI or directly in `app/services/scraping_utils/`)
*   **Web Search:** [Serper API](https://serper.dev/) (via `app/services/search/`)
*   **Reranking:** [Together AI API](https://www.together.ai/) (via `app/agencies/services/ranking.py`)
*   **Chunking:** Custom implementation in `app/services/chunking/`
*   **State Persistence (Optional):** Google Firestore
*   **Language:** Python 3.10+
*   **Async:** `asyncio`

## 💪 Robustness through Pydantic

Pydantic V2 is fundamental to the API's reliability and structure:

*   **API Layer:** Validates incoming requests (`ResearchRequest`) and outgoing WebSocket messages (`WebSocketUpdateHandler`), ensuring schema adherence.
*   **Configuration:** Manages application settings robustly (`app/core/config.py`).
*   **LLM Interaction:** Defines precise Pydantic models (`app/agencies/deep_research/schemas.py`) used as the required output format for LLM agents (e.g., `PlannerOutput`, `WriterOutput`). This allows direct parsing and validation of agent responses, catching errors early.
*   **Internal Data Flow:** Structures data passed between components (e.g., `SearchResult`, `Chunk`, `UsageStatistics` in `app/core/schemas.py`), reducing errors from inconsistent data handling.

## 🧬 Structured Output from LLMs

This project relies heavily on the ability of modern LLMs (like OpenAI's GPT series) to generate output conforming to a specified structure, particularly JSON schemas derived from Pydantic models.

*   **Prompt Engineering:** Prompts for agents (`app/agencies/deep_research/agents.py`) include instructions and often JSON schema definitions or examples to guide the LLM.
*   **API Features:** We utilize LLM API features (e.g., OpenAI's `response_format={"type": "json_object"}` and function/tool calling where appropriate) to encourage structured output.
*   **Validation:** The Pydantic models defined in `schemas.py` act as the final validator. The application attempts to parse the LLM's string output directly into the target Pydantic model. If parsing fails, it indicates the LLM didn't adhere to the requested structure, and appropriate error handling or retries are triggered.

This approach replaces the need for external libraries like LiteLLM solely for managing multiple providers, focusing instead on leveraging provider-specific features for reliable structured output generation guided by Pydantic schemas.

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/odr-api.git # Replace with actual repo URL
    cd odr-api
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    playwright install --with-deps
    ```

## 🔧 Configuration

The API relies on environment variables for configuration, particularly API keys.

1.  Create a `.env` file in the project root directory.
2.  Copy the contents of `.env.example` (you should create this file) into `.env`.
3.  Fill in the required API keys and adjust any other settings:

    ```dotenv
    # .env Example
    LOG_LEVEL=INFO

    # Serper API Key (Required for search)
    SERPER_API_KEY="your_serper_api_key"

    # Together AI API Key (Required for reranking)
    TOGETHER_API_KEY="your_together_api_key"

    # --- LLM API Keys ---
    # Provide keys for the LLM providers you intend to use.
    # Currently uses OpenAI client via OpenRouter, but architecture allows adding others.
    OPENROUTER_API_KEY="your_openai_api_key"

    # --- Firestore Configuration ---
    # Required for task persistence - will work without.
    FIREBASE_SERVICE_ACCOUNT_KEY_JSON="/path/to/your/service-account-key.json"
    ```

## ▶️ Usage

1.  **Run the FastAPI server:**
    ```bash
    uvicorn app.main:app --reload --port 8000
    ```
    The API will be available at `http://localhost:8000`. Access the interactive docs at `http://localhost:8000/docs`.

2.  **Interact via WebSocket:**
    *   The primary endpoint for the deep research agency is `/deep_research/ws/research`.
    *   Connect using a WebSocket client (see `ws_client.py` for a Python example - run with `python ws_client.py }your query here").
    *   Send an initial JSON message matching the `ResearchRequest` schema (`app/core/schemas.py`):
        ```json
        {
          "query": "Your research query here",
          "planner_llm_config": null, // Optional override for agent LLM settings
          "summarizer_llm_config": null, // Optional override
          "writer_llm_config": null, // Optional override
          "refiner_llm_config": null, // Optional override
          // Other config overrides from DeepResearchConfig can go here
          "max_search_tasks": null
        }
        ```
    *   Receive JSON status updates conforming to the structure in `websocket_guide.md`:
        ```json
        {
          "step": "STEP_NAME", // e.g., "PLANNING", "SEARCHING", "RANKING", "PROCESSING", "WRITING", "REFINING", "FINALIZING", "COMPLETE", "ERROR"
          "status": "STATUS", // e.g., "START", "END", "IN_PROGRESS", "SUCCESS", "ERROR", "WARNING", "INFO"
          "message": "Human-readable status message",
          "details": { ... } // Optional dictionary with context (structure varies)
        }
        ```
    *   The final success message (`step: "COMPLETE", status: "END"`) includes final usage statistics in `details`. The actual report content is sent earlier in the `step: "FINALIZING", status: "END"` message's `details`.

3.  **Other Endpoints:**
    *   `/settings`: GET endpoint to view current application settings (excluding secrets).
    *   `/tasks`: GET endpoint to list persisted tasks from Firestore.
    *   `/tasks/{task_id}`: GET endpoint to retrieve details of a specific task.
    *   `/tasks/stop/{task_id}`: POST endpoint to request cancellation of a running task.

## 🤝 Contributing - Build Your Own Agents!

**Contributions are highly encouraged!** This framework is designed to grow. Help us build a diverse ecosystem of powerful research agencies.

**For a detailed guide on creating a new agency from scratch, please see the [Agency Structure & Quickstart Guide](docs/AGENCY_STRUCTURE.md).**

**How to Contribute:**

1.  **Add a New Agency:**
    *   **Create Directory:** Make a new folder `app/agencies/your_agency_name/`.
    *   **Define Components:** Inside, create `__init__.py`, `orchestrator.py`, `agents.py`, and `schemas.py`. Add `prompts.py` or `helpers.py` as needed.
    *   **Implement Logic:**
        *   Write your orchestration flow in `orchestrator.py`.
        *   Define your agent logic in `agents.py`, leveraging LLMs and structured output via Pydantic schemas defined in `schemas.py`.
        *   Reuse core services from `app/services/` (e.g., `SearchService`, `WebScraper`) by importing and calling them in your orchestrator or helpers.
    *   **Define API Endpoint:** Add FastAPI routes (e.g., a WebSocket endpoint) for your new agency in `app/main.py` or by creating a dedicated router in your agency directory and including it in `app/main.py`.
    *   **Add Configuration:** Update `app/core/config.py` if your agency requires specific settings.
    *   **Document:** Add a README or update this one explaining your agency's purpose and workflow.
2.  **Add a New Service:**
    *   Create a new module or directory under `app/services/` (for general services) or potentially `app/agencies/services/` (if strongly tied to agent concepts like ranking).
    *   Implement your service logic (e.g., connecting to a new search API, implementing a data analysis tool).
    *   Ensure it's easily callable, ideally stateless, and potentially asynchronous.
    *   Add necessary configuration to `app/core/config.py`.
3.  **Enhance Existing Components:** Improve agents, services, error handling, add tests, or refine documentation.

**General Contribution Steps:**

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Add tests for your changes (highly recommended!).
5.  Ensure code passes linting and formatting checks (e.g., using Ruff/Black).
6.  Commit your changes (`git commit -m 'Add some feature'`).
7.  Push to the branch (`git push origin feature/your-feature-name`).
8.  Open a Pull Request against the main repository.

**Other Areas for Contribution:**

*   **LLM Support:** Improve compatibility or add configuration options for more LLM providers (especially those with strong structured output support).
*   **Specialized Scrapers:** Add robust scrapers for specific sites or content types in `app/services/scraping_utils/`.
*   **Error Handling & Resilience:** Refine exception handling, retries, and state management across the framework.
*   **Testing:** Add more comprehensive unit, integration, and agent simulation tests.
*   **Documentation:** Improve READMEs, code comments, architecture diagrams, or API documentation (`websocket_guide.md`, `firestore_schema.md`).

## 📜 Citation

This project builds upon concepts and architectures explored in academic research. If you use or extend this work, please consider citing the relevant papers, including:

```bibtex
@misc{alzubi2025opendeepsearchdemocratizing,
      title={Open Deep Search: Democratizing Search with Open-source Reasoning Agents},
      author={Salaheddin Alzubi and Creston Brooks and Purva Chiniya and Edoardo Contente and Chiara von Gerlach and Lucas Irwin and Yihan Jiang and Arda Kaz and Windsor Nguyen and Sewoong Oh and Himanshu Tyagi and Pramod Viswanath},
      year={2025},
      eprint={2503.20201},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.20201},
}
```

## 📧 Contact

Luminary AI Solutions LLC - [info@luminarysolutions.ai](mailto:info@luminarysolutions.ai) - [luminarysolutions.ai](https://luminarysolutions.ai) 