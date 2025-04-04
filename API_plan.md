# API Plan: WebSocket Streaming for Real-time Progress Updates

This document outlines the plan to integrate WebSocket functionality into the FastAPI application to provide clients with real-time updates on the deep research process.

## 1. Goal

-   Replace the current synchronous HTTP request/response cycle for the `/research` endpoint with a WebSocket connection.
-   Stream status updates and intermediate information to the client as the `DeepResearchAgent` progresses through its workflow stages.
-   Enhance user experience by providing transparency into the long-running research task.

## 2. Implementation Strategy

### 2.1. FastAPI Endpoint Modification (`app/main.py`)

-   Define a new WebSocket endpoint (e.g., `/ws/research`).
-   This endpoint will accept the `ResearchRequest` data upon initial connection (perhaps as query parameters or an initial JSON message).
-   It will handle establishing and managing the WebSocket lifecycle (`on_connect`, `on_receive`, `on_disconnect`).
-   Upon connection and receiving the initial request data, it will:
    -   Instantiate the `AppSettings`, `ApiKeys`.
    -   Instantiate the `DeepResearchAgent`.
    -   Start the research process asynchronously.

### 2.2. Agent Modification (`app/core/agent.py`)

-   The `DeepResearchAgent`'s `run_deep_research` method (or a new wrapper method) needs to be adapted to `yield` or `send` status updates back to the calling WebSocket handler.
-   **Callback Mechanism:** Inject the WebSocket connection object or a dedicated sending function into the `DeepResearchAgent` during initialization.
    -   Example: `agent = DeepResearchAgent(settings, api_keys, websocket_send_callback)`
-   **Yielding Updates:** Modify `run_deep_research` to send messages *before* and *after* major steps:
    -   Planning (Start/End + Plan details)
    -   Initial Search (Start/End + Number of results)
    -   Reranking (Start/End + Number of top sources)
    -   Content Fetching/Processing (Start/End + Per-source status: Fetching, Summarizing/Chunking, Done/Failed)
    -   Input Context Management (Filtering status)
    -   Initial Report Generation (Start/End)
    -   Refinement Loop (Start/Iteration N/End + Search query/Results processed)
    -   Final Report Assembly (Start/End)
    -   Completion (Final Report, Stats)
    -   Error Handling (Send error messages if exceptions occur)

### 2.3. Message Format (WebSocket Payloads)

-   Define a consistent JSON structure for messages sent over the WebSocket.
    ```json
    {
      "step": "PLANNING", // PLANNING, SEARCHING, RANKING, PROCESSING, WRITING, REFINING, FINALIZING, ERROR, COMPLETE
      "status": "START", // START, IN_PROGRESS, END, INFO, ERROR
      "message": "Generating search plan...", // Human-readable message
      "details": { ... } // Optional: Step-specific data (e.g., plan, source URL, error details)
    }
    ```
-   **Example Messages:**
    -   `{"step": "PLANNING", "status": "START", "message": "Generating search plan..."}`
    -   `{"step": "PLANNING", "status": "END", "message": "Plan generated.", "details": {"plan": {...}, "tasks": [...]}}`
    -   `{"step": "PROCESSING", "status": "IN_PROGRESS", "message": "Processing source: https://example.com", "details": {"source_url": "https://example.com", "action": "Summarizing"}}`
    -   `{"step": "ERROR", "status": "ERROR", "message": "Failed to fetch source.", "details": {"source_url": "https://example.com", "error": "Timeout"}}`
    -   `{"step": "COMPLETE", "status": "END", "message": "Research complete.", "details": {"report": "...", "usage": {...}}}`

## 3. Considerations

-   **Concurrency:** FastAPI handles WebSocket concurrency well, but ensure agent instantiation and execution are properly managed for simultaneous connections. Each connection should have its own independent agent instance.
-   **Error Handling:** Implement robust error handling within the agent and the WebSocket endpoint. Errors should be sent as specific WebSocket messages, and the connection should be closed gracefully when appropriate.
-   **Client Implementation:** The client receiving these messages will need logic to parse the JSON and update the UI accordingly based on the `step` and `status`.
-   **Security:** Ensure appropriate validation of the initial request data received over the WebSocket connection. Consider authentication/authorization if needed.
-   **Agent Lifecycle:** Manage the lifecycle of the `DeepResearchAgent` instance tied to each WebSocket connection. Ensure resources are cleaned up on disconnect.

## 4. Next Steps

1.  Refactor `app/main.py` to include the `/ws/research` endpoint.
2.  Implement the WebSocket connection handling logic.
3.  Modify `DeepResearchAgent` to accept a callback/sender function.
4.  Integrate status update sending logic within `run_deep_research`.
5.  Define and standardize the WebSocket message JSON schema.
6.  Thoroughly test the WebSocket communication and agent workflow integration. 