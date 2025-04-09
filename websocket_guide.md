# WebSocket API Message Guide (Post-Refactor)

This document outlines the structure and content of messages sent from the backend over the `/deep_research/ws/research` WebSocket connection during a research task, reflecting the **new callback-based architecture**.

## Overall Message Structure

All messages sent from the server **still follow the same fundamental JSON structure**:

```json
{
  "step": "STEP_NAME",
  "status": "STATUS_VALUE",
  "message": "Human-readable status message",
  "details": { ... } // Optional: Contains structured data relevant to the step/status
}
```

-   `step` (string): Represents the major phase or state of the research process.
-   `status` (string): Indicates the specific status within that step (e.g., starting, ending, error).
-   `message` (string): A description suitable for displaying to the user.
-   `details` (object | null): Contains additional structured information. The specific fields within `details` vary depending on the `step` and `status`.

**Key Change:** With the new architecture, the backend orchestrator function now receives a callback and sends *all* progress updates directly via this callback. This means updates are more granular and tied directly to the orchestrator's internal steps.

## ⚠️ Handling Disconnects and Task Persistence

**Important:** By default, the backend research task initiated via this WebSocket connection **continues running even if the client disconnects**. The connection dropping does **not** automatically stop the server-side process.

This has the following implications for the frontend client:

1.  **Store the `task_id`:** The client **must** reliably store the `task_id` received in the `INITIALIZING`/`TASK_ID` message. This ID is the only way to reference the task later.
2.  **No Reconnection for Same Task:** If the WebSocket disconnects, the client **cannot** simply reconnect to the `/deep_research/ws/research` endpoint and expect to receive further updates for the *original* task. A new connection will always start a *new* task instance.
3.  **Use REST for Status/Results:** To check the status or retrieve the final result of a task that may have continued after a disconnect, the client **must** use the stored `task_id` to poll the REST endpoint:
    *   `GET /deep_research/result/{task_id}`
4.  **Interpret REST Response:** The client should examine the JSON response from the GET endpoint:
    *   Check the `status` field (`PENDING`, `PROCESSING`, `COMPLETE`, `ERROR`, `CANCELLED`).
    *   If `PROCESSING`, wait and poll again later.
    *   If `COMPLETE`, the full result (report, usage stats) will be available in the `result` field of the response.
    *   If `ERROR` or `CANCELLED`, handle accordingly, potentially displaying the `error` or `stoppedReason` field.

This polling mechanism ensures the client can retrieve the outcome of long-running tasks even if the initial WebSocket connection is interrupted.

## Possible `step` Values

The `step` field indicates the current phase of the research process. This list reflects the steps explicitly sent by the new callback handler:

-   `INITIALIZING`: Initial connection setup and task acceptance phase (sent from route handler).
-   `STARTING`: Overall research process initiation (first message from orchestrator).
-   `PLANNING`: Generating the research plan and search queries.
-   `SEARCHING`: Executing searches (initial or refinement).
-   `RANKING`: Reranking search results (initial only).
-   `PROCESSING`: Fetching, summarizing, chunking, and reranking content from sources (can occur during initial processing and refinement).
-   `WRITING`: Generating the initial report draft.
-   `REFINING`: Iterative refinement loop (controls refinement-specific searches, processing, and regeneration).
-   `FINALIZING`: Assembling the final report structure and adding references. **The final report content is sent in this step.**
-   `COMPLETE`: The entire process finished successfully. Contains final usage statistics.
-   `ERROR`: An error occurred, potentially terminating the process.

## Possible `status` Values

The `status` field provides more detail within a `step`:

-   `TASK_ID`: Used with `INITIALIZING` to send the assigned Task ID.
-   `START`: Indicates the beginning of a step.
-   `END`: Indicates the successful completion of a step.
-   `IN_PROGRESS`: Indicates the step is actively working (e.g., `PROCESSING` a source, `REFINING` the draft).
-   `SUCCESS`: Used within `PROCESSING` to indicate a single source was processed successfully (summary or chunks).
-   `ERROR`: Indicates an error occurred during the step. This might be recoverable (e.g., a single source failing in `PROCESSING`, an agent failing in `WRITING` or `REFINING`) or fatal.
-   `INFO`: Provides informational updates (e.g., `REFINING`/`INFO` when loop ends because no new info was found).
-   `WARNING`: Indicates a non-critical issue occurred (e.g., `PROCESSING`/`WARNING` if scraping/summarization/chunking fails for a single source).
-   `FATAL`: Used with `step: ERROR` for unrecoverable errors forcing termination (typically sent by the orchestrator wrapper).
-   `VALIDATION_ERROR`: Used with `step: ERROR` if the initial request payload is invalid (sent from route handler).
-   `ORCHESTRATION_ERROR`: Used with `step: ERROR` as a fallback if the orchestrator fails unexpectedly *after* starting but before completing normally (sent from route handler).
-   `CRITICAL_ERROR`: Used with `step: ERROR` as a fallback if the orchestrator fails in a way not caught by its internal handlers (sent from route handler).
-   `HANDLER_ERROR`: Used with `step: ERROR` if the WebSocket *connection handler itself* fails (sent from route handler).


## `details` Structure for Key Messages

The `details` object provides context. The specific fields depend on the `step` and `status`:

---

**1. `step: INITIALIZING`, `status: TASK_ID`**
   - Sent immediately after connection acceptance.
   - `details`:
     ```json
     {
       "task_id": "uuid-string-for-the-task"
     }
     ```

---

**2. `step: STARTING`, `status: START`**
   - First message from the orchestrator indicating the process has begun.
   - `details`: `{}` (Empty object or null)

---

**3. `step: PLANNING`, `status: END`**
   - Research plan generated successfully.
   - `details`:
     ```json
     {
       "plan": {
         // Contains the writing_plan object (serialized) and search_task_count
         "writing_plan": { /* ... */ },
         "search_task_count": 5
       }
     }
     ```

---

**4. `step: PLANNING`, `status: ERROR`**
   - Error during planning.
   - `details`:
     ```json
     {
       "error": "String representation of the exception"
     }
     ```

---

**5. `step: SEARCHING`, `status: START` (Initial)**
   - Starting initial web search.
   - `details`:
     ```json
     {
       "initial_task_count": 5 // Number of planned search tasks
     }
     ```

---

**6. `step: SEARCHING`, `status: END` (Initial)**
   - Initial web search completed.
   - `details`:
     ```json
     {
       "unique_result_count": 45,  // Unique organic results found *before* reranking
       "queries_executed": 5    // Number of tasks sent to search API
     }
     ```
---

**7. `step: RANKING`, `status: START`**
   - Starting reranking of initial results.
   - `details`:
      ```json
      {
          "results_to_rank": 45 // Number of unique results being sent for reranking
      }
      ```
---

**8. `step: RANKING`, `status: END`**
   - Reranking complete, results split for processing.
   - `details`:
     ```json
     {
       "results_for_summary": 10, // Count of top results going to summary path
       "results_for_chunking": 28 // Count of secondary results going to chunking path
     }
     ```

---

**9. `step: PROCESSING`, `status: START` (Overall)**
   - Beginning the phase of fetching/summarizing/chunking sources.
   - `details`:
     ```json
     {
       "urls_to_process": 38 // Total unique URLs identified for processing
     }
     ```

---

**10. `step: PROCESSING`, `status: IN_PROGRESS`**
    - Actively working on a specific source. Message indicates the action.
    - `details`:
      ```json
      {
        "source_url": "http://example.com/article",
        "action": "Fetching" // Or "Summarizing", "Chunking", "Reranking"
        // "chunk_count" might be present if action is "Reranking"
      }
      ```

---

**11. `step: PROCESSING`, `status: SUCCESS`**
    - Successfully processed a single source (either summarized or chunked).
    - `details`:
      ```json
      {
        "source_url": "http://example.com/article",
        "type": "summary" // Or "chunks"
        // "relevant_chunk_count" will be present if type is "chunks"
      }
      ```

---

**12. `step: PROCESSING`, `status: WARNING`**
    - Non-critical error processing a single source (e.g., scraping failed, summarizer failed, no relevant chunks). Processing continues for other sources.
    - `details`:
      ```json
      {
        "source_url": "http://example.com/article",
        "reason": "Scraping failure: Timeout" // Or "Summarizer error: ...", etc.
      }
      ```
---

**13. `step: PROCESSING`, `status: END` (Overall)**
    - Finished processing all designated sources.
    - `details`:
      ```json
      {
        "processed_source_count": 35, // Count of unique sources successfully processed (added to unique_sources map)
        "total_context_items": 88   // Total number of summaries + relevant chunks gathered
      }
      ```
---

**14. `step: WRITING`, `status: START`**
    - Starting the initial draft generation.
    - `details`:
      ```json
      {
        "context_item_count": 88 // Number of summaries/chunks passed to the writer
      }
      ```
---

**15. `step: WRITING`, `status: END`**
    - Initial draft generated.
    - `details`:
      ```json
      {
        "requested_searches_count": 2 // Number of searches requested by the writer (0 if none)
      }
      ```
---

**16. `step: REFINING`, `status: START`**
    - Starting a specific refinement iteration.
    - `details`:
        ```json
        {
            "iteration": 1
        }
        ```
---

**17. `step: SEARCHING`, `status: START` (Refinement)**
    - Starting a search requested during refinement. Message includes iteration & query.
    - `details`:
      ```json
      {
        "iteration": 1,
        "query": "Specific query requested by agent"
      }
      ```

---

**18. `step: SEARCHING`, `status: END` (Refinement)**
    - Refinement search complete.
    - `details`:
      ```json
      {
        "iteration": 1,
        "new_result_count": 15 // New, unique results found this iteration
      }
      ```
---

**19. `step: PROCESSING`, `status: START` (Refinement)**
    - Starting processing of sources found during refinement search.
    - `details`:
      ```json
      {
        "iteration": 1,
        "sources_to_process": 3 // Number of new sources selected for processing
      }
      ```
---

**20. `step: PROCESSING`, `status: END` (Refinement)**
    - Finished processing sources for this refinement iteration.
    - `details`:
      ```json
      {
        "iteration": 1,
        "new_context_items": 12 // Number of new chunks added to context this iteration
      }
      ```
---

**21. `step: REFINING`, `status: IN_PROGRESS` (Refiner Call Start/End)**
    - Calling the refiner LLM or indicating it finished. Message clarifies which.
    - `details`:
      ```json
      {
        "iteration": 1
        // If Refiner LLM just finished:
        // "requested_searches_count": 0
      }
      ```
---

**22. `step: REFINING`, `status: END` or `status: INFO` (Loop End)**
    - The entire refinement loop has finished for a given reason.
    - `details`:
      ```json
      {
        "iteration": 1, // The final iteration number run
        "reason": "Completed normally" // Or "Max iterations reached", "No new info found"
      }
      ```
---

**23. `step: FINALIZING`, `status: START`**
    - Starting final report assembly (citation processing, adding references).
    - `details`: `{}` (Empty object or null)

---

**24. `step: FINALIZING`, `status: END`**
    - Final report assembly complete. **Contains the final report content.**
    - `details`:
      ```json
      {
        "final_report": "# Final Report Title\n\nReport content in Markdown..."
      }
      ```

---

**25. `step: COMPLETE`, `status: END`**
    - Orchestration finished successfully. **Contains usage stats, NOT the report.**
    - `details`:
      ```json
      {
        "final_report_length": 15032, // Character count of the final report
        "usage": { /* Full UsageStatistics object */ }
      }
      ```
---

**26. `step: ERROR`, `status: FATAL` / `ERROR` / `VALIDATION_ERROR` / etc.**
    - An error occurred. Status indicates severity/origin.
    - `details`: Often includes:
      ```json
      {
        "error": "String representation of the error/exception",
        "error_type": "ExceptionClassName" // e.g., AgentExecutionError, ValidationError
        // "error_id" might be present for specific handler errors needing correlation
      }
      ```

---

This updated guide reflects the messages sent by the `WebSocketUpdateHandler` based on the current orchestrator flow. Frontend developers should adapt their logic to handle these specific `step`/`status` combinations and extract data from the `details` payload accordingly, noting especially the changes to how the final report and usage statistics are delivered.

## Connection Stability & Keepalive (Ping/Pong)

WebSocket connections can be closed by intermediate servers or proxies if they remain idle for too long (e.g., 30-120 seconds). During long-running operations on the server (like LLM calls), no actual data messages might be sent, making the connection appear idle.

To prevent premature disconnection, WebSocket protocols use **keepalive** mechanisms, typically involving periodic **ping** and **pong** messages:

-   One side (often the client, like the Python `websockets` library by default) sends a ping message at regular intervals (e.g., every 20 seconds).
-   The other side (the server, like FastAPI/Uvicorn) automatically responds with a pong message.
-   This exchange signals to all intermediaries that the connection is still active, even if no application data is flowing.

**Client-Side:** The `websockets` library used in the example `ws_client.py` handles sending pings automatically by default. You can explicitly configure the `ping_interval` and `ping_timeout` when connecting if needed.

**Server-Side:** FastAPI/Uvicorn typically handles responding to pings automatically.

**Potential Issues:** If you experience connection drops during long tasks despite this mechanism, check for low idle timeout settings on any network proxies, load balancers, or firewalls between the client and the server.