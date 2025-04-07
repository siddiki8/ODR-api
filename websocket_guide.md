# WebSocket API Message Guide

This document outlines the structure and content of messages sent from the backend over the `/ws/research` WebSocket connection during a research task.

## Overall Message Structure

All messages sent from the server follow this JSON structure:

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
-   `details` (object | null): Contains additional structured information. Can be an empty object `{}` or `null`. The specific fields within `details` vary depending on the `step` and `status`.

## Possible `step` Values

The `step` field indicates the current phase of the research process:

-   `INITIALIZING`: Agent setup phase.
-   `STARTING`: Overall research process initiation.
-   `PLANNING`: Generating the research plan and search queries.
-   `SEARCHING`: Executing searches (initial or refinement).
-   `RANKING`: Reranking search results.
-   `PROCESSING`: Fetching, summarizing, chunking, and reranking content from sources.
-   `FILTERING`: Assembling context items (summaries/chunks) for the writer.
-   `WRITING`: Generating the initial report draft.
-   `REFINING`: Iterative refinement loop (searching, processing new info, regenerating draft).
-   `FINALIZING`: Assembling the final report structure, including adding references. **The final report content is sent in this step.**
-   `COMPLETE`: The entire process finished successfully. Contains final usage statistics.
-   `ERROR`: An error occurred, potentially terminating the process.

## Possible `status` Values

The `status` field provides more detail within a `step`:

-   `START`: Indicates the beginning of a step.
-   `END`: Indicates the successful completion of a step.
-   `IN_PROGRESS`: Indicates the step is actively working (often used for long-running tasks like `PROCESSING` or `REFINING`).
-   `SUCCESS`: Used within `PROCESSING` to indicate a single source was processed successfully.
-   `ERROR`: Indicates an error occurred during the step. This might be recoverable (e.g., a single source failing in `PROCESSING`) or fatal (e.g., `INITIALIZING`/`ERROR`).
-   `INFO`: Provides informational updates, often within `REFINING` (e.g., "No further search requested") or `FINALIZING` (e.g., "Using latest draft due to assembly error.").
-   `WARNING`: Indicates a non-critical issue occurred (e.g., `PROCESSING`/`WARNING` if chunking yields no chunks).
-   `FATAL`: Used with `step: ERROR` for unrecoverable errors forcing termination.

## `details` Structure for Key Messages

The `details` object provides context. Here are examples for important messages:

---

**1. `step: INITIALIZING`, `status: ERROR`**
   - Indicates agent setup failed.
   - `details`:
     ```json
     {
       "error": "Error message string" // E.g., "Configuration Error: API Key missing"
     }
     ```

---

**2. `step: PLANNING`, `status: END`**
   - Research plan generated.
   - `details`:
     ```json
     {
       "plan": {
         "writing_plan": { // The full writing plan object (potentially large)
             "overall_goal": "...",
             "desired_tone": "...",
             "sections": [ { "title": "...", "guidance": "..." }, ... ],
             "additional_directives": [ "..." ]
          },
         "search_task_count": 3, // Number of initial search queries generated
         "search_queries": [ "query 1", "query 2", "..." ] // List of initial queries
       }
     }
     ```

---

**3. `step: SEARCHING`, `status: END` (Initial Search)**
   - Initial web search completed.
   - `details`:
     ```json
     {
       "raw_result_count": 30, // Total organic results parsed from API response
       "queries_executed": 3 // Number of search tasks successfully executed
     }
     ```

---

**4. `step: RANKING`, `status: END`**
   - Reranking of initial results is complete.
   - `details`:
     ```json
     {
       // No specific data fields defined currently, message contains counts
     }
     ```
     *(Note: Previously sent counts here, now only in `message`)*

---

**5. `step: PROCESSING`, `status: IN_PROGRESS` (While processing a source)**
   - Provides context about the source being processed.
   - `details`:
     ```json
     {
       "source_url": "http://example.com/article",
       "action": "Fetching" // Or "Summarizing", "Chunking", "Reranking"
     }
     ```
     *(Note: Refinement processing messages include `[Refinement]` prefix in the main `message` field)*

---

**6. `step: PROCESSING`, `status: SUCCESS` (Single source processed)**
   - Indicates one source was fetched and summarized/chunked successfully.
   - `details`:
     ```json
     {
       "source_url": "http://example.com/article"
     }
     ```

---

**7. `step: PROCESSING`, `status: ERROR` (Single source failed)**
   - Indicates an error occurred while processing one source.
   - `details`:
     ```json
     {
       "source_url": "http://example.com/article",
       "error": "ScrapingError" // Or "LLMError", "ValidationError", etc. (The exception type name)
     }
     ```

---

**8. `step: FILTERING`, `status: END`**
   - Context assembly for the writer is complete.
   - `details`:
     ```json
     {
       // No specific data fields defined currently, message contains counts
     }
     ```
     *(Note: Previously sent counts here, now only in `message`)*

---

**9. `step: REFINING`, `status: INFO`, `message: "Refinement search requested..."`**
   - The agent decided to perform a search during refinement.
   - `details`:
     ```json
     {
       // No specific data fields defined currently, message contains the query
     }
     ```
   *(Note: The final report is NOT sent during the REFINING step.)*

---

**10. `step: FINALIZING`, `status: END`**
   - Final report assembly is complete. **This message contains the full final report.**
   - `details`:
      ```json
      {
        "final_report": "# Final Report Title\n\nReport content..." // String containing the full Markdown report
      }
      ```

---

**11. `step: COMPLETE`, `status: END`**
    - The entire research process finished successfully. **This message contains usage statistics but NOT the final report content.**
    - `details`:
      ```json
      {
        "final_report_length": 37066, // Character count of the final report
        "usage": { // Usage statistics object
            "token_usage": {
                "planner": { "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ... },
                "summarizer": { ... },
                "writer": { ... },
                "refiner": { ... },
                "total": { ... }
            },
            "estimated_cost": {
                "planner": 0.0,
                "summarizer": 0.0,
                "writer": 0.0,
                "refiner": 0.0,
                "total": 0.0
            },
            "serper_queries_used": 4,
            "sources_processed_count": 32,
            "refinement_iterations_run": 2
        }
        // Note: "final_report" content is NOT included here. Look for it in the FINALIZING/END message.
      }
      ```

---

**12. `step: ERROR`, `status: ERROR` or `status: FATAL`**
    - A potentially fatal error occurred.
    - `details`: Often includes:
      ```json
      {
        "error_type": "AgentExecutionError", // The exception type name
        "error_id": "uuid-string-here" // A unique ID for server-side log correlation
        // Sometimes specific error messages might be included if safe to expose
      }
      ```

This guide should help in building a client application that can effectively track the progress and handle the results of the research process. Remember that the `message` field is the primary source for human-readable status updates.

## Connection Stability & Keepalive (Ping/Pong)

WebSocket connections can be closed by intermediate servers or proxies if they remain idle for too long (e.g., 30-120 seconds). During long-running operations on the server (like LLM calls), no actual data messages might be sent, making the connection appear idle.

To prevent premature disconnection, WebSocket protocols use **keepalive** mechanisms, typically involving periodic **ping** and **pong** messages:

-   One side (often the client, like the Python `websockets` library by default) sends a ping message at regular intervals (e.g., every 20 seconds).
-   The other side (the server, like FastAPI/Uvicorn) automatically responds with a pong message.
-   This exchange signals to all intermediaries that the connection is still active, even if no application data is flowing.

**Client-Side:** The `websockets` library used in the example `gradio_client.py` handles sending pings automatically by default. You can explicitly configure the `ping_interval` and `ping_timeout` when connecting if needed.

**Server-Side:** FastAPI/Uvicorn typically handles responding to pings automatically.

**Potential Issues:** If you experience connection drops during long tasks despite this mechanism, check for low idle timeout settings on any network proxies, load balancers, or firewalls between the client and the server.