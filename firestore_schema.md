# Firestore `research_tasks` Collection Schema

This document describes the structure of documents stored within the `research_tasks` collection in Firestore. This collection persists the state and results of deep research tasks initiated via the WebSocket endpoint.

## Document ID

The ID for each document is the unique `task_id` (UUID string) assigned when the WebSocket connection is established.

## Fields

Each document contains the following fields:

-   **`taskId`** (String)
    -   Description: The unique identifier for the research task (matches the document ID).
    -   Example: `"a1b2c3d4-e5f6-7890-1234-567890abcdef"`

-   **`query`** (String)
    -   Description: The original user query that initiated the research task.
    -   Example: `"What are the latest advancements in quantum computing for drug discovery?"`

-   **`status`** (String)
    -   Description: The current status of the research task.
    -   Possible Values:
        -   `"PENDING"`: Task created, waiting for orchestration to start.
        -   `"PROCESSING"`: Orchestration is actively running.
        -   `"COMPLETE"`: Orchestration finished successfully.
        -   `"ERROR"`: An error occurred during orchestration or setup, preventing completion.
        -   `"CANCELLED"`: Task was cancelled by user request via the `/stop/{task_id}` endpoint.

-   **`createdAt`** (Timestamp)
    -   Description: Server timestamp indicating when the task document was initially created.

-   **`startedAt`** (Timestamp, Optional)
    -   Description: Server timestamp indicating when the task status was moved to `PROCESSING`.
    -   *Note: May not be present if the task failed before starting.* 

-   **`completedAt`** (Timestamp, Optional)
    -   Description: Server timestamp indicating when the task reached a final state (`COMPLETE` or `ERROR`).
    -   *Note: May not be present if the task is still `PENDING` or `PROCESSING`, or was `CANCELLED` before completion.*

-   **`updatedAt`** (Timestamp, Optional)
    -   Description: Server timestamp indicating the last time the document was updated (e.g., status change, cancellation). 
    -   *Note: Often updated alongside `completedAt` or when status changes to `CANCELLED`.*

-   **`stoppedReason`** (String, Optional)
    -   Description: A brief reason why the task was stopped, typically present only if `status` is `CANCELLED`.
    -   Example: `"Cancelled by user request via API."` or `"Cancelled during execution."`

-   **`error`** (String, Optional)
    -   Description: A detailed error message if the task `status` is `ERROR`.
    -   Example: `"Orchestration failed (ID: ...): AgentExecutionError - Planner agent failed: ..."`

-   **`result`** (Map/Object, Optional)
    -   Description: Contains the final output of the research task. Present only if `status` is `COMPLETE`.
    -   Structure:
        -   **`report`** (String): The final generated report in Markdown format.
        -   **`usage_statistics`** (Map/Object): Detailed resource usage statistics (tokens, cost, API calls). See structure below.

-   **`raw_overrides`** (Map/Object, Optional) - *Potential Field*
    -   Description: If implemented, this could store the raw override parameters provided in the initial `ResearchRequest` for debugging or reference.
    -   *Note: Currently commented out in the `routes.py` example, but could be added.* 

### `result.usage_statistics` Structure

This nested map follows the `UsageStatistics` schema (`app.core.schemas.UsageStatistics`):

```json
{
  "token_usage": {
    "planner": { "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ... },
    "summarizer": { ... }, // Aggregated from all summarizer calls
    "writer_initial": { ... }, // Usage for the initial draft
    "refiner_loop_1": { ... }, // Usage for the first refinement loop call
    "refiner_loop_2": { ... }, // (if applicable)
    // ... other agent roles or specific loop tags
    "total": { ... } // Overall token usage
  },
  "estimated_cost": {
    "planner": 0.0,
    "summarizer": 0.0,
    "writer_initial": 0.0,
    "refiner_loop_1": 0.0,
    // ... other roles/loops
    "total": 0.0 // Overall estimated cost
  },
  "serper_queries_used": 5,
  "sources_processed_count": 35, // Count of unique sources processed (summarized or chunked)
  "refinement_iterations_run": 1
}
```

*Note: Timestamps are stored as Firestore Timestamp objects and will typically be serialized as ISO 8601 date strings (e.g., `"2023-10-27T10:00:00.123456Z"`) when fetched via the HTTP API.* 