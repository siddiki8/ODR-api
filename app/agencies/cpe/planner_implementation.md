# Planner Implementation for Company Profile Extractor (CPE)

This document outlines how to implement the Planner agent for the Company Profile Extractor (CPE) agency, following patterns from the `deep_research` agency. The goal is to generate search tasks that identify multiple potential companies matching a user query and location.

## 1. Update Request Schema (`schemas.py`)

Modify `CPERequest` to include `query`, `location`, and `max_search_tasks`:

```python
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional

class CPERequest(BaseModel):
    query: str = Field(..., description="User query describing the types of companies to find.")
    location: Optional[str] = Field(None, description="Optional geographic location to filter companies.")
    max_search_tasks: Optional[int] = Field(
        default=10, 
        description="Maximum number of search tasks the planner should generate.",
        ge=1, le=20 # Example bounds, adjust as needed
    )
    # Remove start_urls if planning is the only entry point
    # start_urls: List[HttpUrl] = Field(..., description="List of company homepage URLs to extract profiles from") 
```

## 2. Define Planner Output Schema (`schemas.py`)

Add `CPEPlannerOutput` to validate the planner's JSON response:

```python
from app.core.schemas import SearchTask # Use the core SearchTask schema

class CPEPlannerOutput(BaseModel):
    """Validates structured JSON output expected from the Planner LLM."""
    search_tasks: List[SearchTask] = Field(
        ..., 
        description="List of search tasks designed to find websites of companies matching the query and location.", 
        min_items=1
    )
```

## 3. Prompts (`prompts.py`)

Define prompts to guide the planner LLM:

### System Prompt (`CPE_PLANNER_SYSTEM_PROMPT`)
```
You are an expert planning agent specialized in identifying companies.
Given a user query describing the desired type of companies and an optional location, generate a list of diverse SearchTask objects aimed at finding official websites and contact information for *multiple* potential companies matching the criteria. 

Focus on creating targeted search queries that combine the user query and location in different ways (e.g., "[query] companies in [location]", "[query] contact page [location]", "best [query] firms near [location]").

Each SearchTask MUST include:
- query: The specific search string to execute.
- endpoint: Use "/search" for general web searches.
- num_results: Request a reasonable number (e.g., 5 or 10) to increase the chance of finding relevant company URLs.
- reasoning: Briefly explain why this specific search query is likely to find relevant company websites.

Adhere strictly to the number of search tasks requested (maximum {max_search_tasks}).

Output *only* a valid JSON object conforming exactly to the `CPEPlannerOutput` schema. Do not include any explanations or commentary outside the JSON structure.
```

### User Prompt Template (`CPE_PLANNER_USER_TEMPLATE`)
```
Generate search tasks to find companies based on the following criteria:
Query: {query}
Location: {location}
Maximum Search Tasks: {max_search_tasks}
```

## 4. Agent Factory (`agents.py`)

Implement the agent creation function:

```python
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from .config import CPEConfig
from .schemas import CPEPlannerOutput # Import the new output schema
from .prompts import CPE_PLANNER_SYSTEM_PROMPT, CPE_PLANNER_USER_TEMPLATE

# Make sure create_llm_model is defined or imported
# from .agents import create_llm_model 

def create_planner_agent(config: CPEConfig) -> Agent[CPEPlannerOutput]:
    """Creates the CPE Planner Agent instance."""
    model = create_llm_model(config.planner_model_id)
    return Agent[CPEPlannerOutput](
        model=model,
        system_prompt=CPE_PLANNER_SYSTEM_PROMPT, # Use the updated system prompt
        # Note: The user template is usually passed during the .run() call, not here
        result_type=CPEPlannerOutput,
        retries=2 # Or config.planner_retries
    )
```

Update `CPEAgents` to include `planner: Any` and initialize it in `get_cpe_agents`.

## 5. Integrate into Orchestrator (`orchestrator.py`)

Modify `run_cpe` to use the planner:

```python
import logging
# ... other imports
from .schemas import CPERequest, CPEResponse, CPEPlannerOutput # Add CPEPlannerOutput
from .prompts import CPE_PLANNER_USER_TEMPLATE # Add planner user template
from ..services.search import execute_search_queries # Assuming you have this helper
from app.core.schemas import RunUsage # For usage tracking

logger = logging.getLogger(__name__)

async def run_cpe(request: CPERequest, config: CPEConfig) -> CPEResponse:
    agents = get_cpe_agents(config)
    usage_tracker = RunUsage() # Initialize usage tracker
    profiles: List[CompanyProfile] = []
    all_search_results: List[SearchResult] = [] # To store results from all tasks

    logger.info(f"Starting CPE planning for query: '{request.query}', location: '{request.location}'")
    
    # === Step 1: Planning ===
    try:
        # Format the user prompt with actual request values
        planner_user_prompt = CPE_PLANNER_USER_TEMPLATE.format(
            query=request.query,
            location=request.location or "Not specified",
            max_search_tasks=request.max_search_tasks
        )
        
        # Format the system prompt with max_search_tasks if needed (or handle in agent logic)
        # Example: system_prompt = CPE_PLANNER_SYSTEM_PROMPT.format(max_search_tasks=request.max_search_tasks)

        planner_result = await agents.planner.run(planner_user_prompt)
        # usage_tracker.update_agent_usage("planner", planner_result.usage()) # Add usage tracking

        planner_output: CPEPlannerOutput = planner_result.data
        search_tasks = planner_output.search_tasks
        logger.info(f"Planner generated {len(search_tasks)} search tasks.")
        # Add callback/logging for planning completion

    except Exception as e:
        logger.error(f"Planner agent failed: {e}", exc_info=True)
        # Add error callback
        # Return error response
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics()) # Example error return

    # === Step 2: Execute Search Tasks ===
    logger.info(f"Executing {len(search_tasks)} search tasks...")
    # Add search start callback
    try:
        # Use a helper similar to deep_research's execute_search_queries
        # This helper would handle calling Serper/search service for all tasks
        # search_results_map = await execute_search_queries(search_tasks, serper_config, logger)
        # usage_tracker.increment_serper_queries(len(search_tasks))
        
        # TEMP Placeholder: Assume search_results_map is Dict[str, List[SearchResult]]
        search_results_map = {} # Replace with actual search execution

        # Flatten results and deduplicate links
        seen_links = set()
        for task_query, results in search_results_map.items():
            for result in results:
                if result.link not in seen_links:
                    all_search_results.append(result)
                    seen_links.add(result.link)
        logger.info(f"Search tasks yielded {len(all_search_results)} unique results.")
        # Add search end callback

    except Exception as e:
        logger.error(f"Search execution failed: {e}", exc_info=True)
        # Add error callback
        # Return error response
        return CPEResponse(profiles=[], usage_statistics=usage_tracker.get_statistics())

    # === Step 3: Scrape & Extract (Loop through unique results/links) ===
    # The rest of the flow (scraping, email finding, grouping, extraction)
    # needs to be adapted to handle the list of URLs from `all_search_results`
    # instead of just the initial `start_urls`.
    # You might group results by domain first, then process each domain.

    logger.info("Processing search results...")
    start_urls_from_search = [result.link for result in all_search_results]

    # <<< INSERT ADAPTED SCRAPING/EXTRACTION LOGIC HERE >>>
    # Example: Loop through start_urls_from_search, call find_emails_deep, group, aggregate, extract...
    # Remember to update usage_tracker for extractor agent calls.

    # === Step 4: Final Assembly ===
    final_usage_stats = usage_tracker.get_statistics()
    # Add final callback
    logger.info(f"CPE run complete. Extracted {len(profiles)} profiles.")
    return CPEResponse(profiles=profiles, usage_statistics=final_usage_stats.model_dump())

```

---

This revised plan aligns the CPE planner with the `deep_research` pattern, focusing on generating multiple `SearchTask` objects based on a descriptive query and location to discover several relevant companies. 