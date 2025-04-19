# Company Profile Extractor (CPE) Agency Specification

This document outlines the directives for implementing a new "Company Profile Extractor (CPE)" agency, following the conventions and best practices established in the `deep_research` agency. Updates to `emailfinder.py` are allowed; do not modify `scraper.py`.

## 1. Directory & File Structure

Create new directory `app/agencies/cpe/` with the following files:
- `config.py`
- `schemas.py`
- `prompts.py`
- `agents.py`
- `helpers.py`
- `orchestrator.py`

## 2. Schemas (`schemas.py`)

Define Pydantic models:
```python
from pydantic import BaseModel, HttpUrl, EmailStr
from typing import List, Optional

class CompanyProfile(BaseModel):
    domain: HttpUrl
    name: str
    description: str
    location: Optional[str]
    contact_page: Optional[HttpUrl]
    emails: List[EmailStr]
```

## 3. Prompts (`prompts.py`)

- **Extractor System Prompt**: instruct the LLM to parse concatenated HTML into a `CompanyProfile` JSON object, adhering exactly to the schema.
- **Formatter System Prompt** (optional): for human-readable output formats (Markdown tables or enriched JSON).

## 4. Agents (`agents.py`)

Mirror `deep_research/agents.py` structure:

1. `create_planner_agent(config)` → maps input (domains or search terms) to List[`SearchTask`].
2. **Email Finder Step**: wrap `find_emails_deep` from `emailfinder.py`, then group and aggregate HTML per domain.
3. `create_extractor_agent(config)` → `Agent[CompanyProfile]` with system prompt, `result_type=CompanyProfile`, `retries=3`.
4. `create_formatter_agent(config)` (optional) → format list of `CompanyProfile` objects into Markdown or JSON.

Include a result validator (via decorator) to ensure each output matches the `CompanyProfile` schema.

## 5. Helpers (`helpers.py`)

Implement utility functions:
```python
def group_by_domain(pages: List[EmailPageResult]) -> Dict[str, List[EmailPageResult]]:
    # bucket pages by parsed domain
    ...

def aggregate_html(pages: List[EmailPageResult], max_bytes: int = 50000) -> str:
    # concatenate page.html snippets, truncated per page to avoid token limits
    ...
``` 

## 6. Orchestrator Flow (`orchestrator.py`)

Follow the `run_deep_research_orchestration` blueprint:

1. **Planning**: use Planner agent to generate domain list or search tasks.  
2. **Search**: execute all search tasks with `SearchService`.  
3. **Scraping**: fetch pages with `WebScraper` (no changes to `scraper.py`).  
4. **Email Finder & Grouping**:
   - call `find_emails_deep(start_url, ...)` from updated `emailfinder.py`
   - group pages by domain and aggregate HTML blobs
5. **Extraction**: call Extractor agent for each domain's HTML blob → produces `CompanyProfile`.
6. **Formatting**: call Formatter agent (if needed) or assemble JSON list.
7. **Final Assembly**: combine all `CompanyProfile` objects into the final response and attach usage stats.
8. **Callbacks & Persistence**: integrate `update_callback` (WebSocket) and optional Firestore updates exactly as in `deep_research`.

## 7. Logging & Usage Tracking

- Use `RunUsage` to track LLM calls, token usage, crawler pages, and extracted profiles count.  
- Log entry/exit at each major step with clear messages, following the patterns in `deep_research/orchestrator.py`.

## 8. Emailfinder.py Updates

After `find_emails_deep` returns `List[EmailPageResult]`, implement in `emailfinder.py`:
```python
# group and aggregate
from .emailfinder import find_emails_deep
from ..agents.cpe.helpers import group_by_domain, aggregate_html

def extract_profiles_for_url(start_url: str):
    pages, crawled_urls = await find_emails_deep(start_url, max_depth=1, max_pages=50)
    domain_groups = group_by_domain(pages)
    profiles = []
    for domain, pages in domain_groups.items():
        html_blob = aggregate_html(pages)
        # call extractor agent with html_blob to get CompanyProfile
        profile = await agents_collection.extractor.run(html_blob)
        profiles.append(profile.data)
    return profiles
```

> **Important**: Do not alter `scraper.py`; all new logic for grouping, aggregation, and LLM extraction must live in `emailfinder.py`, your new `agents.py`, or the `helpers.py` in the CPE agency.

## Updates

- **Step 1**: Implemented `config.py` (`CPEConfig`) in `app/agencies/cpe/` for agency-specific settings.
- **Step 2**: Created `schemas.py` defining `CompanyProfile`, `CPERequest`, and `CPEResponse` models.
- **Step 3**: Added `helpers.py` with `group_by_domain` and `aggregate_html` utilities for HTML aggregation and domain grouping.
- **Step 4**: Created `prompts.py` containing `EXTRACTOR_SYSTEM_PROMPT` and `EXTRACTOR_USER_MESSAGE_TEMPLATE` for company profile extraction.
- **Step 5**: Implemented `agents.py`, defining `create_extractor_agent` and `get_cpe_agents` to initialize the LLM extractor.
- **Step 6**: Built `orchestrator.py` with `run_cpe`, wiring `find_emails_deep`, grouping, HTML aggregation, and extractor agent calls.
- **Step 7**: Added `test_cpe.py` as a standalone script to load `CPEConfig`, invoke `run_cpe`, and print the JSON response for testing. 