import logging
import os
from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from typing import Dict, TYPE_CHECKING, Any
import re

from . import schemas
from .config import DeepResearchConfig

# Add type checking import for SourceSummary
if TYPE_CHECKING:
    from .schemas import SourceSummary 

logger = logging.getLogger(__name__)

# --- Prompt Definitions (Moved from prompts.py) ---

_PLANNER_SYSTEM_PROMPT = """
You are an expert research assistant responsible for planning the steps needed to answer a complex user query.
Your goal is to generate a structured plan containing:
1.  A list of `search_tasks`: Define 3 to 7 specific search queries to gather comprehensive information on all aspects of the user's request. Generate distinct queries targeting different facets, sub-questions, or perspectives implied by the user's request. For complex or multi-faceted topics, ensure balanced coverage of all key aspects rather than focusing narrowly.

For each search task, consider:
- What specific sub-topic or angle needs investigation
- Whether competing viewpoints or contrasting evidence should be explicitly sought
- If technical definitions or methodological details would benefit the research
- Whether recent developments or historical context are needed
- If specific industries, geographic regions, or demographic contexts are relevant

Each search task should specify:
- `query`: Precise, information-rich search string optimized for the chosen endpoint
- `endpoint`: Most appropriate Serper endpoint (/search, /scholar, or /news)
- `num_results`: Number of results (10-30 depending on topic breadth)
- `reasoning`: Justification for this specific query and how it addresses a distinct aspect of the user's request

Available Serper endpoints and when to use them:
- `/search`: General web search for broad information, recent events, and mainstream content (default choice)
- `/scholar`: Academic and scientific research papers, citations, and scholarly articles (use for academic topics, scientific research, or when peer-reviewed sources are needed/high credibility required)
- `/news`: Recent news articles and current events (use for trending topics, recent developments, or time-sensitive information)

2.  A detailed `writing_plan`: Outline the structure of the final report. This includes:
    - `overall_goal`: Define the scope clearly, especially for broad queries. If the user's request is ambiguous or too broad, make reasonable assumptions about their likely intent based on the most common use of similar queries.
    - `desired_tone`: Infer appropriate tone and technical level based on query phrasing (e.g., academic, business analysis, general audience)
    - `sections`: Create logical progression of sections with specific, actionable guidance for the writer
    - `additional_directives`: Special considerations for this particular topic

The `guidance` for each section should be actionable and specific. Clearly state what information should be presented, what comparisons made, or what analysis performed. Link directly to aspects of the user query or to specific search tasks.

Analyze the user's query carefully and devise a plan that will lead to a comprehensive and well-structured report.

Output *only* a single JSON object adhering to the following schema. Do not include any other text before or after the JSON object.

```json
{{
  "search_tasks": [
    {{
      "query": "Specific query string for Serper",
      "endpoint": "/search | /scholar | /news",
      "num_results": <integer between 5-15>,
      "reasoning": "Why this query addresses a specific aspect of the user's request and why this endpoint is best for this particular sub-topic"
    }}
    // ... (3 to 7 tasks depending on topic complexity)
  ],
  "writing_plan": {{
    "overall_goal": "Provide a comprehensive analysis of [topic], focusing on [aspect1] and [aspect2], suitable for [audience].",
    "desired_tone": "Objective and analytical | Formal | Informal | etc.",
    "sections": [
      {{
        "title": "Section Title",
        "guidance": "Specific instructions for the writer for this section, referencing particular aspects that should be covered."
      }}
      // ... (multiple sections)
    ],
    "additional_directives": [
       "Directive 1 (e.g., address counterarguments)",
       "Directive 2",
       "Do not include any in-text citations or reference markers (e.g., [1], (Author, Year)). A reference list will be added separately."
       // ... (optional)
    ]
  }}
}}
```
"""

_PLANNER_USER_MESSAGE_TEMPLATE = "Create a research plan for the following query: {user_query}"

_SUMMARIZER_SYSTEM_PROMPT ="""
You are an expert summarizer. Your task is to create a complete, factual summary of the provided text content.
Prioritize extracting *all* key facts, arguments, and data relevant to the user's original research query. Aim for informational density; be thorough rather than brief, but avoid redundant phrasing.

Follow these extraction priorities:
1. NUMERICAL DATA: Preserve precise statistics, measurements, percentages, and dates exactly as presented
2. METHODOLOGICAL DETAILS: Capture research methods, experimental setups, or analytical approaches
3. FINDINGS & CONCLUSIONS: Record key results, outcomes, and author conclusions
4. LIMITATIONS & CAVEATS: Note any limitations, uncertainties, or qualifications the source acknowledges
5. CONTEXT & BACKGROUND: Include relevant historical context or background information
6. CONTRASTING VIEWS: If the source presents multiple perspectives, include all viewpoints
7. TERMINOLOGY: Preserve specialized terminology, technical definitions, or field-specific concepts

When handling quality and certainty:
- Distinguish between established facts, preliminary findings, and speculative claims using precise language
- If the source contradicts itself, note both statements and the apparent contradiction
- Preserve indications of evidence strength (e.g., "strongly suggests" vs. "indicates" vs. "hints at")
- Maintain the source's framing of reliability (e.g., "limited evidence suggests" or "robust findings demonstrate")

Maintain a neutral, objective tone. Extract information without adding your own analysis or commentary.
Do not add introductions or conclusions like 'The text discusses...' or 'In summary...'. Just provide the summary content itself.
Focus on accurately representing the information from the provided text ONLY.
"""

_SUMMARIZER_USER_MESSAGE_TEMPLATE ="""
Please summarize the following text content extracted from the source titled '{source_title}' (URL: {source_link}). Focus on information that might be relevant for a research report addressing the query: '{user_query}'

Note: The 'Text Content' below may contain the original page content followed by appended text extracted from a linked PDF document found on the page (often separated by '<hr/>'). Please summarize all relevant information presented.

Text Content:
```
{source_content}
```

Concise Summary:"""


# --- Writer Prompts ---

_WRITER_SYSTEM_PROMPT_BASE ="""
You are an expert research report writer tasked with creating a comprehensive analysis based on scientific literature. Your goal is to synthesize information from provided source materials into a well-structured, coherent, and informative report that specifically addresses the user's query.

When analyzing the provided source materials:
- Evaluate the strength of evidence and methodological rigor in each source
- Prioritize findings from peer-reviewed studies and high-credibility sources
- Consider the recency and relevance of the information to the specific question
- Identify consensus views as well as areas of controversy or uncertainty
- Extract both reported strengths and limitations for each approach or method

SYNTHESIS STRATEGY:
1. MAP the evidence landscape, identifying where sources agree, disagree, or address different aspects
2. EVALUATE source quality, giving appropriate weight to more rigorous or authoritative sources
3. INTEGRATE complementary information to build comprehensive understanding
4. CONTRAST competing findings or interpretations, analyzing possible reasons for differences
5. ACKNOWLEDGE gaps where information is limited or absent from the provided sources
6. ORGANIZE information thematically rather than source-by-source

DATA PRESENTATION:
- Present numerical data consistently (units, significant figures, etc.)
- Use tables to compare related statistics from multiple sources when appropriate
- Put findings in context (e.g., "10% increase" -> "10% increase from 2010 baseline")
- Note uncertainty or confidence intervals when provided

INFORMATION GAPS:
- Acknowledge when critical information called for in the writing plan is not found in any source
- Do not substitute missing information with general knowledge or assumptions
- If *absolutely critical* information is missing that prevents substantial completion of a required section according to the plan, you may request specific searches.

Follow the provided writing plan precisely, including the overall goal, tone, section structure, and specific guidance for each section. Ensure balanced coverage of all aspects of the query.

**Focus on drawing information and insights solely from the provided source materials listed below.** Each source item (summary or chunk) is identified by a unique *reference number* in square brackets in its header, e.g., `[1]`, `[2]`.

**CITATIONS:**
- When you present a specific fact, statistic, finding, or direct claim from a source, you MUST indicate its origin using a citation marker.
- Insert the marker *immediately* after the information, before any punctuation (like periods or commas).
- Use the format `[[CITATION:ref_num]]` where `ref_num` is the *reference number* corresponding to the source item found in its header in the 'Source Materials' list (e.g., `[[CITATION:1]]`, `[[CITATION:3]]`).
- If a single piece of information is supported by multiple sources, list all relevant reference numbers separated by commas: `[[CITATION:1, 2, 5]]`.
- **Cite appropriately:** Prioritize citing specific data points, direct quotes (though avoid overuse), key findings, and controversial claims. Avoid over-citing common knowledge established across multiple sources or highly synthesized statements unless attributing a specific nuanced point.
- A final, numbered list of sources corresponding to these reference numbers will be added automatically later.

Maintain a logical flow with clear transitions between sections. Organize complex information into digestible components while preserving important technical details. **Use Markdown formatting (like tables, code blocks, lists, bolding) to enhance clarity and structure.** Ensure the report directly addresses all aspects of the original user query.

**IMPORTANT: Output ONLY a single JSON object adhering to the following schema. Do not include any conversational text, preamble, or self-description before or after the JSON object.**

```json
{{
  "report_content": "# Report Titlenn## Section 1 TitlennReport content in Markdown format... with [[CITATION:ref_num]] markers as specified...",
  "requested_searches": [
    {{
      "query": "Specific search query needed to fill a critical information gap.",
      "reasoning": "Brief explanation why this search is essential to fulfill the writing plan."
    }}
    // ... include only if critical information is missing
  ] | null // Set to null or omit if no searches are needed
}}
```
"""

# For Initial Draft Generation
_WRITER_USER_MESSAGE_TEMPLATE_INITIAL ="""
Original User Query: {user_query}

Writing Plan:
```json
{writing_plan_json}
```

Source Materials:
Each item below is presented with its unique reference number in its header, e.g., [1]. Use this number for citations like [[CITATION:1]].
{formatted_summaries}

Please generate the research report based *only* on the Writing Plan and the Source Materials provided above.
"""


# For Refinement/Revision
_REFINEMENT_SYSTEM_PROMPT = _WRITER_SYSTEM_PROMPT_BASE # Use the same base prompt

_REFINEMENT_USER_MESSAGE_TEMPLATE ="""
Original User Query: {user_query}

Writing Plan:
```json
{writing_plan_json}
```

Previous Draft:
```markdown
{previous_draft}
```

Refinement Topic/Request: The previous draft was missing information related to '{refinement_topic}'. New source materials focusing on this topic have been gathered.

Newly Added Source Materials:
Each item below is presented with its unique reference number in its header, e.g., [3]. Use this number for citations like [[CITATION:3]].
{formatted_new_summaries}

Please revise the 'Previous Draft' based *only* on the 'Newly Added Source Materials'. Focus on incorporating the new information relevant to the '{refinement_topic}' into the existing structure, following the Writing Plan.
- Integrate the new findings smoothly into the relevant sections.
- Ensure you add citations `[[CITATION:ref_num]]` for all new information using the reference numbers provided with the new materials.
- If the new information contradicts or significantly modifies existing statements, update the draft accordingly, maintaining logical consistency.
- If the new information fills a previously identified gap, ensure the relevant section is now complete.
- You may need to restructure sentences or paragraphs to accommodate the new information naturally.
- Do *not* remove existing content or citations unless the new information directly refutes it or makes it redundant.
- If the new information still doesn't fully address the refinement topic or reveals further critical gaps, you may request additional searches.

Output the *complete, revised report* in the standard JSON format, potentially including new search requests if necessary.
"""


# --- Helper Function for Formatting Context ---
# Updated to use 'source_ref_num' instead of 'rank'

def format_summaries_for_prompt(source_materials: list[Dict[str, Any]]) -> str:
    """
    Formats the collected source materials (summaries and chunks) for inclusion 
    in the Writer or Refiner agent prompts.
    Uses 'source_ref_num' for stable citation identification.
    """
    formatted_string = ""
    for i, item in enumerate(source_materials):
        ref_num = item.get('source_ref_num', f"MISSING_REF_{i+1}") # Use stable ref num
        title = item.get('title', 'Unknown Title')
        link = item.get('link', '#')
        content = item.get('content', '').strip()
        material_type = item.get('type', 'unknown') # e.g., 'summary' or 'chunk'
        score = item.get('score') # For reranked chunks

        # Header for each item using the reference number
        formatted_string += f"--- Source Material [{ref_num}] ---\n"
        formatted_string += f"Title: {title}\n"
        formatted_string += f"Link: {link}\n"
        formatted_string += f"Type: {material_type}"
        if score is not None:
             score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
             formatted_string += f" (Relevance Score: {score_str})"
        formatted_string += "\n"
        formatted_string += f"Content:\n{content}\n"
        formatted_string += f"--- End Source Material [{ref_num}] ---\n\n"

    if not formatted_string:
        return "No source materials provided.\n"

    return formatted_string.strip()


# --- Define OpenRouter Models --- 
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    logger.warning("OPENROUTER_API_KEY environment variable not set. Agent initialization might fail.")
    # You might want to raise an error here depending on application requirements

def create_llm_model(model_id: str) -> OpenAIModel:
    """Helper function to create an OpenAI-compatible model instance for OpenRouter."""
    if not OPENROUTER_API_KEY:
        # Raise error earlier if key is missing, as provider needs it.
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")
    
    provider = OpenAIProvider(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    # Model ID format might need adjustment depending on OpenRouter/Provider expectations
    # Using the provided model_id directly as it seems like the intended format.
    # Pass model_id as the first positional argument
    return OpenAIModel(model_id, provider=provider)

# --- Agent Definitions --- 

def create_planner_agent(config: DeepResearchConfig) -> Agent:
    """Creates the Planner Agent instance."""
    model_id = config.planner_model_id # Get model_id from config
    logger.debug(f"Creating Planner Agent with model: {model_id}") # Log the id directly
    planner_model = create_llm_model(model_id)
    return Agent[
        schemas.PlannerOutput
    ](
        model=planner_model,
        system_prompt=_PLANNER_SYSTEM_PROMPT,
        result_type=schemas.PlannerOutput,
        retries=3
    )

def create_summarizer_agent(config: DeepResearchConfig) -> Agent:
    """Creates the Summarizer Agent instance."""
    model_id = config.summarizer_model_id # Get model_id from config
    logger.debug(f"Creating Summarizer Agent with model: {model_id}") # Log the id directly
    summarizer_model = create_llm_model(model_id)
    return Agent[str](
        model=summarizer_model,
        system_prompt=_SUMMARIZER_SYSTEM_PROMPT,
        result_type=str,
        retries=3
    )

def create_writer_agent(config: DeepResearchConfig) -> Agent:
    """Creates the Writer Agent instance."""
    model_id = config.writer_model_id # Get model_id from config
    logger.debug(f"Creating Writer Agent with model: {model_id}") # Log the id directly
    writer_model = create_llm_model(model_id)
    return Agent[schemas.WriterOutput](
        model=writer_model,
        system_prompt=_WRITER_SYSTEM_PROMPT_BASE,
        deps_type=None,
        result_type=schemas.WriterOutput,
        retries=3
    )

def create_refiner_agent(config: DeepResearchConfig) -> Agent:
    """
    Creates the Refiner Agent instance.
    Handles revisions and refinement. Signals need for external actions via JSON structure.
    """
    model_id = config.refiner_model_id # Get model_id from config
    logger.debug(f"Creating Refiner Agent with model: {model_id}") # Log the id directly
    refiner_model = create_llm_model(model_id)
    return Agent[schemas.WriterOutput](
        model=refiner_model,
        system_prompt=_REFINEMENT_SYSTEM_PROMPT,
        deps_type=None,
        result_type=schemas.WriterOutput,
        retries=3
    )

# Structure to hold all agents for the agency
class AgencyAgents(BaseModel):
    planner: Any
    summarizer: Any
    writer: Any
    refiner: Any

    # Re-added model_config to allow arbitrary Agent types
    model_config = ConfigDict(arbitrary_types_allowed=True)

# --- Agent Initialization Function ---
def get_agency_agents(config: DeepResearchConfig) -> AgencyAgents:
    """
    Initializes and returns all PydanticAI Agent instances for the agency,
    including applying result validators.

    Returns:
        An AgencyAgents object containing the initialized PydanticAI agents.
    """
    planner = create_planner_agent(config=config)
    summarizer = create_summarizer_agent(config=config)
    writer = create_writer_agent(config=config)
    refiner = create_refiner_agent(config=config)

    # --- Apply Result Validator using Decorator Syntax ---
    # Define the validator function here after agents are created
    @writer.result_validator
    @refiner.result_validator
    async def validate_citations_present(ctx: RunContext[None], result: schemas.WriterOutput) -> schemas.WriterOutput:
        """Validates that the report content includes [[CITATION:source_ref_num]] markers."""
        citation_pattern = r"\[\[CITATION:\d+(,\d+)*\]\]"
        if not result.report_content or not re.search(citation_pattern, result.report_content):
            logger.warning(f"Validation failed: Report missing citation markers. Requesting retry {ctx.retry + 1}.")
            raise ModelRetry("The generated report is missing mandatory citation markers like [[CITATION:1]] or [[CITATION:1,2]]. Please revise the report to include citations using the reference numbers provided in the source material headers (e.g., [1], [2]).")
        logger.debug(f"Validation passed: Citations found.")
        return result

    logger.info("Added citation validator to Writer and Refiner agents via decorator.")

    logger.info("Initialized Agency Agents (Planner, Summarizer, Writer, Refiner) using OpenRouter.")

    return AgencyAgents(
        planner=planner,
        summarizer=summarizer,
        writer=writer,
        refiner=refiner
    )
