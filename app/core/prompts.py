import json
from typing import List, Dict, TYPE_CHECKING, Any
from collections import defaultdict
import re

# Add type checking import for SourceSummary
if TYPE_CHECKING:
    from .schemas import SourceSummary 

_PLANNER_SYSTEM_PROMPT = \
"""
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

def get_planner_prompt(user_query: str, max_search_tasks: int = 3) -> List[Dict[str, str]]:
    """Returns the messages for the planner LLM."""
    system_prompt_formatted = _PLANNER_SYSTEM_PROMPT.format(max_search_tasks=max_search_tasks)
    return [
        {"role": "system", "content": system_prompt_formatted},
        {"role": "user", "content": _PLANNER_USER_MESSAGE_TEMPLATE.format(user_query=user_query)}
    ]


_SUMMARIZER_SYSTEM_PROMPT = \
"""
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

_SUMMARIZER_USER_MESSAGE_TEMPLATE = \
"""
Please summarize the following text content extracted from the source titled '{source_title}' (URL: {source_link}). Focus on information that might be relevant for a research report addressing the query: '{user_query}'

Note: The 'Text Content' below may contain the original page content followed by appended text extracted from a linked PDF document found on the page (often separated by '<hr/>'). Please summarize all relevant information presented.

Text Content:
```
{source_content}
```

Concise Summary:"""

def get_summarizer_prompt(user_query: str, source_title: str, source_link: str, source_content: str) -> list[dict[str, str]]:
    """
    Generates the message list for the Summarizer LLM.

    Args:
        user_query: The original user research query (for context).
        source_title: The title of the source document.
        source_link: The URL of the source document.
        source_content: The extracted text content (full or chunked) to summarize.

    Returns:
        A list of messages suitable for litellm.completion.
    """
    return [
        {"role": "system", "content": _SUMMARIZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _SUMMARIZER_USER_MESSAGE_TEMPLATE.format(
                user_query=user_query,
                source_title=source_title,
                source_link=source_link,
                source_content=source_content
            )
        }
    ]

# --- Writer Prompts ---

_WRITER_SYSTEM_PROMPT_BASE = \
"""
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
- Put findings in context (e.g., "10% increase" → "10% increase from 2010 baseline")
- Note uncertainty or confidence intervals when provided

INFORMATION GAPS:
- Acknowledge when critical information called for in the writing plan is not found in any source
- Do not substitute missing information with general knowledge or assumptions
- Use the `<search_request query="...">` tag ONLY for significant gaps preventing substantial completion of a section

Follow the provided writing plan precisely, including the overall goal, tone, section structure, and specific guidance for each section. Ensure balanced coverage of all aspects of the query.

**Focus on drawing information and insights solely from the provided source materials listed below.** Each source item (summary or chunk) is identified by a unique number in square brackets, e.g., `[1]`, `[2]`.

**CITATIONS:**
- When you present a specific fact, statistic, finding, or direct claim from a source, you MUST indicate its origin using a citation marker.
- Insert the marker *immediately* after the information, before any punctuation (like periods or commas).
- Use the format `[[CITATION:rank]]` where `rank` is the number corresponding to the source item in the 'Source Materials' list (e.g., `[[CITATION:1]]`, `[[CITATION:3]]`).
- If a single piece of information is supported by multiple sources, list all relevant ranks separated by commas: `[[CITATION:1, 2, 5]]`.
- **Cite appropriately:** Prioritize citing specific data points, direct quotes (though avoid overuse), key findings, and controversial claims. Avoid over-citing common knowledge established across multiple sources or highly synthesized statements unless attributing a specific nuanced point.
- A final, numbered list of sources corresponding to these ranks will be added automatically later.

Maintain a logical flow with clear transitions between sections. Organize complex information into digestible components while preserving important technical details. **Use Markdown formatting (like tables, code blocks, lists, bolding) to enhance clarity and structure.** Ensure the report directly addresses all aspects of the original user query.

**IMPORTANT: Generate ONLY the report content. Start the report with a clear title formatted as a Markdown H1 (`# Title`), reflecting the 'overall_goal' from the writing plan. Immediately follow the title with the first section or introduction as specified in the plan. Do NOT include any conversational text, preamble, or self-description before the report title.**
"""

# For Initial Draft Generation
_WRITER_USER_MESSAGE_TEMPLATE_INITIAL = \
"""
Original User Query: {user_query}

Writing Plan:
```json
{writing_plan_json}
```

Source Materials:
Each item below is presented with its unique citation number, e.g., [rank].
{formatted_summaries}

---

Your task is to write a comprehensive research report addressing the query: "{user_query}"

For this analysis:
1. Critically evaluate the evidence for both approaches mentioned in the query
2. Assess methodological strengths and limitations based on the provided materials 
3. Compare efficacy metrics and performance when available
4. Analyze potential biases in each approach as documented in the literature
5. Identify scenarios where each approach excels or faces challenges

Follow the writing plan structure while ensuring balanced treatment of all relevant aspects. Draw exclusively from the provided source materials, maintaining scientific objectivity. Present technical concepts accurately while keeping the analysis accessible to an informed but not necessarily specialized audience.

**Remember to insert `[[CITATION:rank]]` markers after specific information drawn from the sources.**

Report Draft:
"""

# --- Helper Function to Format Summaries/Chunks for Prompts ---
def format_summaries_for_prompt(source_materials: list[Dict[str, Any]]) -> str:
    """
    Formats summaries/chunks for the writer prompt, displaying each item
    with its unique rank for citation purposes.

    Args:
        source_materials: List of dictionaries, each representing a summary or chunk
                          with a 'rank', 'link', 'title', and 'content'.

    Returns:
        A formatted string representation of the source materials for the LLM prompt.
    """
    if not source_materials:
        return "No source materials available."

    formatted_output = []
    # Sort by rank to ensure consistent order in the prompt
    # Items should already be ranked sequentially by _assemble_writer_context
    sorted_materials = sorted(source_materials, key=lambda x: x.get('rank', float('inf')))

    for item in sorted_materials:
        rank = item.get('rank')
        link = item.get('link') # Should be string
        title = item.get('title', 'Untitled')
        content = item.get('content', 'N/A')
        item_type = item.get('type', 'Unknown') # 'summary' or 'chunk'
        score = item.get('score') # For chunks

        if rank is None or link is None:
            # Log this? Skip this? For now, skip if rank/link missing.
            continue

        # Header for each item using its rank
        source_header = f"[{rank}] {title} ({link})"
        formatted_output.append(source_header)

        # Display content (summary or chunk)
        if item_type == 'summary':
            formatted_output.append(f"  Summary: {content}")
        elif item_type == 'chunk':
            score_str = f" (Score: {score:.2f})" if score is not None else ""
            formatted_output.append(f"  Relevant Chunk{score_str}: {content}")
        else:
            formatted_output.append(f"  Content: {content}") # Fallback

    return "\n\n".join(formatted_output)

def get_writer_initial_prompt(user_query: str, writing_plan: dict, source_materials: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Generates the message list for the Writer LLM (initial draft).

    Args:
        user_query: The original user query.
        writing_plan: The JSON writing plan from the Planner.
        source_materials: List of dictionaries, containing grouped summaries/chunks.

    Returns:
        A list of messages suitable for litellm.completion.
    """
    formatted_materials_str = format_summaries_for_prompt(source_materials)
    writing_plan_str = json.dumps(writing_plan, indent=2)
    
    return [
        {"role": "system", "content": _WRITER_SYSTEM_PROMPT_BASE},
        {
            "role": "user",
            "content": _WRITER_USER_MESSAGE_TEMPLATE_INITIAL.format(
                user_query=user_query,
                writing_plan_json=writing_plan_str,
                formatted_summaries=formatted_materials_str
            )
        }
    ]

# For Refinement/Revision
_WRITER_USER_MESSAGE_TEMPLATE_REFINEMENT = \
"""
Original User Query: {user_query}

Writing Plan:
```json
{writing_plan_json}
```

Previously Generated Draft:
```
{previous_draft}
```

*New* Source Materials (to address the request for more info on '{refinement_topic}'):
Each item below is presented with its unique citation number, e.g., [rank].
{formatted_new_summaries}

All Available Source Materials (Initial + Previous Refinements):
Each item below is presented with its unique citation number, e.g., [rank].
{formatted_all_summaries}

---

Please revise the previous draft to incorporate critical information from the new source materials about '{refinement_topic}'. 

When integrating this information:
1. Maintain the analytical depth and balance of the existing draft
2. Update any sections where new material provides stronger evidence or contradicts previous assertions
3. Add nuance where the new materials illuminate gaps or limitations in the previous analysis
4. Ensure the logical flow and structure remain coherent after incorporating new information
5. Maintain focus on the original query comparing both approaches for efficacy and bias

Integrate the new information smoothly into the existing structure defined by the writing plan while preserving the scholarly tone and comprehensive nature of the analysis. Prioritize factual accuracy and balanced treatment of all perspectives represented in the sources.

If necessary, you may use the `<search_request query="...">` tag again if *absolutely critical* information for the plan is still missing, but avoid it if possible.

Revised Report Draft:
"""

def get_writer_refinement_prompt(
    user_query: str,
    writing_plan: dict,
    previous_draft: str,
    refinement_topic: str,
    new_summaries: list[dict[str, str]],
    all_summaries: list[dict[str, str]] # Includes initial + all refinement summaries so far
) -> list[dict[str, str]]:
    """
    Generates the message list for the Writer LLM (refinement/revision).

    Args:
        user_query: The original user query.
        writing_plan: The JSON writing plan from the Planner.
        previous_draft: The previous report draft generated by the writer.
        refinement_topic: The specific topic requested via the <search_request> tag.
        new_summaries: List of summaries gathered specifically for this refinement topic.
        all_summaries: List of all summaries gathered so far (initial + all refinements).

    Returns:
        A list of messages suitable for litellm.completion.
    """
    # Format the summaries with their correct numerical indices based on the FULL list
    # The `format_summaries_for_prompt` function now handles displaying ranks correctly for all items.
    # We just need to format the 'new_summaries' list separately for its dedicated section in the prompt.
    # `format_summaries_for_prompt` handles displaying rank for each item.
    formatted_new_summaries_str = format_summaries_for_prompt(new_summaries)
    formatted_all_summaries_str = format_summaries_for_prompt(all_summaries) # All summaries use standard formatting
    writing_plan_str = json.dumps(writing_plan, indent=2)
    
    return [
        {"role": "system", "content": _WRITER_SYSTEM_PROMPT_BASE},
        {
            "role": "user",
            "content": _WRITER_USER_MESSAGE_TEMPLATE_REFINEMENT.format(
                user_query=user_query,
                writing_plan_json=writing_plan_str,
                previous_draft=previous_draft,
                refinement_topic=refinement_topic,
                formatted_new_summaries=formatted_new_summaries_str,
                formatted_all_summaries=formatted_all_summaries_str
            )
        }
    ] 

# Ensure the helper function is available for the agent
format_summaries_for_prompt_template = format_summaries_for_prompt