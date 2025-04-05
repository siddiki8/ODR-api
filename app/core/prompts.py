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
1.  A list of `search_tasks`: Define 1 to {max_search_tasks} specific search queries for a web search engine (like Google via Serper API) to gather the necessary information. Generate distinct queries targeting different facets or sub-questions implied by the user's request. Prioritize quality over quantity; only generate multiple queries if distinct angles are truly needed. For each task, specify the query string, the most appropriate Serper endpoint, the desired number of results (`num_results`, default 10), and a brief reasoning. Consider source credibility when choosing endpoints.

Available Serper endpoints and when to use them:
- `/search`: General web search for broad information, recent events, and mainstream content (default choice)
- `/scholar`: Academic and scientific research papers, citations, and scholarly articles (use for academic topics, scientific research, or when peer-reviewed sources are needed/high credibility required)
- `/news`: Recent news articles and current events (use for trending topics, recent developments, or time-sensitive information)

2.  A detailed `writing_plan`: Outline the structure of the final report. This includes the overall goal, desired tone, specific sections with titles and guidance for each, and any additional directives for the writer.
    - The `guidance` for each section should be actionable. Link it back to specific parts of the user query or specify the *type* of information required (e.g., 'Summarize arguments for X,' 'Detail methodology of Y,' 'Compare Z found in sources').

Analyze the user's query carefully and devise a plan that will lead to a comprehensive and well-structured report.

Output *only* a single JSON object adhering to the following schema. Do not include any other text before or after the JSON object.

```json
{{
  "search_tasks": [
    {{
      "query": "Specific query string for Serper",
      "endpoint": "/search | /scholar | /news",
      "num_results": <integer>, // Default 10
      "reasoning": "Why this query, endpoint, and result count are chosen"
    }}
    // ... (1 to {max_search_tasks} tasks total)
  ],
  "writing_plan": {{
    "overall_goal": "Provide a comprehensive analysis of [topic], focusing on [aspect1] and [aspect2], suitable for [audience].",
    "desired_tone": "Objective and analytical | Formal | Informal | etc.",
    "sections": [
      {{
        "title": "Section Title",
        "guidance": "Specific instructions for the writer for this section."
      }}
      // ... (multiple sections)
    ],
    "additional_directives": [
       "Directive 1 (e.g., citation style)",
       "Directive 2 (e.g., address counterarguments)"
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
Prioritize extracting *all* key facts, arguments, and data relevant to the user's original research query. Aim for informational density; be thorough rather than brief, but avoid redundant phrasing where possible without sacrificing completeness.
The summary is one of many that will be used to generate a comprehensive research report.
Extract key facts, findings, arguments, and data points pertinent to the user's query topic.
Maintain a neutral, objective tone.
The summary should be dense with relevant information but easy to understand.
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

Rather than simply summarizing each source, synthesize insights across multiple sources to develop nuanced analysis. Compare and contrast different methodologies, findings, and perspectives. When sources present conflicting information, analyze potential reasons for these differences and assess the relative strength of supporting evidence.

Follow the provided writing plan precisely, including the overall goal, tone, section structure, and specific guidance for each section. Ensure balanced coverage of all aspects of the query, avoiding over-emphasis on one approach or perspective without sufficient justification.

**Focus on drawing information and insights solely from the provided source materials listed below.**

Maintain a logical flow with clear transitions between sections. Organize complex information into digestible components while preserving important technical details. **Feel free to use Markdown formatting (like tables, code blocks, lists, bolding) where it enhances clarity and structure.** Ensure the report directly addresses all aspects of the original user query.

**IMPORTANT: Generate ONLY the report content itself. Start directly with the first section title or introduction as specified in the writing plan. Do NOT include any conversational text, preamble, or self-description before the report content begins.**

If, while writing, you determine that you lack sufficient specific information on a crucial sub-topic required by the writing plan, you can request a specific web search. To do this, insert the exact tag `<search_request query="...">` at the point in the text where the information is needed. Replace "..." with the specific search query string that would find the missing information. Use this tag *sparingly*, only when fulfilling a core requirement of the writing plan is impossible without it, and *only once* per draft.
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

Report Draft:
"""

def format_summaries_for_prompt(source_materials: list[Dict[str, Any]]) -> str:
    """Formats summaries/chunks for the writer prompt, grouping by original source URL and assigning unique citation numbers."""
    if not source_materials:
        return "No source materials available."
    
    grouped_sources = defaultdict(lambda: {"summaries": [], "chunks": [], "title": "Untitled", "first_seen_order": float('inf')})
    link_order = []

    # Group by link, store original title, track order, separate summaries/chunks
    for idx, item in enumerate(source_materials):
        # Try to get link using various possible key names
        link = item.get('link') or item.get('url')
        if not link: continue # Skip items without links

        group = grouped_sources[link]
        if link not in link_order:
            link_order.append(link)
            group["title"] = item.get('title', 'Untitled')
            # Use original index to maintain relative order for items processed concurrently
            group["first_seen_order"] = idx 

        item_type = item.get('type', 'unknown')
        content = item.get('content') or item.get('chunk_content') or item.get('summary_content')
        score = item.get('score') # Optional score for chunks

        if content:
            data = {"content": content}
            if score is not None: data["score"] = score
            
            if item_type == 'summary':
                group["summaries"].append(data)
            elif item_type == 'chunk':
                group["chunks"].append(data)
            else: # Handle older format or unexpected types
                 # Try to guess if it looks like a summary or chunk based on keys
                 if 'summary' in item: group["summaries"].append({"content": item['summary']})
                 elif 'chunk_content' in item: group["chunks"].append(data)
                 else: group["summaries"].append({"content": str(item)}) # Fallback

    formatted_output = []
    # Sort links by the order they were first encountered
    sorted_links = sorted(link_order, key=lambda l: grouped_sources[l]["first_seen_order"])

    # Assign citation numbers based on sorted order
    for i, link in enumerate(sorted_links):
        citation_marker = f"[{i+1}]"
        group = grouped_sources[link]
        title = group["title"]
        
        source_header = f"Source {citation_marker}: {title} ({link})"
        formatted_output.append(source_header)
        
        # Display Summaries first
        if group["summaries"]:
            # Typically expect only one summary per source
            formatted_output.append(f"  Summary: {group['summaries'][0]['content']}")
        
        # Then display relevant Chunks, potentially sorted by score if available
        if group["chunks"]:
            formatted_output.append("  Relevant Chunks:")
            # Sort chunks by score descending if scores exist, otherwise keep order
            sorted_chunks = sorted(group["chunks"], key=lambda c: c.get('score', 0), reverse=True)
            for chunk_data in sorted_chunks:
                score_str = f" (Score: {chunk_data['score']:.2f})" if 'score' in chunk_data else ""
                formatted_output.append(f"    - Chunk{score_str}: {chunk_data['content']}")
                
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
{formatted_new_summaries}

All Available Source Materials (Initial + Previous Refinements):
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

def format_summaries_for_prompt_with_offset(summaries: list[dict[str, str]], offset: int) -> str:
    """Formats *new* summaries for the refinement prompt context block, using an index offset.
       NOTE: This is ONLY for displaying the NEW summaries concisely in the prompt. 
             The main `formatted_all_summaries` uses the standard grouped formatting.
    """
    if not summaries:
        return "No new summaries available for this topic."
    
    formatted = []
    for i, summary_info in enumerate(summaries):
        # Create numerical citation marker with offset - This marker might not directly correspond 
        # to the final citation if the source was already seen, but gives context.
        context_marker = f"(New Context Item [{offset + i + 1}])"
        title = summary_info.get('title', 'Untitled')
        link = summary_info.get('link', '#')
        summary = summary_info.get('summary', 'No summary content.')
        formatted.append(f"{context_marker} Title: {title} ({link})\nSummary: {summary}")
    
    return "\n\n".join(formatted)

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
    # Find the starting index for new summaries within the all_summaries list
    start_index_new = len(all_summaries) - len(new_summaries)
    formatted_new_summaries_str = format_summaries_for_prompt_with_offset(new_summaries, start_index_new)
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

# --- Refiner Prompt --- 
# REMOVED - This role is being consolidated into iterative calls to the Writer LLM
# using get_writer_refinement_prompt but potentially with a different LLM config.

# _REFINER_SYSTEM_PROMPT = \
# """
# ... (removed content) ...
# """

# _REFINER_USER_MESSAGE_TEMPLATE = \
# """
# ... (removed content) ...
# """

# def get_refiner_prompt(
#     previous_draft: str,
#     search_query: str,
#     new_info: list[Dict[str, Any]] # Updated type hint
# ) -> list[dict[str, str]]:
#    """Generates the message list for the Refiner LLM.""" 
#     # ... (removed function body) ...

# --- End of File --- 