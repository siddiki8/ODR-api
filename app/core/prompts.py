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
1.  A list of `search_tasks`: Define 1 to {max_search_tasks} specific search queries for a web search engine (like Google via Serper API) to gather the necessary information. Prioritize quality over quantity; only generate multiple queries if distinct angles are needed. For each task, specify the query string, the most appropriate Serper endpoint, the desired number of results (`num_results`, default 10), and a brief reasoning.

Available Serper endpoints and when to use them:
- `/search`: General web search for broad information, recent events, and mainstream content (default choice)
- `/scholar`: Academic and scientific research papers, citations, and scholarly articles (use for academic topics, scientific research, or when peer-reviewed sources are needed)
- `/news`: Recent news articles and current events (use for trending topics, recent developments, or time-sensitive information)

2.  A detailed `writing_plan`: Outline the structure of the final report. This includes the overall goal, desired tone, specific sections with titles and guidance for each, and any additional directives for the writer.

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
You are an expert summarizer. Your task is to create a complete, factual summary of the provided text content. You don't have to be particularly concise. Information is the priority, but concisesness where possible helps save tokens. This summary is one of many that will be used to generate a powerful research report - so don't be afraid to include all the important information you can find.
Focus specifically on extracting information relevant to answering the user's original research query, which will be used to generate a comprehensive report.
Extract key facts, findings, arguments, and data points pertinent to the user's query topic.
Maintain a neutral, objective tone.
The summary should be dense with relevant information but easy to understand.
Do not add introductions or conclusions like 'The text discusses...' or 'In summary...'. Just provide the summary content itself.
Focus on accurately representing the information from the provided text ONLY.
"""

_SUMMARIZER_USER_MESSAGE_TEMPLATE = \
"""
Please summarize the following text content extracted from the source titled '{source_title}' (URL: {source_link}). Focus on information that might be relevant for a research report addressing the query: '{user_query}'

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
You are an expert research report writer. Your goal is to synthesize information from provided source materials (summaries and relevant chunks) into a well-structured, coherent, and informative report.
Follow the provided writing plan precisely, including the overall goal, tone, section structure, and specific guidance for each section.
Integrate the information from the source materials naturally into the report narrative.

**Crucially, you MUST cite your sources using the numerical markers provided for the *original source* (e.g., [1], [2]).** Source materials are grouped by their original source, and each original source has a unique citation number. Even if information comes from a specific chunk of a source, cite the main source number. 

When citing:
- Add the corresponding numerical citation marker immediately after the information (e.g., 'Quantum computing poses a threat [1].').
- Use multiple citations if information comes from several distinct original sources (e.g., 'Several sources discuss this [2][3].').
- **Avoid over-citing:** If multiple consecutive sentences or a paragraph clearly draw from the same single source, you may cite it once at the end of the relevant passage rather than after every sentence.

Maintain a logical flow and ensure the report directly addresses the original user query.
**Do NOT generate a bibliography or reference list at the end; this will be added later.**

If, while writing, you determine that you lack sufficient specific information on a crucial sub-topic required by the writing plan, you can request a specific web search. To do this, insert the exact tag `<search_request query="...">` at the point in the text where the information is needed. Replace "..." with the specific search query string that would find the missing information. Use this tag *only* if absolutely necessary to fulfill the writing plan requirements and *only once* per draft.
"""

# For Initial Draft Generation
_WRITER_USER_MESSAGE_TEMPLATE_INITIAL = \
"""
Original User Query: {user_query}

Writing Plan:
```json
{writing_plan_json}
```

Source Summaries (Cite using the numerical markers provided, e.g., [1], [2]):
{formatted_summaries}

---

Please generate the initial draft of the research report based *only* on the provided writing plan and source summaries. Follow all instructions in the system prompt, especially regarding structure, tone, and numerical citations (e.g., [1], [2]). **Do NOT include a reference list.** If necessary, use the `<request_more_info topic="...">` tag as described in the system prompt.

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
        link = item.get('url')
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

def get_writer_initial_prompt(user_query: str, writing_plan: dict, source_summaries: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Generates the message list for the Writer LLM (initial draft).

    Args:
        user_query: The original user query.
        writing_plan: The JSON writing plan from the Planner.
        source_summaries: List of dictionaries, each containing 'title', 'link', 'summary'.

    Returns:
        A list of messages suitable for litellm.completion.
    """
    formatted_summaries_str = format_summaries_for_prompt(source_summaries)
    writing_plan_str = json.dumps(writing_plan, indent=2)
    
    return [
        {"role": "system", "content": _WRITER_SYSTEM_PROMPT_BASE},
        {
            "role": "user",
            "content": _WRITER_USER_MESSAGE_TEMPLATE_INITIAL.format(
                user_query=user_query,
                writing_plan_json=writing_plan_str,
                formatted_summaries=formatted_summaries_str
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

*New* Source Summaries (to address the request for more info on '{refinement_topic}'. Cite using their *new* numerical markers as provided below):
{formatted_new_summaries}

All Available Source Summaries (Initial + Previous Refinements - Use these markers for citation):
{formatted_all_summaries}

---

Please revise the previous draft of the research report.
Your primary goal is to incorporate the *new* source summaries provided above to specifically address the request for more information on the topic: '{refinement_topic}'.
Integrate the new information smoothly into the existing structure defined by the writing plan.
Ensure you *maintain* the overall structure, tone, and guidance from the original writing plan.
Crucially, continue to cite *all* sources accurately using the provided numerical markers (e.g., [1], [2], [15]) for both new and previously used information based on the 'All Available Source Summaries' list. **Do NOT include a reference list.**
If necessary, you may use the `<request_more_info topic="...">` tag again if *absolutely critical* information for the plan is still missing, but avoid it if possible.

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
        refinement_topic: The specific topic requested via the <request_more_info> tag.
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

_REFINER_SYSTEM_PROMPT = \
"""
You are an expert editor tasked with refining a research report draft.
You have been provided with:
1. The previous draft of the report.
2. The specific search query that was performed to find additional information.
3. New relevant information (potentially chunks or summaries) extracted from sources found by that search query.

Your goal is to integrate the *new relevant information* seamlessly into the *previous draft* to improve its completeness and accuracy, specifically addressing the gap identified by the search query.
- Focus on incorporating the substance of the new information.
- Maintain the existing structure, tone, and citation style of the draft.
- **Ensure you handle citations correctly**: If the new information requires citations, determine if the source already exists in the draft (check previous citations) or if it's a new source requiring a new citation number (you may need context from the main writer process to assign the *correct* number, but indicate where a citation is needed).
- Make the report flow naturally after incorporating the changes.
- Output *only* the revised report draft.
"""

_REFINER_USER_MESSAGE_TEMPLATE = \
"""
Previous Draft:
```
{previous_draft}
```

We performed a search for: '{search_query}'

Incorporate the following new information found from that search:
{formatted_new_info}

---

Please provide the revised report draft, integrating the new information logically and maintaining the original style and citations where possible.

Revised Draft:
"""

def get_refiner_prompt(
    previous_draft: str,
    search_query: str,
    new_info: list[Dict[str, Any]] # Updated type hint
) -> list[dict[str, str]]:
    """Generates the message list for the Refiner LLM."""
    
    # Format the new info (chunks) for the prompt
    formatted_new_info_parts = []
    if new_info:
        for i, chunk_data in enumerate(new_info):
            # Use temporary high citation numbers or just indicators for clarity
            # The main citation logic relies on the writer seeing the full history
            source_indicator = f"New Info [{i+1}]"
            title = chunk_data.get('title', 'Untitled')
            url = chunk_data.get('url', '#')
            score = chunk_data.get('score')
            content = chunk_data.get('chunk_content') or chunk_data.get('content')
            
            score_str = f" (Score: {score:.2f})" if score is not None else ""
            
            formatted_new_info_parts.append(f"{source_indicator}: {title}{score_str} ({url})" )
            if content:
                formatted_new_info_parts.append(f"  Content: {content}")
            else:
                formatted_new_info_parts.append("  Content: [Not Available]")
        formatted_new_info = "\n\n".join(formatted_new_info_parts)
    else:
        formatted_new_info = "No new relevant information was found or processed for this query."

    return [
        {"role": "system", "content": _REFINER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _REFINER_USER_MESSAGE_TEMPLATE.format(
                previous_draft=previous_draft,
                search_query=search_query,
                formatted_new_info=formatted_new_info
            )
        }
    ] 