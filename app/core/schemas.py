from pydantic import BaseModel, Field, field_validator, HttpUrl, ConfigDict
from typing import List, Dict, Any, Optional, Literal

# Import LLMConfig for type hinting in ResearchRequest
from .config import LLMConfig

class ExtractionResult(BaseModel):
    """
    Holds the results of a scraping operation, including the source method.
    
    Attributes:
        name: Identifier for the source/method used (e.g., 'wikipedia', 'pdf', 'crawl4ai_markdown').
        content: The extracted text content (Markdown for Crawl4AI, text otherwise), or None if extraction failed.
        raw_markdown_length: The character length of the extracted content, or None.
    """
    model_config = ConfigDict(extra='ignore')
    name: str 
    content: Optional[str] = None
    raw_markdown_length: Optional[int] = None

class SearchResult(BaseModel):
    """
    Represents a single processed search result item from an external API like Serper.
    Provides a structured way to access common fields.
    """
    model_config = ConfigDict(extra='ignore')
    title: str
    link: HttpUrl  # Use HttpUrl for validation
    snippet: str
    position: int
    raw: Dict[str, Any] # Store the original dictionary for flexibility or accessing less common fields

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """
        Factory method to create a SearchResult instance from a raw dictionary 
        (e.g., an item from Serper API 'organic' results).
        
        Performs basic validation and type conversion.
        
        Args:
            data: The dictionary representing a single search result.
            
        Returns:
            A SearchResult instance.
            
        Raises:
            ValidationError: If essential fields like 'link' are missing or invalid.
        """
        # Pydantic will raise ValidationError if 'link' is invalid or missing
        # We assume basic fields are usually present but handle potential missing ones gracefully
        return cls(
            title=data.get('title', 'Untitled'), # Provide a default title
            link=data['link'], # Let Pydantic handle validation/missing key for link
            snippet=data.get('snippet', ''), # Default to empty snippet
            position=data.get('position', -1), # Default position if missing
            raw=data
        )

class SearchTask(BaseModel):
    """Represents a single search task generated by the Planner LLM."""
    model_config = ConfigDict(extra='ignore')
    query: str = Field(..., description="The search query string to execute.", min_length=1)
    endpoint: str = Field("/search", description="The Serper API endpoint to target (e.g., /search, /scholar).")
    num_results: int = Field(5, description="Desired number of search results.", ge=1, le=20)
    reasoning: Optional[str] = Field(None, description="Planner's reasoning for generating this specific search task.")

    @field_validator('endpoint')
    def validate_endpoint(cls, v: str) -> str:
        allowed_endpoints = ['/search', '/scholar', '/news']
        if v not in allowed_endpoints:
            raise ValueError(f"Invalid endpoint '{v}'. Allowed: {allowed_endpoints}")
        return v

class SectionItem(BaseModel):
    """Defines a single section within the writing plan."""
    model_config = ConfigDict(extra='ignore')
    title: str = Field(..., description="Title for this section of the report.", min_length=1)
    guidance: str = Field(..., description="Specific instructions or guidance for writing this section.", min_length=1)

class WritingPlan(BaseModel):
    """Defines the structure and directives for the final report, generated by the Planner LLM."""
    model_config = ConfigDict(extra='ignore')
    overall_goal: str = Field(..., description="The main objective or purpose of the research report.", min_length=1)
    desired_tone: Optional[str] = Field(None, description="Preferred writing style or tone (e.g., formal, neutral, technical).")
    sections: List[SectionItem] = Field(..., description="An ordered list of sections that should constitute the report.", min_items=1)
    additional_directives: Optional[List[str]] = Field(None, description="Any extra high-level instructions for the Writer LLM.")

    @property
    def outline(self) -> List[str]:
        """Provides a simple list of section titles, primarily for backward compatibility or simple display."""
        return [section.title for section in self.sections]

class PlannerOutput(BaseModel):
    """Validates the structured JSON output expected from the Planner LLM."""
    model_config = ConfigDict(extra='ignore')
    search_tasks: List[SearchTask] = Field(..., description="List of search tasks to be executed based on the user query.", min_items=1)
    writing_plan: WritingPlan = Field(..., description="Detailed plan guiding the structure and content of the final report.")

class SourceSummary(BaseModel):
    """Represents processed information derived from a single web source, either a summary or aggregated chunks."""
    model_config = ConfigDict(extra='ignore')
    title: str = Field(..., description="The title of the web source (e.g., from <title> tag or search result).", min_length=1)
    link: HttpUrl = Field(..., description="The URL of the original web source.")
    content: str = Field(..., description="The processed content: either an LLM-generated summary or concatenated relevant text chunks.", min_length=1)
    content_type: Literal['summary', 'chunks'] = Field(..., description="Specifies whether the 'content' field holds a summary or concatenated chunks.")
    original_title: Optional[str] = Field(None, description="The original title if it was modified (e.g., during chunking).") # Consider removing if unused

class Chunk(BaseModel):
    """Represents a single chunk of relevant text extracted from a source document."""
    model_config = ConfigDict(extra='ignore')
    content: str = Field(..., description="The text content of the chunk.", min_length=1)
    link: HttpUrl = Field(..., description="The URL of the original source document.")
    title: str = Field(..., description="The title of the original source document.")
    relevance_score: Optional[float] = Field(None, description="Score assigned by the reranker indicating relevance to the query.")
    rank: Optional[int] = Field(None, description="Sequential rank assigned during processing (e.g., for citation).")

class SearchRequest(BaseModel):
    """Validates a search request extracted from the Writer/Refiner LLM's output (e.g., <search_request query='...'>)."""
    model_config = ConfigDict(extra='ignore')
    query: str = Field(..., description="The specific search query requested by the LLM to gather more information.", min_length=1) 

# --- API Schemas ---

class ResearchRequest(BaseModel):
    """Request schema for the main `/research` endpoint."""
    model_config = ConfigDict(extra='ignore')
    query: str = Field(..., min_length=10, description="The user's research query.")
    # Optional LLM configurations can be added here if the API allows overrides
    # e.g., planner_llm_override: Optional[LLMConfig] = None

    # Optional config overrides for LLMs (Keep these)
    planner_llm_config: Optional[LLMConfig] = Field(None, description="Override default planner LLM settings.")
    summarizer_llm_config: Optional[LLMConfig] = Field(None, description="Override default summarizer LLM settings.")
    writer_llm_config: Optional[LLMConfig] = Field(None, description="Override default writer LLM settings.")
    max_search_tasks: Optional[int] = Field(None, description="Override maximum initial search tasks.")
    llm_provider: Optional[Literal['google', 'openrouter']] = Field(None, description="Override the default LLM provider for this request.")

class TokenUsageCounter(BaseModel):
    """Stores token counts for a specific LLM role or the total."""
    model_config = ConfigDict(frozen=True) # Make immutable
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class UsageStatistics(BaseModel):
    """Detailed statistics about resource usage during the research process."""
    model_config = ConfigDict(frozen=True) # Make immutable

    token_usage: Dict[Literal['planner', 'summarizer', 'writer', 'refiner', 'total'], TokenUsageCounter] = Field(..., description="Token counts broken down by LLM role.")
    estimated_cost: Dict[Literal['planner', 'summarizer', 'writer', 'refiner', 'total'], float] = Field(..., description="Estimated costs broken down by LLM role (USD).")
    serper_queries_used: int = Field(..., description="Total number of calls made to the Serper Search API.")
    sources_processed_count: int = Field(..., description="Total number of unique source URLs fetched and processed.")
    refinement_iterations_run: int = Field(..., description="Number of refinement loops (search -> process -> refine) executed.")

class ResearchResponse(BaseModel):
    """Response schema for the main `/research` endpoint."""
    model_config = ConfigDict(extra='ignore')
    report: str = Field(..., description="The final generated research report in Markdown format.")
    usage_statistics: UsageStatistics = Field(..., description="Detailed statistics about resource usage (tokens, cost, API calls).")
    # Removed old fields:
    # usage_statistics: Dict[str, Any] = Field(..., description="Statistics about token usage, costs, etc.") 
    # llm_token_usage: Dict[str, int] 
    # estimated_llm_cost: float
    # serper_queries_used: int 