from pydantic import BaseModel, Field, field_validator, HttpUrl
from typing import List, Dict, Any, Optional, Literal

class SearchTask(BaseModel):
    """Validates a single search task from the planner."""
    query: str = Field(..., description="The search query to execute", min_length=1)
    endpoint: str = Field("/search", description="The Serper API endpoint to use")
    num_results: int = Field(5, description="Number of results to request", ge=1, le=20)
    reasoning: Optional[str] = Field(None, description="Reasoning for this search task")

    @field_validator('endpoint')
    def validate_endpoint(cls, v):
        allowed_endpoints = ['/search', '/scholar', '/news']
        if v not in allowed_endpoints:
            raise ValueError(f"Invalid endpoint '{v}'. Allowed: {allowed_endpoints}")
        return v

class SectionItem(BaseModel):
    """Validates a section item in the writing plan."""
    title: str = Field(..., description="Title of the section", min_length=1)
    guidance: str = Field(..., description="Guidance for writing this section", min_length=1)

class WritingPlan(BaseModel):
    """Validates the writing plan from the planner."""
    overall_goal: str = Field(..., description="Overall goal of the research report", min_length=1)
    desired_tone: Optional[str] = Field(None, description="Desired tone for the report")
    sections: List[SectionItem] = Field(..., description="Sections of the report", min_items=1)
    additional_directives: Optional[List[str]] = Field(None, description="Additional writing directives")

    # For backward compatibility, derive outline from section titles if needed
    @property
    def outline(self) -> List[str]:
        """Return section titles as outline points for backward compatibility."""
        return [section.title for section in self.sections]

class PlannerOutput(BaseModel):
    """Validates the complete output from the planner LLM."""
    search_tasks: List[SearchTask] = Field(..., description="List of search tasks to execute", min_items=1)
    writing_plan: WritingPlan = Field(..., description="Plan for writing the report")

class SourceSummary(BaseModel):
    """Validates processed information from a source."""
    title: str = Field(..., description="Title of the source", min_length=1)
    link: HttpUrl = Field(..., description="URL of the source")
    content: str = Field(..., description="Generated summary or concatenated relevant chunks of the source content", min_length=1)
    content_type: Literal['summary', 'chunks'] = Field(..., description="Indicates whether the content is a full summary or concatenated chunks")
    original_title: Optional[str] = Field(None, description="Original title if different (e.g., before chunking)")

class SearchRequest(BaseModel):
    """Validates a search request from the writer/refiner LLM."""
    query: str = Field(..., description="Specific search query needed to improve the report", min_length=1) 