from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional


class TokenUsageCounter(BaseModel):
    """Stores token counts for a specific LLM role or the total."""
    model_config = ConfigDict(frozen=True) # Make immutable

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class UsageStatistics(BaseModel):
    """Detailed statistics about resource usage, potentially used by multiple agencies."""
    model_config = ConfigDict(frozen=True) # Make immutable

    token_usage: Optional[Dict[str, TokenUsageCounter]] = Field(None, description="Token counts broken down by LLM role or task.")
    estimated_cost: Optional[Dict[str, float]] = Field(None, description="Estimated costs broken down by LLM role or task (USD)." )
    serper_queries_used: int = Field(..., description="Total number of calls made to the Serper Search API.")
    sources_processed_count: int = Field(..., description="Total number of unique source URLs fetched and processed.")
    refinement_iterations_run: int = Field(..., description="Number of refinement loops (search -> process -> refine) executed.") 