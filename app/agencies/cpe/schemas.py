from pydantic import BaseModel, HttpUrl, EmailStr, Field
from typing import List, Optional
# Import SearchTask and UsageStatistics from core schemas
from app.core.schemas import SearchTask, UsageStatistics

class ExtractedCompanyData(BaseModel):
    """Data expected to be extracted by the LLM from HTML."""
    name: str = Field(..., description="Official company name")
    description: str = Field(..., description="One-sentence description of what the company does")
    location: Optional[str] = Field(None, description="Headquarters or main location if available")
    contact_page: Optional[HttpUrl] = Field(None, description="Canonical contact or 'About' page URL if identified")
    emails: List[EmailStr] = Field(..., description="List of unique email addresses found")

class CompanyProfile(BaseModel):
    """Final structured company profile including the domain."""
    domain: HttpUrl = Field(..., description="The company's base URL")
    name: str = Field(..., description="Official company name")
    description: str = Field(..., description="One-sentence description of what the company does")
    location: Optional[str] = Field(None, description="Headquarters or main location if available")
    contact_page: Optional[HttpUrl] = Field(None, description="Canonical contact or 'About' page URL if identified")
    emails: List[EmailStr] = Field(..., description="List of unique email addresses found")

class CPERequest(BaseModel):
    # Replace start_urls with query, location, and max_search_tasks
    query: str = Field(..., description="User query describing the types of companies to find.")
    location: Optional[str] = Field(None, description="Optional geographic location to filter companies.")
    max_search_tasks: Optional[int] = Field(
        default=10, 
        description="Maximum number of search tasks the planner should generate.",
        ge=1, le=20 # Example bounds, adjust as needed
    )
    # start_urls: List[HttpUrl] = Field(..., description="List of company homepage URLs to extract profiles from")

class CPEPlannerOutput(BaseModel):
    """Validates structured JSON output expected from the Planner LLM."""
    search_tasks: List[SearchTask] = Field(
        ..., 
        description="List of search tasks designed to find websites of companies matching the query and location.", 
        min_items=1
    )

class CPEResponse(BaseModel):
    profiles: List[CompanyProfile] = Field(..., description="Extracted company profiles for each domain")
    # Update usage_statistics to use the core UsageStatistics schema
    usage_statistics: UsageStatistics = Field(..., description="Usage and cost statistics from the run") 