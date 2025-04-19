from pydantic import BaseModel, HttpUrl, EmailStr, Field
from typing import List, Optional

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
    start_urls: List[HttpUrl] = Field(..., description="List of company homepage URLs to extract profiles from")

class CPEResponse(BaseModel):
    profiles: List[CompanyProfile] = Field(..., description="Extracted company profiles for each domain")
    usage_statistics: dict = Field(..., description="Usage and cost statistics from the run") 