from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, model_validator

class CPEConfig(BaseSettings):
    # Keep planner_model_id for potential future use, but extractor is primary
    planner_model_id: str = Field(default="openai/gpt-4.1-mini", description="Model ID for potential future planner agent")
    extractor_model_id: str = Field(..., description="Model ID for the company extractor agent")
    max_crawl_depth: int = Field(default=1, description="Max crawl depth for email finder per start URL")
    max_crawl_pages: int = Field(default=20, description="Max pages to crawl per start URL")

    # Use model_config for Pydantic V2 settings
    model_config = SettingsConfigDict(
        env_prefix = "CPE_",
        env_file = ".env",
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra env vars
    ) 

    @model_validator(mode='after')
    def check_extractor_model_id_present(cls, self):
        if not self.extractor_model_id:
            raise ValueError("CPE_EXTRACTOR_MODEL_ID must be set in the environment or .env file and must not be empty.")
        return self 