from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field, ValidationError, model_validator
from typing import Dict, Any

class ChunkSettings(BaseSettings):
    chunk_size: int = 2048
    chunk_overlap: int = 100
    min_chunk_size: int = 256

class DeepResearchConfig(BaseSettings):
    """Configuration specific to the Deep Research Agency."""
    
    # Sensitive key, loaded from env var DEEP_RESEARCH_TOGETHER_API_KEY
    together_api_key: SecretStr

    # Reranker model, loaded from env var DEEP_RESEARCH_RERANKER_MODEL, with default
    reranker_model: str = Field(default="Salesforce/Llama-Rank-V1")
    
    # Orchestration parameters with defaults, loadable from env vars
    max_refinement_loops: int = Field(default=2)
    rerank_relevance_threshold: float = Field(default=0.25)
    max_total_chunks: int = Field(default=1000)
    max_initial_search_tasks: int = Field(default=3, ge=1, le=10, description="Max searches Planner LLM can generate initially")
    
    # Chunking settings (as a nested model or dict)
    # Option 1: Nested Model (cleaner)
    default_chunk_settings: ChunkSettings = Field(default_factory=ChunkSettings)

    # Model configuration for BaseSettings
    model_config = SettingsConfigDict(
        env_prefix='DEEP_RESEARCH_', # Prefix for environment variables
        env_file='.env',             # Optionally load from .env file
        env_file_encoding='utf-8',
        extra='ignore'               # Ignore extra fields from env/file
    )

    @model_validator(mode='after')
    def check_api_key_present(cls, values):
        # Access values as a dictionary
        together_key = values.get('together_api_key')
        if not together_key:
            raise ValueError("DEEP_RESEARCH_TOGETHER_API_KEY must be set in the environment or .env file")
        return values

# Example usage (for testing if run directly)
if __name__ == "__main__":
    try:
        config = DeepResearchConfig()
        print("Deep Research Config loaded successfully:")
        print(f"  Reranker Model: {config.reranker_model}")
        print(f"  Max Loops: {config.max_refinement_loops}")
        print(f"  Threshold: {config.rerank_relevance_threshold}")
        print(f"  Max Initial Searches: {config.max_initial_search_tasks}")
        print(f"  Chunk Settings: {config.default_chunk_settings}")
        # Avoid printing secret directly
        print(f"  API Key Loaded: {bool(config.together_api_key.get_secret_value())}")
    except ValidationError as e:
        print(f"Error loading DeepResearchConfig: {e}")
    except ValueError as e: # Catch the ValueError from the validator
        print(f"Configuration Error: {e}") 