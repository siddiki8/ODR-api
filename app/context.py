from pydantic import BaseModel, Field, ConfigDict
from fastapi import WebSocket
from typing import Callable, Coroutine, Any

# Attempt to import config/keys from core
try:
    from app.core.config import AppSettings, ApiKeys
except ImportError:
    # Define dummy classes if core config isn't found (should exist)
    class AppSettings: pass
    class ApiKeys: pass

# Attempt to import service classes/types
# Paths might need adjustment based on final service definitions
try:
    from app.services.search_client import SerperClient # Example placeholder
    from app.services.ranking_client import TogetherClient # Example placeholder
    from app.services.web_scraper import WebScraper
except ImportError:
    # Define dummy classes if services aren't defined yet
    class SerperClient: pass
    class TogetherClient: pass
    class WebScraper: pass


# Define the type hint for the WebSocket callback function
WebSocketCallback = Callable[[str, dict | None], Coroutine[Any, Any, None]]

class ResearchDependencies(BaseModel):
    """Dependencies required by research agencies, agents, tools, and helpers."""
    settings: AppSettings
    api_keys: ApiKeys
    serper_client: SerperClient # Instance or client factory
    together_client: TogetherClient # Instance or client factory
    web_scraper: WebScraper # Instance or client factory
    websocket_callback: WebSocketCallback

    class Config:
        # Allow arbitrary types like client instances or callback functions
        arbitrary_types_allowed = True 