import asyncio
from typing import Dict, Any

# Firestore client instance (initialized in main.py)
db: Any = None

# Dictionary to track active background tasks (e.g., agent/orchestrator runs)
active_tasks: Dict[str, asyncio.Task] = {} 