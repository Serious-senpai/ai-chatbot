from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv


__all__ = (
    "ROOT",
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "EMBEDDING_MODEL_READY",
)


ROOT = Path(__file__).parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env", verbose=False)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# Ready events
EMBEDDING_MODEL_READY = asyncio.Event()