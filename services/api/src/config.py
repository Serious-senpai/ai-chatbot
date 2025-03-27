from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


__all__ = ("ROOT", "GROQ_API_KEY", "TAVILY_API_KEY")


ROOT = Path(__file__).parent.parent.parent.parent.resolve()
load_dotenv(ROOT / ".env", verbose=False)

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
