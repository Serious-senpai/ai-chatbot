from __future__ import annotations

from typing import List

from yarl import URL

from ..cli import namespace, parse_args
from ..http import HTTPSessionSingleton


__all__ = ("embed_documents",)


async def embed_documents(documents: List[str]) -> List[List[float]]:
    parse_args()

    url = URL(namespace.ollama).with_path("/api/embed")
    async with HTTPSessionSingleton().session.post(url, json={"model": namespace.embed, "input": documents}) as response:
        response.raise_for_status()
        data = await response.json(encoding="utf-8")
        return data["embeddings"]
