from __future__ import annotations

from langchain_ollama import ChatOllama

from ..cli import namespace, parse_args


__all__ = ("LLM",)


parse_args()
LLM = ChatOllama(
    model=namespace.model,
    base_url=namespace.ollama,
)
