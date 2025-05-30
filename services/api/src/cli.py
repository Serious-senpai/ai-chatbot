from __future__ import annotations

import argparse
from typing import TYPE_CHECKING


__all__ = ("namespace", "parse_args")


class __Namespace(argparse.Namespace):
    if TYPE_CHECKING:
        host: str
        port: int
        log_level: str
        model: str
        embed: str
        ollama: str
        cors: bool


namespace = __Namespace()
__parser = argparse.ArgumentParser(
    description="Run HTTP API server for minichat application",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
__parser.add_argument("--host", type=str, default="0.0.0.0", help="The host to bind the HTTP server to")
__parser.add_argument("--port", type=int, default=80, help="The port to bind the HTTP server to")
__parser.add_argument("--log-level", type=str, default="debug", help="The log level for the application")
__parser.add_argument("--model", type=str, default="gemma2-9b-it", help="The LLM to use in Groq API")
__parser.add_argument("--embed", type=str, default="nomic-embed-text", help="The embedding model to use in Ollama service")
__parser.add_argument("--ollama", type=str, default="http://ollama:11434", help="The base URL for Ollama service")
__parser.add_argument("--cors", action="store_true", help="Enable CORS for the HTTP server")


def parse_args() -> None:
    __parser.parse_args(namespace=namespace)
