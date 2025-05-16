from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from yarl import URL
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src import EMBEDDING_MODEL_READY, HTTPSessionSingleton, ROOT, namespace, parse_args
from src.endpoints import chat
from src.rag import llm_cleanup

try:
    import uvloop
except ImportError:
    print("Unable to import uvloop, asynchronous operations will be slower")
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@asynccontextmanager
async def __lifespan(app: FastAPI) -> AsyncGenerator[None]:
    http = HTTPSessionSingleton()
    await http.prepare()

    async def _pull_embedding_model() -> None:
        ollama = URL(namespace.ollama)
        async with http.session.post(
            ollama.with_path("/api/pull"),
            json={"model": namespace.embed, "stream": False},
            timeout=None,
        ) as response:
            await response.read()
            print(f"Download {namespace.embed!r} {response.status}", file=sys.stderr)
            EMBEDDING_MODEL_READY.set()

    asyncio.create_task(_pull_embedding_model())

    yield

    await http.close()
    await llm_cleanup()


parse_args()
app = FastAPI(
    title="AI Chatbot API",
    summary="HTTP API for AI Chatbot",
    root_path="/api",
    lifespan=__lifespan,
)
app.include_router(chat.router)

outputs_dir = ROOT / "services" / "api" / "outputs"
outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

if namespace.cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )


if __name__ == "__main__":
    print(f"[{os.getpid()}]", namespace, file=sys.stderr)
    uvicorn.run(
        Path(__file__).stem + ":app",
        host=namespace.host,
        port=namespace.port,
        log_level=namespace.log_level,
    )
