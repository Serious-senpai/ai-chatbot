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
from fastapi.middleware.cors import CORSMiddleware

from src import HTTPSessionSingleton, namespace, parse_args
from src.endpoints import chat

try:
    import uvloop
except ImportError:
    print("Unable to import uvloop, asynchronous operations will be slower")
else:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


@asynccontextmanager
async def __lifespan(app: FastAPI) -> AsyncGenerator[None]:
    http = HTTPSessionSingleton()

    ollama = URL(namespace.ollama)
    await http.prepare()
    async with http.session.post(
        ollama.with_path("/api/pull"),
        json={"model": namespace.embed},
    ) as response:
        await response.read()
        print(response.status, file=sys.stderr)

    yield
    await http.close()


parse_args()
app = FastAPI(
    title="AI Chatbot API",
    summary="HTTP API for AI Chatbot",
    root_path="/api",
    lifespan=__lifespan,
)
app.include_router(chat.router)

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
        workers=namespace.workers,
        log_level=namespace.log_level,
    )
