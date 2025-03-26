from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import uvicorn
import uvloop
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import namespace, parse_args
from src.endpoints import chat


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
app = FastAPI(
    title="AI Chatbot API",
    summary="HTTP API for AI Chatbot",
    root_path="/api",
)
app.include_router(chat.router)

parse_args()
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
