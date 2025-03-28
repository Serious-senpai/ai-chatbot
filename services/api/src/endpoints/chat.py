from __future__ import annotations

import asyncio
import binascii
from base64 import b64decode
from collections import deque
from typing import Annotated, Any, AsyncIterable, Deque, Dict, List, Literal, Optional, TypedDict

from fastapi import APIRouter, HTTPException
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessageChunk, HumanMessage
from langchain_core.vectorstores import VectorStoreRetriever
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents.base import Blob

from ..cli import namespace, parse_args
from ..models import Message, Thread
from ..rag import GraphState, graph
from ..stream import ChunkStreamSingleton


__all__ = ("router",)
router = APIRouter(
    prefix="/chat",
)


parse_args()
THREADS: Dict[int, Thread] = {}
MESSAGES: Dict[Thread, Deque[Message]] = {}


@router.post(
    "/",
    name="Create a chat thread",
    description="Create a new chat thread",
)
async def create_thread() -> Thread:
    thread = await Thread.create()
    THREADS[thread.id] = thread
    return thread


@router.get(
    "/",
    name="Get all chat threads",
    description="Get a list of all chat threads",
)
async def get_threads() -> List[Thread]:
    return list(THREADS.values())


@router.get(
    "/{thread_id}",
    name="Get a chat thread",
    description="Get a chat thread by ID",
)
async def get_thread(thread_id: int) -> Thread:
    try:
        return THREADS[thread_id]

    except KeyError:
        raise HTTPException(status_code=404, detail="Thread not found")


class __MessageStreamPayload(TypedDict):
    event: Literal["message", "chunk"]
    data: Any


class __CreateMessageBody(BaseModel):
    content: Annotated[str, Field(description="The content of the message")]
    file: Annotated[Optional[str], Field(description="The base64-encoded file to send")]


async def __yield_messages(content: str, file: Optional[bytes], thread: Thread) -> AsyncIterable[__MessageStreamPayload]:
    message = await Message.create(HumanMessage(content), thread)
    yield __MessageStreamPayload(event="message", data=message.model_dump_json())

    history = MESSAGES.setdefault(thread, deque())
    history.appendleft(message)

    retriever: Optional[VectorStoreRetriever] = None
    if file is not None:
        try:
            blob = Blob.from_data(b64decode(file))

        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")

        parser = PyPDFParser()
        retriever = Chroma.from_documents(
            documents=list(parser.lazy_parse(blob)),
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=namespace.ollama,
            ),
        ).as_retriever()

    task = asyncio.create_task(
        graph.ainvoke(
            GraphState(messages=[message.data], temperature=1.0, retriever=retriever),
            {
                "configurable": {"thread_id": thread.id},
            },
        ),
    )

    stream = ChunkStreamSingleton()
    queue = stream.subscribe(thread.id)
    try:
        # Stop streaming immediately when a complete answer is available
        event = asyncio.Event()
        stream.listen(thread.id, lambda _: event.set())

        to_cancel: List[asyncio.Task[Any]] = []
        while not event.is_set():
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(event.wait()),
                    asyncio.create_task(queue.get()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            to_cancel.extend(pending)

            for t in done:
                result = await t
                if isinstance(result, BaseMessageChunk):
                    yield __MessageStreamPayload(event="chunk", data=result.model_dump_json())

        for t in to_cancel:
            if not t.done():
                t.cancel()

    finally:
        stream.unsubscribe(thread.id, queue)

    data = await task
    ai_message = await Message.create(data["messages"][-1], thread)
    history.appendleft(ai_message)
    yield __MessageStreamPayload(event="message", data=ai_message.model_dump_json())


@router.post(
    "/{thread_id}/messages",
    name="Send a message",
    description="Send a message to a chat thread",
)
async def send_message(
    thread_id: int,
    body: __CreateMessageBody,
) -> EventSourceResponse:
    try:
        thread = THREADS[thread_id]
    except KeyError:
        raise HTTPException(status_code=404, detail="Thread not found")

    try:
        base64 = body.file
        file = b64decode(base64) if base64 else None
    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    return EventSourceResponse(__yield_messages(body.content, file, thread))


@router.get(
    "/{thread_id}/messages",
    name="Get all messages in a thread",
    description="Get a list of all messages in a chat thread",
)
async def get_messages(thread_id: int) -> List[Message]:
    try:
        thread = THREADS[thread_id]

    except KeyError:
        raise HTTPException(status_code=404, detail="Thread not found")

    else:
        return list(MESSAGES.get(thread, deque()))
