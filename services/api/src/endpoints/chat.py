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
from sse_starlette.sse import EventSourceResponse
from langchain_core.documents.base import Blob

from ..cli import namespace, parse_args
from ..config import EMBEDDING_MODEL_READY
from ..models import Message, Thread
from ..rag import GraphState, graph
from ..state import ThreadStateSingleton
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
    event: Literal["ai", "event", "chunk"]
    data: str


class __AttachmentPayload(BaseModel):
    filename: Annotated[str, Field(description="File name")]
    data: Annotated[str, Field(description="The base64-encoded file data")]


class __CreateMessageBody(BaseModel):
    content: Annotated[str, Field(description="The content of the message")]
    file: Annotated[Optional[__AttachmentPayload], Field(description="The file to send")] = None


async def __yield_messages(
    content: str,
    filename: Optional[str],
    data: Optional[bytes],
    thread: Thread,
) -> AsyncIterable[__MessageStreamPayload]:
    state = ThreadStateSingleton()
    async with state.lock(thread.id):
        message = await Message.create(HumanMessage(content), filename, thread)
        yield __MessageStreamPayload(event="ai", data=message.model_dump_json())

        await asyncio.sleep(0)

        history = MESSAGES.setdefault(thread, deque())
        history.appendleft(message)

        if data is not None:
            blob = Blob.from_data(data)

            parser = PyPDFParser()

            yield __MessageStreamPayload(event="event", data=f"Pulling model {namespace.embed!r} to Ollama server...")
            await EMBEDDING_MODEL_READY.wait()

            yield __MessageStreamPayload(event="event", data=f"Reading {filename!r}...")
            await asyncio.sleep(0)

            chroma = await asyncio.to_thread(
                Chroma.from_documents,
                documents=list(parser.lazy_parse(blob)),
                collection_name=str(thread.id),
                embedding=OllamaEmbeddings(
                    model=namespace.embed,
                    base_url=namespace.ollama,
                ),
            )
            retriever = chroma.as_retriever()
            state.retrievers[thread.id] = retriever

        yield __MessageStreamPayload(event="event", data="Generating response...")
        await asyncio.sleep(0)

        task = asyncio.create_task(
            graph.ainvoke(
                GraphState(
                    messages=[message.data],
                    temperature=1.0,
                    documents=[],
                    rag_generation="",
                ),
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

        response = await task
        state.retrievers.pop(thread.id, None)

        ai_message = await Message.create(response["messages"][-1], None, thread)
        history.appendleft(ai_message)

        yield __MessageStreamPayload(event="ai", data=ai_message.model_dump_json())


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
        file = body.file
        filename = file.filename if file else None
        data = b64decode(file.data.encode("utf-8"), validate=True) if file else None
    except binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")

    return EventSourceResponse(__yield_messages(body.content, filename, data, thread))


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
