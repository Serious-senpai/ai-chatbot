from __future__ import annotations

import asyncio
from collections import deque
from typing import Annotated, Any, AsyncIterable, Deque, Dict, List, Literal, TypedDict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessageChunk, HumanMessage
from sse_starlette.sse import EventSourceResponse

from ..models import Message, Thread
from ..rag import graph
from ..stream import ChunkStreamSingleton


__all__ = ("router",)
router = APIRouter(
    prefix="/chat",
)


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


async def __yield_messages(content: str, thread: Thread) -> AsyncIterable[__MessageStreamPayload]:
    message = await Message.create(HumanMessage(content), thread)
    yield __MessageStreamPayload(event="message", data=message.model_dump_json())

    history = MESSAGES.setdefault(thread, deque())
    history.appendleft(message)

    task = asyncio.create_task(
        graph.ainvoke(
            {"messages": [message.data]},
            {"configurable": {"thread_id": thread.id}},
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

            for item in done:
                if isinstance(item, BaseMessageChunk):
                    chunk = await queue.get()
                    yield __MessageStreamPayload(event="chunk", data=chunk.model_dump_json())

        for task in to_cancel:
            if not task.done():
                task.cancel()

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

    else:
        return EventSourceResponse(__yield_messages(body.content, thread))


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
