from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import AsyncIterable, ClassVar, DefaultDict, Deque, List, Optional, TYPE_CHECKING

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.messages.utils import _chunk_to_msg


__all__ = ("ChunkStreamSingleton",)


class ChunkStreamSingleton:

    __slots__ = ("__chunks", "__messages", "__listeners")
    __instance__: ClassVar[Optional[ChunkStreamSingleton]] = None
    if TYPE_CHECKING:
        __chunks: DefaultDict[int, Deque[BaseMessageChunk]]
        __messages: DefaultDict[int, Deque[BaseMessage]]
        __listeners: DefaultDict[int, List[asyncio.Queue[BaseMessageChunk]]]

    def __new__(cls) -> ChunkStreamSingleton:
        if cls.__instance__ is None:
            self = super().__new__(cls)
            self.__chunks = defaultdict(deque)
            self.__messages = defaultdict(deque)
            self.__listeners = defaultdict(list)

            cls.__instance__ = self

        return cls.__instance__

    def add_chunk(self, thread_id: int, chunk: BaseMessageChunk) -> None:
        chunks = self.__chunks[thread_id]  # Fast attr lookup
        chunks.append(chunk)

        for q in self.__listeners[thread_id]:
            q.put_nowait(chunk)

        if chunk.response_metadata.get("done", False):
            final_chunk = chunks.popleft()
            while len(chunks) > 0:
                final_chunk += chunks.popleft()

            message = _chunk_to_msg(final_chunk)
            self.__messages[thread_id].append(message)

    def consume(self, thread_id: int) -> BaseMessage:
        return self.__messages[thread_id].popleft()

    def subscribe(self, thread_id: int) -> asyncio.Queue[BaseMessageChunk]:
        q: asyncio.Queue[BaseMessageChunk] = asyncio.Queue()
        self.__listeners[thread_id].append(q)
        return q

    def unsubscribe(self, thread_id: int, q: asyncio.Queue[BaseMessageChunk]) -> None:
        self.__listeners[thread_id].remove(q)
