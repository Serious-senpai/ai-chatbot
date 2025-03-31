from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Callable, ClassVar, DefaultDict, Deque, List, Optional, TYPE_CHECKING

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.messages.utils import _chunk_to_msg


__all__ = ("ChunkStreamSingleton",)


class ChunkStreamSingleton:

    __slots__ = ("__chunks", "__consumers", "__listeners")
    __instance__: ClassVar[Optional[ChunkStreamSingleton]] = None
    if TYPE_CHECKING:
        __chunks: DefaultDict[int, Deque[BaseMessageChunk]]
        __consumers: DefaultDict[int, List[asyncio.Queue[BaseMessageChunk]]]
        __listeners: DefaultDict[int, List[Callable[[BaseMessage], None]]]

    def __new__(cls) -> ChunkStreamSingleton:
        if cls.__instance__ is None:
            self = super().__new__(cls)
            self.__chunks = defaultdict(deque)
            self.__consumers = defaultdict(list)
            self.__listeners = defaultdict(list)

            cls.__instance__ = self

        return cls.__instance__

    def add_chunk(self, thread_id: int, chunk: BaseMessageChunk) -> None:
        """Add a chunk to the associated thread's queue."""
        self.__chunks[thread_id].append(chunk)

        for q in self.__consumers[thread_id]:
            q.put_nowait(chunk)

    def consume(self, thread_id: int) -> BaseMessage:
        """Consume all chunks for the associated thread and return the combined message."""
        chunks = self.__chunks[thread_id]  # Fast attr lookup
        final_chunk = chunks.popleft()
        while len(chunks) > 0:
            final_chunk += chunks.popleft()

        message = _chunk_to_msg(final_chunk)
        for listener in self.__listeners[thread_id]:
            listener(message)

        self.__listeners[thread_id].clear()
        return message

    def listen(self, thread_id: int, callback: Callable[[BaseMessage], None]) -> None:
        """Add a listener to the associated thread.

        The listener will be called when the next message is consumed.
        """
        self.__listeners[thread_id].append(callback)

    def subscribe(self, thread_id: int) -> asyncio.Queue[BaseMessageChunk]:
        """Subscribe to the associated thread."""
        q: asyncio.Queue[BaseMessageChunk] = asyncio.Queue()
        self.__consumers[thread_id].append(q)
        return q

    def unsubscribe(self, thread_id: int, q: asyncio.Queue[BaseMessageChunk]) -> None:
        """Unsubscribe from the associated thread."""
        self.__consumers[thread_id].remove(q)