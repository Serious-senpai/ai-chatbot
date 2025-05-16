from __future__ import annotations

import asyncio
from typing import ClassVar, Dict, Optional, TYPE_CHECKING

import chromadb


__all__ = ("ThreadStateSingleton",)


class ThreadStateSingleton:

    __slots__ = ("chroma", "locks")
    __instance__: ClassVar[Optional[ThreadStateSingleton]] = None
    if TYPE_CHECKING:
        chroma: chromadb.ClientAPI
        locks: Dict[int, asyncio.Lock]

    def __new__(cls) -> ThreadStateSingleton:
        if cls.__instance__ is None:
            self = super().__new__(cls)
            self.chroma = chromadb.Client()
            self.locks = {}

            cls.__instance__ = self

        return cls.__instance__

    def lock(self, thread_id: int) -> asyncio.Lock:
        try:
            return self.locks[thread_id]

        except KeyError:
            lock = asyncio.Lock()
            self.locks[thread_id] = lock
            return lock
