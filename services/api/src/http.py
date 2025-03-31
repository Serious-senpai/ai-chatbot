from __future__ import annotations

import asyncio
from typing import ClassVar, Optional, TYPE_CHECKING

import aiohttp


__all__ = ("HTTPSessionSingleton",)


class HTTPSessionSingleton:

    __slots__ = ("__session", "__prepare_lock", "__prepared")
    __instance__: ClassVar[Optional[HTTPSessionSingleton]] = None
    if TYPE_CHECKING:
        __session: Optional[aiohttp.ClientSession]
        __prepare_lock: asyncio.Lock
        __prepared: bool

    def __new__(cls) -> HTTPSessionSingleton:
        if cls.__instance__ is None:
            cls.__instance__ = self = super().__new__(cls)
            self.__session = None
            self.__prepare_lock = asyncio.Lock()
            self.__prepared = False

        return cls.__instance__

    async def prepare(self) -> None:
        async with self.__prepare_lock:
            if not self.__prepared:
                self.__session = aiohttp.ClientSession()
                self.__prepared = True

    @property
    def session(self) -> aiohttp.ClientSession:
        if self.__session is None:
            raise RuntimeError("HTTP session hasn't been prepared yet. Did you call `prepare`?")

        return self.__session

    async def close(self) -> None:
        if self.__session is not None:
            await self.__session.close()
            self.__session = None

        self.__prepared = False