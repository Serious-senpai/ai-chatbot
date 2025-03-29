from __future__ import annotations

from typing import ClassVar, Dict, Optional, TYPE_CHECKING

from langchain_core.vectorstores import VectorStoreRetriever


__all__ = ("RetrieverSingleton",)


class RetrieverSingleton:

    __slots__ = ("retrievers",)
    __instance__: ClassVar[Optional[RetrieverSingleton]] = None
    if TYPE_CHECKING:
        retrievers: Dict[int, VectorStoreRetriever]

    def __new__(cls) -> RetrieverSingleton:
        if cls.__instance__ is None:
            self = super().__new__(cls)
            self.retrievers = {}
            cls.__instance__ = self

        return cls.__instance__
