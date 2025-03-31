from __future__ import annotations

from typing import Optional

from langchain_core.messages import BaseMessage

from .snowflake import Snowflake


__all__ = ("Thread", "Message")


class Thread(Snowflake):
    """Represents a chat thread."""

    @classmethod
    async def create(cls) -> Thread:
        """Create a new chat thread."""
        return cls(id=Snowflake.new_id())


class Message(Snowflake):
    """Wrapper around a message from LangChain"""

    data: BaseMessage
    attachment: Optional[str]
    thread: Thread

    @classmethod
    async def create(cls, data: BaseMessage, attachment: Optional[str], thread: Thread) -> Message:
        """Create a new chat message."""
        return cls(id=Snowflake.new_id(), data=data, attachment=attachment, thread=thread)