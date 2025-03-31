from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar

from pydantic import BaseModel


__all__ = ("Snowflake",)
EPOCH = datetime(2025, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)


class Snowflake(BaseModel):
    """Represents a snowflake model"""

    id: int
    __id_counter: ClassVar[int] = 0

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Snowflake):
            return self.id == other.id

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def created_at(self) -> datetime:
        return EPOCH + timedelta(milliseconds=self.id >> 16)

    @staticmethod
    def new_id() -> int:
        now = datetime.now(tz=timezone.utc)
        milliseconds = int(1000 * (now - EPOCH).total_seconds())

        Snowflake.__id_counter = (Snowflake.__id_counter + 1) & 0xFFFF
        return (milliseconds << 16) | Snowflake.__id_counter