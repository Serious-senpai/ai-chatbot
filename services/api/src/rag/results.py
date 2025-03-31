from __future__ import annotations

from typing import Annotated, Generic, Optional, TypeVar

from pydantic import BaseModel, Field


__all__ = ("Result",)
T = TypeVar("T")


class Result(BaseModel, Generic[T]):
    """Schema representing result from a tool call."""
    result: Annotated[Optional[T], Field(description="The result of the operation")] = None
    error: Annotated[Optional[str], Field(description="A string describing the error, or `None` if the operation succeeded")] = None