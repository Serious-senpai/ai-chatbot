from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, NotRequired, Optional, Sequence, TypedDict, Union

import aiohttp
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import agenerate_from_stream
from langchain_core.utils.function_calling import convert_to_openai_tool

from ..config import GROQ_API_KEY


__all__ = ("Groq", "Ollama")


class CustomChatModel(BaseChatModel):

    _tools: List[Dict[str, Any]] = []
    _session: Optional[aiohttp.ClientSession] = None

    base_url: str

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(self.base_url)

        return self._session

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, Literal["any"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        for tool in tools:
            if isinstance(tool, BaseTool):
                self._tools.append(convert_to_openai_tool(tool))
            elif isinstance(tool, dict):
                self._tools.append(tool)
            else:
                raise TypeError(f"Unsupported tool: {tool!r}")

        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("This implementation does not support synchronous generation")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(self._astream(messages, stop, run_manager, **kwargs))


class Groq(CustomChatModel):

    base_url: str = "https://api.groq.com"
    model: str

    @property
    def _llm_type(self) -> str:
        return f"Groq[{self.model}]"

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        temperature: Optional[float] = kwargs.get("temperature")

        async with self.session.post(
            "/openai/v1/chat/completions",
            json={
                "messages": [self.__dump_message(message) for message in messages],
                "model": self.model,
                "n": 1,
                "stop": stop,
                "stream": True,
                "temperature": temperature,
                "tools": self._tools,
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
        ) as response:
            response.raise_for_status()

            while data := await response.content.readline():
                line = data.decode("utf-8").strip()
                if len(line) == 0:
                    continue

                line = line.removeprefix("data: ")
                chunk = json.loads(line)

                message = chunk["choices"][0]
                if message.get("finish_reason") is not None:
                    break

                delta = message["delta"]
                if tool_calls := delta.get("tool_calls"):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    name=t["function"]["name"],
                                    args=json.loads(t["function"]["arguments"]),
                                    id=t["id"],
                                ) for t in tool_calls
                            ],
                            id=chunk["id"],
                        )
                    )

                else:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=message["delta"]["content"] or "",
                            id=chunk["id"],
                        ),
                    )

    @staticmethod
    def __dump_message(message: BaseMessage) -> Dict[str, Any]:
        message_dict: Dict[str, Any]
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}

        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}

        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}

            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
                # If function call only, content is None not empty string
                if message_dict["content"] == "":
                    message_dict["content"] = None

            if message.tool_calls or message.invalid_tool_calls:
                message_dict["tool_calls"] = []
            elif "tool_calls" in message.additional_kwargs:
                message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
                # If tool calls only, content is None not empty string
                if message_dict["content"] == "":
                    message_dict["content"] = None

        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}

        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }

        elif isinstance(message, ToolMessage):
            message_dict = {
                "role": "tool",
                "content": message.content,
                "tool_call_id": message.tool_call_id,
            }

        else:
            raise TypeError(f"Got unknown type {message}")

        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]

        return message_dict


class OllamaMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: NotRequired[Dict[str, Any]]


class Ollama(CustomChatModel):

    base_url: str
    model: str

    @property
    def _llm_type(self) -> str:
        return f"Ollama[{self.model}]"

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        temperature: Optional[float] = kwargs.get("temperature")

        async with self.session.post(
            "/api/chat",
            json={
                "messages": [self.__dump_message(message) for message in messages],
                "model": self.model,
                "stream": True,
                "options": {
                    "temperature": temperature,
                },
                "tools": self._tools,
            },
        ) as response:
            response.raise_for_status()

            while data := await response.content.readline():
                line = data.decode("utf-8").strip()
                if len(line) == 0:
                    continue

                chunk = json.loads(line)

                if chunk.get("done", False):
                    break

                message = chunk["message"]
                if tool_calls := message.get("tool_calls"):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content="",
                            tool_calls=[
                                ToolCall(
                                    name=t["function"]["name"],
                                    args=t["function"]["arguments"],
                                    id=None,
                                ) for t in tool_calls
                            ],
                        )
                    )

                else:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=message["content"],
                        ),
                    )

    @staticmethod
    def __dump_message(message: BaseMessage) -> OllamaMessage:
        message_dict: OllamaMessage
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": str(message.content)}  # type: ignore

        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": str(message.content)}

        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": str(message.content)}

        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": str(message.content)}

        elif isinstance(message, ToolMessage):
            message_dict = {
                "role": "tool",
                "content": str(message.content),
            }

        else:
            raise TypeError(f"Got unknown type {message}")

        print(f"From {message}\nTo {message_dict}")
        return message_dict
