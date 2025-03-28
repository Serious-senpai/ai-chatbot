from __future__ import annotations

import asyncio
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from .llm import Groq
from .results import Result
from .tools import f, TOOLS
from ..cli import namespace, parse_args
from ..stream import ChunkStreamSingleton


__all__ = ("graph",)


parse_args()
LLM_F = Groq(model=namespace.model, tools=[f])


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


async def chatbot(state: State, config: RunnableConfig) -> State:
    thread_id = config["metadata"]["thread_id"]
    stream = ChunkStreamSingleton()
    async for chunk in LLM_F.astream(state["messages"]):
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)
    return State(messages=[message])


async def tools(state: State) -> State:
    outputs: List[BaseMessage] = []

    async def _process(call: ToolCall) -> None:
        try:
            tool = TOOLS[call["name"]]
            result = await tool.ainvoke(call["args"])

        except Exception as e:
            result = Result(error=str(e))

        assert isinstance(result, Result), "all tools must return `Result`"
        message = ToolMessage(
            result.model_dump_json(),
            name=call["name"],
            tool_call_id=call["id"],
        )

        outputs.append(message)

    message = state["messages"][-1]
    if isinstance(message, AIMessage):
        await asyncio.gather(*[_process(c) for c in message.tool_calls])

    return State(messages=outputs)


def route_tools(state: State) -> str:
    if len(state["messages"]) == 0:
        return END

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        return END

    if len(message.tool_calls) == 0:
        return END

    return "tools"


graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, ["tools", END])
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = graph_builder.compile(memory)
# with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
