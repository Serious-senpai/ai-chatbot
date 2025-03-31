from __future__ import annotations

import asyncio
from typing import Annotated, List, Literal, TypedDict, cast

from langchain import hub
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

from .llm import Groq
from .results import Result
from .tools import search, TOOLS
from ..cli import namespace, parse_args
from ..state import ThreadStateSingleton
from ..stream import ChunkStreamSingleton


__all__ = ("GraphState", "graph")


parse_args()


class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    temperature: float
    documents: List[Document]
    rag_generation: str


async def __tools(state: GraphState) -> GraphState:
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

    return GraphState(
        messages=outputs,
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=state["rag_generation"],
    )


__ROUTE_QUERY_LLM = Groq(model=namespace.model)
__ROUTE_QUERY_SYSTEM = SystemMessage(
    "You are an expert at routing a user's question to an appropriate tool. Analyze the user's question and determine:\n"
    "- If the question requires searching the provided document, answer \"vectorstore\".\n"
    "- If the question requires information you do not know or you are unsure of, answer \"search\".\n"
    "- Otherwise, answer \"chat\".\n"
    "Your answer must contain exactly ONE word \"vectorstore\", \"search\", or \"chat\". No yapping.\n"
)


async def __route_query(state: GraphState) -> Literal["vectorstore", "search", "chat"]:
    # Do not stream message chunks from this node

    # Insert system message at position -2
    state["messages"].insert(-2, __ROUTE_QUERY_SYSTEM)
    message = await __ROUTE_QUERY_LLM.ainvoke(state["messages"], temperature=state["temperature"])

    # Remove system message
    state["messages"].pop(-2)

    content = message.content
    if content not in ["vectorstore", "search", "chat"]:
        raise ValueError(f"Unexpected route_query {content!r}")

    return content  # type: ignore


__VECTORSTORE_PROMPT = hub.pull("rlm/rag-prompt")
assert isinstance(__VECTORSTORE_PROMPT, ChatPromptTemplate)

__VECTORSTORE_LLM = Groq(model=namespace.model)
__VECTORSTORE_CHAIN = __VECTORSTORE_PROMPT | __VECTORSTORE_LLM | StrOutputParser()


async def __vectorstore(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    retriever = ThreadStateSingleton().retrievers.get(thread_id)
    if retriever is not None:
        documents = await retriever.ainvoke(cast(str, state["messages"][-1].content))
    else:
        documents = []

    generation = await __VECTORSTORE_CHAIN.ainvoke({"context": documents, "question": state["messages"][-1].content})
    return GraphState(
        messages=[],
        temperature=state["temperature"],
        documents=documents,
        rag_generation=generation,
    )


__VECTORSTORE_SUMMARY_LLM = Groq(model=namespace.model)
__VECTORSTORE_SUMMARY_SYSTEM = SystemMessage("You are an expert at answering user's question based on the provided document.")


async def __vectorstore_summary(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    state["messages"].append(__VECTORSTORE_SUMMARY_SYSTEM)
    state["messages"].append(
        HumanMessage("Summarize the documents below to answer my previous question:\n" + state["rag_generation"]),
    )

    async for chunk in __VECTORSTORE_SUMMARY_LLM.astream(
        state["messages"],
        temperature=state["temperature"],
    ):
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)

    state["messages"].pop()  # Remove the summary message
    state["messages"].pop()  # Remove the system message

    return GraphState(
        messages=[message],
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=state["rag_generation"],
    )


__LLM_SEARCH = Groq(model=namespace.model)
__LLM_SEARCH.bind_tools([search])


async def __search(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    async for chunk in __LLM_SEARCH.astream(
        state["messages"],
        temperature=state["temperature"],
    ):
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)
    return GraphState(
        messages=[message],
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=state["rag_generation"],
    )


def __route_search(state: GraphState) -> str:
    if len(state["messages"]) == 0:
        return END

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        return END

    if len(message.tool_calls) == 0:
        return END

    return "search-tool"


__CHAT_LLM = Groq(model=namespace.model)


async def __chat(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    async for chunk in __CHAT_LLM.astream(
        state["messages"],
        temperature=state["temperature"],
    ):
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)
    return GraphState(
        messages=[message],
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=state["rag_generation"],
    )


graph_builder = StateGraph(GraphState)
graph_builder.add_node("vectorstore", __vectorstore)
graph_builder.add_node("vector_store_summary", __vectorstore_summary)
graph_builder.add_node("search", __search)
graph_builder.add_node("search-tool", __tools)
graph_builder.add_node("chat", __chat)

# Routing
graph_builder.add_conditional_edges(
    START,
    __route_query,
    ["vectorstore", "search", "chat"],
)

# Vectorstore branch
graph_builder.add_edge("vectorstore", "vector_store_summary")
graph_builder.add_edge("vector_store_summary", END)

# Search branch
graph_builder.add_conditional_edges(
    "search",
    __route_search,
    ["search-tool", END],
)
graph_builder.add_edge("search-tool", "search")


# Chat branch
graph_builder.add_edge("chat", END)


memory = MemorySaver()
graph = graph_builder.compile(memory)
# with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
