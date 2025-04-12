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


__all__ = ("GraphState", "graph", "llm_cleanup")


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


async def __route_query(state: GraphState) -> Literal["vectorstore", "chat"]:
    # Copy the thread history
    history = state["messages"][:]

    # Insert system message at position -2
    history.insert(
        -2,
        SystemMessage(
            "You are an expert at routing a user's question to an appropriate tool. Analyze the user's question and determine:\n"
            "- If the question requires searching the provided document, answer \"vectorstore\".\n"
            "- Otherwise, answer \"chat\".\n\n"
            "Your answer must contain exactly ONE word \"vectorstore\" or \"chat\". No yapping.\n"
        ),
    )

    retry = 3
    for _ in range(retry):
        message = await __ROUTE_QUERY_LLM.ainvoke(history, temperature=state["temperature"])

        content = message.content
        if content in ["vectorstore", "chat"]:
            return content  # type: ignore

        history.append(message)
        history.append(
            HumanMessage(
                "Your previous response is invalid. Please answer again with exactly ONE word \"vectorstore\" or \"chat\"."
            ),
        )

    raise ValueError(f"Failed to route query after {retry} attempts. Last response: {content!r}")


async def __vectorstore_retriever(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    chroma = ThreadStateSingleton().chroma.get(thread_id)

    if chroma is not None:
        retriever = chroma.as_retriever()
        documents = await retriever.ainvoke(cast(str, state["messages"][-1].content))
    else:
        documents = []

    return GraphState(
        messages=[],
        temperature=state["temperature"],
        documents=documents,
        rag_generation=state["rag_generation"],
    )


__ROUTE_GRADER_LLM = Groq(model=namespace.model)


async def __route_grader(state: GraphState) -> Literal["vectorstore-generate", "chat"]:
    # Copy the thread history
    history: List[BaseMessage] = [
        SystemMessage(
            "You are a grader assessing relevance of a retrieved document to a user question.\n"
            "It does not need to be a stringent test. The goal is to filter out erroneous retrievals.\n"
            "- If the document contains keyword(s) or semantic meaning related to the user question, answer \"relevant\"\n"
            "- Otherwise, answer \"irrelevant\".\n\n"
            "Your answer must contain exactly ONE word \"relevant\" or \"irrelevant\". No yapping.\n",
        ),
        HumanMessage(f"Retrieved documents:\n\n{state['documents']}\n\nUser question:\n\n{state['messages'][-1]}"),
    ]

    retry = 3
    for _ in range(retry):
        message = await __ROUTE_GRADER_LLM.ainvoke(history, temperature=state["temperature"])

        content = message.content
        if content == "relevant":
            return "vectorstore-generate"

        if content == "irrelevant":
            return "chat"

        history.append(message)
        history.append(
            HumanMessage(
                "Your previous response is invalid. Please answer again with exactly ONE word \"relevant\" or \"irrelevant\"."
            ),
        )

    raise ValueError(f"Failed to route query after {retry} attempts. Last response: {content!r}")


__VECTORSTORE_PROMPT = hub.pull("rlm/rag-prompt")
assert isinstance(__VECTORSTORE_PROMPT, ChatPromptTemplate)

__VECTORSTORE_LLM = Groq(model=namespace.model)
__VECTORSTORE_CHAIN = __VECTORSTORE_PROMPT | __VECTORSTORE_LLM | StrOutputParser()


async def __vectorstore_generate(state: GraphState) -> GraphState:
    generation = await __VECTORSTORE_CHAIN.ainvoke({"context": state["documents"], "question": state["messages"][-1].content})
    return GraphState(
        messages=[],
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=generation,
    )


__VECTORSTORE_SUMMARY_LLM = Groq(model=namespace.model)


async def __vectorstore_summary(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    history = state["messages"][:]
    history.append(SystemMessage("You are an expert at answering user's question based on the provided document."))
    history.append(
        HumanMessage("Summarize the documents below to answer my previous question:\n" + state["rag_generation"]),
    )

    async for chunk in __VECTORSTORE_SUMMARY_LLM.astream(
        history,
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


__CHAT_LLM = Groq(model=namespace.model)
__CHAT_LLM.bind_tools([search])


async def __chat(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    async for chunk in __CHAT_LLM.astream(
        state["messages"],
        temperature=state["temperature"],
    ):
        # print(chunk, file=sys.stderr)
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)

    return GraphState(
        messages=[message],
        temperature=state["temperature"],
        documents=state["documents"],
        rag_generation=state["rag_generation"],
    )


def __route_chat(state: GraphState) -> str:
    if len(state["messages"]) == 0:
        return END

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        return END

    if len(message.tool_calls) == 0:
        return END

    return "search-tool"


graph_builder = StateGraph(GraphState)
graph_builder.add_node("vectorstore", __vectorstore_retriever)
graph_builder.add_node("vectorstore-generate", __vectorstore_generate)
graph_builder.add_node("vectorstore-summary", __vectorstore_summary)
graph_builder.add_node("search-tool", __tools)
graph_builder.add_node("chat", __chat)

# Routing
graph_builder.add_conditional_edges(
    START,
    __route_query,
    ["vectorstore", "chat"],
)

# Vectorstore branch
graph_builder.add_conditional_edges(
    "vectorstore",
    __route_grader,
    ["vectorstore-generate", "chat"],
)
graph_builder.add_edge("vectorstore-generate", "vectorstore-summary")
graph_builder.add_edge("vectorstore-summary", END)

# Chat branch
graph_builder.add_conditional_edges(
    "chat",
    __route_chat,
    ["search-tool", END],
)
graph_builder.add_edge("search-tool", "chat")


memory = MemorySaver()
graph = graph_builder.compile(memory)


async def llm_cleanup() -> None:
    for llm in (__ROUTE_QUERY_LLM, __VECTORSTORE_LLM, __ROUTE_GRADER_LLM, __VECTORSTORE_SUMMARY_LLM, __CHAT_LLM):
        await llm.session.close()
