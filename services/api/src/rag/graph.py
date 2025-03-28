from __future__ import annotations

import asyncio
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain import hub

from .llm import Groq
from .results import Result
from .tools import f, TOOLS
from ..cli import namespace, parse_args
from ..stream import ChunkStreamSingleton


__all__ = ("GraphState", "graph")


parse_args()
LLM_F = Groq(model=namespace.model)
LLM_F.bind_tools([f])

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    temperature: float
    retriever: Optional[VectorStoreRetriever]
    question: Optional[str]
    generation: Optional[str]
    documents: Optional[List[str]]

# Initialize graders and rewriters
def initialize_graders():
    llm = Groq(model=namespace.model, temperature=0)
    
    # Document relevance grader
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    grade_prompt_template = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", grade_prompt_template),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
    retrieval_grader = grade_prompt | structured_llm_grader
    
    # Hallucination grader
    hallucination_grader_llm = llm.with_structured_output(GradeHallucinations)
    hallucination_prompt_template = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", hallucination_prompt_template),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | hallucination_grader_llm
    
    # Answer grader
    answer_grader_llm = llm.with_structured_output(GradeAnswer)
    answer_prompt_template = """You are a grader assessing whether an answer addresses / resolves a question \n 
         Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", answer_prompt_template),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ])
    answer_grader = answer_prompt | answer_grader_llm
    
    # Question rewriter
    rewrite_prompt_template = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", rewrite_prompt_template),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ])
    question_rewriter = rewrite_prompt | llm | StrOutputParser()
    
    # RAG chain
    rag_prompt = hub.pull("rlm/rag-prompt")
    rag_chain = rag_prompt | llm | StrOutputParser()
    
    return {
        "retrieval_grader": retrieval_grader,
        "hallucination_grader": hallucination_grader,
        "answer_grader": answer_grader,
        "question_rewriter": question_rewriter,
        "rag_chain": rag_chain
    }


GRADERS = initialize_graders()

async def chatbot(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config["configurable"]["thread_id"]
    stream = ChunkStreamSingleton()

    async for chunk in LLM_F.astream(
        state["messages"],
        temperature=state["temperature"],
        retriever=state["retriever"],
    ):
        stream.add_chunk(thread_id, chunk)

    message = stream.consume(thread_id)
    return GraphState(messages=[message], 
                      temperature=state["temperature"], 
                      retriever=state["retriever"],
                      question=state["question"],
                      generation=state["generation"],
                      documents=state["documents"])


async def tools(state: GraphState) -> GraphState:
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

    return GraphState(messages=outputs, 
                      temperature=state["temperature"], 
                      retriever=state["retriever"],
                      question=state["question"],
                      generation=state["generation"],
                      documents=state["documents"])

async def retrieve(state: GraphState) -> GraphState:
    print("---RETRIEVE---")
    question = state["question"]
    retriever = state["retriever"]
    
    if retriever:
        documents = await retriever.ainvoke(question)
        return GraphState(
            messages=state["messages"],
            temperature=state["temperature"],
            retriever=retriever,
            question=question,
            generation=state["generation"],
            documents=documents
        )
    
    return state


async def grade_documents(state: GraphState) -> GraphState:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return state
    
    # Score each doc
    filtered_docs = []
    for doc in documents:
        score = await GRADERS["retrieval_grader"].ainvoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    return GraphState(
        messages=state["messages"],
        temperature=state["temperature"],
        retriever=state["retriever"],
        question=question,
        generation=state["generation"],
        documents=filtered_docs
    )


async def transform_query(state: GraphState) -> GraphState:
    print("---TRANSFORM QUERY---")
    question = state["question"]
    
    better_question = await GRADERS["question_rewriter"].ainvoke({"question": question})
    
    return GraphState(
        messages=state["messages"],
        temperature=state["temperature"],
        retriever=state["retriever"],
        question=better_question,
        generation=state["generation"],
        documents=state["documents"]
    )


async def generate_rag(state: GraphState) -> GraphState:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    if not documents:
        return state
    
    generation = await GRADERS["rag_chain"].ainvoke({"context": documents, "question": question})
    
    return GraphState(
        messages=state["messages"],
        temperature=state["temperature"],
        retriever=state["retriever"],
        question=question,
        generation=generation,
        documents=documents
    )


def route_tools(state: GraphState) -> str:
    if len(state["messages"]) == 0:
        return END

    message = state["messages"][-1]
    if not isinstance(message, AIMessage):
        return END

    if len(message.tool_calls) == 0:
        return END

    return "tools"

def decide_to_generate(state: GraphState) -> str:
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate_rag"


async def grade_generation(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    if not documents or not generation:
        return "not_useful"

    score = await GRADERS["hallucination_grader"].ainvoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = await GRADERS["answer_grader"].ainvoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not_useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not_supported"


# Build the graph
graph_builder = StateGraph(GraphState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tools)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("generate_rag", generate_rag)
graph_builder.add_node("transform_query", transform_query)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", route_tools, ["tools", END])
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate_rag": "generate_rag",
    },
)
graph_builder.add_edge("transform_query", "retrieve")
graph_builder.add_edge("generate_rag", "chatbot")

memory = MemorySaver()
graph = graph_builder.compile(memory)
# with open("graph.png", "wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())
