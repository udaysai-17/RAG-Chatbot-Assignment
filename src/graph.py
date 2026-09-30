from typing import TypedDict

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.config import COHERE_API_KEY, PINECONE_INDEX_NAME


class RAGState(TypedDict):
    question: str
    context: str
    retrieved_context_chunks: list[str]
    answer: str
    confidence_score: float


embeddings = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=COHERE_API_KEY
)


vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-a-03-2025",
    temperature=0
)


def retrieve(state: RAGState):
    question = state["question"]

    results = vectorstore.similarity_search_with_score(
        question,
        k=4
    )

    chunks = [
        doc.page_content
        for doc, score in results
    ]

    context = "\n\n".join(chunks)

    if results:
        scores = [
            float(score)
            for doc, score in results
        ]

        confidence_score = sum(scores) / len(scores)

        confidence_score = max(
            0.0,
            min(1.0, confidence_score)
        )
    else:
        confidence_score = 0.0

    return {
        "context": context,
        "retrieved_context_chunks": chunks,
        "confidence_score": confidence_score
    }


def generate_answer(state: RAGState):
    question = state["question"]
    context = state["context"]

    confidence_score = state.get(
        "confidence_score",
        0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context
from the Agentic AI eBook.

If the answer cannot be found in the context, say exactly:

"I couldn't find that information in the provided document."

Do not use outside knowledge.
Do not invent information.
Keep the answer clear and concise.

Context:
{context}
"""
        ),
        (
            "human",
            "{question}"
        )
    ])

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    answer = response.content

    if "I couldn't find that information" in answer:
        confidence_score = 0.0

    return {
        "answer": answer,
        "confidence_score": confidence_score
    }


workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()