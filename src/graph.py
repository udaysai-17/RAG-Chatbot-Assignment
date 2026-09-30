from typing import TypedDict

from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from src.config import COHERE_API_KEY, PINECONE_INDEX_NAME


class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# Embedding model
embeddings = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=COHERE_API_KEY
)


# Pinecone vector store
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)


# Cohere chat model
llm = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-a-03-2025",
    temperature=0
)


def retrieve(state: RAGState):
    question = state["question"]

    docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    return {
        "context": context
    }


def generate_answer(state: RAGState):
    question = state["question"]
    context = state["context"]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context
from the Agentic AI eBook.

If the answer cannot be found in the context, say:
"I couldn't find that information in the provided document."

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

    return {
        "answer": response.content
    }


# Create LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

rag_graph = workflow.compile()