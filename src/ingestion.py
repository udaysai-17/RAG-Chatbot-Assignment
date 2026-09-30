from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import COHERE_API_KEY, PINECONE_INDEX_NAME


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "Ebook-Agentic-AI.pdf"


def ingest_pdf():
    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    embeddings = CohereEmbeddings(
       model="embed-v4.0",
       cohere_api_key=COHERE_API_KEY
    )

    print("Uploading embeddings to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("PDF ingestion completed successfully.")


if __name__ == "__main__":
    ingest_pdf()