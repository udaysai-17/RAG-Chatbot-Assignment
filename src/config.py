import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY is missing from .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

print("Environment variables loaded successfully.")
print("Pinecone index:", PINECONE_INDEX_NAME)