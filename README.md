# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangGraph, Cohere, Pinecone, and Streamlit.

## Project Overview

This application answers questions based only on the provided Agentic AI eBook PDF.

The system:

1. Loads the PDF document.
2. Splits the document into smaller chunks.
3. Generates embeddings using Cohere.
4. Stores the embeddings in Pinecone.
5. Retrieves relevant document chunks for a user question.
6. Uses Cohere Chat to generate an answer.
7. Uses LangGraph to manage the RAG workflow.
8. Provides a Streamlit web interface.

## Technologies Used

- Python
- LangChain
- LangGraph
- Cohere
- Pinecone
- Streamlit
- PyPDF
- Python-dotenv

## Setup

Create and activate a virtual environment:

```bash
py -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add:

```env
COHERE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=agentic-ai-index
```

## Ingest the PDF

Run:

```bash
python -m src.ingestion
```

The PDF is loaded, split into chunks, embedded using Cohere, and stored in Pinecone.

## Run the Chatbot

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## RAG Workflow

```text
PDF
  ↓
Text Chunks
  ↓
Cohere Embeddings
  ↓
Pinecone Vector Database
  ↓
Similarity Search
  ↓
LangGraph
  ↓
Cohere LLM
  ↓
Answer
```

## Project Structure

```text
RAG-Chatbot-Assignment/
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   └── ingestion.py
├── app.py
├── test_graph.py
├── test_sample_queries.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Features

- PDF-based question answering
- Semantic similarity search
- Pinecone vector database
- Cohere embeddings
- Cohere LLM generation
- LangGraph RAG workflow
- Streamlit web interface
- Context-grounded responses
- Prevents unsupported answers when information is not available in the document

## Example

### Question

```text
What is Agentic AI?
```

### Answer

The chatbot retrieves relevant content from the Agentic AI eBook and generates an answer based on the retrieved document context.

For questions unrelated to the provided document, the chatbot responds:

```text
I couldn't find that information in the provided document.
```

## Testing

The RAG pipeline can be tested using:

```bash
python test_graph.py
```

The application was tested with both relevant questions from the document and unrelated questions to verify that the chatbot does not generate unsupported answers.