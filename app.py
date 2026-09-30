import streamlit as st

from src.graph import rag_graph


st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI RAG Chatbot")
st.write("Ask questions based on the Agentic AI eBook.")

question = st.text_input(
    "Enter your question:",
    placeholder="What is Agentic AI?"
)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            result = rag_graph.invoke({
                "question": question,
                "context": "",
                "answer": ""
            })

        st.subheader("Answer")
        st.write(result["answer"])