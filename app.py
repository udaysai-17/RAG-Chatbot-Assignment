import streamlit as st

from src.graph import rag_graph


st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)


st.title("🤖 Agentic AI RAG Chatbot")

st.write(
    "Ask questions based on the Agentic AI eBook."
)


question = st.text_input(
    "Enter your question:",
    placeholder="What is Agentic AI?"
)


if st.button("Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Generating answer..."
        ):

            result = rag_graph.invoke({
                "question": question,
                "context": "",
                "retrieved_context_chunks": [],
                "answer": "",
                "confidence_score": 0.0
            })


        # Get confidence safely
        confidence = result.get(
            "confidence_score",
            0.0
        )


        # Answer
        st.subheader("Answer")

        st.write(
            result.get(
                "answer",
                "No answer generated."
            )
        )


        # Confidence score
        st.subheader("Confidence Score")

        st.write(
            f"{confidence:.2f}"
        )


        # Retrieved context
        st.subheader("Retrieved Context")

        chunks = result.get(
            "retrieved_context_chunks",
            []
        )


        if chunks:

            for i, chunk in enumerate(
                chunks,
                start=1
            ):

                st.write(
                    f"**Chunk {i}:**"
                )

                st.write(chunk)

        else:

            st.write(
                "No context chunks retrieved."
            )


        # Structured response
        st.subheader("Response JSON")

        response_payload = {
            "query": question,
            "final_answer": result.get(
                "answer",
                ""
            ),
            "retrieved_context_chunks": chunks,
            "confidence_score": round(
                confidence,
                2
            )
        }


        st.json(response_payload)