from src.graph import rag_graph


result = rag_graph.invoke({
    "question": "What is Agentic AI?",
    "context": "",
    "answer": ""
})

print("\nANSWER:\n")
print(result["answer"])