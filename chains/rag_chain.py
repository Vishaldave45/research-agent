"""
Phase 6 - Retrieval-Augmented Generation (RAG)
Retrieves relevant chunks from FAISS, then answers using only that context.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from models.llm import llm
from models.vectorstore import load_vectorstore
from prompts.chat_prompt import rag_prompt

def _format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def get_rag_chain_with_llm(llm_instance):
    """Build the RAG chain with the provided LLM instance."""
    store = load_vectorstore()
    retriever = store.as_retriever(search_kwargs={"k": 4})
    chain = (
        RunnableParallel(
            context=retriever | _format_docs,
            question=RunnablePassthrough(),
        )
        | rag_prompt
        | llm_instance
        | StrOutputParser()
    )
    return chain, retriever

def get_rag_chain():
    """Build RAG chain using the default LLM (for backwards compatibility)."""
    return get_rag_chain_with_llm(llm)

def ask_with_sources(question: str, llm_override=None) -> dict:
    """Returns both the answer and the source chunks used - good for the API layer."""
    target_llm = llm_override if llm_override else llm
    chain, retriever = get_rag_chain_with_llm(target_llm)
    sources = retriever.invoke(question)
    answer = chain.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:300] for doc in sources],
    }

def ask_with_sources_stream(question: str, llm_override=None):
    """Yields sources first, then answer chunks in real-time."""
    target_llm = llm_override if llm_override else llm
    chain, retriever = get_rag_chain_with_llm(target_llm)
    sources = retriever.invoke(question)
    yield {"sources": [doc.page_content[:300] for doc in sources]}
    for chunk in chain.stream(question):
        yield {"answer_chunk": chunk}

if __name__ == "__main__":
    # uv run python -m chains.rag_chain
    print("Testing invoke:")
    result = ask_with_sources("What is the goal of this project?")
    print("Answer:", result["answer"])
    print("\nSources used:")
    for s in result["sources"]:
        print("-", s[:100], "...")

    print("\nTesting stream:")
    for chunk in ask_with_sources_stream("What is the goal of this project?"):
        if "sources" in chunk:
            print("Sources:", chunk["sources"])
        elif "answer_chunk" in chunk:
            print(chunk["answer_chunk"], end="", flush=True)
    print()
