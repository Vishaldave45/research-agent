"""
Quick manual test runner for Phases 1-10.
uv run python main.py
"""
from chains.chatbot_chain import ask
from chains.lcel_pipelines import sequential_chain, parallel_chain
from loaders.document_loader import load_directory
from loaders.text_splitter import split_documents
from models.vectorstore import build_vectorstore
from chains.rag_chain import ask_with_sources
from chains.memory_chain import chat
from chains.agent_chain import run_agent
from chains.structured_chain import research, summarize, fact_check



def main():
    print("=" * 60)
    print("PHASE 1 - Basic Chatbot")
    print("=" * 60)
    print(ask("What is LangChain in one sentence?"))

    print("\n" + "=" * 60)
    print("PHASE 2 - LCEL Pipelines")
    print("=" * 60)
    print(parallel_chain.invoke({"text": "LangChain helps developers build LLM apps."}))

    print("\n" + "=" * 60)
    print("PHASE 3-5 - Load, Split, Embed, Store")
    print("=" * 60)
    docs = load_directory("data/raw")
    if docs:
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    else:
        print("No documents found in data/raw - skipping RAG demo.")

    print("\n" + "=" * 60)
    print("PHASE 6 - RAG")
    print("=" * 60)
    if docs:
        result = ask_with_sources("What is the goal of this project?")
        print(result["answer"])

    print("\n" + "=" * 60)
    print("PHASE 7 - Memory")
    print("=" * 60)
    print(chat("My favorite language is Python.", session_id="demo"))
    print(chat("Give me an example.", session_id="demo"))

    print("\n" + "=" * 60)
    print("PHASE 8-9 - Agent with Tools")
    print("=" * 60)
    result = run_agent("What is 15 * 8, and what's today's date?")
    print(f"\nAnswer: {result['answer']}")
    print(f"Tools used: {result['tools_used']}")

    print("\n" + "=" * 60)
    print("PHASE 10 - Structured Output")
    print("=" * 60)
    
    print("Running Structured Research Chain...")
    res_struct = research("LangChain Agents")
    print(f"Topic: {res_struct.topic}")
    print(f"Findings: {res_struct.findings}")
    print(f"Confidence: {res_struct.confidence}")
    
    print("\nRunning Structured Fact Checking Chain...")
    fc_struct = fact_check("The Sun is a star.")
    print(f"Verdict: {fc_struct.verdict}")
    print(f"Evidence: {fc_struct.evidence}")
    print(f"Confidence: {fc_struct.confidence}")


if __name__ == "__main__":
    main()

