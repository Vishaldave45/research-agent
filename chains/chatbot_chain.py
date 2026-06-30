"""
Phase 1 - Basic Chatbot
A minimal LCEL chain: prompt -> llm -> parser
"""
from langchain_core.output_parsers import StrOutputParser
from models.llm import llm
from prompts.chat_prompt import basic_chat_prompt

def build_basic_chat_chain(llm_instance):
    return basic_chat_prompt | llm_instance | StrOutputParser()

# Default instance for backwards compatibility / non-UI usage
basic_chat_chain = build_basic_chat_chain(llm)

def ask(question: str, llm_override=None) -> str:
    """Convenience wrapper for direct calls / quick testing."""
    chain = build_basic_chat_chain(llm_override) if llm_override else basic_chat_chain
    return chain.invoke({"question": question})

def ask_stream(question: str, llm_override=None):
    """Yield response chunks in real-time."""
    chain = build_basic_chat_chain(llm_override) if llm_override else basic_chat_chain
    for chunk in chain.stream({"question": question}):
        yield chunk

if __name__ == "__main__":
    # Quick manual test: uv run python -m chains.chatbot_chain
    print("Testing invoke:")
    print(ask("What is LangChain in one sentence?"))
    print("\nTesting stream:")
    for chunk in ask_stream("What is LangChain in one sentence?"):
        print(chunk, end="", flush=True)
    print()
