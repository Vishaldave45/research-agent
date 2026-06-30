"""
Phase 1 - Basic Chatbot
A minimal LCEL chain: prompt -> llm -> parser
"""
from langchain_core.output_parsers import StrOutputParser
from models.llm import llm
from prompts.chat_prompt import basic_chat_prompt

# The core chain. This is the pattern every later chain builds on.
basic_chat_chain = basic_chat_prompt | llm | StrOutputParser()


def ask(question: str) -> str:
    """Convenience wrapper for direct calls / quick testing."""
    return basic_chat_chain.invoke({"question": question})


def ask_stream(question: str):
    """Yield response chunks in real-time."""
    for chunk in basic_chat_chain.stream({"question": question}):
        yield chunk


if __name__ == "__main__":
    # Quick manual test: uv run python -m chains.chatbot_chain
    print("Testing invoke:")
    print(ask("What is LangChain in one sentence?"))
    print("\nTesting stream:")
    for chunk in ask_stream("What is LangChain in one sentence?"):
        print(chunk, end="", flush=True)
    print()

