"""
Phase 7 - Conversation Memory
Wraps the basic chat chain with per-session message history.
"""
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from models.llm import llm
from prompts.chat_prompt import conversational_prompt

# In-memory session store. Swap for Redis/DB in production.
_session_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


_base_chain = conversational_prompt | llm | StrOutputParser()

memory_chain = RunnableWithMessageHistory(
    _base_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


def chat(question: str, session_id: str = "default") -> str:
    return memory_chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )


def chat_stream(question: str, session_id: str = "default"):
    """Yield response chunks in real-time with memory history."""
    for chunk in memory_chain.stream(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    ):
        yield chunk


if __name__ == "__main__":
    # uv run python -m chains.memory_chain
    print("Testing chat:")
    print(chat("My favorite programming language is Python.", session_id="test"))
    print("\nTesting chat_stream:")
    for chunk in chat_stream("Generate a short code example.", session_id="test"):
        print(chunk, end="", flush=True)
    print()

