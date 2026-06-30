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

def build_memory_chain(llm_instance):
    base_chain = conversational_prompt | llm_instance | StrOutputParser()
    return RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

# Default instance for backwards compatibility / non-UI usage
memory_chain = build_memory_chain(llm)

def chat(question: str, session_id: str = "default", llm_override=None) -> str:
    chain = build_memory_chain(llm_override) if llm_override else memory_chain
    return chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}},
    )

def chat_stream(question: str, session_id: str = "default", llm_override=None):
    """Yield response chunks in real-time with memory history."""
    chain = build_memory_chain(llm_override) if llm_override else memory_chain
    for chunk in chain.stream(
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
