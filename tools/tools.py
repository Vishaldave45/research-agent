"""
Phase 8 - Custom Tools
Five tools the agent can choose to invoke.
"""
import datetime
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression, e.g. '12 * (4 + 3)'. Returns the numeric result."""
    try:
        # NOTE: eval is fine here for a learning project; sandbox it in production.
        allowed = "0123456789+-*/(). "
        if not all(c in allowed for c in expression):
            return "Error: expression contains disallowed characters."
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def text_summarizer(text: str) -> str:
    """Summarize a block of text into 2-3 sentences."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from models.llm import llm_precise

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following text in 2-3 sentences."),
        ("human", "{text}"),
    ])
    chain = prompt | llm_precise | StrOutputParser()
    return chain.invoke({"text": text})


@tool
def keyword_extractor(text: str) -> str:
    """Extract the 5 most important keywords from a block of text, comma-separated."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from models.llm import llm_precise

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract exactly 5 keywords from this text. Return only "
                   "a comma-separated list, nothing else."),
        ("human", "{text}"),
    ])
    chain = prompt | llm_precise | StrOutputParser()
    return chain.invoke({"text": text})


@tool
def document_search(query: str) -> str:
    """Search the document knowledge base (FAISS vector store) for relevant content."""
    from models.vectorstore import load_vectorstore

    try:
        store = load_vectorstore()
    except FileNotFoundError:
        return "No documents have been indexed yet."

    results = store.similarity_search(query, k=3)
    if not results:
        return "No relevant documents found."
    return "\n---\n".join(d.page_content[:300] for d in results)


ALL_TOOLS = [calculator, current_datetime, text_summarizer, keyword_extractor, document_search]


if __name__ == "__main__":
    # uv run python -m tools.tools
    print(calculator.invoke({"expression": "12 * (4 + 3)"}))
    print(current_datetime.invoke({}))
