"""
Single shared LLM instance used across the whole app.
Supports dynamic provider switching between Gemini, Groq, and OpenRouter.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm(provider: str, temperature: float = 0.7):
    """
    Factory function to retrieve LLM instances dynamically by provider name.
    Supports 'gemini' | 'groq' | 'openrouter'.
    """
    provider = provider.lower()
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment.")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=temperature,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        if not os.getenv("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY not found in environment.")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )
    elif provider == "openrouter":
        try:
            import importlib
            openai_module = importlib.import_module("langchain_openai")
            ChatOpenAI = openai_module.ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is required for OpenRouter. Please run 'uv add langchain-openai' to install it."
            )

        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found in environment.")
        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            model="meta-llama/llama-3.3-70b-instruct",
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Default instances for backwards compatibility and standalone tests
try:
    if os.getenv("GOOGLE_API_KEY") and not os.getenv("GOOGLE_API_KEY").startswith("AQ."):
        # Use Gemini if a valid API key exists
        llm = get_llm("gemini", temperature=0.7)
        llm_precise = get_llm("gemini", temperature=0.1)
    elif os.getenv("GROQ_API_KEY"):
        # Fallback to Groq if Google key is missing or invalid
        llm = get_llm("groq", temperature=0.7)
        llm_precise = get_llm("groq", temperature=0.1)
    else:
        # Prevent crash at import time by using a placeholder
        from langchain_core.language_models.fake import FakeListLLM
        llm = FakeListLLM(responses=["Please configure GOOGLE_API_KEY or GROQ_API_KEY in your .env file."])
        llm_precise = llm
except Exception:
    from langchain_core.language_models.fake import FakeListLLM
    llm = FakeListLLM(responses=["Failed to initialize default LLM. Check API keys."])
    llm_precise = llm
