"""
Single shared LLM instance used across the whole app.
Import `llm` anywhere you need to call Gemini.
"""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found. Add it to your .env file:\n"
        "GOOGLE_API_KEY=your_key_here"
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.7,
)

# Lower temperature variant - useful for structured/deterministic tasks (Phase 9+)
llm_precise = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.1,
)
