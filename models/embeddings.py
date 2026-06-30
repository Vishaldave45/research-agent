"""
Phase 5 - Embeddings
Wraps Google's embedding model so it's importable as a singleton.
"""
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
