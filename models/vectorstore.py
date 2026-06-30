"""
Phase 5 - Vector Store (FAISS)
Builds, persists, and loads a FAISS index from document chunks.
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from models.embeddings import embeddings

VECTORSTORE_DIR = "data/vectorstore"


def build_vectorstore(chunks: list[Document], save_path: str = VECTORSTORE_DIR) -> FAISS:
    """Create a new FAISS index from document chunks and persist it to disk."""
    if not chunks:
        raise ValueError("No chunks provided to build vector store.")

    store = FAISS.from_documents(chunks, embeddings)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    store.save_local(save_path)
    print(f"Vector store saved to {save_path} ({len(chunks)} chunks)")
    return store


def load_vectorstore(save_path: str = VECTORSTORE_DIR) -> FAISS:
    """Load an existing FAISS index from disk."""
    if not Path(save_path).exists():
        raise FileNotFoundError(
            f"No vector store found at {save_path}. Run build_vectorstore() first."
        )
    return FAISS.load_local(
        save_path, embeddings, allow_dangerous_deserialization=True
    )


def add_documents(chunks: list[Document], save_path: str = VECTORSTORE_DIR) -> FAISS:
    """Add new chunks to an existing store, creating one if it doesn't exist."""
    if Path(save_path).exists():
        store = load_vectorstore(save_path)
        store.add_documents(chunks)
        store.save_local(save_path)
        return store
    return build_vectorstore(chunks, save_path)


if __name__ == "__main__":
    # uv run python -m models.vectorstore
    from loaders.document_loader import load_directory
    from loaders.text_splitter import split_documents

    docs = load_directory("data/raw")
    chunks = split_documents(docs)
    store = build_vectorstore(chunks)

    results = store.similarity_search("What is LangChain?", k=3)
    for r in results:
        print(r.page_content[:150], "\n---")
