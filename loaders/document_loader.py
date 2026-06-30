"""
Phase 3 - Document Loading
Loads PDF, TXT, Markdown, and HTML files into a unified list of Documents.
"""
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    BSHTMLLoader,
)

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_file(file_path: str) -> list[Document]:
    """Load a single file using the appropriate loader based on extension."""
    path = Path(file_path)
    loader_cls = LOADER_MAP.get(path.suffix.lower())

    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    loader = loader_cls(str(path))
    return loader.load()


def load_directory(dir_path: str) -> list[Document]:
    """Load every supported file in a directory into one unified collection."""
    documents: list[Document] = []
    dir_path = Path(dir_path)

    for file_path in dir_path.rglob("*"):
        if file_path.suffix.lower() in LOADER_MAP:
            try:
                docs = load_file(str(file_path))
                documents.extend(docs)
                print(f"Loaded {len(docs)} doc(s) from {file_path.name}")
            except Exception as e:
                print(f"Failed to load {file_path.name}: {e}")

    return documents


if __name__ == "__main__":
    # uv run python -m loaders.document_loader
    docs = load_directory("data/raw")
    print(f"\nTotal documents loaded: {len(docs)}")
    if docs:
        print(f"First doc metadata: {docs[0].metadata}")
