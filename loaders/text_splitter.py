"""
Phase 4 - Text Splitting
Splits loaded documents into chunks ready for embedding.
"""
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Recursive character splitting - the default, works well for most text."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def split_documents_by_tokens(
    documents: list[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Token-based splitting - more accurate for staying within model limits."""
    splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def compare_chunk_sizes(documents: list[Document]) -> None:
    """Experiment helper: print chunk counts for different size configs."""
    for size in [500, 1000, 2000]:
        chunks = split_documents(documents, chunk_size=size, chunk_overlap=int(size * 0.2))
        print(f"chunk_size={size}: {len(chunks)} chunks")
        if chunks:
            print(f"  sample: {chunks[0].page_content[:120]!r}\n")


if __name__ == "__main__":
    # uv run python -m loaders.text_splitter
    from loaders.document_loader import load_directory

    docs = load_directory("data/raw")
    if docs:
        compare_chunk_sizes(docs)
