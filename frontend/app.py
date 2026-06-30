"""
Streamlit frontend for the AI Research Assistant (Phases 1-7).

Run with:
    uv run streamlit run frontend/app.py
"""
import sys
from pathlib import Path

# Allow imports from the project root (chains/, loaders/, models/, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from chains.chatbot_chain import ask
from chains.memory_chain import chat as memory_chat
from chains.rag_chain import ask_with_sources
from loaders.document_loader import load_file
from loaders.text_splitter import split_documents
from models.vectorstore import build_vectorstore, add_documents

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI Research Assistant")
st.caption("LangChain + Gemini + FAISS  ·  Phases 1–7")

tab_chat, tab_memory, tab_upload, tab_rag = st.tabs(
    ["💬 Basic Chat", "🧠 Memory Chat", "📄 Upload Documents", "📚 Ask Your Documents (RAG)"]
)

# ── Tab 1: Phase 1 - Basic Chatbot ──────────────────────────────────────────
with tab_chat:
    st.subheader("Basic Chatbot")
    st.caption("Stateless — every question is independent, no memory.")

    question = st.text_input("Ask anything", key="basic_question")
    if st.button("Ask", key="basic_ask_btn"):
        if question.strip():
            with st.spinner("Thinking..."):
                try:
                    answer = ask(question)
                    st.success(answer)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Type a question first.")

# ── Tab 2: Phase 7 - Conversation Memory ────────────────────────────────────
with tab_memory:
    st.subheader("Memory Chat")
    st.caption("Remembers earlier turns in this session.")

    if "memory_messages" not in st.session_state:
        st.session_state.memory_messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit-session"

    for role, text in st.session_state.memory_messages:
        with st.chat_message(role):
            st.write(text)

    user_msg = st.chat_input("Type a message...")
    if user_msg:
        st.session_state.memory_messages.append(("user", user_msg))
        with st.chat_message("user"):
            st.write(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    reply = memory_chat(user_msg, session_id=st.session_state.session_id)
                    st.write(reply)
                    st.session_state.memory_messages.append(("assistant", reply))
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.button("Clear conversation"):
        st.session_state.memory_messages = []
        st.rerun()

# ── Tab 3: Phase 3-5 - Document Upload, Split, Embed ────────────────────────
with tab_upload:
    st.subheader("Upload Documents")
    st.caption("Supports PDF, TXT, Markdown, and HTML. Files are split into chunks and embedded into FAISS.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md", "html", "htm"],
        accept_multiple_files=True,
    )

    chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)

    if st.button("Process and index"):
        if not uploaded_files:
            st.warning("Upload at least one file first.")
        else:
            data_dir = Path("data/raw")
            data_dir.mkdir(parents=True, exist_ok=True)

            all_chunks = []
            for f in uploaded_files:
                save_path = data_dir / f.name
                save_path.write_bytes(f.getbuffer())

                with st.spinner(f"Loading {f.name}..."):
                    try:
                        docs = load_file(str(save_path))
                        chunks = split_documents(
                            docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                        )
                        all_chunks.extend(chunks)
                        st.write(f"✅ {f.name} → {len(chunks)} chunks")
                    except Exception as e:
                        st.error(f"Failed to process {f.name}: {e}")

            if all_chunks:
                with st.spinner("Embedding and saving to FAISS..."):
                    try:
                        add_documents(all_chunks)
                        st.success(f"Indexed {len(all_chunks)} chunks total. Ready for RAG queries.")
                    except Exception as e:
                        st.error(f"Failed to build vector store: {e}")

# ── Tab 4: Phase 6 - RAG ─────────────────────────────────────────────────────
with tab_rag:
    st.subheader("Ask Your Documents")
    st.caption("Retrieves relevant chunks from FAISS, then answers using only that context.")

    rag_question = st.text_input("Ask a question about your uploaded documents", key="rag_question")
    if st.button("Search & Answer", key="rag_btn"):
        if rag_question.strip():
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    result = ask_with_sources(rag_question)
                    st.markdown("### Answer")
                    st.success(result["answer"])

                    st.markdown("### Sources used")
                    for i, src in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}"):
                            st.write(src)
                except FileNotFoundError:
                    st.error("No documents indexed yet. Upload some in the 'Upload Documents' tab first.")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Type a question first.")

st.divider()
st.caption("Built with LangChain · Gemini · FAISS · Streamlit")
