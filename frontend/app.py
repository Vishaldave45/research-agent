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

from chains.chatbot_chain import ask, ask_stream
from chains.memory_chain import chat as memory_chat, chat_stream
from chains.rag_chain import ask_with_sources, ask_with_sources_stream
from loaders.document_loader import load_file
from loaders.text_splitter import split_documents
from models.vectorstore import build_vectorstore, add_documents
from chains.structured_chain import (
    research as run_research,
    summarize as run_summarize,
    fact_check as run_fact_check,
    ask_structured as run_qa,
    generate_report as run_report,
)
from chains.multi_agent_workflow import coordinate



st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔎",
    layout="wide",
)

# ── Custom Styling Injection ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import modern premium font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif !important;
    }


    /* Radial gradient background */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827 0%, #030712 100%) !important;
        color: #F3F4F6 !important;
    }
    
    /* Make headers glow with indigo gradient */
    h1, h2, h3 {
        background: linear-gradient(135deg, #C7D2FE 0%, #818CF8 50%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    
    h1 {
        font-size: 2.8rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* Tabs list styled as glassmorphic capsule */
    div[data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        margin-bottom: 25px !important;
        gap: 8px !important;
    }
    
    /* Tab buttons style */
    button[data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 10px 20px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #9CA3AF !important;
        border: none !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(165, 180, 252, 0.15) 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25) !important;
    }
    
    /* Input fields and selectors style */
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
        background-color: rgba(17, 24, 39, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        color: #F9FAFB !important;
        padding: 12px !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #818CF8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }

    /* Button enhancements */
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 22px rgba(99, 102, 241, 0.5) !important;
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%) !important;
    }

    /* Custom containers / cards (st.container border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 18px !important;
        padding: 24px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3) !important;
        margin-bottom: 20px !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(99, 102, 241, 0.2) !important;
        box-shadow: 0 10px 40px 0 rgba(99, 102, 241, 0.08) !important;
    }

    /* Metric values custom style */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #818CF8 !important;
    }
    
    /* Notification alerts */
    div[data-testid="stNotification"] {
        border-radius: 12px !important;
        background-color: rgba(17, 24, 39, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Chat message backgrounds */
    div[data-testid="stChatMessage"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        margin-bottom: 10px !important;
    }
    
    div[data-testid="stChatMessage"][data-classname="stChatMessage-user"] {
        background-color: rgba(99, 102, 241, 0.08) !important;
    }

    div[data-testid="stChatMessage"][data-classname="stChatMessage-assistant"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔎 AI Research Assistant")
st.caption("LangChain + Gemini + FAISS  ·  Phases 1–12")

# ── Sidebar Configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("⚙️ System Configuration")
    
    # 1. Provider Selectbox
    if "provider" not in st.session_state:
        st.session_state.provider = "gemini"
        
    provider_choice = st.selectbox(
        "Select LLM Provider",
        ["gemini", "groq", "openrouter"],
        key="provider"
    )
    
    # 2. Dynamic instantiation & caching
    if st.session_state.get("provider_built") != st.session_state.provider or "active_llm" not in st.session_state:
        from models.llm import get_llm
        try:
            st.session_state.active_llm = get_llm(st.session_state.provider)
            st.session_state.provider_built = st.session_state.provider
        except Exception as e:
            st.error(f"Error loading {st.session_state.provider}: {e}")
            from langchain_core.language_models.fake import FakeListLLM
            st.session_state.active_llm = FakeListLLM(responses=[f"Configuration Error: {e}"])
            st.session_state.provider_built = st.session_state.provider
            
    # 3. Read model name dynamically
    model_name = ""
    if hasattr(st.session_state.active_llm, "model_name"):
        model_name = st.session_state.active_llm.model_name
    elif hasattr(st.session_state.active_llm, "model"):
        model_name = st.session_state.active_llm.model
        
    with st.container(border=True):
        st.markdown("**Active LLM Engine:**")
        if model_name:
            st.info(f"{st.session_state.provider} ({model_name})")
            st.caption(f"Active: {st.session_state.provider} ({model_name})")
        else:
            st.info(f"{st.session_state.provider}")
            st.caption(f"Active: {st.session_state.provider}")
        
        st.markdown("**Embeddings Engine:**")
        st.info("Gemini Embedding")
        
        st.markdown("**Vector Store:**")
        st.success("FAISS Database")

    st.divider()
    st.markdown("### 📚 Document Library")
    try:
        import os
        raw_files = os.listdir("data/raw")
        raw_files = [f for f in raw_files if f != ".gitkeep"]
        if raw_files:
            for f in raw_files:
                st.caption(f"📄 {f}")
        else:
            st.caption("No documents uploaded yet.")
    except Exception:
        st.caption("No document library found.")

tab_chat, tab_memory, tab_upload, tab_rag, tab_structured, tab_multi_agent = st.tabs(
    [
        "💬 Basic Chat",
        "🧠 Memory Chat",
        "📄 Upload Documents",
        "📚 Ask Your Documents (RAG)",
        "🧩 Structured Output",
        "🤖 Multi-Agent Coordinator",
    ]
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
                    st.write_stream(ask_stream(question, llm_override=st.session_state.active_llm))
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
                    reply = st.write_stream(
                        chat_stream(user_msg, session_id=st.session_state.session_id, llm_override=st.session_state.active_llm)
                    )
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

    with st.container(border=True):
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
                try:
                    sources_holder = []
                    def get_answer_chunks():
                        for chunk in ask_with_sources_stream(rag_question, llm_override=st.session_state.active_llm):
                            if "sources" in chunk:
                                sources_holder.extend(chunk["sources"])
                            elif "answer_chunk" in chunk:
                                yield chunk["answer_chunk"]
                                
                    st.markdown("### Answer")
                    st.write_stream(get_answer_chunks())

                    st.markdown("### Sources used")
                    if sources_holder:
                        for i, src in enumerate(sources_holder, 1):
                            with st.expander(f"Source {i}"):
                                st.write(src)
                    else:
                        st.info("No sources retrieved.")

                except FileNotFoundError:
                    st.error("No documents indexed yet. Upload some in the 'Upload Documents' tab first.")
                except Exception as e:
                    st.error(f"Error: {e}")


        else:
            st.warning("Type a question first.")

# ── Tab 5: Phase 10 - Structured Output ──────────────────────────────────────
with tab_structured:
    st.subheader("Structured Agent Outputs")
    st.caption("Validates LLM responses against strict Pydantic schemas aligned with Phase 12 agents.")

    agent_type = st.selectbox(
        "Select Agent / Output Schema",
        [
            "🔍 Research Agent (ResearchResult)",
            "📝 Summarization Agent (SummaryResult)",
            "✅ Fact Checking Agent (FactCheckResult)",
            "💬 Question Answering Agent (QAResult)",
            "📄 Report Writing Agent (ReportResult)"
        ]
    )

    if agent_type == "🔍 Research Agent (ResearchResult)":
        topic = st.text_input("Enter research topic:", placeholder="e.g. LangGraph Framework")
        if st.button("Research Topic", key="btn_research"):
            if topic.strip():
                with st.spinner("Researching..."):
                    try:
                        res = run_research(topic)
                        st.markdown("### 🔍 Research Result")
                        st.metric("Confidence Score", f"{res.confidence * 100:.1f}%")
                        st.info(f"**Topic:** {res.topic}")
                        st.markdown("#### Key Findings")
                        for finding in res.findings:
                            st.write(f"- {finding}")
                        if res.sources:
                            st.markdown("#### Sources")
                            for src in res.sources:
                                st.write(f"- {src}")
                        
                        with st.expander("Raw Pydantic JSON"):
                            st.json(res.model_dump())
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a topic.")

    elif agent_type == "📝 Summarization Agent (SummaryResult)":
        text_to_sum = st.text_area("Enter text to summarize:", height=150)
        if st.button("Summarize", key="btn_sum"):
            if text_to_sum.strip():
                with st.spinner("Summarizing..."):
                    try:
                        res = run_summarize(text_to_sum)
                        st.markdown("### 📝 Summary Result")
                        st.success(f"**Title:** {res.title}")
                        st.write(f"**Summary:** {res.summary}")
                        st.markdown("#### Key Points")
                        for pt in res.key_points:
                            st.write(f"- {pt}")
                        st.write(f"**Estimated Word Count:** {res.word_count}")

                        with st.expander("Raw Pydantic JSON"):
                            st.json(res.model_dump())
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text.")

    elif agent_type == "✅ Fact Checking Agent (FactCheckResult)":
        claim = st.text_input("Enter claim to verify:", placeholder="e.g. LangChain was released in 2022")
        if st.button("Check Claim", key="btn_fc"):
            if claim.strip():
                with st.spinner("Fact checking..."):
                    try:
                        res = run_fact_check(claim)
                        st.markdown("### ✅ Fact Check Result")
                        st.metric("Verdict Confidence", f"{res.confidence * 100:.1f}%")
                        
                        if res.verdict == "TRUE":
                            st.success(f"**Verdict:** {res.verdict}")
                        elif res.verdict == "FALSE":
                            st.error(f"**Verdict:** {res.verdict}")
                        else:
                            st.warning(f"**Verdict:** {res.verdict}")
                            
                        st.info(f"**Claim Checked:** {res.claim}")
                        st.markdown("#### Evidence")
                        for ev in res.evidence:
                            st.write(f"- {ev}")

                        with st.expander("Raw Pydantic JSON"):
                            st.json(res.model_dump())
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a claim.")

    elif agent_type == "💬 Question Answering Agent (QAResult)":
        qa_q = st.text_input("Enter question:", placeholder="e.g. How does vector similarity search work?")
        if st.button("Answer Question", key="btn_qa"):
            if qa_q.strip():
                with st.spinner("Answering..."):
                    try:
                        res = run_qa(qa_q)
                        st.markdown("### 💬 QA Result")
                        st.metric("Confidence Score", f"{res.confidence * 100:.1f}%")
                        st.success(f"**Answer:** {res.answer}")
                        st.markdown("#### Reasoning")
                        st.write(res.reasoning)

                        with st.expander("Raw Pydantic JSON"):
                            st.json(res.model_dump())
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

    elif agent_type == "📄 Report Writing Agent (ReportResult)":
        topic_report = st.text_input("Enter report topic:", placeholder="e.g. The Impact of RAG on LLM Hallucinations")
        if st.button("Generate Report", key="btn_report"):
            if topic_report.strip():
                with st.spinner("Generating Report..."):
                    try:
                        res = run_report(topic_report)
                        st.markdown(f"## {res.title}")
                        
                        for section in res.sections:
                            heading = section.get("heading", "Section")
                            content = section.get("content", "")
                            st.markdown(f"### {heading}")
                            st.write(content)
                            
                        st.markdown("### Conclusion")
                        st.info(res.conclusion)
                        
                        if res.references:
                            st.markdown("### References")
                            for ref in res.references:
                                st.write(f"- {ref}")

                        with st.expander("Raw Pydantic JSON"):
                            st.json(res.model_dump())
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a topic.")

# ── Tab 6: Phase 12 - Multi-Agent Workflow ────────────────────────────────────
with tab_multi_agent:
    st.subheader("Multi-Agent Orchestrator")
    st.caption("A coordinator agent decides which specialized agent is best equipped to handle your query.")
    
    multi_query = st.text_input("Enter your request:", placeholder="e.g. Write a report on quantum computing OR verify if water boils at 100 degrees Celsius.")
    
    if st.button("Route & Execute", key="btn_multi_agent"):
        if multi_query.strip():
            with st.spinner("Coordinator routing query..."):
                try:
                    outcome = coordinate(multi_query)
                    
                    st.info(f"🤖 **Selected Agent:** {outcome['agent'].upper()}")
                    st.markdown(f"**Reasoning:** {outcome['reason']}")
                    
                    agent_result = outcome["result"]
                    
                    st.divider()
                    st.markdown("### Agent Output")
                    
                    if outcome["agent"] == "research":
                        st.subheader("🔍 Research Result")
                        st.metric("Confidence", f"{agent_result.confidence * 100:.1f}%")
                        st.write(f"**Topic:** {agent_result.topic}")
                        st.markdown("#### Key Findings")
                        for finding in agent_result.findings:
                            st.write(f"- {finding}")
                        if agent_result.sources:
                            st.markdown("#### Sources")
                            for src in agent_result.sources:
                                st.write(f"- {src}")
                                
                    elif outcome["agent"] == "summarization":
                        st.subheader("📝 Summary Result")
                        st.success(f"**Title:** {agent_result.title}")
                        st.write(f"**Summary:** {agent_result.summary}")
                        st.markdown("#### Key Takeaways")
                        for pt in agent_result.key_points:
                            st.write(f"- {pt}")
                        st.write(f"**Original Word Count:** {agent_result.word_count}")
                        
                    elif outcome["agent"] == "fact_checking":
                        st.subheader("✅ Fact Check Result")
                        st.metric("Verdict Confidence", f"{agent_result.confidence * 100:.1f}%")
                        if agent_result.verdict == "TRUE":
                            st.success(f"**Verdict:** {agent_result.verdict}")
                        elif agent_result.verdict == "FALSE":
                            st.error(f"**Verdict:** {agent_result.verdict}")
                        else:
                            st.warning(f"**Verdict:** {agent_result.verdict}")
                        st.write(f"**Claim Checked:** {agent_result.claim}")
                        st.markdown("#### Evidence")
                        for ev in agent_result.evidence:
                            st.write(f"- {ev}")
                            
                    elif outcome["agent"] == "report_writer":
                        st.subheader(f"📄 Report: {agent_result.title}")
                        for section in agent_result.sections:
                            st.markdown(f"#### {section.get('heading', 'Section')}")
                            st.write(section.get("content", ""))
                        st.markdown("#### Conclusion")
                        st.info(agent_result.conclusion)
                        if agent_result.references:
                            st.markdown("#### References")
                            for ref in agent_result.references:
                                st.write(f"- {ref}")
                                
                    else: # qa
                        st.subheader("💬 QA Result")
                        st.metric("Confidence", f"{agent_result.confidence * 100:.1f}%")
                        st.success(f"**Answer:** {agent_result.answer}")
                        st.markdown("#### Reasoning")
                        st.write(agent_result.reasoning)
                        
                    with st.expander("Raw Pydantic JSON"):
                        st.json(agent_result.model_dump())
                        
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please type a query first.")

st.divider()
st.caption("Built with LangChain · Gemini · FAISS · Streamlit")

