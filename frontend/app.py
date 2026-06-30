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

st.title("🔎 AI Research Assistant")
st.caption("LangChain + Gemini + FAISS  ·  Phases 1–12")

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
                    st.write_stream(ask_stream(question))
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
                        chat_stream(user_msg, session_id=st.session_state.session_id)
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
                        for chunk in ask_with_sources_stream(rag_question):
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

