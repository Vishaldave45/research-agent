<div align="center">

# 🔎 AI Research Assistant

### _Your intelligent multi-agent document companion powered by LangChain, LangGraph, Groq & Gemini_

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-1C3C3C?style=for-the-badge)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-🕸️-000000?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

---

**Multi-LLM Switcher · Document Upload · LangGraph Multi-Agent Orchestrator · Real-Time Streaming**

</div>

<br>

## ✨ Features

<table>
<tr>
<td width="50%" valign="top">

### 💬 Conversational & Multi-LLM AI
- **LLM Provider Switcher** — Swap between Gemini, Groq, and OpenRouter in the sidebar without restarting the app.
- **Stateless Chat** — Quick Q&A powered by the active LLM.
- **Memory Chat** — Context-aware conversations that remember history.
- **Real-Time Streaming** — Fast typewriter-style response rendering.

</td>
<td width="50%" valign="top">

### 📚 Document Intelligence & Agents
- **Multi-format Ingestion** — PDF, TXT, Markdown, HTML.
- **Smart Chunking** — Configurable text splitting with overlap.
- **RAG Pipeline** — Retrieval-augmented answers with source citations.
- **LangGraph Orchestrator** — A coordinator agent analyzes the query and conditionally routes the request to specialized nodes (Research, Summarize, Fact Check, QA, Report).

</td>
</tr>
</table>

---

## 🚀 Quick Start

### 1️⃣ Clone the repo

```bash
git clone https://github.com/Vishaldave45/research-agent.git
cd research-agent
```

### 2️⃣ Install dependencies

```bash
uv sync
```

> 💡 Don't have `uv`? Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 3️⃣ Configure your API keys

Create a `.env` file in the project root:

```env
# Optional (needed for Gemini models & embeddings)
GOOGLE_API_KEY=AIzaSy...

# Optional (needed for Groq Llama 3.3 model execution)
GROQ_API_KEY=gsk_...

# Optional (needed for OpenRouter models)
OPENROUTER_API_KEY=sk-or-...
```

### 4️⃣ Run the app

```bash
# Terminal test — validates components & LangGraph workflow execution
uv run python main.py

# Web UI — full interactive experience with custom glassmorphism styling
uv run streamlit run frontend/app.py
```

---

## 🖥️ Streamlit Web UI Tabs

| Tab | Description |
|:---:|:---|
| 💬 **Basic Chat** | Stateless Q&A — ask anything, get instant streaming answers |
| 🧠 **Memory Chat** | Conversations with history — the AI remembers context |
| 📄 **Upload Documents** | Drag & drop PDFs, TXT, MD, HTML → auto-indexed into FAISS |
| 📚 **Ask Your Documents** | RAG-powered answers sourced directly from your uploads |
| 🧩 **Structured Output** | Validates individual agent responses against strict Pydantic schemas |
| 🤖 **Multi-Agent Coordinator** | Graph-orchestrated query router powered by **LangGraph** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    🖥️  Streamlit Frontend                    │
│             (Config Sidebar · Tabs 1-6 · Glass UI)            │
│                              │                               │
│                              ▼                               │
│                🕸️  LangGraph Coordinator Router              │
└──────────────────────────────┬───────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
     ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
     │ 🔍 Research  │   │ ✅ Fact-Check│   │  💬 General  │
     │    Agent     │   │    Agent     │   │   QA Agent   │
     └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               ▼
     ┌────────────────────────────────────────────────────────┐
     │                     🛠️ Tools Layer                      │
     │  Document search (FAISS)  ·  Calculator  ·  Datetime   │
     └────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
research-agent/
│
├── 📂 chains/                    # LangChain processing chains & workflows
│   ├── chatbot_chain.py          # Basic chatbot (prompt → LLM → parser)
│   ├── lcel_pipelines.py         # Sequential & parallel LCEL chains
│   ├── rag_chain.py              # Retrieval-augmented generation
│   ├── memory_chain.py           # Conversation memory with history
│   ├── agent_chain.py            # Tool-calling agent loop
│   ├── structured_chain.py       # Pydantic schema validation chains
│   ├── multi_agent_workflow.py   # Multi-agent orchestrator logic
│   └── langgraph_workflow.py     # StateGraph-based orchestrator (LangGraph)
│
├── 📂 loaders/                   # Document ingestion
│   ├── document_loader.py        # Multi-format file loader
│   └── text_splitter.py          # Recursive text chunking
│
├── 📂 models/                    # AI model configuration
│   ├── llm.py                    # Provider switcher factory & instances
│   ├── embeddings.py             # Text embedding model
│   └── vectorstore.py            # FAISS vector store operations
│
├── 📂 tools/                     # Agent tools
│   └── tools.py                  # Document search, calculator, datetime
│
├── 📂 prompts/                   # Prompt templates
│   └── chat_prompt.py            # Shared chat prompts
│
├── 📂 schemas/                   # Data models
│   └── response_schema.py        # Pydantic response schemas
│
├── 📂 frontend/                  # Web interface
│   └── app.py                    # Streamlit app (6 tabs + custom styling)
│
├── 📂 data/raw/                  # Raw reference documents (gitignored)
├── main.py                       # CLI test runner
├── pyproject.toml                # Project config & dependencies
└── .env                          # API keys (gitignored)
```

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:---:|:---|
| 🦜 **LangChain** | LLM orchestrations, chains, and parser mappings |
| 🕸️ **LangGraph** | Multi-agent state graph routing & execution flow |
| 🚀 **Groq (Llama 3.3)** | Super fast open-source model execution |
| ✨ **Google Gemini** | Multimodal reasoning and content production |
| 🔢 **FAISS** | High-performance vector similarity search |
| 🎨 **Streamlit** | Premium Glassmorphic Web Dashboard |

</div>

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using LangChain · LangGraph · Streamlit · FAISS**

⭐ Star this repo if you found it useful!

</div>
