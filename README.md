<div align="center">

# 🔎 AI Research Assistant

### _Your intelligent document companion powered by LangChain & Google Gemini_

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)

---

**Upload documents · Ask questions · Get cited answers · All in one place**

</div>

<br>

## ✨ Features

<table>
<tr>
<td width="50%">

### 💬 Conversational AI
- **Stateless Chat** — Quick Q&A powered by Gemini
- **Memory Chat** — Context-aware conversations that remember history
- **Tool-Calling Agent** — Autonomously uses calculators, search, summarizers & more

</td>
<td width="50%">

### 📚 Document Intelligence
- **Multi-format Ingestion** — PDF, TXT, Markdown, HTML
- **Smart Chunking** — Configurable text splitting with overlap
- **RAG Pipeline** — Retrieval-augmented answers with source citations

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

### 3️⃣ Add your API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key_here
```

🔑 Get a free key at **[aistudio.google.com](https://aistudio.google.com/app/apikey)**

### 4️⃣ Run the app

```bash
# Terminal test — validates all components
uv run python main.py

# Web UI — full interactive experience
uv run streamlit run frontend/app.py
```

---

## 🖥️ Streamlit Web UI

The frontend provides four interactive tabs:

| Tab | Description |
|:---:|:---|
| 💬 **Basic Chat** | Stateless Q&A — ask anything, get instant answers |
| 🧠 **Memory Chat** | Conversations with history — the AI remembers context |
| 📄 **Upload Documents** | Drag & drop PDFs, TXT, MD, HTML → auto-indexed into FAISS |
| 📚 **Ask Your Documents** | RAG-powered answers sourced directly from your uploads |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    🖥️  Streamlit Frontend                    │
│              (Basic Chat · Memory · Upload · RAG)            │
└──────────────────────────┬───────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   ┌─────────────┐  ┌───────────┐  ┌──────────────┐
   │  💬 Chains   │  │ 🛠️ Agent  │  │  📄 Loaders  │
   │             │  │           │  │              │
   │ • Chatbot   │  │ • Calc    │  │ • PDF        │
   │ • RAG       │  │ • Date    │  │ • TXT / MD   │
   │ • Memory    │  │ • Search  │  │ • HTML       │
   │ • LCEL      │  │ • Summary │  │ • Splitter   │
   └──────┬──────┘  └─────┬─────┘  └──────┬───────┘
          │               │               │
          ▼               ▼               ▼
   ┌─────────────────────────────────────────────┐
   │              🧠 Models Layer                 │
   │                                             │
   │  Gemini 2.5 Flash  ·  Embeddings  ·  FAISS │
   └─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
research-agent/
│
├── 📂 chains/                    # LangChain processing chains
│   ├── chatbot_chain.py          # Basic chatbot (prompt → LLM → parser)
│   ├── lcel_pipelines.py         # Sequential & parallel LCEL chains
│   ├── rag_chain.py              # Retrieval-augmented generation
│   ├── memory_chain.py           # Conversation memory with history
│   └── agent_chain.py            # Tool-calling agent loop
│
├── 📂 loaders/                   # Document ingestion
│   ├── document_loader.py        # Multi-format file loader
│   └── text_splitter.py          # Recursive text chunking
│
├── 📂 models/                    # AI model configuration
│   ├── llm.py                    # Gemini 2.5 Flash instances
│   ├── embeddings.py             # Google embedding model
│   └── vectorstore.py            # FAISS vector store ops
│
├── 📂 tools/                     # Agent tools
│   └── tools.py                  # Calculator, datetime, summarizer, etc.
│
├── 📂 prompts/                   # Prompt templates
│   └── chat_prompt.py            # Shared chat prompts
│
├── 📂 schemas/                   # Data models
│   └── response_schema.py        # Pydantic response schemas
│
├── 📂 frontend/                  # Web interface
│   └── app.py                    # Streamlit app (4 tabs)
│
├── 📂 data/raw/                  # Drop documents here (or upload via UI)
├── main.py                       # Terminal test runner
├── pyproject.toml                # Project config & dependencies
└── .env                          # API key (not committed)
```

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|:---:|:---|
| 🦜 **LangChain** | Orchestration framework for LLM chains, agents & tools |
| ✨ **Google Gemini 2.5 Flash** | Fast, capable LLM for generation & reasoning |
| 🔢 **FAISS** | High-performance vector similarity search |
| 🎨 **Streamlit** | Interactive web frontend |
| 📦 **uv** | Ultra-fast Python package manager |

</div>

---

## 🔧 Configuration

### Model Settings

Models are configured in [`models/llm.py`](models/llm.py):

```python
# Creative tasks (chat, summarization)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# Deterministic tasks (agents, extraction)
llm_precise = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
```

### Embedding Model

Configured in [`models/embeddings.py`](models/embeddings.py):

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

> ⚠️ **Rate Limits:** Free-tier API keys have usage quotas. Check your usage at [ai.dev/rate-limit](https://ai.dev/rate-limit). If you hit `429 RESOURCE_EXHAUSTED`, wait a minute or upgrade your plan.

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ using LangChain · Google Gemini · FAISS · Streamlit**

⭐ Star this repo if you found it useful!

</div>
