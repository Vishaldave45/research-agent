# AI Research Assistant — Phases 1–7

LangChain + Google Gemini + FAISS, with a Streamlit frontend.

## Phases included

| Phase | Feature | File |
|---|---|---|
| 1 | Basic Chatbot | `chains/chatbot_chain.py` |
| 2 | LCEL Pipelines | `chains/lcel_pipelines.py` |
| 3 | Document Loading | `loaders/document_loader.py` |
| 4 | Text Splitting | `loaders/text_splitter.py` |
| 5 | Embeddings + Vector Store (FAISS) | `models/embeddings.py`, `models/vectorstore.py` |
| 6 | RAG | `chains/rag_chain.py` |
| 7 | Conversation Memory | `chains/memory_chain.py` |

## Setup

```bash
cd research-agent-code

# Install dependencies
uv add langchain langchain-google-genai langchain-community \
       faiss-cpu pypdf unstructured python-dotenv pydantic streamlit
```

Add your Gemini API key to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

> **Model note:** `models/llm.py` uses `gemini-flash-latest`. If you hit a `404 NOT_FOUND` or `429 RESOURCE_EXHAUSTED` error, your key may not have access to that exact alias yet — check which models your key supports at [ai.dev/rate-limit](https://ai.dev/rate-limit), then update the `model=` value in `models/llm.py` accordingly.

## Run the backend test script

Walks through Phases 1–7 in the terminal:

```bash
uv run python main.py
```

## Run the Streamlit frontend

```bash
uv run streamlit run frontend/app.py
```

This opens a browser UI with four tabs:

- **Basic Chat** — Phase 1, stateless Q&A
- **Memory Chat** — Phase 7, remembers conversation history
- **Upload Documents** — Phases 3–5, upload PDF/TXT/MD/HTML, adjust chunk size, index into FAISS
- **Ask Your Documents (RAG)** — Phase 6, ask questions answered from your uploaded documents, with sources shown

## Project structure

```
research-agent-code/
├── chains/
│   ├── chatbot_chain.py      Phase 1
│   ├── lcel_pipelines.py     Phase 2
│   ├── rag_chain.py          Phase 6
│   └── memory_chain.py       Phase 7
├── loaders/
│   ├── document_loader.py    Phase 3
│   └── text_splitter.py      Phase 4
├── models/
│   ├── llm.py                Shared Gemini LLM
│   ├── embeddings.py         Phase 5
│   └── vectorstore.py        Phase 5
├── prompts/
│   └── chat_prompt.py        Shared prompt templates
├── schemas/
│   └── response_schema.py    Shared Pydantic schemas
├── frontend/
│   └── app.py                Streamlit UI
├── data/raw/                 Drop documents here (or upload via UI)
├── main.py                   Terminal test runner
└── .env                      GOOGLE_API_KEY=...
```

## Run order

1. `uv run python main.py` — confirm everything works in the terminal first
2. `uv run streamlit run frontend/app.py` — then explore via the browser UI
