# Chat With Your PDF — RAG Chatbot

Built a RAG chatbot. Upload any PDF and chat with it.

## Setup

```bash
python -m venv venv
source venv/bin/activate      
pip install -r requirements.txt
cp .env.example .env        
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

## How it's organized

| File | Handles |
|---|---|
| `loaders.py` | Level 1 (PDF → Documents) and Level 2 (table-aware extraction via pdfplumber) |
| `rag_pipeline.py` | Chunking/embeddings/vectorstore (L1), question condensing for follow-ups (L5), retrieval + citation formatting (L4), streaming generation (L3) |
| `app.py` | Gradio UI wiring it all together |

## What each level looks like in the code

- **Level 1**: `file_upload` in the UI → `ingest()` → `load_pdf()` → `split_documents()` → `build_vectorstore()`. Tune `chunk_size`/`k` live with the sliders.
- **Level 2**: the "PDF extraction method" dropdown switches between `PyPDFLoader` (simple) and the `pdfplumber` table-aware loader in `loaders.py`, so you can A/B test the same question against both.
- **Level 3**: `answer_stream()` uses `llm.stream(...)` and yields the growing answer; `bot_respond()` in `app.py` re-renders the chat bubble on every yield.
- **Level 4**: `format_citations()` reads `metadata["source"]`/`metadata["page"]` off the retrieved chunks and appends them after the answer.
- **Level 5**: `condense_question()` rewrites a new question into a standalone one using the last few turns before retrieval ever happens, so "when was he born?" becomes "when was [Player Name] born?".

## Submission notes (fill these in after you run it on your own document)

**Document used:** _[fill in — e.g. "my linear algebra lecture notes.pdf"]_

**Levels reached:** _[fill in]_

**One thing that surprised me:** _[fill in — e.g. how much retrieval quality changed with chunk_size, or how the table-aware loader handled/mishandled a specific table]_

## Level 6 (bonus) — not included here

This repo currently ships as a self-contained Gradio app (Level 6's "keep Gradio as a standalone app.py" option is already satisfied). If you want to push further into a FastAPI backend + separate frontend, `rag_pipeline.py` is already decoupled from the UI — wrapping `answer_stream()` in a `POST /ask` FastAPI route (using `StreamingResponse`) is the main remaining step. Ask if you'd like that scaffolded too.
