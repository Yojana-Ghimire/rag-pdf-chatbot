"""
rag_pipeline.py — everything that isn't "load the PDF" or "draw the UI".
"""

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config (override via env vars if you want to experiment) ---
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
CHAT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from a document. If the answer is not contained in the context,
say "I don't know based on this document." Do not use outside knowledge.
Be concise and cite specifics (numbers, names) exactly as they appear in the context."""


# ---------------------------------------------------------------------------
# Level 1: chunking + embedding + vectorstore
# ---------------------------------------------------------------------------

def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 150):
    """Bigger chunk_size = more context per chunk but coarser retrieval
    (you may pull in irrelevant text alongside the relevant bit).
    Smaller chunk_size = more precise retrieval but risks splitting a fact
    away from the context it needs (e.g. a table row from its header)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def get_llm(streaming: bool = True):
    return ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.2, streaming=streaming)


# ---------------------------------------------------------------------------
# Level 5: conversational memory
# ---------------------------------------------------------------------------

def condense_question(question: str, chat_history: list[tuple[str, str]], llm) -> str:
    """Rewrite `question` as a standalone question, resolving pronouns /
    implicit references using the last few turns. If there's no history,
    or the question is already standalone, this just returns it unchanged.
    """
    if not chat_history:
        return question

    history_text = "\n".join(
        f"User: {q}\nAssistant: {a}" for q, a in chat_history[-4:]
    )
    prompt = f"""Given this conversation history:

{history_text}

And this new question: "{question}"

Rewrite the new question as a fully standalone question that makes sense
without the conversation history — resolve any pronouns (he/she/it/they/that)
or implicit references to what they actually refer to. If the question is
already standalone, return it unchanged. Reply with ONLY the rewritten
question, nothing else."""

    result = llm.invoke(prompt)
    return result.content.strip()


# ---------------------------------------------------------------------------
# Level 1: retrieval
# ---------------------------------------------------------------------------

def retrieve(vectorstore, query: str, k: int = 4):
    return vectorstore.similarity_search(query, k=k)


# ---------------------------------------------------------------------------
# Level 4: citations
# ---------------------------------------------------------------------------

def format_citations(source_docs) -> str:
    """Dedupe (source, page) pairs and format them for display under an answer."""
    seen = []
    for d in source_docs:
        key = (d.metadata.get("source", "unknown"), d.metadata.get("page", "?"))
        if key not in seen:
            seen.append(key)

    if not seen:
        return ""

    lines = [f"📄 {source} — page {page}" for source, page in seen]
    return "\n\n**Sources:**\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Level 3: streaming answer generator (wires together everything above)
# ---------------------------------------------------------------------------

def answer_stream(question: str, vectorstore, llm, chat_history: list[tuple[str, str]], k: int = 4):
    """
    Generator that yields the growing answer string, chunk by chunk, and
    finally appends the citations. Used directly by app.py's Gradio callback.
    """
    standalone_question = condense_question(question, chat_history, llm)

    docs = retrieve(vectorstore, standalone_question, k=k)
    context = "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source')}, page: {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {standalone_question}

Answer:"""

    partial = ""
    for chunk in llm.stream(prompt):
        token = chunk.content or ""
        partial += token
        yield partial  # yield the growing answer, no citations yet

    citations = format_citations(docs)
    yield partial + citations  # final yield includes citations
