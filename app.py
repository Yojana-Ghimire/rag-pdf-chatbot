"""
app.py — the Gradio front-end.

Run with:
    python app.py

Covers, in one screen:
  Level 1 - upload a PDF, ingest, chat (with chunk_size / k sliders to tune)
  Level 2 - loader_mode dropdown to compare PyPDFLoader vs pdfplumber (tables)
  Level 3 - answers stream token-by-token
  Level 4 - every answer is followed by its source document + page number(s)
  Level 5 - pronouns / follow-ups resolve using conversation history
"""

import os
import tempfile
import gradio as gr
from dotenv import load_dotenv

from loaders import load_pdf
from rag_pipeline import split_documents, build_vectorstore, get_llm, answer_stream

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    # langchain-google-genai looks for GOOGLE_API_KEY; alias it from GEMINI_API_KEY
    # so you can use either name in your .env
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

LLM = get_llm(streaming=True)


def ingest(file, loader_mode, chunk_size, chunk_overlap):
    """Runs when a PDF is uploaded. Builds a fresh vectorstore and resets
    the conversation (a new document shouldn't inherit old chat history)."""
    if file is None:
        return None, [], "Upload a PDF to get started.", gr.update(interactive=False)

    docs = load_pdf(file.name, mode=loader_mode)
    if not docs:
        return None, [], "⚠️ Couldn't extract any text from that PDF (is it scanned/image-only?).", gr.update(interactive=False)

    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = build_vectorstore(chunks)

    status = (
        f"✅ Loaded **{os.path.basename(file.name)}** — "
        f"{len(docs)} page-level document(s) → {len(chunks)} chunks. Ask away!"
    )
    return vectorstore, [], status, gr.update(interactive=True)


def chat(message, history, vectorstore, k):
    """`history` is Gradio's chatbot state, list of {"role":..,"content":..} dicts
    (type="messages" format). We convert it to (question, answer) tuples for
    the condense-question step, then stream the new answer back in."""
    if vectorstore is None:
        yield "Please upload a PDF first."
        return

    qa_pairs = []
    pending_user_msg = None
    for turn in history:
        if turn["role"] == "user":
            pending_user_msg = turn["content"]
        elif turn["role"] == "assistant" and pending_user_msg is not None:
            qa_pairs.append((pending_user_msg, turn["content"]))
            pending_user_msg = None

    for partial_answer in answer_stream(message, vectorstore, LLM, qa_pairs, k=k):
        yield partial_answer


with gr.Blocks(title="Chat With Your PDF") as demo:
    gr.Markdown("# 📄 Chat With Your PDF\nUpload a document, then ask it questions.")

    vectorstore_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload a PDF", file_types=[".pdf"])
            loader_mode = gr.Dropdown(
                choices=[("pdfplumber — tables-aware (Level 2)", "table_aware"),
                         ("PyPDFLoader — simple text only", "simple")],
                value="table_aware",
                label="PDF extraction method",
            )
            chunk_size = gr.Slider(200, 2000, value=1000, step=100, label="chunk_size")
            chunk_overlap = gr.Slider(0, 400, value=150, step=25, label="chunk_overlap")
            k_slider = gr.Slider(1, 10, value=4, step=1, label="k (chunks retrieved per question)")
            ingest_btn = gr.Button("Ingest PDF", variant="primary")
            status = gr.Markdown("Upload a PDF to get started.")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", type="messages", height=500)
            msg_box = gr.Textbox(label="Ask a question", placeholder="What is this document about?", interactive=False)
            clear_btn = gr.ClearButton([msg_box, chatbot])

    ingest_btn.click(
        ingest,
        inputs=[file_upload, loader_mode, chunk_size, chunk_overlap],
        outputs=[vectorstore_state, chatbot, status, msg_box],
    )

    def user_submit(message, history):
        return "", history + [{"role": "user", "content": message}]

    def bot_respond(history, vectorstore, k):
        history = history + [{"role": "assistant", "content": ""}]
        user_message = history[-2]["content"]
        for partial in chat(user_message, history[:-2], vectorstore, k):
            history[-1]["content"] = partial
            yield history

    msg_box.submit(user_submit, [msg_box, chatbot], [msg_box, chatbot]).then(
        bot_respond, [chatbot, vectorstore_state, k_slider], [chatbot]
    )

if __name__ == "__main__":
    demo.launch()
