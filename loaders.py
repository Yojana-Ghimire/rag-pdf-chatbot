"""
loaders.py — turning a PDF into LangChain Documents.

Two loading strategies are provided so you can literally compare them
(that's the Level 2 assignment):

    load_pdf_simple(path)     -> plain text via PyPDFLoader
    load_pdf_table_aware(path)-> text + tables via pdfplumber

Both return a list of langchain_core.documents.Document, each with
metadata = {"source": <filename>, "page": <1-indexed page number>, "content_type": "text" | "table"}
so that citations (Level 4) always have something to point to.
"""

import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber


def load_pdf_simple(path: str) -> list[Document]:
    """
    The baseline loader from the original session. Fast, but it flattens
    tables into whatever order the raw text stream puts them in — numbers
    and labels frequently end up disconnected from each other.
    """
    loader = PyPDFLoader(path)
    docs = loader.load()
    filename = os.path.basename(path)
    for d in docs:
        # PyPDFLoader already sets metadata["page"], just normalize "source"
        d.metadata["source"] = filename
        d.metadata["content_type"] = "text"
    return docs


def _table_to_text(table: list[list]) -> str:
    """Render a pdfplumber table (list of rows) as a simple markdown-ish
    string, e.g.:

        | Year | Revenue | Growth |
        | 2023 | 4.2M    | 12%    |

    This keeps row/column relationships intact for the LLM, which a plain
    text dump of the page usually destroys.
    """
    rows = []
    for row in table:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def load_pdf_table_aware(path: str) -> list[Document]:
    """
    Extraction strategy for messy PDFs (Level 2):
      - pulls normal text with pdfplumber's extract_text()
      - separately pulls tables with extract_tables() and renders them as
        markdown-style rows so numbers stay attached to their column headers
      - each table becomes its OWN Document chunk (content_type="table"),
        so it can be retrieved and cited independently of the surrounding
        prose, and so a big table doesn't get chopped mid-row by the
        text splitter later on.
    """
    filename = os.path.basename(path)
    docs: list[Document] = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1

            text = (page.extract_text() or "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "content_type": "text",
                        },
                    )
                )

            tables = page.extract_tables() or []
            for t_idx, table in enumerate(tables):
                if not table or not any(any(cell for cell in row) for row in table):
                    continue
                table_text = _table_to_text(table)
                docs.append(
                    Document(
                        page_content=f"[Table {t_idx + 1} on page {page_num}]\n{table_text}",
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "content_type": "table",
                        },
                    )
                )

    return docs


def load_pdf(path: str, mode: str = "table_aware") -> list[Document]:
    """Convenience dispatcher used by app.py.

    mode: "simple" (PyPDFLoader) or "table_aware" (pdfplumber, Level 2 default)
    """
    if mode == "simple":
        return load_pdf_simple(path)
    elif mode == "table_aware":
        return load_pdf_table_aware(path)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# --- Stretch idea noted in the assignment (not implemented here) ---
# For scanned/image-only PDFs, neither loader above will find text because
# there isn't a text layer to extract. You'd instead:
#   1. Rasterize each page to an image (pdfplumber's page.to_image(), or
#      pdf2image.convert_from_path)
#   2. Run pytesseract.image_to_string(image) on each page image, OR
#   3. Send the page image directly to a multimodal model (Gemini can take
#      images) and ask it to transcribe / answer directly.
# Either result just becomes another Document with content_type="ocr".
