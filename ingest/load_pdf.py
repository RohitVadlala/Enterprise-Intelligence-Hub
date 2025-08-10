# ingest/load_pdf.py
from __future__ import annotations
import io
import os
import re
from typing import List, Dict, Optional

# Try PyMuPDF first (faster, better layout), then pypdf
try:
    import fitz  
    _PDF_BACKEND = "pymupdf"
except Exception:
    fitz = None
    _PDF_BACKEND = None

try:
    from pypdf import PdfReader
    _PDF_BACKEND = _PDF_BACKEND or "pypdf"
except Exception:
    PdfReader = None

_WHITESPACE_RX = re.compile(r"[ \t\r\f\v]+")

def _normalize_text(s: str) -> str:
    # collapse excessive spaces but keep newlines
    s = _WHITESPACE_RX.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def load_pdf(path: str) -> List[Dict]:
    """
    Extract text from a PDF file on disk.

    Returns:
        List[Dict]: [{"content": str, "metadata": {"source": str, "page_number": int}}, ...]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    source = os.path.basename(path)

    if _PDF_BACKEND == "pymupdf":
        doc = fitz.open(path)  
        pages: List[Dict] = []
        for i, page in enumerate(doc):
            
            text = page.get_text("text") or ""
            text = _normalize_text(text)
            if text:
                pages.append({"content": text,
                              "metadata": {"source": source, "page_number": i + 1}})
        doc.close()
        return pages

    if _PDF_BACKEND == "pypdf":
        reader = PdfReader(path)  
        pages: List[Dict] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = _normalize_text(text)
            if text:
                pages.append({"content": text,
                              "metadata": {"source": source, "page_number": i + 1}})
        return pages

    # Neither library available
    raise ImportError(
        "No PDF backend found. Install one of:\n"
        "  pip install pymupdf   # faster, better layout\n"
        "  pip install pypdf     # pure-Python fallback"
    )

def load_pdf_bytes(file_bytes: bytes, filename: str = "uploaded.pdf") -> List[Dict]:
    """
    Extract text from a PDF given raw bytes (for Streamlit uploads).

    Args:
        file_bytes: PDF file bytes
        filename: used only for metadata.source

    Returns:
        Same structure as load_pdf(path).
    """
    source = filename

    if _PDF_BACKEND == "pymupdf":
        doc = fitz.open(stream=file_bytes, filetype="pdf")  
        pages: List[Dict] = []
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            text = _normalize_text(text)
            if text:
                pages.append({"content": text,
                              "metadata": {"source": source, "page_number": i + 1}})
        doc.close()
        return pages

    if _PDF_BACKEND == "pypdf":
        reader = PdfReader(io.BytesIO(file_bytes))  
        pages: List[Dict] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = _normalize_text(text)
            if text:
                pages.append({"content": text,
                              "metadata": {"source": source, "page_number": i + 1}})
        return pages

    raise ImportError(
        "No PDF backend found. Install one of:\n"
        "  pip install pymupdf\n"
        "  pip install pypdf"
    )

def join_pages(pages: List[Dict]) -> str:
    """Convenience: return one big string from page dicts."""
    return "\n\n".join(p["content"] for p in pages if p.get("content"))
