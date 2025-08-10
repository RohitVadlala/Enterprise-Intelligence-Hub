# ingest/split_text.py
from __future__ import annotations
from typing import List, Dict

# Prefer the new package; fallback to old import if not installed
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_chunks(
    pages: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] | None = None,
) -> List[Dict]:
    """
    Split page-wise text into smaller chunks, preserving metadata.

    Args:
        pages: Output from load_pdf/load_pdf_bytes (list of dicts).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap in characters between adjacent chunks.
        separators: Optional custom split priority (largest -> smallest).

    Returns:
        List[Dict]: [{"content": str, "metadata": {..., "chunk_index": int}}, ...]
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    chunks: List[Dict] = []
    for page in pages:
        text = (page.get("content") or "").strip()
        if not text:
            continue

        pieces = splitter.split_text(text)
        for i, piece in enumerate(pieces):
            piece = piece.strip()
            if not piece:
                continue
            meta = dict(page.get("metadata", {}))
            meta["chunk_index"] = i
            chunks.append({"content": piece, "metadata": meta})

    return chunks
