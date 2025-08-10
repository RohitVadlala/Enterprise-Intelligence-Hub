# rag/query_engine.py
from __future__ import annotations
import re
from typing import List, Tuple, Dict, Union

TextChunk = Union[str, Dict]  

def _extract_text(c: TextChunk) -> str:
    if isinstance(c, dict):
        return (c.get("content") or "").strip()
    return (c or "").strip()

def _score_chunk(text: str, q_terms: set[str]) -> float:
    """
    Very small heuristic: keyword hits + a tiny length prior
    so 5-word fragments don't always outrank fuller context.
    """
    if not text:
        return 0.0
    toks = re.findall(r"\w+", text.lower())
    hits = sum(1 for t in toks if t in q_terms)
    length_prior = min(len(text) / 8000.0, 0.5)  
    return hits + length_prior

def pick_context(
    chunks: List[TextChunk],
    question: str,
    top_k: int = 4,
) -> Tuple[str, List[TextChunk]]:
    """
    Rank chunks by simple keyword overlap and return:
      - joined context string for prompting
      - the selected chunk objects (for UI evidence)

    Args:
        chunks: list[str] or list[{"content": str, "metadata": {...}}]
        question: user question
        top_k: number of chunks to keep

    Returns:
        (context_string, selected_chunks)
    """
    q_terms = set(re.findall(r"\w+", (question or "").lower()))
    scored: List[Tuple[float, TextChunk]] = []

    for ch in chunks:
        text = _extract_text(ch)
        if not text:
            continue
        scored.append((_score_chunk(text, q_terms), ch))

    # If nothing scored (e.g., empty doc), avoid crashing
    if not scored:
        return "", []

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [c for _, c in scored[:max(1, top_k)]]

    # Join text versions with separators for clarity
    joined = "\n\n---\n\n".join(_extract_text(c) for c in chosen)
    return joined, chosen
