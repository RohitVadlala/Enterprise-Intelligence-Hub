
from __future__ import annotations
import os
from typing import List, Optional

# ---- Gemini (public API)
import google.generativeai as genai

# ---- FAISS + embeddings
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    # fallback for older envs
    from langchain.embeddings import HuggingFaceEmbeddings  


def get_model(api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
    """Configure and return a Gemini model (public API)."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Pass api_key or set env var.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def summarize(model, text: str) -> str:
    prompt = f"""You are a financial analyst. Summarize the document into 6–10 crisp bullets.
Document:
\"\"\"{text[:18000]}\"\"\""""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()

def answer(model, question: str, context: str) -> str:
    prompt = f"""Answer ONLY from the context. Provide a concise answer plus 2–4 quoted snippets.
Question: {question}

Context:
\"\"\"{context[:18000]}\"\"\""""
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", None) or "").strip()


def load_vectorstore(db_path: str = "vectorstore/faiss_index") -> FAISS:
    """Load an existing FAISS index (expects index.faiss and index.pkl in db_path)."""
    if not os.path.isdir(db_path):
        raise FileNotFoundError(f"FAISS directory not found: {db_path}")
    needed = {"index.faiss", "index.pkl"}
    have = set(os.listdir(db_path))
    missing = needed - have
    if missing:
        raise FileNotFoundError(f"FAISS index missing files {missing} in {db_path}")

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # allow_dangerous_deserialization required for LC v0.2.x saved stores
    return FAISS.load_local(db_path, embeddings=emb, allow_dangerous_deserialization=True)

def get_top_k_docs(query: str, db: FAISS, k: int = 3) -> List[str]:
    """Return the top-k page contents from FAISS for a query."""
    docs = db.similarity_search(query, k=k)
    return [d.page_content for d in docs]

def ask_gemini_with_retrieval(
    query: str,
    db: FAISS,
    model=None,
    api_key: Optional[str] = None,
    k: int = 3,
) -> str:
    """Retrieve context from FAISS, then answer with Gemini."""
    if model is None:
        model = get_model(api_key=api_key)
    top = get_top_k_docs(query, db, k=k)
    context = "\n\n---\n\n".join(top)
    return answer(model, query, context)
