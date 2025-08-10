
import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import os
from typing import List, Dict, Tuple

import streamlit as st

from ingest.load_pdf import load_pdf                 
from ingest.split_text import split_chunks
from rag.query_engine import pick_context


import google.generativeai as genai

from pathlib import Path
import tempfile, uuid


DEFAULT_LOCAL_PDF = str(Path(ROOT_DIR) / "sample_data" / "test_budget.pdf")



APP_TITLE = "🧠 Enterprise Intelligence Chatbot"

def save_uploaded_to_tmp(uploaded_file: "UploadedFile") -> str:
    """Save a Streamlit UploadedFile to a temp path and return that path."""
    tmp_dir = Path(tempfile.gettempdir())
    safe_name = Path(uploaded_file.name).name
    tmp_path = tmp_dir / f"eih_{uuid.uuid4().hex}_{safe_name}"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(tmp_path)

def need_package_msg(pkg: str, pip_cmd: str):
    st.error(
        f"`{pkg}` is not installed. In Colab/terminal run:\n\n"
        f"```bash\n{pip_cmd}\n```"
    )


def ensure_deps() -> bool:
    """Quick dependency checks for PyMuPDF and google-generativeai."""

    try:
        import fitz  
    except Exception:
        need_package_msg("PyMuPDF", "pip install pymupdf")
        return False

    try:
        import google.generativeai as _  
    except Exception:
        need_package_msg("google-generativeai", "pip install google-generativeai")
        return False

    return True


def configure_gemini(api_key: str, model_name: str = "gemini-1.5-flash"):
    genai.configure(api_key=api_key.strip())
    return genai.GenerativeModel(model_name)


def to_text_list(chunks: List[Dict]) -> List[str]:
    """Convert split_chunks output -> list[str] of text content."""
    out = []
    for c in chunks:
        txt = (c.get("content") or "").strip()
        if txt:
            out.append(txt)
    return out


def gemini_summarize(model, text: str) -> str:
    prompt = f"""
You are a financial analyst. Summarize the following document into crisp bullet points.
Be specific with numbers, trends, and risks. Keep to 6–10 bullets.

Document:
\"\"\"{text[:18000]}\"\"\"
"""
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


def gemini_qa(model, question: str, context_text: str) -> str:
    prompt = f"""
Answer the question **only** using the provided context. Start with a short answer,
then show 2–4 quoted snippets as evidence (with page/chunk cues if available).
If the answer isn't in the context, say you can't find it.

Question:
{question}

Context:
\"\"\"{context_text[:18000]}\"\"\"
"""
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

st.set_page_config(page_title="Enterprise Intelligence Chatbot", page_icon="🧠", layout="centered")
st.title(APP_TITLE)
st.caption("Upload a PDF or use the local sample. Summarize and ask questions with Gemini.")

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key_default = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("🔑 GEMINI_API_KEY", value=api_key_default, type="password")
    model_name = st.selectbox("Gemini model", ["gemini-1.5-flash", "gemini-1.5-pro"], index=0)
    show_debug = st.toggle("Show debug info", value=False)

if not ensure_deps():
    st.stop()

# Source selector
st.subheader("1) Choose your document")
uploaded = st.file_uploader("📄 Upload a PDF", type=["pdf"])
load_sample = st.button("Load sample PDF")

pages: List[Dict] = []
doc_label = ""

if uploaded:
    try:
        tmp_path = save_uploaded_to_tmp(uploaded)
        pages = load_pdf(tmp_path)
        doc_label = Path(uploaded.name).name
    except Exception as e:
        st.error(f"Failed to read uploaded PDF: {e}")

elif load_sample:
    if os.path.exists(DEFAULT_LOCAL_PDF):
        try:
            pages = load_pdf(DEFAULT_LOCAL_PDF)
            doc_label = Path(DEFAULT_LOCAL_PDF).name
        except Exception as e:
            st.error(f"Failed to read `{DEFAULT_LOCAL_PDF}`: {e}")
    else:
        st.error(f"Sample not found at: {DEFAULT_LOCAL_PDF}")

if not pages:
    st.info("Upload a PDF above or click **Load sample PDF** to try the app.")
    st.stop()

if show_debug:
    st.success(f"Loaded **{doc_label}** with {len(pages)} pages.")

# Split into chunks (your ingest/split_text.py)
st.subheader("2) Prepare chunks")
with st.spinner("Splitting text…"):
    chunks = split_chunks(pages, chunk_size=500, chunk_overlap=50)

if show_debug:
    st.write(f"Total chunks: {len(chunks)}")
    if chunks:
        st.code(chunks[0]["content"][:500] + ("…" if len(chunks[0]["content"]) > 500 else ""))

# Configure Gemini
if not api_key.strip():
    st.error("Please paste your Gemini API key in the sidebar.")
    st.stop()

try:
    model = configure_gemini(api_key, model_name=model_name)
except Exception as e:
    st.error(f"Gemini init error: {e}")
    st.stop()

# Summarize
st.subheader("3) Summarize the document")
if st.button("Generate Summary"):
    with st.spinner("Summarizing with Gemini…"):
        # Join a reasonable slice of text for the summary
        full_text = "\n".join([c["content"] for c in chunks])[:30000]
        summary = gemini_summarize(model, full_text)
    st.markdown("#### Summary")
    st.markdown(summary)

st.divider()

# Q&A
st.subheader("4) Ask a question about this document")
question = st.text_input("Your question")
top_k = st.slider("How many chunks to ground on (top_k)", min_value=1, max_value=8, value=4, step=1)

if question:
    with st.spinner("Retrieving context and answering…"):
        # Use your simple retriever
        context_text, selected = pick_context(chunks, question, top_k=top_k)
        answer = gemini_qa(model, question, context_text)

    st.markdown("#### Answer")
    st.markdown(answer)

    with st.expander("Show selected evidence chunks"):
        for i, ch in enumerate(selected, 1):
            meta = ch.get("metadata", {})
            src = meta.get("source", doc_label)
            page = meta.get("page_number", "?")
            idx = meta.get("chunk_index", "?")
            st.markdown(f"**Chunk {i}** — source: `{src}`, page: `{page}`, chunk_index: `{idx}`")
            st.code(ch.get("content", "")[:1200] + ("…" if len(ch.get("content", "")) > 1200 else ""))

st.divider()
st.caption(
)
