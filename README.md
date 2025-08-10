🧠 Enterprise Intelligence Hub
A minimal, production‑leaning RAG (Retrieval‑Augmented Generation) app for PDFs.
Upload a document (or use the sample), chunk & index it, then ask grounded questions powered by Gemini.

Tech: Streamlit · Python 3.11+ · PyMuPDF · FAISS · google‑generativeai
Deploy: Docker + Cloud Run (GCP)

✨ Features
Upload or Sample: Try with your own PDF or the included sample.

Chunking pipeline: Clean page text → overlapping chunks.

Lightweight retrieval: FAISS cosine similarity + top‑K evidence.

Grounded answers: Gemini responses constrained by retrieved context.

Evidence viewer: Inspect the chunks used for each answer.

Cloud‑ready: Dockerfile + Cloud Run instructions.

📂 Project structure
bash
Copy
Edit
enterprise_intelligence_hub/
├─ app/
│  ├─ streamlit_app.py        # Streamlit UI (entrypoint)
│  └─ query_gemini.py         # (optional helpers)
├─ ingest/
│  ├─ load_pdf.py             # PyMuPDF page extraction
│  └─ split_text.py           # chunking logic
├─ rag/
│  ├─ query_engine.py         # FAISS retrieval + top_k selection
│  └─ rag_infer.py            # (optional helpers)
├─ sample_data/
│  └─ test_budget.pdf         # demo file (small)
├─ vectorstore/               # (gitignored) FAISS index lives here locally
│  ├─ index.faiss
│  └─ index.pkl
├─ requirements.txt
└─ Dockerfile
ℹ️ vectorstore/ is ignored in Git to keep the repo small and reproducible. See the section Why vector stores aren’t in the repo below.
