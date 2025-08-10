from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List, Dict

def embed_store(chunks: List[Dict], db_path: str = "vectorstore/faiss_index") -> FAISS:
    """
    Embeds chunks into a FAISS vector store using HuggingFace embeddings.

    Args:
        chunks (List[Dict]): Output from split_chunks.
        db_path (str): Where to save the FAISS index.

    Returns:
        FAISS: The FAISS vector store instance.
    """
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    db.save_local(db_path)
    return db
