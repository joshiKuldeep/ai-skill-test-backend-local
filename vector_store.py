"""
Vector store module.
Lightweight in-memory vector store using numpy for cosine similarity.
No heavy dependencies - fully Vercel serverless compatible.
"""

import numpy as np
from config import settings
from chunker import Chunk
from embeddings import embed_texts, embed_query

# In-memory store: {doc_id: {"embeddings": np.array, "texts": [...], "metadatas": [...]}}
_store: dict[str, dict] = {}


def store_chunks(chunks: list[Chunk], doc_id: str) -> int:
    """Embed and store chunks in memory. Returns number stored."""
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    metadatas = [
        {
            "doc_id": doc_id,
            "chunk_index": c.index,
            "pages": str(c.metadata.get("pages", [])),
            "word_count": c.metadata.get("word_count", 0),
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        for c in chunks
    ]

    _store[doc_id] = {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "texts": texts,
        "metadatas": metadatas,
    }

    return len(chunks)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(b_norm, a_norm)


def retrieve_chunks(
    query: str,
    doc_id: str,
    top_k: int = None,
) -> list[dict]:
    """Retrieve the most relevant chunks for a query. Returns list of {text, metadata, similarity_score}."""
    top_k = top_k or settings.RETRIEVAL_TOP_K

    if doc_id not in _store:
        return []

    doc_data = _store[doc_id]
    query_embedding = np.array(embed_query(query), dtype=np.float32)
    similarities = _cosine_similarity(query_embedding, doc_data["embeddings"])

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "text": doc_data["texts"][idx],
            "metadata": doc_data["metadatas"][idx],
            "similarity_score": round(float(similarities[idx]), 4),
        })

    return retrieved


def delete_document(doc_id: str) -> None:
    """Remove all data for a document."""
    _store.pop(doc_id, None)


def document_exists(doc_id: str) -> bool:
    """Check if a document has been indexed."""
    return doc_id in _store
