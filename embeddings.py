"""
Embedding module.
Uses sentence-transformers with BAAI/bge-small-en-v1.5 for LOCAL embeddings.
No external API calls — runs entirely on the machine.
"""

import logging

from sentence_transformers import SentenceTransformer

from config import settings
from exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_model = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        try:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{settings.EMBEDDING_MODEL}': {str(e)}. "
                "Make sure sentence-transformers is installed: pip install sentence-transformers"
            )
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Runs locally — no API calls, no rate limits.
    """
    if not texts:
        return []

    try:
        model = get_model()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        return embeddings.tolist()
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Embedding generation failed: {str(e)}")


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a search query.
    BGE recommends prefixing queries with 'Represent this sentence:' for retrieval.
    """
    try:
        model = get_model()
        prefixed_query = f"Represent this sentence: {query}"
        embedding = model.encode(
            [prefixed_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"Query embedding failed: {str(e)}")