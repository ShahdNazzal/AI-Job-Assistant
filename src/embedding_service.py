"""Sentence-Transformer embeddings with lazy loading and caching.

The model (~90 MB for all-MiniLM-L6-v2) is downloaded on first use and then
cached. Long documents are split into word chunks, embedded separately and
mean-pooled, because MiniLM truncates input at 256 word-piece tokens.

All vectors are L2-normalized, so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import numpy as np

from config.settings import SETTINGS
from src.exceptions import ModelNotAvailableError
from src.utils import get_logger

logger = get_logger("embedding_service")


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ModelNotAvailableError(
            "sentence-transformers is not installed. Run: "
            "pip install sentence-transformers"
        ) from exc
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # noqa: BLE001 - network, disk, HF outages...
        raise ModelNotAvailableError(
            f"Could not load embedding model '{model_name}'. First use "
            f"requires internet access to download ~90 MB from Hugging Face "
            f"(then it is cached locally). Underlying error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class EmbeddingService:
    """Thin, testable wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or SETTINGS.embedding_model
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        if self._model is None:
            logger.info("Loading embedding model %s ...", self.model_name)
            self._model = _load_sentence_transformer(self.model_name)

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts; returns an (n, dim) L2-normalized array."""
        self._ensure_model()
        cleaned = [t if isinstance(t, str) and t.strip() else " " for t in texts]
        vectors = self._model.encode(
            cleaned,
            batch_size=SETTINGS.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text; returns a 1-D normalized vector."""
        return self.embed([text])[0]

    def embed_long_text(self, text: str) -> np.ndarray:
        """Embed long text by chunking into word windows and mean-pooling."""
        chunks = self.split_long_text(text)
        if not chunks:
            return self.embed_one(" ")
        if len(chunks) == 1:
            return self.embed_one(chunks[0])
        vecs = self.embed(chunks)
        pooled = vecs.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    @staticmethod
    def split_long_text(text: str) -> list[str]:
        words = (text or "").split()
        size = SETTINGS.long_text_chunk_words
        return [
            " ".join(words[i : i + size]) for i in range(0, len(words), size)
        ]

    @staticmethod
    def similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity of two (already normalized) vectors, in [0, 1]."""
        dot = float(np.dot(vec_a, vec_b))
        return max(0.0, min(1.0, dot))

    def embed_taxonomy_names(self, names: list[str], taxonomy_version: str) -> np.ndarray:
        """Embed taxonomy skill names with a process-level cache.

        Skill names change only when the taxonomy changes, so the vectors are
        cached by (model, taxonomy version, names) to avoid re-encoding.
        """
        return _cached_taxonomy_embeddings(
            self.model_name, taxonomy_version, tuple(names), self
        )


class DummyEmbeddingService(EmbeddingService):
    """Deterministic hash-based embeddings for tests (no model, no network).

    Bag-of-words hashing: shared tokens -> shared vector components, so
    cosine similarity roughly tracks lexical overlap. Good enough to test
    matching logic deterministically; never used by the real app.
    """

    def __init__(self, dim: int = 96):
        super().__init__(model_name="dummy")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _token_vector(self, token: str) -> np.ndarray:
        import hashlib

        digest = hashlib.md5(token.lower().encode("utf-8")).digest()
        vec = np.zeros(self._dim, dtype=np.float32)
        for i in range(3):
            idx = int.from_bytes(digest[i * 2 : i * 2 + 2], "big") % self._dim
            vec[idx] += 1.0
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        import re

        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            vec = np.zeros(self._dim, dtype=np.float32)
            tokens = re.findall(r"[a-z0-9]+", str(text).lower())
            for token in tokens:
                vec += self._token_vector(token)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            out[row] = vec
        return out


try:  # Streamlit-aware factory: one shared model instance across pages.
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _build_service(model_name: str) -> EmbeddingService:
        return EmbeddingService(model_name)

except Exception:  # pragma: no cover - outside Streamlit (tests, CLI)

    def _build_service(model_name: str) -> EmbeddingService:
        return EmbeddingService(model_name)


_SERVICE_SINGLETON: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Return the shared EmbeddingService (Streamlit-cached when available)."""
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        try:
            _SERVICE_SINGLETON = _build_service(SETTINGS.embedding_model)
        except Exception:
            _SERVICE_SINGLETON = EmbeddingService(SETTINGS.embedding_model)
    return _SERVICE_SINGLETON


from functools import lru_cache  # noqa: E402  (kept close to its only user)


@lru_cache(maxsize=8)
def _cached_taxonomy_embeddings(
    model_name: str,
    taxonomy_version: str,
    names: tuple[str, ...],
    service: EmbeddingService,
) -> np.ndarray:
    return service.embed(list(names))
