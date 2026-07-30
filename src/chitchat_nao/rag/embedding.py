from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must be non-zero")
    return values / norms


class SentenceTransformerProvider:
    """Production provider with model construction isolated to this seam."""

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return normalize_vectors(
            self._model.encode(texts, convert_to_numpy=True)
        )
