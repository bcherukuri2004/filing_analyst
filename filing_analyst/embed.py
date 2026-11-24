"""Turn text into vectors. Two embedders share one tiny interface: a real
model for actual use, and a fast hashing one that tests use so they run
without downloading anything."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended by the BGE model authors


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (m / norms).astype(np.float32)


DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # the embedding model the index is built with


class FastEmbedder:
    """BAAI/bge-small-en-v1.5 through the fastembed library (runs on CPU)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        from fastembed import TextEmbedding

        self.name = model
        self._model = TextEmbedding(model_name=model)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed(texts, batch_size=32))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed([QUERY_PREFIX + text]))))[0]


class HashingEmbedder:
    """Bag of words hashed into a fixed number of buckets. It knows nothing
    about meaning, but it is deterministic and instant, which is what tests
    need."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            v[h % self.dim] += 1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text)[None, :])[0]
