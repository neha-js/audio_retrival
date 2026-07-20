"""
Embedding provider abstraction.

Any component that needs vector embeddings (the dense retriever, the
cross-encoder fallback) depends only on the `EmbeddingProvider`
interface below, not on a specific model. That keeps the swap between
"TF-IDF running locally" and "a real sentence-transformers /
OpenAI / Vertex embedding model" a one-line change at the call site.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

import numpy as np

from src.utils.text import tokenize


class EmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """Fit any corpus-level statistics (vocab, IDF, etc.)."""

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Return an (n_texts, dim) matrix of embeddings."""


class TfidfEmbeddingProvider(EmbeddingProvider):
    """Dependency-free stand-in for a dense embedding model.

    In production, replace with e.g.:

        from sentence_transformers import SentenceTransformer
        class SentenceTransformerProvider(EmbeddingProvider):
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)
            def fit(self, corpus): pass  # no corpus fitting needed
            def encode(self, texts):
                return self.model.encode(texts, normalize_embeddings=True)

    Everything downstream (HybridRetriever, etc.) is unaffected by the
    swap since both implementations satisfy EmbeddingProvider.
    """

    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, corpus: List[str]) -> None:
        doc_token_sets = [set(tokenize(doc)) for doc in corpus]
        vocab_counter: Counter = Counter()
        for tokens in doc_token_sets:
            vocab_counter.update(tokens)

        self.vocab = {term: i for i, term in enumerate(sorted(vocab_counter))}
        n_docs = len(corpus)
        df = np.zeros(len(self.vocab))
        for tokens in doc_token_sets:
            for term in tokens:
                df[self.vocab[term]] += 1
        # smoothed idf
        self.idf = np.log((n_docs + 1) / (df + 1)) + 1

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("Call fit() before encode().")

        vectors = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            tf = Counter(tokens)
            max_tf = max(tf.values())
            for term, count in tf.items():
                idx = self.vocab.get(term)
                if idx is None:
                    continue  # unseen term (query-time OOV)
                vectors[i, idx] = (count / max_tf) * self.idf[idx]

        # L2-normalize so cosine similarity == dot product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def cosine_similarity_matrix(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """query_vec: (dim,) already-normalized; doc_matrix: (n, dim) already-normalized."""
    return doc_matrix @ query_vec
