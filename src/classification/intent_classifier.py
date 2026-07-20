"""
Intent classification for query routing.

Classifies an incoming query into one of a few intents (factual
lookup, summarization, comparison, how-to) so the pipeline can route
cheap queries around the full hybrid-retrieval + reranking path and
save both latency and LLM tokens.

Ships two backends behind the same `IntentClassifier` interface:
  - TensorFlowIntentClassifier: a small tf.keras Sequential model
    (used when TensorFlow is available / use_tensorflow=True).
  - NumpyIntentClassifier: a from-scratch single-hidden-layer network
    trained with plain-numpy gradient descent, used as a dependency-
    free fallback so the project runs anywhere.

Both learn a bag-of-words -> softmax mapping from a small labeled
seed set; in production this would be fine-tuned on real logged
queries.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.utils.text import tokenize

SEED_TRAINING_DATA: List[Tuple[str, str]] = [
    ("what does bm25 stand for", "factual_lookup"),
    ("who is the speaker in episode 4", "factual_lookup"),
    ("when was hybrid retrieval discussed", "factual_lookup"),
    ("what is a cross encoder", "factual_lookup"),
    ("summarize the episode about token costs", "summarization"),
    ("give me a summary of the graph database episode", "summarization"),
    ("tl;dr the intent classification discussion", "summarization"),
    ("condense what was said about reranking", "summarization"),
    ("compare bm25 and dense retrieval", "comparison"),
    ("what is the difference between neo4j and a vector store", "comparison"),
    ("bm25 vs dense embeddings which is better", "comparison"),
    ("compare cross encoder reranking to embedding similarity", "comparison"),
    ("how do i reduce llm token consumption", "how_to"),
    ("how to build a hybrid retriever", "how_to"),
    ("steps to set up metadata filtering", "how_to"),
    ("how can i cut retrieval latency", "how_to"),
]


class _BagOfWordsVectorizer:
    def __init__(self):
        self.vocab: Dict[str, int] = {}

    def fit(self, texts: List[str]) -> None:
        vocab = set()
        for t in texts:
            vocab.update(tokenize(t))
        self.vocab = {term: i for i, term in enumerate(sorted(vocab))}

    def transform(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)))
        for i, t in enumerate(texts):
            for term in tokenize(t):
                idx = self.vocab.get(term)
                if idx is not None:
                    mat[i, idx] = 1.0
        return mat


class IntentClassifier:
    """Public interface used by the RAG pipeline. Delegates to a
    TensorFlow or numpy backend depending on config."""

    def __init__(self, labels: Tuple[str, ...], use_tensorflow: bool = False):
        self.labels = list(labels)
        self.use_tensorflow = use_tensorflow
        self._backend = None

        if use_tensorflow:
            try:
                self._backend = _TensorFlowBackend(self.labels)
            except ImportError:
                self.use_tensorflow = False

        if self._backend is None:
            self._backend = _NumpyBackend(self.labels)

        self._backend.train(SEED_TRAINING_DATA)

    def predict(self, query: str) -> Tuple[str, float]:
        return self._backend.predict(query)


class _NumpyBackend:
    """Single-hidden-layer softmax classifier, trained with vanilla
    gradient descent — a dependency-free stand-in for the tf.keras
    model used in production."""

    def __init__(self, labels: List[str], hidden_dim: int = 16, lr: float = 0.5, epochs: int = 300):
        self.labels = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.vectorizer = _BagOfWordsVectorizer()
        self.W1 = self.b1 = self.W2 = self.b2 = None

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def train(self, data: List[Tuple[str, str]]) -> None:
        texts = [t for t, _ in data]
        labels = [l for _, l in data]
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        y_idx = np.array([self.label_to_idx[l] for l in labels])
        n_samples, n_features = X.shape
        n_classes = len(self.labels)

        rng = np.random.default_rng(42)
        self.W1 = rng.normal(0, 0.5, (n_features, self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.normal(0, 0.5, (self.hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)

        Y = np.zeros((n_samples, n_classes))
        Y[np.arange(n_samples), y_idx] = 1.0

        for _ in range(self.epochs):
            h = np.tanh(X @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            probs = self._softmax(logits)

            grad_logits = (probs - Y) / n_samples
            grad_W2 = h.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)

            grad_h = grad_logits @ self.W2.T * (1 - h**2)
            grad_W1 = X.T @ grad_h
            grad_b1 = grad_h.sum(axis=0)

            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2

    def predict(self, query: str) -> Tuple[str, float]:
        x = self.vectorizer.transform([query])
        h = np.tanh(x @ self.W1 + self.b1)
        probs = self._softmax(h @ self.W2 + self.b2)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


class _TensorFlowBackend:
    """Production backend. Requires `pip install tensorflow`.

    Mirrors _NumpyBackend's interface exactly so IntentClassifier can
    swap between them with zero changes to calling code.
    """

    def __init__(self, labels: List[str]):
        import tensorflow as tf  # noqa: F401  (raises ImportError if absent)

        self.tf = tf
        self.labels = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.vectorizer = _BagOfWordsVectorizer()
        self.model = None

    def train(self, data: List[Tuple[str, str]]) -> None:
        tf = self.tf
        texts = [t for t, _ in data]
        labels = [l for _, l in data]
        self.vectorizer.fit(texts)
        X = self.vectorizer.transform(texts)
        y = np.array([self.label_to_idx[l] for l in labels])

        n_features = X.shape[1]
        n_classes = len(self.labels)

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(n_features,)),
                tf.keras.layers.Dense(16, activation="tanh"),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        self.model.fit(X, y, epochs=50, verbose=0)

    def predict(self, query: str) -> Tuple[str, float]:
        x = self.vectorizer.transform([query])
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])
