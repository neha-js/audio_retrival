"""
Cross-encoder reranking.

A cross-encoder jointly encodes (query, passage) pairs and outputs a
single relevance score — slower than embedding similarity but far
more accurate, so it's applied only to the small hybrid-retrieval
shortlist, not the whole corpus.

This module ships a dependency-free lexical fallback so the pipeline
runs without internet access. Flip `use_real_cross_encoder=True` in
config.py (with sentence-transformers installed) to use an actual
MS MARCO cross-encoder — the call site in rag_pipeline.py doesn't
change either way.
"""
from __future__ import annotations

from typing import List

from src.retrieval.hybrid_retriever import ScoredSegment
from src.utils.text import tokenize


class CrossEncoderReranker:
    def __init__(self, model_name: str, use_real_cross_encoder: bool = False):
        self.model_name = model_name
        self.use_real_cross_encoder = use_real_cross_encoder
        self._model = None

        if self.use_real_cross_encoder:
            # Production path:
            #   from sentence_transformers import CrossEncoder
            #   self._model = CrossEncoder(self.model_name)
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                self._model = CrossEncoder(self.model_name)
            except ImportError:
                self.use_real_cross_encoder = False  # graceful fallback

    def _lexical_pair_score(self, query: str, passage: str) -> float:
        """Fallback (query, passage) relevance score: token overlap
        with a small bonus for phrase adjacency, roughly mimicking
        what a cross-encoder rewards (fine-grained query/passage
        interaction) without needing a trained model."""
        q_tokens = set(tokenize(query))
        p_tokens = tokenize(passage)
        if not q_tokens or not p_tokens:
            return 0.0

        p_token_set = set(p_tokens)
        overlap = len(q_tokens & p_token_set) / len(q_tokens)

        # bonus if query tokens appear close together in the passage
        positions = [i for i, t in enumerate(p_tokens) if t in q_tokens]
        proximity_bonus = 0.0
        if len(positions) > 1:
            span = positions[-1] - positions[0] + 1
            proximity_bonus = len(positions) / span

        return overlap + 0.25 * proximity_bonus

    def rerank(self, query: str, candidates: List[ScoredSegment], top_k: int = 3) -> List[ScoredSegment]:
        if not candidates:
            return []

        if self.use_real_cross_encoder and self._model is not None:
            pairs = [(query, c.segment.text) for c in candidates]
            raw_scores = self._model.predict(pairs)
        else:
            raw_scores = [self._lexical_pair_score(query, c.segment.text) for c in candidates]

        rescored = [
            ScoredSegment(
                segment=c.segment,
                bm25_score=c.bm25_score,
                dense_score=c.dense_score,
                fused_score=float(score),
            )
            for c, score in zip(candidates, raw_scores)
        ]
        rescored.sort(key=lambda s: s.fused_score, reverse=True)
        return rescored[:top_k]
