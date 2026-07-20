"""
BM25 keyword retriever, implemented from scratch (no rank_bm25
dependency) so the project runs anywhere. Swappable for rank_bm25 or
an Elasticsearch/OpenSearch BM25 index in production — see the
`BM25Retriever` docstring.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from src.ingestion.transcript_loader import TranscriptSegment
from src.utils.text import tokenize


class BM25Retriever:
    """Classic Okapi BM25 over transcript segment text.

    Production swap: point this at an OpenSearch/Elasticsearch BM25
    index and keep the same `search(query, top_k) -> List[(segment, score)]`
    signature; nothing in HybridRetriever needs to change.
    """

    def __init__(self, segments: List[TranscriptSegment], k1: float = 1.5, b: float = 0.75):
        self.segments = segments
        self.k1 = k1
        self.b = b

        self.doc_tokens = [tokenize(s.text) for s in segments]
        self.doc_lengths = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

        self.doc_freq: dict = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.n_docs = len(segments)
        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in self.doc_freq.items()
        }

    def _score(self, query_tokens: List[str], doc_idx: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        term_counts: dict = {}
        for term in self.doc_tokens[doc_idx]:
            term_counts[term] = term_counts.get(term, 0) + 1

        for term in query_tokens:
            if term not in term_counts:
                continue
            idf = self.idf.get(term, 0.0)
            freq = term_counts[term]
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TranscriptSegment, float]]:
        query_tokens = tokenize(query)
        scored = [
            (self.segments[i], self._score(query_tokens, i))
            for i in range(self.n_docs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
