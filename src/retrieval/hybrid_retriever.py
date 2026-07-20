"""
Hybrid retriever: fuses BM25 (lexical) and dense (semantic) retrieval,
with optional metadata filtering (speaker / topic / episode) applied
before scoring so the search space shrinks up front instead of being
filtered after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.ingestion.transcript_loader import TranscriptSegment
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


@dataclass
class ScoredSegment:
    segment: TranscriptSegment
    bm25_score: float
    dense_score: float
    fused_score: float


class HybridRetriever:
    def __init__(
        self,
        segments: List[TranscriptSegment],
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        self.segments = segments
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return scores
        max_score = max(scores.values()) or 1.0
        return {k: v / max_score for k, v in scores.items()}

    def _apply_metadata_filter(
        self, candidates: List[TranscriptSegment], metadata_filter: Optional[Dict[str, str]]
    ) -> List[TranscriptSegment]:
        if not metadata_filter:
            return candidates
        filtered = []
        for seg in candidates:
            if all(getattr(seg, key, None) == value for key, value in metadata_filter.items()):
                filtered.append(seg)
        return filtered

    def search(
        self,
        query: str,
        top_k: int = 6,
        metadata_filter: Optional[Dict[str, str]] = None,
        candidate_pool: int = 10,
    ) -> List[ScoredSegment]:
        """Retrieve top_k segments fusing BM25 + dense scores.

        metadata_filter narrows the candidate pool *before* scoring,
        e.g. {"speaker": "Priya Nair"} or {"topic": "rag"}.
        """
        bm25_hits = self.bm25.search(query, top_k=candidate_pool)
        dense_hits = self.dense.search(query, top_k=candidate_pool)

        bm25_scores = {seg.id: score for seg, score in bm25_hits}
        dense_scores = {seg.id: score for seg, score in dense_hits}

        bm25_scores = self._normalize(bm25_scores)
        dense_scores = self._normalize(dense_scores)

        candidate_ids = set(bm25_scores) | set(dense_scores)
        id_to_segment = {s.id: s for s in self.segments}
        candidates = [id_to_segment[cid] for cid in candidate_ids]
        candidates = self._apply_metadata_filter(candidates, metadata_filter)

        scored = []
        for seg in candidates:
            b = bm25_scores.get(seg.id, 0.0)
            d = dense_scores.get(seg.id, 0.0)
            fused = self.bm25_weight * b + self.dense_weight * d
            scored.append(ScoredSegment(segment=seg, bm25_score=b, dense_score=d, fused_score=fused))

        scored.sort(key=lambda s: s.fused_score, reverse=True)
        return scored[:top_k]
