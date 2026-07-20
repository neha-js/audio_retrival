"""
Dense (semantic) retriever. Embeds all transcript segments once at
index time and scores queries by cosine similarity.
"""
from __future__ import annotations

from typing import List, Tuple

from src.ingestion.transcript_loader import TranscriptSegment
from src.utils.embeddings import EmbeddingProvider, cosine_similarity_matrix


class DenseRetriever:
    """Semantic retriever over a pluggable EmbeddingProvider.

    Production swap: pass a SentenceTransformerProvider (or a FAISS /
    pgvector / Pinecone-backed index) instead of TfidfEmbeddingProvider.
    HybridRetriever is unaffected either way.
    """

    def __init__(self, segments: List[TranscriptSegment], embedding_provider: EmbeddingProvider):
        self.segments = segments
        self.embedder = embedding_provider
        texts = [s.text for s in segments]
        self.embedder.fit(texts)
        self.doc_matrix = self.embedder.encode(texts)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[TranscriptSegment, float]]:
        query_vec = self.embedder.encode([query])[0]
        scores = cosine_similarity_matrix(query_vec, self.doc_matrix)
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [(self.segments[i], float(scores[i])) for i in ranked_idx]
