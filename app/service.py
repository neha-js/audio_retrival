"""Wraps your existing RAGPipeline so it can grow as audio is uploaded.

The retrievers in src/ index a fixed list of segments at construction
time, so after each upload we rebuild them from the database. That is
fine for hundreds of episodes; for much larger corpora swap in a
persistent vector store (see README "Going to production").
"""
import threading
from typing import Optional

from config import CONFIG
from src.classification.intent_classifier import IntentClassifier
from src.graph.metadata_graph import build_graph_store
from src.pipeline.rag_pipeline import RAGPipeline, RAGResult
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.embeddings import TfidfEmbeddingProvider  # swap for a sentence-transformers provider

from app.store import Store


class PipelineService:
    def __init__(self, store: Store):
        self.store = store
        self._lock = threading.RLock()
        self.pipeline: Optional[RAGPipeline] = None
        # Trained once, reused across rebuilds.
        self._intent = IntentClassifier(CONFIG.intent.labels, CONFIG.intent.use_tensorflow)
        self._reranker = CrossEncoderReranker(
            CONFIG.reranker.model_name, CONFIG.reranker.use_real_cross_encoder
        )
        self.rebuild()

    def rebuild(self) -> None:
        segments = self.store.all_segments()
        with self._lock:
            if not segments:
                self.pipeline = None
                return
            bm25 = BM25Retriever(segments)
            dense = DenseRetriever(segments, TfidfEmbeddingProvider())
            hybrid = HybridRetriever(
                segments, bm25, dense,
                bm25_weight=CONFIG.retrieval.bm25_weight,
                dense_weight=CONFIG.retrieval.dense_weight,
            )
            graph = build_graph_store(
                CONFIG.neo4j.use_real_neo4j, CONFIG.neo4j.uri,
                CONFIG.neo4j.user, CONFIG.neo4j.password,
            )
            graph.index_segments(segments)
            self.pipeline = RAGPipeline(CONFIG, hybrid, self._reranker, self._intent, graph)

    def run(self, question: str) -> Optional[RAGResult]:
        with self._lock:
            pipeline = self.pipeline
        return pipeline.run(question) if pipeline else None
