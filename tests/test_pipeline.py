import unittest

from config import CONFIG
from src.classification.intent_classifier import IntentClassifier
from src.graph.metadata_graph import build_graph_store
from src.ingestion.transcript_loader import TranscriptLoader
from src.pipeline.rag_pipeline import RAGPipeline
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.utils.embeddings import TfidfEmbeddingProvider


class TestPipelineComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segments = TranscriptLoader(CONFIG.data_path).load()
        cls.bm25 = BM25Retriever(cls.segments)
        cls.dense = DenseRetriever(cls.segments, TfidfEmbeddingProvider())
        cls.hybrid = HybridRetriever(cls.segments, cls.bm25, cls.dense)
        cls.reranker = CrossEncoderReranker(CONFIG.reranker.model_name, use_real_cross_encoder=False)
        cls.classifier = IntentClassifier(CONFIG.intent.labels, use_tensorflow=False)
        cls.graph = build_graph_store(False)
        cls.graph.index_segments(cls.segments)
        cls.pipeline = RAGPipeline(CONFIG, cls.hybrid, cls.reranker, cls.classifier, cls.graph)

    def test_loader_reads_all_segments(self):
        self.assertEqual(len(self.segments), 10)

    def test_bm25_finds_relevant_segment(self):
        hits = self.bm25.search("token consumption", top_k=1)
        self.assertIn("token", hits[0][0].text.lower())

    def test_hybrid_retriever_returns_ranked_results(self):
        results = self.hybrid.search("hybrid retrieval embeddings", top_k=3)
        self.assertGreater(len(results), 0)
        scores = [r.fused_score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_metadata_filter_narrows_results(self):
        results = self.hybrid.search("retrieval", top_k=5, metadata_filter={"speaker": "David Kim"})
        self.assertTrue(all(r.segment.speaker == "David Kim" for r in results))

    def test_reranker_reduces_to_top_k(self):
        candidates = self.hybrid.search("cross encoder reranking", top_k=6)
        top = self.reranker.rerank("cross encoder reranking", candidates, top_k=2)
        self.assertLessEqual(len(top), 2)

    def test_intent_classifier_predicts_known_label(self):
        label, confidence = self.classifier.predict("how do i reduce token costs")
        self.assertIn(label, CONFIG.intent.labels)
        self.assertGreaterEqual(confidence, 0.0)

    def test_graph_store_topic_lookup(self):
        speakers = self.graph.top_speakers_for_topic("rag")
        self.assertIn("David Kim", speakers)

    def test_pipeline_end_to_end_reduces_tokens(self):
        result = self.pipeline.run("how do i reduce llm token consumption")
        self.assertLessEqual(result.tokens_actual, result.tokens_if_unranked)
        self.assertGreaterEqual(result.token_savings_pct, 0.0)
        self.assertIn(result.intent, CONFIG.intent.labels)


if __name__ == "__main__":
    unittest.main()
