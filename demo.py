"""
End-to-end demo of the Audio Retrieval Backend.

Run with:
    python demo.py
"""
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


def build_pipeline() -> RAGPipeline:
    segments = TranscriptLoader(CONFIG.data_path).load()

    bm25 = BM25Retriever(segments)
    dense = DenseRetriever(segments, TfidfEmbeddingProvider())
    hybrid = HybridRetriever(
        segments,
        bm25,
        dense,
        bm25_weight=CONFIG.retrieval.bm25_weight,
        dense_weight=CONFIG.retrieval.dense_weight,
    )

    reranker = CrossEncoderReranker(
        CONFIG.reranker.model_name, CONFIG.reranker.use_real_cross_encoder
    )

    intent_classifier = IntentClassifier(CONFIG.intent.labels, CONFIG.intent.use_tensorflow)

    graph_store = build_graph_store(
        CONFIG.neo4j.use_real_neo4j, CONFIG.neo4j.uri, CONFIG.neo4j.user, CONFIG.neo4j.password
    )
    graph_store.index_segments(segments)

    return RAGPipeline(CONFIG, hybrid, reranker, intent_classifier, graph_store)


def print_result(result) -> None:
    print(f"\nQuery: {result.query}")
    print(f"  Routed intent: {result.intent}  (confidence {result.intent_confidence:.2f})")
    print(f"  Hybrid candidates: {len(result.retrieved)} -> reranked to: {len(result.reranked)}")
    for rank, s in enumerate(result.reranked, start=1):
        print(
            f"    #{rank} [{s.fused_score:.3f}] {s.segment.title} "
            f"({s.segment.speaker}, {s.segment.timestamp_sec}s): {s.segment.text[:80]}..."
        )
    print(
        f"  Tokens — naive: {result.tokens_if_unranked}, actual: {result.tokens_actual} "
        f"({result.token_savings_pct:.1f}% saved by reranking/trimming)"
    )


if __name__ == "__main__":
    pipeline = build_pipeline()

    demo_queries = [
        "how do i reduce llm token consumption",
        "what did David Kim say about reranking",
        "compare bm25 and dense retrieval",
        "summarize the episode about graph databases",
    ]

    total_naive = 0
    total_actual = 0

    for q in demo_queries:
        result = pipeline.run(q)
        print_result(result)
        total_naive += result.tokens_if_unranked
        total_actual += result.tokens_actual

    overall_savings = 100.0 * (total_naive - total_actual) / total_naive
    print(f"\n--- Overall across {len(demo_queries)} queries ---")
    print(f"Naive token spend: {total_naive}")
    print(f"Optimized token spend: {total_actual}")
    print(f"Total token savings: {overall_savings:.1f}%")

    print("\n--- Graph store: top speakers on 'rag' topic ---")
    print(pipeline.graph_store.top_speakers_for_topic("rag"))
