"""
Central configuration for the Audio Retrieval Backend.

Swap these values (or wire them to environment variables) when moving
from local/demo mode to a production deployment with real TensorFlow
models, a hosted LLM, and a live Neo4j instance.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_bm25: int = 5
    top_k_dense: int = 5
    top_k_hybrid: int = 6          # candidates passed into the reranker
    top_k_final: int = 3           # passages actually sent to the LLM
    bm25_weight: float = 0.5
    dense_weight: float = 0.5


@dataclass(frozen=True)
class RerankerConfig:
    # In production: "cross-encoder/ms-marco-MiniLM-L-6-v2" via
    # sentence_transformers.CrossEncoder. Falls back to a lexical
    # scorer when that library isn't available (see reranking/).
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_real_cross_encoder: bool = False


@dataclass(frozen=True)
class IntentConfig:
    labels: tuple = ("factual_lookup", "summarization", "comparison", "how_to")
    # In production: a small tf.keras Sequential model, see
    # classification/intent_classifier.py for the swap point.
    use_tensorflow: bool = False


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "changeme"
    use_real_neo4j: bool = False   # falls back to an in-memory graph store


@dataclass(frozen=True)
class Config:
    retrieval: RetrievalConfig = RetrievalConfig()
    reranker: RerankerConfig = RerankerConfig()
    intent: IntentConfig = IntentConfig()
    neo4j: Neo4jConfig = Neo4jConfig()
    data_path: str = "data/sample_transcripts.json"


CONFIG = Config()
