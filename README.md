# AI-Powered Audio Retrieval Backend

A modular Python backend for semantic search over audio transcripts, using
Retrieval-Augmented Generation (RAG) with metadata-aware hybrid retrieval,
cross-encoder reranking, and TensorFlow-based intent routing.

## Architecture

```
Query
  │
  ▼
IntentClassifier ──────► routes query (factual / summarization / comparison / how-to)
  │
  ▼
HybridRetriever (BM25 + dense embeddings, metadata-filtered via GraphStore)
  │
  ▼
CrossEncoderReranker ───► rescopes shortlist down to top-k
  │
  ▼
Prompt assembly ────────► sent to LLM (LangChain-style chain)
```

```
audio_rag_backend/
├── config.py                          # central config, all swap points documented
├── demo.py                            # runnable end-to-end demo
├── data/sample_transcripts.json       # sample transcript + metadata corpus
├── src/
│   ├── ingestion/transcript_loader.py     # TranscriptSegment model + loader
│   ├── retrieval/
│   │   ├── bm25_retriever.py              # sparse/lexical retrieval (BM25, from scratch)
│   │   ├── dense_retriever.py             # semantic retrieval over an EmbeddingProvider
│   │   └── hybrid_retriever.py            # fuses BM25 + dense, metadata pre-filtering
│   ├── reranking/cross_encoder_reranker.py# cross-encoder rescoring of the shortlist
│   ├── classification/intent_classifier.py# query-routing classifier (TF or numpy backend)
│   ├── graph/metadata_graph.py            # Neo4j-backed (or in-memory) metadata graph
│   ├── pipeline/rag_pipeline.py           # orchestration (LangChain-style chain)
│   └── utils/                             # tokenization, embedding abstractions
└── tests/test_pipeline.py             # unit tests for every layer
```

## Design principles

- **OOP, interface-driven**: every swappable piece (`EmbeddingProvider`,
  `GraphStore`, the reranker/classifier backends) is defined against an
  abstract interface, so retrieval, reranking, and routing logic never
  depend on a specific vendor or model.
- **Metadata-aware retrieval**: queries that name a speaker or topic are
  pre-filtered through the metadata graph *before* scoring, shrinking the
  candidate pool instead of filtering after the fact.
- **Token-conscious**: the pipeline reports token spend for a naive
  "stuff everything retrieved into the prompt" baseline vs. the actual
  reranked/trimmed prompt, so the token savings from reranking are
  measurable, not just claimed.

## Running it

This environment has no external network access, so the demo runs with
dependency-free fallbacks by default — no install step required:

```bash
python demo.py
python -m unittest tests.test_pipeline -v
```

- **Dense retrieval** uses a from-scratch TF-IDF `EmbeddingProvider`
  instead of a hosted embedding model.
- **Reranking** uses a lexical overlap/proximity scorer instead of a
  trained cross-encoder.
- **Intent classification** uses a from-scratch numpy softmax classifier
  instead of `tf.keras`.
- **Metadata graph** uses an in-memory dict-backed store instead of a
  live Neo4j instance.

## Going to production

Every fallback has a documented one-line swap to the real dependency:

| Component | Local fallback | Production swap |
|---|---|---|
| Embeddings | `TfidfEmbeddingProvider` | `sentence-transformers` / OpenAI embeddings via `EmbeddingProvider` |
| Reranker | lexical overlap scorer | `sentence_transformers.CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` — set `use_real_cross_encoder=True` in `config.py` |
| Intent classifier | numpy softmax net | `tf.keras.Sequential` — set `use_tensorflow=True` in `config.py` |
| Metadata store | `InMemoryGraphStore` | `Neo4jGraphStore` — set `use_real_neo4j=True` and connection details in `config.py` |
| Orchestration | plain Python chain in `rag_pipeline.py` | `langchain_core.runnables` / `LLMChain` (see docstring in `rag_pipeline.py`) |

Install real dependencies with:

```bash
pip install tensorflow langchain langchain-openai sentence-transformers neo4j
```

Then flip the relevant flags in `config.py` — no other code changes required.

## Tech stack

Python, TensorFlow, LangChain, Neo4j, RAG, Hybrid Retrieval, Git
