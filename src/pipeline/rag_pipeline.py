"""
End-to-end RAG orchestration.

Structured to mirror LangChain's Runnable/Chain composition
(query -> route -> retrieve -> rerank -> build prompt -> generate) so
swapping in real `langchain` primitives (LLMChain, RunnableSequence,
PromptTemplate) later is a refactor of this one file, not a rewrite
of the retrieval/reranking/classification layers underneath it.

Production swap sketch, using real LangChain:

    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_openai import ChatOpenAI

    prompt = PromptTemplate.from_template(RAGPipeline.PROMPT_TEMPLATE)
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = (
        RunnableLambda(self._retrieve_and_rerank)
        | prompt
        | llm
    )

Everything upstream of prompt construction (classifier, hybrid
retriever, reranker) is untouched by that swap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from config import Config
from src.classification.intent_classifier import IntentClassifier
from src.graph.metadata_graph import GraphStore
from src.ingestion.transcript_loader import TranscriptSegment
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.hybrid_retriever import HybridRetriever, ScoredSegment
from src.utils.text import estimate_token_count


@dataclass
class RAGResult:
    query: str
    intent: str
    intent_confidence: float
    retrieved: List[ScoredSegment]        # hybrid shortlist
    reranked: List[ScoredSegment]         # final passages sent to the LLM
    prompt: str
    tokens_if_unranked: int               # what we'd have spent without reranking/trimming
    tokens_actual: int
    token_savings_pct: float


class RAGPipeline:
    """Wires: IntentClassifier -> HybridRetriever -> CrossEncoderReranker
    -> prompt assembly. GraphStore is used for metadata-aware
    pre-filtering when the query names a speaker or topic.
    """

    PROMPT_TEMPLATE = (
        "You are answering questions about a podcast archive using only "
        "the transcript excerpts below.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n"
        "Answer using only the context above, citing episode titles where relevant."
    )

    def __init__(
        self,
        config: Config,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        intent_classifier: IntentClassifier,
        graph_store: GraphStore,
    ):
        self.config = config
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.intent_classifier = intent_classifier
        self.graph_store = graph_store

    def _resolve_metadata_filter(self, query: str) -> Optional[Dict[str, str]]:
        """Very light entity spotting against known speakers/topics so a
        query like 'what did David Kim say about reranking' narrows the
        candidate pool via the graph store before scoring. A production
        system would use NER or the intent classifier's slot-filling
        output here instead of substring matching.
        """
        query_lower = query.lower()
        for speaker in getattr(self.graph_store, "_by_speaker", {}):
            if speaker.lower() in query_lower:
                return {"speaker": speaker}
        for topic in getattr(self.graph_store, "_by_topic", {}):
            if topic.lower().replace("-", " ") in query_lower:
                return {"topic": topic}
        return None

    def run(self, query: str) -> RAGResult:
        intent, confidence = self.intent_classifier.predict(query)

        metadata_filter = self._resolve_metadata_filter(query)

        retrieved = self.hybrid_retriever.search(
            query,
            top_k=self.config.retrieval.top_k_hybrid,
            metadata_filter=metadata_filter,
        )

        reranked = self.reranker.rerank(
            query, retrieved, top_k=self.config.retrieval.top_k_final
        )

        context = "\n\n".join(
            f"[{s.segment.title} — {s.segment.speaker}, {s.segment.timestamp_sec}s] {s.segment.text}"
            for s in reranked
        )
        prompt = self.PROMPT_TEMPLATE.format(context=context, query=query)

        # Token accounting: "unranked" = every hybrid candidate stuffed
        # into the prompt (the naive approach), vs. "actual" = only the
        # reranked top-k that the reranker decided were worth sending.
        unranked_context = "\n\n".join(s.segment.text for s in retrieved)
        tokens_if_unranked = estimate_token_count(unranked_context) + estimate_token_count(query)
        tokens_actual = estimate_token_count(prompt)
        savings_pct = (
            100.0 * (tokens_if_unranked - tokens_actual) / tokens_if_unranked
            if tokens_if_unranked
            else 0.0
        )

        return RAGResult(
            query=query,
            intent=intent,
            intent_confidence=confidence,
            retrieved=retrieved,
            reranked=reranked,
            prompt=prompt,
            tokens_if_unranked=tokens_if_unranked,
            tokens_actual=tokens_actual,
            token_savings_pct=savings_pct,
        )
