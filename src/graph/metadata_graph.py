"""
Metadata graph store: models (Episode)-[:HAS_SEGMENT]->(Segment),
(Segment)-[:SPOKEN_BY]->(Speaker), and (Segment)-[:ABOUT]->(Topic) so
the retriever can answer graph-shaped questions like "which speakers
most often discuss retrieval" or pre-filter candidates by speaker/topic
before scoring.

Ships a Neo4jGraphStore (real driver, used when a live instance is
configured) and an InMemoryGraphStore fallback with the same
interface so the rest of the system is storage-agnostic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from src.ingestion.transcript_loader import TranscriptSegment


class GraphStore:
    def index_segments(self, segments: List[TranscriptSegment]) -> None:
        raise NotImplementedError

    def segments_by_speaker(self, speaker: str) -> List[str]:
        raise NotImplementedError

    def segments_by_topic(self, topic: str) -> List[str]:
        raise NotImplementedError

    def top_speakers_for_topic(self, topic: str, top_n: int = 3) -> List[str]:
        raise NotImplementedError


class InMemoryGraphStore(GraphStore):
    """Dependency-free fallback used when no live Neo4j instance is
    configured. Same query surface as the real graph, backed by
    plain dicts instead of Cypher."""

    def __init__(self):
        self._by_speaker: Dict[str, List[str]] = defaultdict(list)
        self._by_topic: Dict[str, List[str]] = defaultdict(list)
        self._speaker_topic_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def index_segments(self, segments: List[TranscriptSegment]) -> None:
        for seg in segments:
            self._by_speaker[seg.speaker].append(seg.id)
            self._by_topic[seg.topic].append(seg.id)
            self._speaker_topic_counts[seg.topic][seg.speaker] += 1

    def segments_by_speaker(self, speaker: str) -> List[str]:
        return list(self._by_speaker.get(speaker, []))

    def segments_by_topic(self, topic: str) -> List[str]:
        return list(self._by_topic.get(topic, []))

    def top_speakers_for_topic(self, topic: str, top_n: int = 3) -> List[str]:
        counts = self._speaker_topic_counts.get(topic, {})
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [speaker for speaker, _ in ranked[:top_n]]


class Neo4jGraphStore(GraphStore):
    """Production backend. Requires `pip install neo4j` and a running
    Neo4j instance. Builds the same (Episode)-(Segment)-(Speaker)/
    (Topic) graph via Cypher MERGE statements.
    """

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase  # raises ImportError if absent

        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def index_segments(self, segments: List[TranscriptSegment]) -> None:
        with self._driver.session() as session:
            for seg in segments:
                session.run(
                    """
                    MERGE (e:Episode {id: $episode_id, title: $title})
                    MERGE (sp:Speaker {name: $speaker})
                    MERGE (t:Topic {name: $topic})
                    MERGE (s:Segment {id: $id, text: $text, timestamp_sec: $timestamp_sec})
                    MERGE (e)-[:HAS_SEGMENT]->(s)
                    MERGE (s)-[:SPOKEN_BY]->(sp)
                    MERGE (s)-[:ABOUT]->(t)
                    """,
                    episode_id=seg.episode_id,
                    title=seg.title,
                    speaker=seg.speaker,
                    topic=seg.topic,
                    id=seg.id,
                    text=seg.text,
                    timestamp_sec=seg.timestamp_sec,
                )

    def segments_by_speaker(self, speaker: str) -> List[str]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Segment)-[:SPOKEN_BY]->(:Speaker {name: $speaker}) RETURN s.id AS id",
                speaker=speaker,
            )
            return [r["id"] for r in result]

    def segments_by_topic(self, topic: str) -> List[str]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Segment)-[:ABOUT]->(:Topic {name: $topic}) RETURN s.id AS id",
                topic=topic,
            )
            return [r["id"] for r in result]

    def top_speakers_for_topic(self, topic: str, top_n: int = 3) -> List[str]:
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (sp:Speaker)<-[:SPOKEN_BY]-(s:Segment)-[:ABOUT]->(:Topic {name: $topic})
                RETURN sp.name AS speaker, count(s) AS n
                ORDER BY n DESC
                LIMIT $top_n
                """,
                topic=topic,
                top_n=top_n,
            )
            return [r["speaker"] for r in result]


def build_graph_store(use_real_neo4j: bool, uri: str = "", user: str = "", password: str = "") -> GraphStore:
    """Factory: returns a live Neo4jGraphStore if configured and
    reachable, otherwise falls back to InMemoryGraphStore."""
    if use_real_neo4j:
        try:
            return Neo4jGraphStore(uri, user, password)
        except Exception:
            pass  # fall through to in-memory
    return InMemoryGraphStore()
