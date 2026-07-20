"""
Ingestion layer: loads audio transcript segments (already
speech-to-text'd upstream) plus their metadata into TranscriptSegment
objects the rest of the system operates on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class TranscriptSegment:
    """A single retrievable unit: one chunk of an audio transcript."""

    id: str
    episode_id: str
    title: str
    speaker: str
    topic: str
    timestamp_sec: int
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "episode_id": self.episode_id,
            "title": self.title,
            "speaker": self.speaker,
            "topic": self.topic,
            "timestamp_sec": self.timestamp_sec,
            "text": self.text,
        }


class TranscriptLoader:
    """Loads transcript segments from a JSON source.

    Swap `load()` for a real ASR pipeline output (e.g. Whisper JSON,
    AWS Transcribe, or a database query) without touching any
    downstream retrieval code — everything consumes TranscriptSegment.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> List[TranscriptSegment]:
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        segments = [
            TranscriptSegment(
                id=r["id"],
                episode_id=r["episode_id"],
                title=r["title"],
                speaker=r["speaker"],
                topic=r["topic"],
                timestamp_sec=r["timestamp_sec"],
                text=r["text"],
            )
            for r in raw
        ]
        return segments
