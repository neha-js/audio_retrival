"""Audio -> (start_second, text) chunks using faster-whisper."""
from pathlib import Path
from typing import List, Tuple

from app import settings

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # imported lazily: heavy

        _model = WhisperModel(settings.WHISPER_MODEL, compute_type="int8")
    return _model


def transcribe_to_chunks(path: Path, chunk_seconds: int) -> List[Tuple[int, str]]:
    """Group Whisper's short segments into ~chunk_seconds windows, which
    retrieve better than 2-3 second fragments."""
    segments, _info = _get_model().transcribe(str(path))
    chunks, start, texts = [], None, []
    for seg in segments:
        if start is None:
            start = seg.start
        texts.append(seg.text.strip())
        if seg.end - start >= chunk_seconds:
            chunks.append((int(start), " ".join(texts)))
            start, texts = None, []
    if texts:
        chunks.append((int(start), " ".join(texts)))
    return chunks
