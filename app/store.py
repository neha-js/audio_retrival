"""SQLite persistence for episodes (uploaded audio) and transcript segments."""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from src.ingestion.transcript_loader import TranscriptSegment

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes(
    id TEXT PRIMARY KEY, title TEXT, filename TEXT, audio_path TEXT,
    speaker TEXT, topic TEXT, status TEXT, error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS segments(
    id TEXT PRIMARY KEY, episode_id TEXT, title TEXT, speaker TEXT,
    topic TEXT, timestamp_sec INTEGER, text TEXT);
"""


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_episode(self, ep_id, title, filename, audio_path, speaker, topic):
        with self._conn() as c:
            c.execute(
                "INSERT INTO episodes(id,title,filename,audio_path,speaker,topic,status) "
                "VALUES(?,?,?,?,?,?, 'processing')",
                (ep_id, title, filename, str(audio_path), speaker, topic),
            )

    def set_status(self, ep_id: str, status: str, error: Optional[str] = None):
        with self._conn() as c:
            c.execute("UPDATE episodes SET status=?, error=? WHERE id=?", (status, error, ep_id))

    def add_segments(self, segments: List[TranscriptSegment]):
        with self._conn() as c:
            c.executemany(
                "INSERT INTO segments VALUES(?,?,?,?,?,?,?)",
                [(s.id, s.episode_id, s.title, s.speaker, s.topic, s.timestamp_sec, s.text)
                 for s in segments],
            )

    def list_episodes(self) -> List[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT e.id,e.title,e.filename,e.speaker,e.topic,e.status,e.error,e.created_at,"
                "(SELECT COUNT(*) FROM segments s WHERE s.episode_id=e.id) AS segment_count "
                "FROM episodes e ORDER BY e.created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_episode(self, ep_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM episodes WHERE id=?", (ep_id,)).fetchone()
        return dict(row) if row else None

    def delete_episode(self, ep_id: str):
        with self._conn() as c:
            c.execute("DELETE FROM segments WHERE episode_id=?", (ep_id,))
            c.execute("DELETE FROM episodes WHERE id=?", (ep_id,))

    def all_segments(self) -> List[TranscriptSegment]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT s.* FROM segments s JOIN episodes e ON e.id=s.episode_id "
                "ORDER BY s.episode_id, s.timestamp_sec").fetchall()
        return [TranscriptSegment(**dict(r)) for r in rows]
