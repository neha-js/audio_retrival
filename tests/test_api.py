"""API tests. Whisper and Ollama are stubbed, so no models are needed."""
import os
import tempfile
import unittest
from unittest import mock

os.environ["APP_DATA_DIR"] = tempfile.mkdtemp()

from fastapi.testclient import TestClient  # noqa: E402

from app import main  # noqa: E402

FAKE_CHUNKS = [
    (0, "Hybrid retrieval combines BM25 keyword search with dense embeddings."),
    (30, "A cross encoder reranker rescores the shortlist to save LLM tokens."),
]


class ApiTests(unittest.TestCase):
    def test_full_flow(self):
        with mock.patch.object(main, "transcribe_to_chunks", return_value=FAKE_CHUNKS), \
             mock.patch.object(main.llm, "generate", return_value=None), \
             TestClient(main.app) as client:

            r = client.post("/query", json={"question": "what is hybrid retrieval"})
            self.assertEqual(r.json()["sources"], [])          # nothing indexed yet

            r = client.post("/upload", files={"file": ("talk.mp3", b"fake-audio", "audio/mpeg")},
                            data={"title": "Search Talk", "speaker": "Neha"})
            self.assertEqual(r.status_code, 202)
            ep_id = r.json()["id"]

            files = client.get("/files").json()
            self.assertEqual(files[0]["status"], "ready")
            self.assertEqual(files[0]["segment_count"], 2)

            r = client.post("/query", json={"question": "how does hybrid retrieval work"})
            body = r.json()
            self.assertFalse(body["llm_used"])
            self.assertEqual(body["sources"][0]["title"], "Search Talk")
            self.assertIn("timestamp_sec", body["sources"][0])

            self.assertEqual(client.get(f"/files/{ep_id}/audio").content, b"fake-audio")

            client.delete(f"/files/{ep_id}")
            self.assertEqual(client.get("/files").json(), [])

    def test_rejects_bad_type(self):
        with TestClient(main.app) as client:
            r = client.post("/upload", files={"file": ("notes.txt", b"x", "text/plain")})
            self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
