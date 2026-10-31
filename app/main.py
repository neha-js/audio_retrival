"""FastAPI app: upload audio, transcribe + index it, ask questions.

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import llm, settings
from app.service import PipelineService
from app.store import Store
from app.transcribe import transcribe_to_chunks
from src.ingestion.transcript_loader import TranscriptSegment

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    store = Store(settings.DB_PATH)
    state["store"] = store
    state["service"] = PipelineService(store)
    yield


app = FastAPI(title="Audio Retrieval API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def ingest(ep_id: str, path: Path, title: str, speaker: str, topic: str) -> None:
    """Background job: transcribe -> store segments -> rebuild indexes."""
    store, service = state["store"], state["service"]
    try:
        chunks = transcribe_to_chunks(path, settings.CHUNK_SECONDS)
        if not chunks:
            raise ValueError("No speech detected in this audio file.")
        segments = [
            TranscriptSegment(
                id=f"{ep_id}_seg{i:03d}", episode_id=ep_id, title=title,
                speaker=speaker, topic=topic, timestamp_sec=ts, text=text,
            )
            for i, (ts, text) in enumerate(chunks)
        ]
        store.add_segments(segments)
        service.rebuild()
        store.set_status(ep_id, "ready")
    except Exception as e:
        store.set_status(ep_id, "failed", str(e))


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload", status_code=202)
def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(""),
    speaker: str = Form("Unknown"),
    topic: str = Form("uncategorized"),
):
    name = file.filename or "audio"
    if not name.lower().endswith(settings.ALLOWED_EXT):
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(settings.ALLOWED_EXT)}")

    ep_id = uuid.uuid4().hex[:12]
    path = settings.AUDIO_DIR / f"{ep_id}{Path(name).suffix.lower()}"
    limit, written = settings.MAX_UPLOAD_MB * 1024 * 1024, 0
    with path.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"File larger than {settings.MAX_UPLOAD_MB} MB.")
            out.write(chunk)

    title = title.strip() or Path(name).stem
    state["store"].create_episode(ep_id, title, name, path, speaker.strip() or "Unknown",
                                  topic.strip() or "uncategorized")
    background.add_task(ingest, ep_id, path, title, speaker.strip() or "Unknown",
                        topic.strip() or "uncategorized")
    return {"id": ep_id, "status": "processing"}


@app.get("/files")
def list_files():
    return state["store"].list_episodes()


@app.get("/files/{ep_id}/audio")
def get_audio(ep_id: str):
    ep = state["store"].get_episode(ep_id)
    if not ep:
        raise HTTPException(404, "Not found")
    return FileResponse(ep["audio_path"])


@app.delete("/files/{ep_id}")
def delete_file(ep_id: str):
    store = state["store"]
    ep = store.get_episode(ep_id)
    if not ep:
        raise HTTPException(404, "Not found")
    Path(ep["audio_path"]).unlink(missing_ok=True)
    store.delete_episode(ep_id)
    state["service"].rebuild()
    return {"deleted": ep_id}


@app.post("/query")
def query(q: Query):
    result = state["service"].run(q.question)
    if result is None:
        return {"answer": "No audio has been indexed yet. Upload a file first.",
                "llm_used": False, "intent": None, "sources": []}

    answer = llm.generate(result.prompt)
    return {
        "answer": answer or "The language model is not reachable, so here are the most relevant passages.",
        "llm_used": answer is not None,
        "intent": result.intent,
        "tokens": {"naive": result.tokens_if_unranked, "actual": result.tokens_actual},
        "sources": [
            {
                "episode_id": s.segment.episode_id,
                "title": s.segment.title,
                "speaker": s.segment.speaker,
                "timestamp_sec": s.segment.timestamp_sec,
                "text": s.segment.text,
                "score": round(s.fused_score, 3),
            }
            for s in result.reranked
        ],
    }
