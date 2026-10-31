"""App-level settings, all overridable via environment variables."""
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("APP_DATA_DIR", "storage"))
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "app.db"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "30"))   # transcript chunk length
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
ALLOWED_EXT = (".mp3", ".wav", ".m4a", ".flac", ".ogg")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))
