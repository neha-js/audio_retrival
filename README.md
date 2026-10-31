# Earmark: Audio Retrieval App

Upload audio, and ask questions about it in plain English. Earmark transcribes the audio with Whisper, finds the most relevant passages with hybrid retrieval and reranking, has Llama write the answer, and shows you the exact moments it came from, ready to play.

<!-- Add a screenshot: save it as docs/screenshot.png, then uncomment the next line.
![Earmark UI](docs/screenshot.png)
-->

## Features

- **Audio in, answers out.** Upload mp3, wav, m4a, flac or ogg. Whisper (faster-whisper) transcribes it in the background.
- **Hybrid retrieval.** BM25 keyword search fused with dense embeddings, with metadata filtering by speaker or topic before scoring.
- **Cross-encoder reranking.** The shortlist is rescored so only the best passages are sent to the LLM.
- **Token accounting.** Each query reports tokens used against the naive "stuff everything into the prompt" baseline.
- **Local Llama.** Answers are written by Llama through Ollama. If Ollama isn't running, you still get the retrieved passages.
- **Cited moments.** Every source shows its speaker and timestamp, with a player that starts at that point.
- **Web UI.** Animated black-and-mustard interface with a dark/light switch. No frontend build step.
- **Persistent.** Uploads, transcripts and metadata are stored in SQLite and survive restarts.

## How it works

```
Audio file
  │
  ▼
Whisper (faster-whisper) ──► ~30 second transcript chunks with timestamps
  │
  ▼
SQLite  ◄── indexes are rebuilt from here after each upload
  │
Question
  │
  ▼
IntentClassifier ──► labels the query (factual / summarization / comparison / how-to)
  │
  ▼
HybridRetriever ──► BM25 + dense embeddings, metadata-filtered via GraphStore
  │
  ▼
CrossEncoderReranker ──► shortlist narrowed to the top passages
  │
  ▼
Prompt assembly ──► Llama (Ollama) ──► answer + cited moments
```

## Project structure

```
├── app/
│   ├── main.py            # FastAPI routes
│   ├── transcribe.py      # Whisper: audio -> timestamped chunks
│   ├── service.py         # wraps RAGPipeline, rebuilds indexes after uploads
│   ├── llm.py             # Llama generation through Ollama
│   ├── store.py           # SQLite storage
│   ├── settings.py        # env-var settings
│   └── static/index.html  # the web UI
├── src/
│   ├── ingestion/         # TranscriptSegment model + loader
│   ├── retrieval/         # BM25, dense and hybrid retrievers
│   ├── reranking/         # cross-encoder reranker
│   ├── classification/    # query intent classifier
│   ├── graph/             # metadata graph (in-memory or Neo4j)
│   ├── pipeline/          # RAG orchestration
│   └── utils/             # tokenization, embedding providers
├── config.py              # retrieval pipeline configuration
├── demo.py                # command-line demo on sample transcripts
├── tests/                 # unit and API tests
├── Dockerfile
└── docker-compose.yml
```

## Quick start (Windows, macOS, Linux)

You need Python 3.10+ and [Ollama](https://ollama.com).

```bash
pip install -r requirements-app.txt
ollama pull llama3.2
uvicorn app.main:app --reload
```

Open http://localhost:8000. The first upload is slow because Whisper downloads its model once.

Ollama usually starts automatically after install. If `ollama serve` says the port is already in use, it's already running.

## Docker

```bash
docker compose up --build
```

Open http://localhost:8000. The container talks to the Ollama you already run on your machine (`ollama pull llama3.2` first). Uploads and the database live in a Docker volume, and the Whisper model is cached in another, so neither is re-downloaded or lost between runs.

To run Ollama in a container too:

```bash
docker compose --profile ollama up --build -d
docker compose exec ollama ollama pull llama3.2
```

Set `OLLAMA_URL=http://ollama:11434` first (in a `.env` file next to `docker-compose.yml`, or in your shell) so the app uses it. Docker Desktop on Windows and macOS runs this on CPU, so the containerized Ollama will be slower than a native install.

## API

Interactive docs are at http://localhost:8000/docs.

| Endpoint | Purpose |
| --- | --- |
| `POST /upload` | Upload audio (optional `title`, `speaker`, `topic`). Transcribed and indexed in the background |
| `POST /query` | `{"question": "..."}` returns the answer and sources with timestamps |
| `GET /files` | List uploads and their status |
| `GET /files/{id}/audio` | Stream the original audio |
| `DELETE /files/{id}` | Remove an upload and its transcript |
| `GET /health` | Health check |

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_MODEL` | `llama3.2` | Ollama model used for answers |
| `OLLAMA_URL` | `http://localhost:11434` | Where Ollama is running |
| `WHISPER_MODEL` | `small` | Whisper size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `CHUNK_SECONDS` | `30` | Length of each transcript chunk |
| `MAX_UPLOAD_MB` | `200` | Upload size limit |
| `APP_DATA_DIR` | `storage` | Where audio and the database are kept |

Retrieval settings (top-k, fusion weights, reranker and classifier backends) are in `config.py`.

## Local fallbacks and production swaps

To run anywhere without extra installs, several components use dependency-free fallbacks by default. Each has a documented swap to the real thing:

| Component | Default | Production swap |
| --- | --- | --- |
| Embeddings | `TfidfEmbeddingProvider` | `sentence-transformers` or OpenAI embeddings through `EmbeddingProvider` |
| Reranker | lexical overlap scorer | `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`: set `use_real_cross_encoder=True` in `config.py` |
| Intent classifier | numpy softmax network | `tf.keras.Sequential`: set `use_tensorflow=True` in `config.py` |
| Metadata store | in-memory graph | `Neo4jGraphStore`: set `use_real_neo4j=True` and connection details in `config.py` |

Install the real dependencies with `pip install tensorflow langchain sentence-transformers neo4j` (or `pip install -r requirements.txt`).

## Tests

```bash
python -m unittest tests.test_pipeline tests.test_api -v
python demo.py     # command-line demo on sample transcripts
```

The API tests stub out Whisper and Ollama, so they run without any models installed.

## Known limitations

- Whisper doesn't identify speakers, so speaker defaults to "Unknown" unless you enter one at upload.
- The search index is rebuilt in memory after every upload or delete. This is fine for hundreds of episodes; for much larger collections, move to a persistent vector store such as Chroma or Qdrant.
- The intent classifier labels each query, but the label doesn't yet change how retrieval runs.
- There is no login. Run it locally or behind your own authentication.

## Tech stack

Python, FastAPI, faster-whisper, Llama (Ollama), SQLite, BM25, dense retrieval, cross-encoder reranking, Docker. Optional: TensorFlow, LangChain, Neo4j, sentence-transformers.
