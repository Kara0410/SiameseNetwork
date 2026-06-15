# Deployment

## Local development (no Docker)

```bash
# ai-services + backend
cd ai-services && pip install -e ".[dev]" && cd ..
cd backend && pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Defaults (`VISIONIQ_EMBEDDING_DEFAULT=dummy`, `VISIONIQ_REASONING_PROVIDER=template`)
need no API keys, GPU, or model downloads.

## Docker Compose (self-hosted, recommended for a full local stack)

```bash
cd docker
cp .env.example .env   # edit as needed
docker compose up --build

# with a local Ollama server for the `ollama` reasoning provider:
docker compose --profile ollama up --build
```

This builds `backend/Dockerfile` and `frontend/Dockerfile` from the repo root (both
Dockerfiles expect that build context so the backend can `pip install -e ../ai-services`),
and persists SQLite + FAISS data in the `visioniq-data` volume.

- Frontend: http://localhost:3000
- Backend: http://localhost:8000 (`/healthz`, `/api/v1/...`)
- Ollama (optional): http://localhost:11434

## Frontend on Vercel

1. Import the repo, set the project root to `frontend/`.
2. Set `NEXT_PUBLIC_API_BASE_URL` to your deployed backend URL (e.g.
   `https://visioniq-api.onrender.com`).
3. Vercel auto-detects Next.js; no extra build configuration is required
   (`output: "standalone"` in `next.config.mjs` is harmless on Vercel - it's used by the
   Docker image instead).

## Backend on Render / Fly.io / Railway

All three support deploying directly from `backend/Dockerfile` with the repo root as
build context:

- **Render**: New Web Service → Docker → set "Dockerfile Path" to `backend/Dockerfile`
  and "Docker Build Context Directory" to the repo root. Add a persistent disk mounted
  at `/data` and set `VISIONIQ_VECTOR_STORE_DIR=/data/vector_store`,
  `VISIONIQ_DATABASE_URL=sqlite:////data/visioniq.db`.
- **Fly.io**: `fly launch` from the repo root, point `fly.toml` at
  `dockerfile = "backend/Dockerfile"`, attach a volume at `/data`.
- **Railway**: New Service → Dockerfile, set the Dockerfile path to
  `backend/Dockerfile` with repo-root context, add a volume at `/data`.

In every case, set `VISIONIQ_CORS_ORIGINS` to a JSON array containing your frontend's
deployed origin, e.g. `["https://visioniq.vercel.app"]`.

## Switching to real models and an LLM reasoning provider

Everything defaults to the `dummy` embedding backend and the `template` reasoning
engine so the app runs instantly with no downloads or API keys. To run the full stack:

### Embedding backend

Set `VISIONIQ_EMBEDDING_DEFAULT` to one of `clip`, `vit`, `efficientnet`, or `siamese`.
The first request downloads weights into `VISIONIQ_MODEL_CACHE`
(default `~/.cache/visioniq/models`, or `/data/model_cache` in the Docker image) -
pre-fetch them instead with:

```bash
python scripts/download_models.py clip vit efficientnet siamese
```

For `siamese`, optionally set `VISIONIQ_SIAMESE_CHECKPOINT` to a checkpoint produced by
`python -m ai_services.siamese.train` (see `ai-services/ai_services/siamese`).

### Reasoning provider

- **Ollama**: set `VISIONIQ_REASONING_PROVIDER=ollama`, `VISIONIQ_OLLAMA_HOST`
  (`http://localhost:11434` locally, `http://ollama:11434` in docker-compose), and
  `VISIONIQ_OLLAMA_MODEL` (e.g. `llama3.2`). Pull the model first:
  `docker compose --profile ollama exec ollama ollama pull llama3.2`.
- **OpenAI-compatible**: set `VISIONIQ_REASONING_PROVIDER=openai_compatible`,
  `VISIONIQ_OPENAI_API_KEY`, `VISIONIQ_OPENAI_BASE_URL` (any OpenAI-compatible chat
  completions endpoint), and `VISIONIQ_OPENAI_MODEL`.

Both adapters fall back to the `template` engine on any request error, so misconfigured
LLM infrastructure degrades gracefully instead of breaking `/verify`.

### Seeding demo data

```bash
python scripts/seed_vector_store.py --model dummy --count 24
```

Populates the embedding map and vector-search panel before any real `/verify` calls
have been made. Run with `--model <name>` to seed a different backend's store.
