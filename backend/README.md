# VisionIQ API

FastAPI backend for the VisionIQ Engine identity-intelligence platform. Wraps the
[`ai-services`](../ai-services) package with REST + WebSocket endpoints, request
history, and per-model vector search.

## Endpoints (`/api/v1`)

| Method | Path            | Description                                              |
| ------ | --------------- | --------------------------------------------------------- |
| POST   | `/verify`        | Compare two images: similarity, distance, verdict, liveness, heatmaps, LLM reasoning. |
| POST   | `/search`        | Embed an image and return nearest neighbors + a 2D embedding map. |
| GET    | `/history`       | Recent `/verify` calls.                                   |
| GET    | `/models`        | Available embedding backends and the active default.     |
| GET    | `/architecture`  | Pipeline metadata + measured latency breakdown.           |
| WS     | `/live`          | Streaming webcam verification (similarity + liveness per frame). |

`GET /healthz` is available outside the `/api/v1` prefix for container health checks.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

cp .env.example .env
uvicorn app.main:app --reload
```

## Configuration

All settings are environment variables prefixed `VISIONIQ_` (see `.env.example`):

- `VISIONIQ_EMBEDDING_DEFAULT`: `dummy` (offline/dev default) | `clip` | `vit` | `efficientnet` | `siamese`.
- `VISIONIQ_REASONING_PROVIDER`: `template` (offline default) | `ollama` | `openai_compatible`.
- `VISIONIQ_DATABASE_URL`, `VISIONIQ_VECTOR_STORE_DIR`: persistence locations.

Switching `VISIONIQ_EMBEDDING_DEFAULT` to a real backend downloads the corresponding
model weights on first use (see `../scripts/download_models.py`).

## Tests

```bash
pytest
```

Tests force `VISIONIQ_EMBEDDING_DEFAULT=dummy` and `VISIONIQ_REASONING_PROVIDER=template`
so they run instantly with no model downloads or network access.
