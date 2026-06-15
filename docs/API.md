# API Reference

Base URL: `http://localhost:8000` (configurable via `NEXT_PUBLIC_API_BASE_URL` on the
frontend). All `/api/v1/*` responses are JSON; `GET /healthz` is outside the versioned
prefix for container health checks.

## `GET /healthz`

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}
```

## `POST /api/v1/verify`

Compare two images and return a similarity score, verdict, liveness/spoof signals,
explainability heatmaps, vector neighbors, and an LLM reasoning trace. Persists an
`InferenceRecord` and adds the candidate (`image_b`) embedding to the vector store.

**Multipart form fields:** `image_a`, `image_b` (image files).
**Query params:** `model` (optional, defaults to `VISIONIQ_EMBEDDING_DEFAULT`).

```bash
curl -X POST "http://localhost:8000/api/v1/verify?model=dummy" \
  -F "image_a=@reference.jpg" \
  -F "image_b=@candidate.jpg"
```

```json
{
  "trace_id": "viq_7fd21a3c",
  "model": "dummy",
  "similarity": 0.973,
  "distance": 0.232,
  "verdict": "verified",
  "spoof_risk": 0.0,
  "anomalies": ["no spoof signal"],
  "reasoning": "The DUMMY encoder detected high embedding similarity (97.3%, distance 0.232) ...",
  "latency_ms": 8.8,
  "heatmap_a": "data:image/png;base64,...",
  "heatmap_b": "data:image/png;base64,...",
  "created_at": "2026-06-15T12:30:15.789136"
}
```

`verdict` is `verified` (similarity ≥ 0.85), `review` (≥ 0.7), or `blocked`.
`heatmap_a`/`heatmap_b` are `null` for backends without an explainability hook (none
currently - `dummy` uses a heuristic overlay).

## `POST /api/v1/search`

Embed an image, return its nearest neighbors from the model's vector store, and a 2D
projection of the whole store for the embedding map.

**Multipart form fields:** `image`.
**Query params:** `model` (optional), `k` (optional, default 5).

```bash
curl -X POST "http://localhost:8000/api/v1/search?model=dummy&k=5" \
  -F "image=@candidate.jpg"
```

```json
{
  "model": "dummy",
  "matches": [
    { "id": "viq_7fd21a3c", "score": 1.0, "metadata": { "label": "viq_7fd21a3c", "verdict": "verified", "similarity": 0.973 } }
  ],
  "embedding_map": [
    { "id": "viq_7fd21a3c", "x": 0.0, "y": 0.0, "label": "viq_7fd21a3c", "verdict": "verified", "similarity": 0.973 }
  ]
}
```

## `GET /api/v1/history`

Most recent `/verify` calls, newest first.

**Query params:** `limit` (optional, 1-100, default 20).

```bash
curl "http://localhost:8000/api/v1/history?limit=10"
```

```json
{
  "items": [
    {
      "trace_id": "viq_7fd21a3c",
      "model": "dummy",
      "similarity": 0.973,
      "distance": 0.232,
      "verdict": "verified",
      "spoof_risk": 0.0,
      "anomalies": ["no spoof signal"],
      "reasoning": "...",
      "latency_ms": 8.8,
      "created_at": "2026-06-15T12:30:15.789136"
    }
  ]
}
```

## `GET /api/v1/models`

Available embedding backends and the configured default.

```bash
curl http://localhost:8000/api/v1/models
```

```json
{
  "models": [
    { "name": "dummy", "display_name": "Dummy (offline dev)", "dimension": 64, "description": "...", "explainability": "heuristic" },
    { "name": "clip", "display_name": "CLIP ViT-B/32", "dimension": 512, "description": "...", "explainability": "attention" }
  ],
  "default": "dummy"
}
```

## `GET /api/v1/architecture`

Static six-stage pipeline metadata plus a latency breakdown derived from the average of
the last 20 `InferenceRecord`s (falling back to a plausible default before any history
exists).

```bash
curl http://localhost:8000/api/v1/architecture
```

```json
{
  "pipeline": [
    { "id": "ingest", "layer": "01", "title": "Capture", "metric": "quality gate", "detail": "...", "stack": ["FastAPI", "Pillow", "signed upload"] }
  ],
  "latency": [
    { "stage": "capture", "ms": 13.1 },
    { "stage": "encode", "ms": 23.4 },
    { "stage": "search", "ms": 5.8 },
    { "stage": "reason", "ms": 21.9 },
    { "stage": "api", "ms": 8.8 }
  ],
  "deployment": ["UI", "API", "Encoder", "Vector DB", "LLM", "Monitor"]
}
```

## `WS /api/v1/live`

Streaming webcam verification. The first message (or any message with `"reset": true`)
sets the reference embedding; subsequent frames are compared against it.

**Client → server** (JSON per frame):

```json
{ "image": "data:image/jpeg;base64,...", "reset": false }
```

**Server → client:**

```json
{ "model": "dummy", "similarity": 0.97, "spoof_risk": 0.0, "flags": ["no spoof signal"], "latency_ms": 6.2 }
```

or, if `image` is missing:

```json
{ "error": "missing 'image' field" }
```

Example with `websocat`:

```bash
websocat ws://localhost:8000/api/v1/live
{"image": "data:image/jpeg;base64,...", "reset": true}
```
