# VisionIQ — Multimodal AI Identity Intelligence Platform

VisionIQ is an end-to-end identity verification platform: upload or stream two faces,
get a similarity score, a verdict, liveness/spoof signals, GradCAM/attention
explainability heatmaps, nearest-neighbor vector search, and an LLM-generated reasoning
trace — all from a dark, dashboard-first UI.

## About this project

The goal behind VisionIQ was to build a genuinely deep, practical understanding of
**how computer vision models process images and how to train them** — not just call a
pretrained API, but work through the full pipeline from raw pixels to a decision.

That meant going through every layer of the stack:

- **Image preprocessing** — decoding, EXIF handling, resizing, and face cropping
  before anything touches a model.
- **Representation learning** — training a Siamese network from scratch to turn a
  face image into an embedding vector, and comparing **binary cross-entropy**,
  **contrastive loss**, and **triplet loss** as training objectives (with ReLU vs.
  SeLU activations) to see how each shapes the embedding space.
- **Modern pretrained backbones** — integrating CLIP, ViT, and EfficientNet alongside
  the custom-trained Siamese/ResNet18 model so the same pipeline can run on multiple
  vision encoders and the results can be compared directly.
- **Explainability** — using Grad-CAM (for CNN backbones) and attention-rollout (for
  transformer backbones) to visualize *what* a model focused on when producing an
  embedding, instead of treating it as a black box.
- **Vector search** — storing every embedding in a FAISS index and projecting it to
  2D, to see how faces cluster in the learned representation space.
- **Putting it all together** — wrapping preprocessing, the embedding models,
  explainability, vector search, liveness checks, and an LLM reasoning layer into a
  real FastAPI backend and Next.js dashboard, so the whole pipeline is observable and
  usable end-to-end rather than living in a notebook.

The project grew out of [`legacy/`](legacy/), an earlier research project focused
purely on training Siamese networks for face verification and comparing loss
functions/activations. Those findings (triplet loss + ResNet18 as the strongest
combination) became the default configuration for the `siamese` embedding backend in
the current platform — see [Research lineage](#research-lineage) below.

## Features

- **Identity verification** — upload two images or stream from a webcam; get a
  similarity score, `verified`/`review`/`blocked` verdict, and spoof-risk flags.
- **Pluggable vision encoders** — `dummy` (offline, zero-download default), `clip`,
  `vit`, `efficientnet`, and `siamese` (the modernized ResNet18 successor to the
  original research), selectable per request via `?model=`.
- **Explainable AI** — Grad-CAM (CNN backbones) or attention-rollout (ViT/CLIP) overlays
  rendered as heatmaps for both images.
- **Vector search** — FAISS-backed nearest-neighbor search over every embedding ever
  computed, plus a 2D embedding map for the dashboard.
- **LLM reasoning** — a natural-language explanation of the verdict, from an offline
  template engine by default, or Ollama / any OpenAI-compatible endpoint.
- **Real-time webcam verification** — WebSocket streaming with a reference-frame /
  live-comparison flow.
- **AI dashboard + architecture view** — telemetry sparklines, latency breakdown, and an
  animated 6-stage pipeline view, all driven by live backend data.

## Architecture

```
frontend/      Next.js 14 (App Router) + TypeScript + Tailwind + shadcn/ui
backend/       FastAPI app (REST + WebSocket), SQLModel, services layer
ai-services/   installable Python package: embeddings, explainability,
                vector store, reasoning, liveness, preprocessing, siamese
legacy/        original Siamese-network research that informed ai_services.siamese
docker/        docker-compose + env templates
docs/          architecture, API, and deployment docs
scripts/       model pre-download and vector-store seeding
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full data-flow diagram and
pipeline breakdown, and [`docs/API.md`](docs/API.md) for the REST/WebSocket reference.

## Quickstart

### Docker Compose (full stack, one command)

```bash
cd docker
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend: http://localhost:8000

Everything runs with offline defaults (`dummy` embeddings, `template` reasoning) — no
API keys, GPU, or model downloads required.

### Local development

```bash
# ai-services + backend
cd ai-services && pip install -e ".[dev]" && cd ..
cd backend && pip install -r requirements-dev.txt && cd ..
cd backend && cp .env.example .env && uvicorn app.main:app --reload

# frontend (separate terminal)
cd frontend && npm install && cp .env.example .env.local && npm run dev
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Vercel/Render/Fly/Railway deployment
and how to switch to real embedding models (`clip`/`vit`/`efficientnet`/`siamese`) and
LLM reasoning providers (`ollama`/`openai_compatible`).

## Training the Siamese model

The `siamese` embedding backend (`ai_services.siamese`) is a ResNet18 trunk with an
MLP projection head, trained with triplet-margin loss to produce L2-normalized
embeddings. Given directories of anchor/positive/negative face crops, it can be
fine-tuned with:

```bash
cd ai-services
python -m ai_services.siamese.train \
    --anchor data/anchor --positive data/positive --negative data/negative \
    --epochs 5 --out models/siamese.pth
```

The resulting checkpoint is picked up by the embedding registry via the
`VISIONIQ_SIAMESE_CHECKPOINT` environment variable. Contrastive loss
(`ai_services.siamese.losses.ContrastiveLoss`) is also available for experimenting
with pairwise (rather than triplet) training.

## Testing

```bash
cd ai-services && pytest
cd backend && pytest
cd frontend && npm run build
```

## Research lineage

The `siamese` embedding backend and its training pipeline (`ai_services.siamese`) are a
direct continuation of the original face-verification research in
[`legacy/`](legacy/) — see that directory's README for the experiments (Binary
Cross-Entropy vs. Contrastive vs. Triplet loss, ReLU vs. SeLU) that shaped the current
ResNet18 + triplet-loss default.

## Author

Boran Cihan Polat
