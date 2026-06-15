# ai-services

Installable Python package providing the AI engine behind VisionIQ: pluggable
embedding backends (CLIP, ViT, EfficientNet, a modernized Siamese network, and a
zero-dependency `dummy` backend for tests/dev), GradCAM/attention explainability,
FAISS-backed vector search, liveness heuristics, and a pluggable LLM reasoning layer.

## Install

```bash
pip install -e .[dev]
```

## Run tests

```bash
pytest
```

Tests rely only on the `dummy` embedding backend and synthetic tensors, so they run
without downloading any model weights.

## Layout

- `ai_services/embeddings/` — `EmbeddingModel` ABC + registry (`clip`, `vit`,
  `efficientnet`, `siamese`, `dummy`).
- `ai_services/siamese/` — modernized Siamese embedding network, losses
  (contrastive + triplet), dataset, and training CLI.
- `ai_services/explainability/` — GradCAM (CNN backbones) and attention-rollout
  (transformer backbones), plus heatmap-to-overlay helpers.
- `ai_services/vector_store/` — FAISS-backed vector store with 2D PCA projection for
  embedding-map visualizations.
- `ai_services/reasoning/` — LLM reasoning engine: offline template provider (default),
  Ollama adapter, OpenAI-compatible adapter, with automatic fallback to the template
  provider on any failure.
- `ai_services/liveness/` — OpenCV-based anti-spoofing heuristics.
- `ai_services/preprocessing/` — image decode/resize/face-crop utilities.
