"""Static pipeline metadata plus a latency breakdown derived from recent inferences.

The pipeline stages mirror the six-stage story in the design mockup (capture -> vision
encoder -> embedding head -> vector memory -> reasoning -> serve/observe); the latency
split is an illustrative proportional breakdown of the measured end-to-end average.
"""

from sqlmodel import Session, select

from ..db.models import InferenceRecord

PIPELINE = [
    {
        "id": "ingest",
        "layer": "01",
        "title": "Capture",
        "metric": "preprocessing",
        "detail": (
            "Every image enters the same way, whether it's a drag-and-drop upload or a webcam "
            "snapshot: a multipart POST handled by FastAPI. Pillow decodes the bytes, EXIF "
            "metadata is stripped for privacy, and the frame is resized to a maximum dimension "
            "so every downstream model sees a consistent input size."
        ),
        "stack": ["FastAPI", "Pillow", "EXIF stripping", "resize_max_dim"],
    },
    {
        "id": "encoder",
        "layer": "02",
        "title": "Vision encoder",
        "metric": "pluggable backbone",
        "detail": (
            "The preprocessed image is run through a swappable PyTorch backbone - OpenAI's CLIP "
            "ViT-B/32, a plain ViT-B/16, EfficientNet-B0, or a custom Siamese ResNet18 - each "
            "loaded via Hugging Face Transformers or torchvision and producing a dense feature "
            "vector. In parallel, OpenCV-based heuristics check sharpness, glare, and texture for "
            "liveness/anti-spoof signals."
        ),
        "stack": ["PyTorch", "Transformers (CLIP / ViT)", "torchvision", "OpenCV liveness"],
    },
    {
        "id": "metric",
        "layer": "03",
        "title": "Embedding head",
        "metric": "64-768d, unit-normalized",
        "detail": (
            "Backbone features are projected by a small MLP head into a 64-768 dimensional "
            "embedding and L2-normalized onto the unit sphere. The Siamese variant is trained "
            "end-to-end with either contrastive loss - pulling matching pairs together and "
            "pushing non-matching pairs apart past a margin - or triplet loss, which keeps an "
            "anchor closer to a positive example than a negative one. At inference time, cosine "
            "similarity and Euclidean distance between the two embeddings drive the verdict."
        ),
        "stack": ["Siamese ResNet18", "Contrastive loss", "Triplet loss", "Cosine similarity"],
    },
    {
        "id": "retrieval",
        "layer": "04",
        "title": "Vector memory",
        "metric": "k-NN search",
        "detail": (
            "Every candidate embedding is written to a per-model FAISS flat index together with "
            "its verdict and metadata. A verification run searches that index for the nearest "
            "neighbors by cosine similarity, surfacing prior identities the new image most "
            "resembles. For visualization, a PCA projection collapses the high-dimensional "
            "vectors down to 2D so the embedding map can be plotted."
        ),
        "stack": ["FAISS flat index", "Cosine k-NN", "PCA projection", "JSON metadata"],
    },
    {
        "id": "reasoning",
        "layer": "05",
        "title": "Reasoning",
        "metric": "LLM + fallback",
        "detail": (
            "Similarity, distance, spoof risk, and the nearest neighbors are packaged into a "
            "verification context and handed to a language model - a local Ollama model or any "
            "OpenAI-compatible API - which writes a short, plain-language explanation of the "
            "decision. If no LLM is configured, a deterministic template engine produces the "
            "same structure directly from the numbers, so the pipeline degrades gracefully."
        ),
        "stack": ["Ollama", "OpenAI-compatible API", "Prompt templating", "Template fallback"],
    },
    {
        "id": "serve",
        "layer": "06",
        "title": "Serve + observe",
        "metric": "REST + WS",
        "detail": (
            "FastAPI exposes the REST surface (/verify, /search, /history, /architecture) plus a "
            "WebSocket endpoint for live webcam streaming. Every inference - embeddings, "
            "similarity, verdict, latency, and the generated explanation - is persisted via "
            "SQLModel to SQLite, and the whole stack (frontend, API, and optional local LLM) is "
            "packaged as Docker containers via docker-compose."
        ),
        "stack": ["FastAPI", "WebSockets", "SQLModel + SQLite", "Docker Compose"],
    },
]

DEPLOYMENT = [
    {
        "category": "Frontend",
        "items": ["Next.js 14", "React 18", "TypeScript", "Tailwind CSS", "Framer Motion"],
    },
    {
        "category": "Backend API",
        "items": ["FastAPI", "Uvicorn", "SQLModel", "SQLite", "WebSockets"],
    },
    {
        "category": "AI / ML",
        "items": ["PyTorch", "Transformers", "torchvision", "FAISS", "OpenCV", "NumPy"],
    },
    {
        "category": "Reasoning & ops",
        "items": ["Ollama", "OpenAI-compatible LLMs", "Docker", "Docker Compose"],
    },
]

# Proportional split of end-to-end latency across pipeline stages.
_LATENCY_SHARE = {
    "capture": 0.18,
    "encode": 0.32,
    "search": 0.08,
    "reason": 0.30,
    "api": 0.12,
}

_DEFAULT_AVG_LATENCY_MS = 73.0
_HISTORY_SAMPLE_SIZE = 20


def get_architecture(session: Session) -> dict:
    records = session.exec(
        select(InferenceRecord).order_by(InferenceRecord.created_at.desc()).limit(_HISTORY_SAMPLE_SIZE)
    ).all()

    avg_latency = (
        sum(record.latency_ms for record in records) / len(records) if records else _DEFAULT_AVG_LATENCY_MS
    )

    latency = [{"stage": stage, "ms": round(avg_latency * share, 1)} for stage, share in _LATENCY_SHARE.items()]

    return {"pipeline": PIPELINE, "latency": latency, "deployment": DEPLOYMENT}
