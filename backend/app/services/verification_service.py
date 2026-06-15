"""Orchestrates a single `/verify` call: embed both images, score their similarity,
run liveness checks, build explainability overlays, and produce an LLM explanation.
"""

import time
import uuid

import numpy as np
from PIL import Image
from sqlmodel import Session

from ai_services.embeddings import get_model
from ai_services.explainability import heatmap_to_overlay
from ai_services.liveness import assess_liveness
from ai_services.preprocessing import resize_max_dim
from ai_services.reasoning import VerificationContext

from ..db.models import InferenceRecord
from .reasoning_provider import get_reasoning_provider
from .vector_stores import get_vector_store

# Cosine-similarity thresholds for the verdict classifier.
VERIFIED_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.7


async def verify_images(image_a: Image.Image, image_b: Image.Image, model_name: str, session: Session) -> dict:
    start = time.perf_counter()

    model = get_model(model_name)
    image_a = resize_max_dim(image_a)
    image_b = resize_max_dim(image_b)

    vector_a = model.embed(image_a)
    vector_b = model.embed(image_b)

    similarity = float(np.dot(vector_a, vector_b))
    distance = float(np.linalg.norm(vector_a - vector_b))
    verdict = _classify(similarity)

    liveness_a = assess_liveness(image_a)
    liveness_b = assess_liveness(image_b)
    spoof_risk = max(liveness_a.risk, liveness_b.risk)
    anomalies = _merge_flags(liveness_a.flags, liveness_b.flags)

    heatmap_a = _build_heatmap(model, image_a)
    heatmap_b = _build_heatmap(model, image_b)

    vector_store = get_vector_store(model_name)
    neighbors = [
        {"id": match.id, "score": match.score, **match.metadata}
        for match in vector_store.search(vector_b, k=3)
    ]

    latency_ms = (time.perf_counter() - start) * 1000

    context = VerificationContext(
        model_name=model_name,
        similarity=similarity,
        distance=distance,
        verdict=verdict,
        spoof_risk=spoof_risk,
        anomalies=anomalies,
        latency_ms=latency_ms,
        neighbors=neighbors,
    )
    reasoning = await get_reasoning_provider().explain(context)

    trace_id = f"viq_{uuid.uuid4().hex[:8]}"

    record = InferenceRecord(
        trace_id=trace_id,
        model_name=model_name,
        similarity=similarity,
        distance=distance,
        verdict=verdict,
        spoof_risk=spoof_risk,
        anomalies=anomalies,
        reasoning=reasoning,
        latency_ms=latency_ms,
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    vector_store.add(trace_id, vector_b, {"label": trace_id, "verdict": verdict, "similarity": similarity})
    vector_store.persist()

    return {
        "trace_id": trace_id,
        "model": model_name,
        "similarity": similarity,
        "distance": distance,
        "verdict": verdict,
        "spoof_risk": spoof_risk,
        "anomalies": anomalies,
        "reasoning": reasoning,
        "latency_ms": latency_ms,
        "heatmap_a": heatmap_a,
        "heatmap_b": heatmap_b,
        "created_at": record.created_at,
    }


def _classify(similarity: float) -> str:
    if similarity >= VERIFIED_THRESHOLD:
        return "verified"
    if similarity >= REVIEW_THRESHOLD:
        return "review"
    return "blocked"


def _merge_flags(flags_a: list[str], flags_b: list[str]) -> list[str]:
    combined = sorted({*flags_a, *flags_b} - {"no spoof signal"})
    return combined or ["no spoof signal"]


def _build_heatmap(model, image: Image.Image) -> str | None:
    heatmap = model.explain(image)
    if heatmap is None:
        return None
    return heatmap_to_overlay(heatmap, image)
