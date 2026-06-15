"""Shared prompt construction for LLM-backed reasoning providers."""

from .schemas import VerificationContext

SYSTEM_PROMPT = "You are an AI verification analyst writing concise explanations for a security dashboard."


def build_prompt(context: VerificationContext) -> str:
    anomalies = ", ".join(context.anomalies) or "none"
    neighbor = ""
    if context.neighbors:
        top = context.neighbors[0]
        label = top.get("label") or top.get("id", "unknown")
        neighbor = f" Nearest vector-store neighbor: '{label}' (similarity {top.get('score', 0):.3f})."

    return (
        "Explain this identity-verification result in 2-3 concise sentences for a security "
        f"dashboard. Embedding model: {context.model_name}. Similarity: {context.similarity:.3f}. "
        f"Distance: {context.distance:.3f}. Verdict: {context.verdict}. Spoof risk: "
        f"{context.spoof_risk:.3f}. Anomalies: {anomalies}.{neighbor} Mention embedding "
        "consistency and any lighting/pose factors, and end with a confidence statement."
    )
