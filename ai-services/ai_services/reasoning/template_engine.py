"""Offline, deterministic reasoning provider.

This is the default reasoning backend: it requires no network access or external
service, and every other provider falls back to it on failure, so the API never
breaks due to missing LLM infrastructure.
"""

from .base import ReasoningProvider
from .schemas import VerificationContext


class TemplateReasoningProvider(ReasoningProvider):
    name = "template"

    async def explain(self, context: VerificationContext) -> str:
        similarity_pct = context.similarity * 100
        sentences = [_verdict_sentence(context, similarity_pct)]

        if context.anomalies and context.anomalies != ["no spoof signal"]:
            sentences.append("Detected signals: " + ", ".join(context.anomalies) + ".")

        sentences.append(_spoof_sentence(context))

        if context.neighbors:
            top = context.neighbors[0]
            label = top.get("label") or top.get("id", "unknown")
            sentences.append(
                f"Nearest vector-store neighbor is '{label}' at similarity {top.get('score', 0):.3f}."
            )

        return " ".join(sentences)


def _verdict_sentence(context: VerificationContext, similarity_pct: float) -> str:
    encoder = context.model_name.upper()
    if context.verdict == "verified":
        return (
            f"The {encoder} encoder detected high embedding similarity ({similarity_pct:.1f}%, "
            f"distance {context.distance:.3f}), consistent with the same identity. Embedding "
            "consistency remained stable across key feature regions."
        )
    if context.verdict == "review":
        return (
            f"The {encoder} encoder found moderate similarity ({similarity_pct:.1f}%, "
            f"distance {context.distance:.3f}) - close enough to flag for manual review "
            "rather than auto-approve."
        )
    return (
        f"The {encoder} encoder measured low similarity ({similarity_pct:.1f}%, "
        f"distance {context.distance:.3f}), indicating these are likely different identities."
    )


def _spoof_sentence(context: VerificationContext) -> str:
    if context.spoof_risk > 0.5:
        return (
            f"Liveness heuristics flag an elevated spoof risk ({context.spoof_risk:.2f}); this "
            "result should not be trusted without a step-up check."
        )
    if context.spoof_risk > 0.2:
        return (
            f"Liveness heuristics show a mild spoof risk ({context.spoof_risk:.2f}), likely "
            "from lighting or pose variance rather than a presentation attack."
        )
    return "No spoof signal was detected by the liveness heuristics."
