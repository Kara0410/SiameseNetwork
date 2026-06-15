"""Shared context object passed from the verification service to a reasoning provider."""

from dataclasses import dataclass, field


@dataclass
class VerificationContext:
    """Everything a reasoning provider needs to explain a verification result."""

    model_name: str
    similarity: float  # 0..1, higher = more similar
    distance: float  # raw embedding distance
    verdict: str  # "verified" | "review" | "blocked"
    spoof_risk: float = 0.0  # 0..1, higher = more likely spoofed
    anomalies: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    neighbors: list[dict] = field(default_factory=list)
