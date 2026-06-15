from datetime import datetime

from pydantic import BaseModel


class VerifyResponse(BaseModel):
    trace_id: str
    model: str
    similarity: float
    distance: float
    verdict: str
    spoof_risk: float
    anomalies: list[str]
    reasoning: str
    latency_ms: float
    heatmap_a: str | None = None
    heatmap_b: str | None = None
    created_at: datetime
