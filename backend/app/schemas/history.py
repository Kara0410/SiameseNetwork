from datetime import datetime

from pydantic import BaseModel


class HistoryItem(BaseModel):
    trace_id: str
    model: str
    similarity: float
    distance: float
    verdict: str
    spoof_risk: float
    anomalies: list[str]
    reasoning: str
    latency_ms: float
    created_at: datetime


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
