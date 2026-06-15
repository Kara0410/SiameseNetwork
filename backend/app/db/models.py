"""Persistence models."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class InferenceRecord(SQLModel, table=True):
    """A single `/verify` call, persisted for the history and architecture endpoints."""

    id: int | None = Field(default=None, primary_key=True)
    trace_id: str = Field(index=True, unique=True)
    model_name: str
    similarity: float
    distance: float
    verdict: str
    spoof_risk: float
    anomalies: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    reasoning: str
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
