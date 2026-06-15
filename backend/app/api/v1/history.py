from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ...db.session import get_session
from ...db.models import InferenceRecord
from ...schemas.history import HistoryItem, HistoryResponse

router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def history(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
) -> HistoryResponse:
    records = session.exec(
        select(InferenceRecord).order_by(InferenceRecord.created_at.desc()).limit(limit)
    ).all()

    return HistoryResponse(
        items=[
            HistoryItem(
                trace_id=record.trace_id,
                model=record.model_name,
                similarity=record.similarity,
                distance=record.distance,
                verdict=record.verdict,
                spoof_risk=record.spoof_risk,
                anomalies=record.anomalies,
                reasoning=record.reasoning,
                latency_ms=record.latency_ms,
                created_at=record.created_at,
            )
            for record in records
        ]
    )
