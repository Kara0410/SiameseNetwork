from fastapi import APIRouter, Depends
from sqlmodel import Session

from ...db.session import get_session
from ...schemas.architecture import ArchitectureResponse
from ...services.architecture_service import get_architecture

router = APIRouter()


@router.get("/architecture", response_model=ArchitectureResponse)
def architecture(session: Session = Depends(get_session)) -> dict:
    return get_architecture(session)
