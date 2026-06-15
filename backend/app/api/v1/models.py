from dataclasses import asdict

from fastapi import APIRouter

from ai_services.embeddings import list_models

from ...core.config import get_settings
from ...schemas.models import ModelInfo, ModelsResponse

router = APIRouter()


@router.get("/models", response_model=ModelsResponse)
def models() -> ModelsResponse:
    settings = get_settings()
    return ModelsResponse(
        models=[ModelInfo(**asdict(info)) for info in list_models()],
        default=settings.embedding_default,
    )
