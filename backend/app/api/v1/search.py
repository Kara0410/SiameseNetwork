from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ai_services.embeddings.registry import available_models
from ai_services.preprocessing import load_image_from_bytes

from ...core.config import get_settings
from ...schemas.search import SearchResponse
from ...services.search_service import search_image

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    image: UploadFile = File(...),
    model: str | None = Query(default=None),
    k: int = Query(default=5, ge=1, le=20),
) -> dict:
    settings = get_settings()
    model_name = model or settings.embedding_default
    if model_name not in available_models():
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

    pil_image = load_image_from_bytes(await image.read())
    return search_image(pil_image, model_name, k=k)
