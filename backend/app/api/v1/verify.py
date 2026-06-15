from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlmodel import Session

from ai_services.embeddings.registry import available_models
from ai_services.preprocessing import load_image_from_bytes

from ...core.config import get_settings
from ...db.session import get_session
from ...schemas.verify import VerifyResponse
from ...services.verification_service import verify_images

router = APIRouter()


@router.post("/verify", response_model=VerifyResponse)
async def verify(
    image_a: UploadFile = File(...),
    image_b: UploadFile = File(...),
    model: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> dict:
    settings = get_settings()
    model_name = model or settings.embedding_default
    if model_name not in available_models():
        raise HTTPException(status_code=400, detail=f"Unknown model '{model_name}'")

    pil_a = load_image_from_bytes(await image_a.read())
    pil_b = load_image_from_bytes(await image_b.read())

    return await verify_images(pil_a, pil_b, model_name, session)
