from fastapi import APIRouter

from . import architecture, history, live, models, search, verify

api_router = APIRouter()
api_router.include_router(verify.router, tags=["verify"])
api_router.include_router(search.router, tags=["search"])
api_router.include_router(history.router, tags=["history"])
api_router.include_router(models.router, tags=["models"])
api_router.include_router(architecture.router, tags=["architecture"])
api_router.include_router(live.router, tags=["live"])
