from pydantic import BaseModel


class SearchMatch(BaseModel):
    id: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    model: str
    matches: list[SearchMatch]
    embedding_map: list[dict]
