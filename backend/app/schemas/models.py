from pydantic import BaseModel


class ModelInfo(BaseModel):
    name: str
    display_name: str
    dimension: int
    description: str
    explainability: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    default: str
