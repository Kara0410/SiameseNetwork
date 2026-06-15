from pydantic import BaseModel


class PipelineStage(BaseModel):
    id: str
    layer: str
    title: str
    metric: str
    detail: str
    stack: list[str]


class LatencyStage(BaseModel):
    stage: str
    ms: float


class DeploymentGroup(BaseModel):
    category: str
    items: list[str]


class ArchitectureResponse(BaseModel):
    pipeline: list[PipelineStage]
    latency: list[LatencyStage]
    deployment: list[DeploymentGroup]
