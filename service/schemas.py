from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class TranslateRequest(BaseModel):
    source_lang: str = Field(..., min_length=2, max_length=16, examples=["en"])
    target_lang: str = Field(..., min_length=2, max_length=16, examples=["zh"])
    text: str = Field(..., min_length=1, max_length=5000, examples=["Where is the nearest hospital?"])
    warmup: bool = False

class TranslateResponse(BaseModel):
    direction: str
    source_lang: str
    target_lang: str
    model: str
    architecture: str
    text: str
    translation: str
    generation_latency_seconds: float
    request_latency_seconds: float
    device: str

class ModelInfo(BaseModel):
    direction: str
    model_name: str
    architecture: str
    status: str
    source_lang: str
    target_lang: str
    path: str
    path_exists: bool
    runtime_available: bool

class ModelsResponse(BaseModel):
    models: list[ModelInfo]

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    registry_loaded: bool
    ready_directions: list[str]
