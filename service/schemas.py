from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    source_lang: str = Field(..., min_length=2, max_length=8, examples=["zh"])
    target_lang: str = Field(..., min_length=2, max_length=8, examples=["uz"])
    text: str = Field(..., min_length=1, max_length=5000, examples=["今天天气怎么样"])


class TranslateResponse(BaseModel):
    direction: str
    source_language: str
    target_language: str
    model_name: str
    text: str
    translation: str
    latency_ms: float
    device: str


class ModelInfo(BaseModel):
    direction: str
    model_name: str
    model_path: str
    available: bool
    loaded: bool


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    directions: list[str]
