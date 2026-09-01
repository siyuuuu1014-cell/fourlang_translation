from __future__ import annotations
import time
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference.engine import TranslatorEngine
from inference.registry import ModelRegistry
from .dependencies import get_translator_engine
from .schemas import HealthResponse, ModelInfo, ModelsResponse, TranslateRequest, TranslateResponse

SERVICE_NAME = "FourLang Translation API"
API_VERSION = "0.1.0"

def normalize_lang(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")

def build_direction(source_lang: str, target_lang: str) -> str:
    source = normalize_lang(source_lang)
    target = normalize_lang(target_lang)
    if source == target:
        raise HTTPException(status_code=400, detail="source_lang and target_lang must be different.")
    return f"{source}_{target}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.registry = ModelRegistry()
    yield

app = FastAPI(
    title=SERVICE_NAME,
    version=API_VERSION,
    description="Unified HTTP inference service for FourLang specialist translation models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health():
    registry = ModelRegistry()
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        registry_loaded=True,
        ready_directions=registry.directions(ready_only=True),
    )

@app.get("/models", response_model=ModelsResponse)
def models():
    registry = ModelRegistry()
    ready_runtime = set(registry.directions(ready_only=True))
    rows = []
    for row in registry.describe():
        rows.append(
            ModelInfo(
                direction=row["direction"],
                model_name=row["model_name"],
                architecture=row["architecture"],
                status=row["status"],
                source_lang=row["source_lang"],
                target_lang=row["target_lang"],
                path=row["path"],
                path_exists=bool(row["path_exists"]),
                runtime_available=row["direction"] in ready_runtime,
            )
        )
    return ModelsResponse(models=rows)

@app.post("/translate", response_model=TranslateResponse)
def translate(
    request: TranslateRequest,
    engine: TranslatorEngine = Depends(get_translator_engine),
):
    direction = build_direction(request.source_lang, request.target_lang)
    ready = set(engine.available_directions(ready_only=True))
    if direction not in ready:
        raise HTTPException(
            status_code=404,
            detail=f"Translation direction {direction!r} is not available. Ready directions: {sorted(ready)}",
        )

    start = time.perf_counter()
    try:
        if request.warmup:
            engine.warmup(direction)
        result = engine.translate(direction=direction, text=request.text)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    request_latency = time.perf_counter() - start
    return TranslateResponse(
        direction=result["direction"],
        source_lang=result["source_lang"],
        target_lang=result["target_lang"],
        model=result["model"],
        architecture=result["architecture"],
        text=result["text"],
        translation=result["translation"],
        generation_latency_seconds=float(result["generation_latency_seconds"]),
        request_latency_seconds=float(request_latency),
        device=result["device"],
    )
