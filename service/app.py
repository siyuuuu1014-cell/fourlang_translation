from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .dependencies import get_pair_service
from .schemas import HealthResponse, ModelInfo, ModelsResponse, TranslateRequest, TranslateResponse

SERVICE_NAME = "FourLang Translation API"
API_VERSION = "0.2.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PACKAGE_TAR = PROJECT_ROOT / "onnx_export" / "m2m100_fourlang_mobile.tar"


def build_direction(source_lang: str, target_lang: str) -> str:
    source = str(source_lang).strip().lower().replace("_", "-").replace("->", "-")
    target = str(target_lang).strip().lower().replace("_", "-").replace("->", "-")
    if source == target:
        raise HTTPException(status_code=400, detail="source_lang and target_lang must be different.")
    return f"{source}-{target}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = get_pair_service()
    yield


app = FastAPI(
    title=SERVICE_NAME,
    version=API_VERSION,
    description="Unified HTTP inference service for the six FourLang pair specialists.",
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
def health(service=Depends(get_pair_service)):
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        directions=service.directions(),
    )


@app.get("/models", response_model=ModelsResponse)
def models(service=Depends(get_pair_service)):
    return ModelsResponse(
        models=[ModelInfo(**row) for row in service.describe()]
    )


@app.post("/translate", response_model=TranslateResponse)
def translate(request: TranslateRequest, service=Depends(get_pair_service)):
    try:
        direction = build_direction(request.source_lang, request.target_lang)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid direction")

    start = time.perf_counter()
    try:
        result = service.translate(direction, request.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    request_latency = (time.perf_counter() - start) * 1000

    return TranslateResponse(
        direction=result["direction"],
        source_language=result["source_language"],
        target_language=result["target_language"],
        model_name=result["model_name"],
        text=result["input"],
        translation=result["translation"],
        latency_ms=round(request_latency, 3),
        device=result["device"],
    )


@app.get("/download/model-package")
def download_model_package():
    if not MODEL_PACKAGE_TAR.exists():
        raise HTTPException(status_code=404, detail="model package not built yet")
    return FileResponse(
        str(MODEL_PACKAGE_TAR),
        filename=MODEL_PACKAGE_TAR.name,
        media_type="application/x-tar",
    )
