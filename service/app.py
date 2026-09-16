from __future__ import annotations
import json
import tarfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .dependencies import get_pair_service
from .schemas import HealthResponse, ModelInfo, ModelsResponse, TranslateRequest, TranslateResponse

SERVICE_NAME = "FourLang Translation API"
API_VERSION = "0.3.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PACKAGES_DIR = PROJECT_ROOT / "onnx_export"


def model_directions() -> dict[str, list[str]]:
    """Map model_id -> covered directions (pair specialists + the four-language model)."""
    directions: dict[str, list[str]] = {}
    manifest = PROJECT_ROOT / "configs/specialists/current_pair_models.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for item in payload["pairs"]:
            directions[item["id"]] = sorted(item["directions"])
    directions["m2m100_fourlang"] = sorted(
        f"{s}-{t}"
        for s in ("zh", "en", "ru", "uz")
        for t in ("zh", "en", "ru", "uz")
        if s != t
    )
    return directions


def list_model_manifest() -> list[dict]:
    """Scan onnx_export/*_mobile/MANIFEST.json and return a versioned model list."""
    rows: list[dict] = []
    if not MODEL_PACKAGES_DIR.is_dir():
        return rows
    dir_map = model_directions()
    for child in sorted(MODEL_PACKAGES_DIR.iterdir()):
        if not child.is_dir() or not child.name.endswith("_mobile"):
            continue
        manifest_path = child / "MANIFEST.json"
        if not manifest_path.is_file():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        model_id = child.name[: -len("_mobile")]
        rows.append(
            {
                "model_id": model_id,
                "family": m.get("family"),
                "version": m.get("version"),
                "source_sha256": m.get("source_sha256"),
                "size_bytes": sum(f["size_bytes"] for f in m.get("files", {}).values()),
                "directions": dir_map.get(model_id, []),
                "download_url": f"/download/{model_id}",
            }
        )
    return rows


def ensure_tar(model_id: str) -> Path:
    """Return the tar path for a package, creating it on first use."""
    pkg_dir = MODEL_PACKAGES_DIR / f"{model_id}_mobile"
    if not pkg_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"model package {model_id!r} not found")
    tar_path = MODEL_PACKAGES_DIR / f"{model_id}_mobile.tar"
    if not tar_path.exists():
        with tarfile.open(tar_path, "w") as tf:
            tf.add(pkg_dir, arcname=pkg_dir.name)
    return tar_path


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


@app.get("/models/manifest")
def model_manifest():
    return {"models": list_model_manifest()}


@app.get("/download/{model_id}")
def download_model_package(model_id: str):
    tar_path = ensure_tar(model_id)
    return FileResponse(
        str(tar_path),
        filename=tar_path.name,
        media_type="application/x-tar",
    )
