"""Lazy-loading inference service for the six bidirectional pair specialists.

Reads the authoritative mapping in ``configs/specialists/current_pair_models.json``
and caches one loaded model per pair, so switching between the two directions of a
pair reuses the same weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .engine import TranslationEngine, parse_direction
from .loader import load_translation_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs/specialists/current_pair_models.json"


def load_routes(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, dict[str, str]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    routes: dict[str, dict[str, str]] = {}
    for item in payload["pairs"]:
        for direction in item["directions"]:
            routes[direction] = {
                "model_name": item["model_name"],
                "path": item["model_path"],
            }
    return routes


class PairModelService:
    def __init__(
        self,
        manifest_path: Path = DEFAULT_MANIFEST,
        device: str = "auto",
        dtype: str = "float16",
        num_beams: int = 5,
        max_source_length: int = 256,
        max_new_tokens: int = 256,
    ) -> None:
        self.manifest_path = manifest_path
        self.routes = load_routes(manifest_path)
        self.device = device
        self.dtype = dtype
        self.num_beams = num_beams
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens
        self._cache: dict[str, tuple[str, TranslationEngine]] = {}

    def directions(self) -> list[str]:
        return sorted(self.routes)

    def describe(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for direction in self.directions():
            _, path = self._resolve(direction)
            rows.append(
                {
                    "direction": direction,
                    "model_name": self.routes[direction]["model_name"],
                    "model_path": str(path),
                    "available": path.is_dir(),
                    "loaded": str(path) in self._cache,
                }
            )
        return rows

    def _resolve(self, direction: str) -> tuple[dict[str, str], Path]:
        source, target = parse_direction(direction)
        key = f"{source}-{target}"
        route = self.routes.get(key)
        if route is None:
            raise KeyError(f"direction {key!r} is not configured")
        path = Path(route["path"]).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return route, path.resolve()

    def _engine_for(self, direction: str) -> tuple[str, TranslationEngine]:
        route, path = self._resolve(direction)
        key = str(path)
        entry = self._cache.get(key)
        if entry is None:
            loaded = load_translation_model(
                path,
                device=self.device,
                dtype=self.dtype,
            )
            engine = TranslationEngine(
                loaded,
                direction=direction,
                max_source_length=self.max_source_length,
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
            entry = (route["model_name"], engine)
            self._cache[key] = entry
        return entry

    def translate(self, direction: str, text: str) -> dict[str, Any]:
        model_name, engine = self._engine_for(direction)
        result = engine.translate(text, direction=direction)
        result["model_name"] = model_name
        return result
