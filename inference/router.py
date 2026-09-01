"""Automatic direction-to-model discovery, routing, and lazy model caching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .engine import SUPPORTED_DIRECTIONS, TranslationEngine, parse_direction
from .loader import LoadedModel, PROJECT_ROOT, load_translation_model


@dataclass(frozen=True)
class ModelRoute:
    direction: str
    model_name: str
    model_path: Path | str
    specialized: bool
    fallback: bool = False

    def as_dict(self) -> dict[str, Any]:
        path = str(self.model_path)
        return {
            "direction": self.direction,
            "model_name": self.model_name,
            "model_path": path,
            "specialized": self.specialized,
            "fallback": self.fallback,
            "available": Path(path).exists(),
        }


def _first_existing(candidates: list[Path | None]) -> Path | None:
    return next((path for path in candidates if path and path.is_dir()), None)


class ModelRouter:
    """Resolve each direction and lazily cache models used in one process."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        base_model_path: str | Path | None = None,
        specialists_root: str | Path | None = None,
        adapter_path: str | Path | None = None,
        device: str = "auto",
        dtype: str = "auto",
        project_root: str | Path = PROJECT_ROOT,
        load_fn: Callable[..., LoadedModel] = load_translation_model,
    ) -> None:
        if adapter_path and not model_path:
            raise ValueError("--adapter-path requires --model-path in single-model mode")

        self.project_root = Path(project_root).resolve()
        self.model_path = str(model_path) if model_path else None
        self.base_model_path = str(base_model_path) if base_model_path else None
        self.specialists_root = (
            Path(specialists_root).expanduser().resolve()
            if specialists_root
            else self.project_root / "models" / "final_specialists"
        )
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.device = device
        self.dtype = dtype
        self.load_fn = load_fn
        self._cache: dict[str, LoadedModel] = {}
        self._routes = self._discover_routes()

    @property
    def mode(self) -> str:
        return "single" if self.model_path else "auto"

    @property
    def loaded_model_count(self) -> int:
        return len(self._cache)

    def _discover_base_model(self) -> Path:
        env_path = os.environ.get("FOURLANG_BASE_MODEL_PATH")
        candidates = [
            Path(self.base_model_path).expanduser() if self.base_model_path else None,
            Path(env_path).expanduser() if env_path else None,
            self.project_root / "models" / "small100",
            self.project_root.parent / "models" / "small100",
        ]
        selected = _first_existing(candidates)
        if selected:
            return selected.resolve()
        checked = ", ".join(str(path) for path in candidates if path)
        raise FileNotFoundError(
            "No SMaLL-100 base model found. Checked: " + checked
        )

    def _specialist_candidates(
        self, direction: str
    ) -> tuple[str, list[Path]] | None:
        env_name = "FOURLANG_" + direction.upper().replace("-", "_") + "_MODEL_PATH"
        env_path = os.environ.get(env_name)
        explicit = [Path(env_path).expanduser()] if env_path else []
        if direction in {"en-uz", "uz-en"}:
            return (
                "en_uz_small100_v1",
                explicit + [self.specialists_root / "en_uz_small100_v1"],
            )
        if direction == "en-zh":
            return (
                "en_zh_v1",
                explicit + [self.specialists_root / "en_zh_v1"],
            )
        if direction == "zh-en":
            return (
                "zh_en_exp2_kd_v1",
                explicit
                + [
                    self.specialists_root / "zh_en_v1",
                    self.project_root
                    / "results"
                    / "specialists"
                    / "zh_en"
                    / "opus_mt_zh_en"
                    / "exp2_kd_v1"
                    / "best_model",
                ],
            )
        return None

    def _discover_routes(self) -> dict[str, ModelRoute]:
        if self.model_path:
            return {
                direction: ModelRoute(
                    direction=direction,
                    model_name="explicit_model",
                    model_path=self.model_path,
                    specialized=False,
                )
                for direction in SUPPORTED_DIRECTIONS
            }

        base_path = self._discover_base_model()
        routes: dict[str, ModelRoute] = {}
        for direction in SUPPORTED_DIRECTIONS:
            specialist = self._specialist_candidates(direction)
            if specialist:
                name, candidates = specialist
                selected = _first_existing(candidates)
                if selected:
                    routes[direction] = ModelRoute(
                        direction=direction,
                        model_name=name,
                        model_path=selected.resolve(),
                        specialized=True,
                    )
                    continue
            routes[direction] = ModelRoute(
                direction=direction,
                model_name="small100_base",
                model_path=base_path,
                specialized=False,
                fallback=specialist is not None,
            )
        return routes

    def route_for(self, direction: str) -> ModelRoute:
        source, target = parse_direction(direction)
        return self._routes[f"{source}-{target}"]

    def route_manifest(self) -> list[dict[str, Any]]:
        return [self._routes[direction].as_dict() for direction in SUPPORTED_DIRECTIONS]

    def load_for(self, direction: str) -> tuple[LoadedModel, ModelRoute]:
        route = self.route_for(direction)
        cache_key = f"{route.model_path}|{self.adapter_path or ''}"
        if cache_key not in self._cache:
            self._cache[cache_key] = self.load_fn(
                route.model_path,
                adapter_path=self.adapter_path,
                device=self.device,
                dtype=self.dtype,
            )
        return self._cache[cache_key], route


@dataclass
class RoutedTranslationEngine:
    router: ModelRouter
    direction: str = "zh-en"
    max_source_length: int = 128
    max_new_tokens: int = 128
    num_beams: int = 1

    def __post_init__(self) -> None:
        self.set_direction(self.direction)
        if self.max_source_length <= 0 or self.max_new_tokens <= 0:
            raise ValueError("token limits must be positive")
        if self.num_beams <= 0:
            raise ValueError("num_beams must be positive")

    def set_direction(self, direction: str) -> str:
        source, target = parse_direction(direction)
        self.direction = f"{source}-{target}"
        return self.direction

    def current_route(self) -> dict[str, Any]:
        return self.router.route_for(self.direction).as_dict()

    def translate(self, text: str, direction: str | None = None) -> dict[str, Any]:
        if direction is not None:
            self.set_direction(direction)
        loaded, route = self.router.load_for(self.direction)
        engine = TranslationEngine(
            loaded=loaded,
            direction=self.direction,
            max_source_length=self.max_source_length,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        result = engine.translate(text)
        result.update(
            {
                "model_name": route.model_name,
                "routing_mode": self.router.mode,
                "specialized_model": route.specialized,
                "loaded_model_count": self.router.loaded_model_count,
            }
        )
        return result
