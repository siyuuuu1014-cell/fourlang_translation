from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import get_project_root, resolve_project_path


VALID_STATUSES = {
    "ready",
    "staged",
    "disabled",
}

VALID_ARCHITECTURES = {
    "marian",
    "m2m100",
    "pending",
    "small100",
}


@dataclass(frozen=True)
class ModelSpec:
    direction: str
    model_name: str
    architecture: str
    path: Path
    source_lang: str
    target_lang: str
    status: str
    generation: dict[str, Any]


class ModelRegistry:
    """
    Read-only runtime registry for frozen translation specialists.

    The registry file stores only project-relative paths. Runtime paths
    are resolved against the repository root, so the same code works on:

      Windows:
        D:\\dev\\projects\\fourlang_translation

      Linux server:
        /root/autodl-tmp/fourlang_translation
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        registry_path: str | Path | None = None,
    ):
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else get_project_root()
        )

        self.registry_path = (
            resolve_project_path(
                registry_path,
                self.project_root,
            )
            if registry_path is not None
            else (
                self.project_root
                / "models"
                / "model_registry.json"
            ).resolve()
        )

        if not self.registry_path.exists():
            raise FileNotFoundError(
                f"Model registry not found: {self.registry_path}"
            )

        payload = json.loads(
            self.registry_path.read_text(
                encoding="utf-8"
            )
        )

        models = payload.get("models")

        if not isinstance(models, dict):
            raise RuntimeError(
                "Invalid model registry: 'models' must be an object."
            )

        self.version = int(
            payload.get("version", 1)
        )

        self._models = models

        self._validate_schema()

    def _validate_schema(self) -> None:
        required = {
            "model_name",
            "architecture",
            "path",
            "source_lang",
            "target_lang",
            "status",
        }

        for direction, raw in self._models.items():
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"Registry entry {direction!r} must be an object."
                )

            missing = required - set(raw)

            if missing:
                raise RuntimeError(
                    f"Registry entry {direction!r} missing: "
                    f"{sorted(missing)}"
                )

            architecture = str(
                raw["architecture"]
            ).strip().lower()

            if architecture not in VALID_ARCHITECTURES:
                raise RuntimeError(
                    f"Unsupported architecture in registry: "
                    f"{direction} -> {architecture}"
                )

            status = str(
                raw["status"]
            ).strip().lower()

            if status not in VALID_STATUSES:
                raise RuntimeError(
                    f"Invalid registry status: "
                    f"{direction} -> {status}"
                )

    def directions(
        self,
        *,
        ready_only: bool = False,
    ) -> list[str]:
        directions = []

        for direction, raw in self._models.items():
            status = str(
                raw["status"]
            ).strip().lower()

            if ready_only:
                if status != "ready":
                    continue

                model_path = resolve_project_path(
                    raw["path"],
                    self.project_root,
                )

                if not model_path.exists():
                    continue

            directions.append(
                str(direction).strip().lower()
            )

        return sorted(directions)

    def get(
        self,
        direction: str,
    ) -> ModelSpec:
        direction = str(
            direction
        ).strip().lower()

        raw = self._models.get(direction)

        if raw is None:
            available = ", ".join(
                self.directions()
            ) or "(none)"

            raise KeyError(
                f"Unknown direction {direction!r}. "
                f"Registered: {available}"
            )

        return ModelSpec(
            direction=direction,
            model_name=str(
                raw["model_name"]
            ).strip(),
            architecture=str(
                raw["architecture"]
            ).strip().lower(),
            path=resolve_project_path(
                raw["path"],
                self.project_root,
            ),
            source_lang=str(
                raw["source_lang"]
            ).strip().lower(),
            target_lang=str(
                raw["target_lang"]
            ).strip().lower(),
            status=str(
                raw["status"]
            ).strip().lower(),
            generation=dict(
                raw.get(
                    "generation",
                    {}
                )
            ),
        )

    def require_ready(
        self,
        direction: str,
    ) -> ModelSpec:
        spec = self.get(direction)

        if spec.status != "ready":
            raise RuntimeError(
                f"Direction {direction!r} is not ready. "
                f"Current status: {spec.status!r}"
            )

        if not spec.path.exists():
            raise FileNotFoundError(
                "Registered model directory does not exist:\n"
                f"{spec.path}"
            )

        return spec

    def describe(self) -> list[dict[str, Any]]:
        rows = []

        for direction in self.directions():
            spec = self.get(direction)

            rows.append(
                {
                    "direction": spec.direction,
                    "model_name": spec.model_name,
                    "architecture": spec.architecture,
                    "status": spec.status,
                    "path": str(spec.path),
                    "path_exists": spec.path.exists(),
                    "source_lang": spec.source_lang,
                    "target_lang": spec.target_lang,
                }
            )

        return rows
