from __future__ import annotations

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
COMMERCIAL_LICENSES = {"apache-2.0", "mit", "cc-by-4.0", "mixed_apache2_ccby4"}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return expand_environment(raw)


def expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise RuntimeError(f"Required environment variable is not set: {name}")
        return os.environ[name]

    return ENV_PATTERN.sub(replace, value)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pair_info(config: dict[str, Any]) -> tuple[str, str, str, str]:
    direction = config["direction"]
    return (
        str(direction["pair"]),
        str(direction["source_lang"]),
        str(direction["target_lang"]),
        str(direction["version"]),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_by_id(
    config: dict[str, Any], role: str, candidate_id: str
) -> dict[str, Any]:
    key = f"{role}_candidates"
    matches = [item for item in config[key] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one {role} candidate {candidate_id!r}.")
    return dict(matches[0])


def commercial_candidates(config: dict[str, Any], role: str) -> list[dict[str, Any]]:
    candidates = [dict(item) for item in config[f"{role}_candidates"]]
    if config["direction"].get("commercial_use", False):
        candidates = [
            item
            for item in candidates
            if item.get("commercial_allowed") is True
            and str(item.get("license", "")).lower() in COMMERCIAL_LICENSES
        ]
    if not candidates:
        raise RuntimeError(
            f"No commercially eligible {role} candidates are configured."
        )
    return candidates


def direction_key(source: str, target: str) -> str:
    return f"{source}-{target}"


def parquet_columns(frame: Any, source: str, target: str) -> tuple[str, str]:
    columns = set(frame.columns)
    if {source, target}.issubset(columns):
        return source, target
    if {"source_text", "target_text"}.issubset(columns):
        return "source_text", "target_text"
    raise ValueError(
        f"Benchmark must contain [{source}, {target}] or [source_text, target_text]; "
        f"found {sorted(columns)}"
    )
