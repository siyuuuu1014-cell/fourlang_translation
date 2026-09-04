from __future__ import annotations

import tomllib
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        config = _expand_environment(tomllib.load(handle))
    config["_config_path"] = str(config_path)
    return config


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    missing = sorted(
        {name for name in _ENV_PATTERN.findall(value) if name not in os.environ}
    )
    if missing:
        raise RuntimeError(
            "Missing required environment variables in config: " + ", ".join(missing)
        )
    return os.path.expandvars(value)


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def positive_limit(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None
