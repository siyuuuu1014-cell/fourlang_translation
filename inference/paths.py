from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """
    Resolve the repository root from this package location.

    Expected layout:
        <project_root>/
            inference/
                paths.py
            models/
            scripts/

    This deliberately avoids hard-coded Windows or Linux absolute paths.
    """
    return Path(__file__).resolve().parents[1]


def resolve_project_path(
    value: str | Path,
    project_root: str | Path | None = None,
) -> Path:
    """
    Resolve a project-relative path against the repository root.

    Absolute paths are accepted for explicit overrides, but production
    configuration should normally use project-relative paths.
    """
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else get_project_root()
    )

    path = Path(value)

    if not path.is_absolute():
        path = root / path

    return path.resolve()
