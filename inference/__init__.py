from __future__ import annotations

__all__ = [
    "TranslatorEngine",
    "ModelRegistry",
    "ModelSpec",
]


def __getattr__(name: str):
    if name == "TranslatorEngine":
        from .engine import TranslatorEngine
        return TranslatorEngine

    if name in {
        "ModelRegistry",
        "ModelSpec",
    }:
        from .registry import (
            ModelRegistry,
            ModelSpec,
        )

        return {
            "ModelRegistry": ModelRegistry,
            "ModelSpec": ModelSpec,
        }[name]

    raise AttributeError(name)
