from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .loaders import (
    LoadedTranslationModel,
    load_translation_model,
)
from .registry import (
    ModelRegistry,
    ModelSpec,
)


class TranslatorEngine:
    """
    Unified lazy-loading translation engine.

    - Registry-driven
    - Project-relative paths
    - Architecture-aware
    - Lazy model loading
    - In-process model cache
    - CUDA/CPU selection
    - Deterministic generation
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        registry_path: str | Path | None = None,
        device: str | None = None,
    ):
        self.registry = ModelRegistry(
            project_root=project_root,
            registry_path=registry_path,
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else
                "cpu"
            )

        self.device = torch.device(
            device
        )

        if (
            self.device.type == "cuda"
            and
            not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA was requested, but CUDA is unavailable."
            )

        self._cache: dict[
            tuple[str, str],
            LoadedTranslationModel,
        ] = {}

    def available_directions(
        self,
        *,
        ready_only: bool = True,
    ) -> list[str]:
        return self.registry.directions(
            ready_only=ready_only
        )

    def registry_info(
        self,
    ) -> list[dict[str, Any]]:
        return self.registry.describe()

    def _cache_key(
        self,
        spec: ModelSpec,
    ) -> tuple[str, str]:
        return (
            spec.architecture,
            str(
                spec.path
            ),
        )

    def _get_loaded(
        self,
        spec: ModelSpec,
    ) -> LoadedTranslationModel:
        key = self._cache_key(
            spec
        )

        cached = self._cache.get(
            key
        )

        if cached is not None:
            return cached

        print(
            f"[FourLang] loading {spec.direction}: "
            f"{spec.model_name}"
        )

        print(
            f"[FourLang] model path: "
            f"{spec.path}"
        )

        print(
            f"[FourLang] architecture: "
            f"{spec.architecture}"
        )

        print(
            f"[FourLang] device: "
            f"{self.device}"
        )

        loaded = load_translation_model(
            spec,
            self.device,
        )

        parameter_count = sum(
            p.numel()
            for p in loaded.model.parameters()
        )

        print(
            f"[FourLang] loaded: "
            f"{parameter_count:,} parameters"
        )

        self._cache[
            key
        ] = loaded

        return loaded

    @staticmethod
    def _generation_kwargs(
        spec: ModelSpec,
    ) -> dict[str, Any]:
        generation = {
            "num_beams": 4,
            "max_new_tokens": 128,
            "do_sample": False,
        }

        generation.update(
            spec.generation
        )

        # Production translation must be deterministic.
        generation[
            "do_sample"
        ] = False

        return generation

    @torch.inference_mode()
    def translate(
        self,
        direction: str,
        text: str,
    ) -> dict[str, Any]:
        text = str(
            text
        ).strip()

        if not text:
            raise ValueError(
                "Input text is empty."
            )

        spec = (
            self.registry
            .require_ready(
                direction
            )
        )

        loaded = self._get_loaded(
            spec
        )

        # SMaLL-100 switches direction using tgt_lang.
        if spec.architecture == "small100":
            loaded.tokenizer.tgt_lang = (
                spec.target_lang
            )

        encoded = loaded.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )

        encoded = {
            key: value.to(
                loaded.device
            )
            for key, value
            in encoded.items()
        }

        if loaded.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        output_ids = loaded.model.generate(
            **encoded,
            **self._generation_kwargs(
                spec
            ),
        )

        if loaded.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            -
            start
        )

        translation = (
            loaded.tokenizer
            .decode(
                output_ids[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        return {
            "direction": spec.direction,
            "source_lang": spec.source_lang,
            "target_lang": spec.target_lang,
            "model": spec.model_name,
            "architecture": spec.architecture,
            "text": text,
            "translation": translation,
            "generation_latency_seconds": float(
                elapsed
            ),
            "device": str(
                loaded.device
            ),
            "model_path": str(
                spec.path
            ),
        }

    def warmup(
        self,
        direction: str,
    ) -> dict[str, Any]:
        spec = (
            self.registry
            .require_ready(
                direction
            )
        )

        dummy_text = {
            "en": "Hello.",
            "zh": "你好。",
            "uz": "Salom.",
            "ru": "Привет.",
        }.get(
            spec.source_lang,
            "Hello.",
        )

        return self.translate(
            direction,
            dummy_text,
        )

    def unload(
        self,
        direction: str,
    ) -> None:
        """
        Remove one model from the in-process cache.

        Shared bidirectional models (e.g. EN<->UZ SMaLL-100) use the
        same cache key, so unloading either direction unloads the pair.
        """
        spec = self.registry.get(
            direction
        )

        key = self._cache_key(
            spec
        )

        loaded = self._cache.pop(
            key,
            None,
        )

        if loaded is not None:
            del loaded.model
            del loaded.tokenizer

            if (
                self.device.type
                ==
                "cuda"
            ):
                torch.cuda.empty_cache()
