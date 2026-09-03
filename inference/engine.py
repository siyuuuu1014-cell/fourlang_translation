from __future__ import annotations

import time
import threading
from dataclasses import dataclass
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
from .loader import LoadedModel


SUPPORTED_LANGUAGES = ("zh", "en", "ru", "uz")
SUPPORTED_DIRECTIONS = tuple(
    f"{source}-{target}"
    for source in SUPPORTED_LANGUAGES
    for target in SUPPORTED_LANGUAGES
    if source != target
)

NLLB_LANGUAGE_CODES = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl",
    "uz": "uzn_Latn",
}


def parse_direction(value: str) -> tuple[str, str]:
    normalized = (
        value.strip().lower().replace("->", "-").replace("_", "-").replace(":", "-")
    )
    parts = normalized.split("-")
    if len(parts) != 2:
        raise ValueError("direction must look like zh-en, zh_en, or zh->en")
    source, target = parts
    if source not in SUPPORTED_LANGUAGES or target not in SUPPORTED_LANGUAGES:
        raise ValueError(
            "direction languages must be one of: " + ", ".join(SUPPORTED_LANGUAGES)
        )
    if source == target:
        raise ValueError("source and target languages must be different")
    return source, target


def _language_id(tokenizer: Any, language: str) -> int:
    if hasattr(tokenizer, "get_lang_id"):
        return int(tokenizer.get_lang_id(language))
    try:
        return int(tokenizer.lang_code_to_id[language])
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"tokenizer does not support target language: {language}") from exc


@dataclass
class TranslationEngine:
    """Compatibility engine used by the automatic 12-direction router."""

    loaded: LoadedModel
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

    def translate(self, text: str, direction: str | None = None) -> dict[str, Any]:
        if direction is not None:
            self.set_direction(direction)
        if not text.strip():
            raise ValueError("text must not be empty")

        source, target = parse_direction(self.direction)
        tokenizer = self.loaded.tokenizer
        tokenizer_kind = self.loaded.tokenizer_kind
        if tokenizer_kind == "small100":
            tokenizer.tgt_lang = target
        elif tokenizer_kind == "nllb":
            tokenizer.src_lang = NLLB_LANGUAGE_CODES[source]
        elif tokenizer_kind != "marian":
            tokenizer.src_lang = source

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.loaded.device)
        generation: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": self.num_beams,
        }
        if tokenizer_kind == "nllb":
            generation["forced_bos_token_id"] = _language_id(
                tokenizer, NLLB_LANGUAGE_CODES[target]
            )
        elif tokenizer_kind not in {"small100", "marian"}:
            generation["forced_bos_token_id"] = _language_id(tokenizer, target)

        if self.loaded.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.loaded.model.generate(**inputs, **generation)
        if self.loaded.device.type == "cuda":
            torch.cuda.synchronize()

        translation = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return {
            "ok": True,
            "direction": self.direction,
            "source_language": source,
            "target_language": target,
            "input": text,
            "translation": translation,
            "model_path": self.loaded.model_path,
            "adapter_path": self.loaded.adapter_path,
            "device": str(self.loaded.device),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }


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
        # One process owns one GPU model cache. Serializing load + tokenizer
        # mutation + generation prevents duplicate cold loads and tgt_lang races.
        self._inference_lock = threading.RLock()

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
        with self._inference_lock:
            return self._translate_locked(direction, text)

    def _translate_locked(
        self,
        direction: str,
        text: str,
    ) -> dict[str, Any]:
        text = str(text).strip()

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
        elif spec.architecture == "m2m100":
            loaded.tokenizer.src_lang = spec.source_lang
        elif spec.architecture == "nllb":
            loaded.tokenizer.src_lang = NLLB_LANGUAGE_CODES[spec.source_lang]

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

        generation = self._generation_kwargs(spec)
        if spec.architecture == "m2m100":
            generation["forced_bos_token_id"] = loaded.tokenizer.get_lang_id(spec.target_lang)
        elif spec.architecture == "nllb":
            generation["forced_bos_token_id"] = loaded.tokenizer.convert_tokens_to_ids(
                NLLB_LANGUAGE_CODES[spec.target_lang]
            )
        output_ids = loaded.model.generate(**encoded, **generation)

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
