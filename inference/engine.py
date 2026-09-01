"""Direction-aware translation engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from .loader import LoadedModel


SUPPORTED_LANGUAGES = ("zh", "en", "ru", "uz")
SUPPORTED_DIRECTIONS = tuple(
    f"{source}-{target}"
    for source in SUPPORTED_LANGUAGES
    for target in SUPPORTED_LANGUAGES
    if source != target
)


def parse_direction(value: str) -> tuple[str, str]:
    normalized = value.strip().lower().replace("->", "-").replace("_", "-").replace(":", "-")
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
        tokenizer.src_lang = source
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        ).to(self.loaded.device)

        if self.loaded.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.loaded.model.generate(
                **inputs,
                forced_bos_token_id=_language_id(tokenizer, target),
                max_new_tokens=self.max_new_tokens,
                num_beams=self.num_beams,
            )
        if self.loaded.device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000

        translation = tokenizer.batch_decode(
            generated, skip_special_tokens=True
        )[0]
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
            "latency_ms": round(latency_ms, 3),
        }
