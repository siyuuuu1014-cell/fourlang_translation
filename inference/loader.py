"""Shared model loading for inference only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep using the model directory already present in this project.
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "small100"


@dataclass(frozen=True)
class LoadedModel:
    """Tokenizer, model, and runtime metadata returned by the shared loader."""

    tokenizer: Any
    model: Any
    device: torch.device
    model_path: str
    adapter_path: str | None


def _resolve_source(value: str | Path) -> str:
    """Resolve project-relative local paths while leaving Hub IDs untouched."""

    raw = str(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return str(project_path)
    return raw


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _select_dtype(device: torch.device, requested: str) -> torch.dtype | None:
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[requested]


def load_translation_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    adapter_path: str | Path | None = None,
    device: str = "auto",
    dtype: str = "auto",
) -> LoadedModel:
    """Load a full seq2seq model and, optionally, a PEFT/LoRA adapter.

    ``model_path`` may be an existing project-relative path, an absolute path,
    or a Hugging Face model ID. No model files are moved or rewritten.
    """

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if dtype not in {"auto", "float32", "float16", "bfloat16"}:
        raise ValueError(
            "dtype must be one of: auto, float32, float16, bfloat16"
        )

    model_source = _resolve_source(model_path)
    adapter_source = _resolve_source(adapter_path) if adapter_path else None
    runtime_device = _select_device(device)
    runtime_dtype = _select_dtype(runtime_device, dtype)

    # Prefer adapter tokenizer files when supplied; fall back to the base model.
    tokenizer_source = model_source
    if adapter_source and Path(adapter_source).is_dir():
        if (Path(adapter_source) / "tokenizer_config.json").exists():
            tokenizer_source = adapter_source

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if runtime_dtype is not None:
        load_kwargs["torch_dtype"] = runtime_dtype
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, **load_kwargs)
    if adapter_source:
        model = PeftModel.from_pretrained(model, adapter_source)

    model = model.to(runtime_device).eval()
    model.config.use_cache = True
    return LoadedModel(
        tokenizer=tokenizer,
        model=model,
        device=runtime_device,
        model_path=str(model_path),
        adapter_path=str(adapter_path) if adapter_path else None,
    )
