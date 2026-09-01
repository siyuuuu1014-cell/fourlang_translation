from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
)

from .registry import ModelSpec


@dataclass
class LoadedTranslationModel:
    tokenizer: Any
    model: Any
    device: torch.device
    architecture: str
    model_path: Path


def _common_model_kwargs(
    device: torch.device,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": True,
    }

    if device.type == "cuda":
        kwargs["dtype"] = torch.float16

    return kwargs


def _load_small100_tokenizer_class(
    model_path: Path,
):
    """
    EN-UZ Exp2 was trained with the project-local SMALL100Tokenizer
    implementation saved alongside the frozen model.
    """
    tokenizer_py = (
        model_path
        / "tokenization_small100.py"
    )

    if not tokenizer_py.exists():
        raise FileNotFoundError(
            "SMaLL-100 model is missing its custom tokenizer file:\n"
            f"{tokenizer_py}"
        )

    module_name = (
        "fourlang_small100_tokenizer_"
        + str(
            abs(
                hash(
                    str(
                        model_path.resolve()
                    )
                )
            )
        )
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        tokenizer_py,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load tokenizer module: {tokenizer_py}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    tokenizer_cls = getattr(
        module,
        "SMALL100Tokenizer",
        None,
    )

    if tokenizer_cls is None:
        raise RuntimeError(
            f"SMALL100Tokenizer not found in {tokenizer_py}"
        )

    return tokenizer_cls


def load_translation_model(
    spec: ModelSpec,
    device: torch.device,
) -> LoadedTranslationModel:
    if spec.architecture == "marian":
        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(
                    spec.path
                ),
                local_files_only=True,
                use_fast=False,
            )
        )

        model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(
                str(
                    spec.path
                ),
                **_common_model_kwargs(
                    device
                ),
            )
        )

    elif spec.architecture == "small100":
        tokenizer_cls = (
            _load_small100_tokenizer_class(
                spec.path
            )
        )

        tokenizer = (
            tokenizer_cls
            .from_pretrained(
                str(
                    spec.path
                ),
                tgt_lang=(
                    spec.target_lang
                ),
                local_files_only=True,
            )
        )

        model = (
            M2M100ForConditionalGeneration
            .from_pretrained(
                str(
                    spec.path
                ),
                **_common_model_kwargs(
                    device
                ),
            )
        )

    else:
        raise ValueError(
            f"Unsupported architecture: "
            f"{spec.architecture!r}"
        )

    model = model.to(
        device
    )

    model.eval()

    try:
        model.config.use_cache = True
    except Exception:
        pass

    return LoadedTranslationModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        architecture=(
            spec.architecture
        ),
        model_path=(
            spec.path
        ),
    )
