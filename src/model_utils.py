from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, M2M100ForConditionalGeneration


def _small100_tokenizer_class(model_path: Path):
    tokenizer_file = model_path / "tokenization_small100.py"
    if not tokenizer_file.is_file():
        raise FileNotFoundError(
            f"SMaLL-100 tokenizer implementation is missing: {tokenizer_file}"
        )
    module_name = f"fourlang_training_small100_{abs(hash(str(model_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, tokenizer_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import tokenizer from {tokenizer_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    tokenizer_class = getattr(module, "SMALL100Tokenizer", None)
    if tokenizer_class is None:
        raise RuntimeError(f"SMALL100Tokenizer is missing from {tokenizer_file}")
    return tokenizer_class


def load_tokenizer(model_name: str, architecture: str):
    architecture = architecture.lower()
    if architecture != "small100":
        return AutoTokenizer.from_pretrained(model_name)

    model_path = Path(model_name).expanduser().resolve()
    tokenizer_class = _small100_tokenizer_class(model_path)
    vocab_file = model_path / "vocab.json"
    spm_file = model_path / "sentencepiece.bpe.model"
    for required in (vocab_file, spm_file):
        if not required.is_file():
            raise FileNotFoundError(required)
    return tokenizer_class(
        vocab_file=str(vocab_file),
        spm_file=str(spm_file),
        tgt_lang="en",
    )


def _dtype_keyword() -> str:
    try:
        major_minor = tuple(int(part) for part in transformers.__version__.split(".")[:2])
    except ValueError:
        major_minor = (4, 46)
    return "dtype" if major_minor >= (4, 56) else "torch_dtype"


def load_base_model(
    model_name: str,
    architecture: str,
    *,
    dtype: torch.dtype | None = None,
):
    kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if dtype is not None:
        kwargs[_dtype_keyword()] = dtype
    if architecture.lower() == "small100":
        return M2M100ForConditionalGeneration.from_pretrained(model_name, **kwargs)
    return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)


def prepare_generation(tokenizer: Any, architecture: str, source: str, target: str) -> dict[str, int]:
    if architecture.lower() == "small100":
        tokenizer.tgt_lang = target
        return {}
    tokenizer.src_lang = source
    return {"forced_bos_token_id": int(tokenizer.get_lang_id(target))}

