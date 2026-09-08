from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
PROJECT = PACKAGE.parents[1]
LANGUAGES = ("en", "zh", "uz", "ru")
DIRECTIONS = tuple(f"{a}-{b}" for a in LANGUAGES for b in LANGUAGES if a != b)
GROUPS = ("human_only", "human_kd")


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def writable(path: Path) -> Path:
    """All writes, including temp files, are fenced inside this experiment."""
    path = path.resolve()
    boundary = PACKAGE / "artifacts"
    if not path.is_relative_to(boundary) or path == boundary:
        raise ValueError(f"Output must stay inside {boundary}: {path}")
    return path


def atomic_json(path: Path, value) -> None:
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable(path.with_name(path.name + ".tmp"))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def source_path(value: str) -> Path:
    path = (PROJECT / value).resolve()
    if not path.is_relative_to(PROJECT / "data") or not path.is_file():
        raise FileNotFoundError(
            f"Read-only source data is missing or outside project/data: {path}"
        )
    return path


def load_config(path: Path = PACKAGE / "config.toml", suite: str | None = None) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if suite:
        config["experiment"]["suite"] = suite
    if not re.fullmatch(
        r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", config["experiment"]["suite"]
    ):
        raise ValueError("Suite must be a simple directory name, not a path.")
    if tuple(config["experiment"]["languages"]) != LANGUAGES:
        raise ValueError("These experiments require en, zh, uz, ru in the fixed order.")
    model = config["model"]
    for key in (
        "d_model",
        "nhead",
        "encoder_layers",
        "decoder_layers",
        "dim_feedforward",
        "max_length",
    ):
        if not isinstance(model[key], int) or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer.")
    if model["d_model"] % model["nhead"] or model["max_length"] < 4:
        raise ValueError("Invalid Transformer dimensions/maximum length.")
    for key in (
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "checkpoint_steps",
        "eval_steps",
        "log_steps",
        "keep_checkpoints",
    ):
        if config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive.")
    if not 0 < config["sampling"]["kd_ratio"] < 1:
        raise ValueError("KD ratio must be between 0 and 1.")
    if config["sampling"]["rows_per_direction"] < 2:
        raise ValueError("Sampling quota must be at least 2 per direction.")
    if (
        not 0
        < round(
            config["sampling"]["rows_per_direction"] * config["sampling"]["kd_ratio"]
        )
        < config["sampling"]["rows_per_direction"]
    ):
        raise ValueError(
            "Rounded sampling quotas must include both human and Teacher rows."
        )
    if config["sampling"]["max_teacher_repeats_per_block"] < 1:
        raise ValueError("Teacher repetition cap must be positive.")
    for key in ("validation_per_direction", "batch_size", "max_new_tokens"):
        if config["evaluation"][key] < 1:
            raise ValueError(f"evaluation.{key} must be positive.")
    if (
        not 0 <= model["dropout"] < 1
        or not 0 <= config["training"]["label_smoothing"] < 1
    ):
        raise ValueError("Invalid dropout/label smoothing.")
    if (
        config["training"]["learning_rate"] <= 0
        or config["training"]["max_grad_norm"] <= 0
        or config["training"]["weight_decay"] < 0
    ):
        raise ValueError("Invalid optimizer settings.")
    if (
        config["tokenizer"]["vocab_size"] < 16
        or config["tokenizer"]["max_sentences_per_language"] < 1
        or not 0.98 <= config["tokenizer"]["character_coverage"] <= 1
    ):
        raise ValueError("Invalid tokenizer settings.")
    expected_pairs = {frozenset((a, b)) for a in LANGUAGES for b in LANGUAGES if a != b}
    actual_pairs = [frozenset(item["pair"].split("_")) for item in config["pair_data"]]
    if (
        len(actual_pairs) != 6
        or set(actual_pairs) != expected_pairs
        or set(config["benchmarks"]) != {"dev", "test"}
    ):
        raise ValueError("Require six distinct language pairs and dev/test benchmarks.")
    if config["evaluation"]["max_new_tokens"] >= model["max_length"]:
        raise ValueError("max_new_tokens must be less than model.max_length.")
    return config


def suite_root(config: dict) -> Path:
    return writable(PACKAGE / "artifacts" / config["experiment"]["suite"])


def versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "pandas",
            "pyarrow",
            "sentencepiece",
            "sacrebleu",
            "opencc-python-reimplemented",
            "filelock",
        )
    }


def implementation() -> dict:
    return {path.name: file_hash(path) for path in sorted(PACKAGE.glob("*.py"))}


def verify_files(root: Path, hashes: dict) -> None:
    for name, expected in hashes.items():
        path = writable(root / name)
        if not path.is_file() or file_hash(path) != expected:
            raise RuntimeError(
                f"Missing/changed artifact: {path}. Use a new suite; do not bypass the manifest."
            )
