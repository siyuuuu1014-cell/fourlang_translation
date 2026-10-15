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


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def writable(path: Path) -> Path:
    path = path.resolve()
    boundary = (PACKAGE / "artifacts").resolve()
    if path == boundary or not path.is_relative_to(boundary):
        raise ValueError(f"v2 may write only inside {boundary}: {path}")
    return path


def atomic_json(path: Path, value) -> None:
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable(path.with_name(path.name + ".tmp"))
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_config(path: Path = PACKAGE / "config.toml", suite: str | None = None) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if suite:
        config["experiment"]["suite"] = suite
    name = config["experiment"]["suite"]
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", name):
        raise ValueError("Suite must be a simple directory name.")
    if tuple(config["experiment"]["languages"]) != LANGUAGES:
        raise ValueError("v2 requires en, zh, uz, ru in fixed order.")
    model = config["model"]
    if any(not isinstance(model[key], int) or model[key] < 1 for key in
           ("d_model", "nhead", "encoder_layers", "decoder_layers", "dim_feedforward", "max_length")):
        raise ValueError("Model dimensions must be positive integers.")
    if model["d_model"] % model["nhead"] or not 0 <= model["dropout"] < 1:
        raise ValueError("Invalid Transformer dimensions/dropout.")
    sampling = config["sampling"]
    if set(sampling["kd_ratio"]) != set(DIRECTIONS):
        raise ValueError("KD ratios must specify all 12 directions exactly.")
    if any(not 0 < ratio < 1 for ratio in sampling["kd_ratio"].values()):
        raise ValueError("Every mixed-stage KD ratio must be between 0 and 1.")
    phases = config["training"]["phases"]
    if [phase["name"] for phase in phases] != ["mixed_kd", "human_polish"] or [phase["data"] for phase in phases] != ["mixed", "human"]:
        raise ValueError("v2 requires the frozen mixed_kd -> human_polish phase order.")
    if any(phase["steps"] < 1 or phase["learning_rate"] <= 0 or phase["warmup_steps"] < 1 for phase in phases):
        raise ValueError("Invalid phase schedule.")
    training = config["training"]
    for key in ("batch_size", "gradient_accumulation_steps", "checkpoint_steps", "eval_steps", "log_steps", "keep_checkpoints"):
        if training[key] < 1:
            raise ValueError(f"training.{key} must be positive.")
    evaluation = config["evaluation"]
    if evaluation["beam_size"] < 1 or evaluation["max_new_tokens"] >= model["max_length"] or evaluation["length_penalty"] <= 0:
        raise ValueError("Invalid decoding settings.")
    return config


def total_steps(config):
    return sum(phase["steps"] for phase in config["training"]["phases"])


def suite_root(config):
    return writable(PACKAGE / "artifacts" / config["experiment"]["suite"])


def implementation():
    return {path.name: file_hash(path) for path in sorted(PACKAGE.glob("*.py"))}


def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "numpy", "pandas", "pyarrow", "sentencepiece", "sacrebleu", "filelock")}


def frozen_prepared(config):
    relative = Path(config["experiment"]["prepared_source"])
    path = (PROJECT / relative).resolve()
    boundary = (PROJECT / "experiments" / "scratch_translation" / "artifacts").resolve()
    expected = (boundary / "fourlang_scratch_v1" / "prepared").resolve()
    if path != expected or not path.is_dir():
        raise FileNotFoundError(f"Expected frozen v1 prepared data at {expected}")
    manifest = read_json(path / "manifest.json")
    for name, expected_hash in manifest["files"].items():
        item = (path / name).resolve()
        if not item.is_relative_to(path) or not item.is_file() or file_hash(item) != expected_hash:
            raise RuntimeError(f"Frozen v1 prepared artifact is missing/changed: {item}")
    return path, manifest
