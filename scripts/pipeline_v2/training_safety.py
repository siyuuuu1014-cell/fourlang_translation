"""Fail-closed guards for local models and single-process resumable training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import TrainerCallback


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def model_files(path: Path, *, tokenizer: bool = True) -> list[Path]:
    """Check exported model files, including every shard referenced by an index."""
    path = path.resolve()
    files = {path / "config.json"}
    indexes = list(path.glob("*.index.json"))
    weights = [
        path / name
        for name in ("model.safetensors", "pytorch_model.bin")
        if (path / name).is_file()
    ]
    if not weights and not indexes:
        raise FileNotFoundError(f"Trained model weights are missing: {path}")
    files.update(weights)
    for index in indexes:
        mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
        if not mapping:
            raise ValueError(f"Empty model shard index: {index}")
        files.add(index)
        for name in set(mapping.values()):
            shard = (path / name).resolve()
            if shard.parent != path:
                raise ValueError(f"Shard must be inside the model directory: {name}")
            files.add(shard)
    if tokenizer:
        vocab = [
            item
            for name in (
                "tokenizer.json",
                "sentencepiece.bpe.model",
                "sentencepiece.model",
                "spiece.model",
                "vocab.json",
            )
            if (item := path / name).is_file()
        ]
        if not vocab:
            raise FileNotFoundError(f"Trained model tokenizer is missing: {path}")
        files.update(vocab)
        files.update(path.glob("*token*.json"))
        files.update(path.glob("tokenization_*.py"))
        files.update(path.glob("*.model"))
        if (path / "generation_config.json").exists():
            files.add(path / "generation_config.json")
    for item in files:
        if not item.is_file() or item.stat().st_size == 0:
            raise FileNotFoundError(f"Model artifact is missing or empty: {item}")
    json.loads((path / "config.json").read_text(encoding="utf-8"))
    return sorted(files)


def model_signature(path: Path) -> dict[str, str]:
    return {item.name: file_sha256(item) for item in model_files(path)}


def bind_run(directory: Path, manifest: dict) -> str:
    """Never silently adopt checkpoints from different data/settings/model/code."""
    path = directory / "run_manifest.json"
    signature = fingerprint(manifest)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != {"fingerprint": signature, "manifest": manifest}:
            raise RuntimeError(
                "Training inputs/settings/model changed. Refusing old checkpoints. "
                f"Use a separate experiment directory; preserve the old run: {directory}"
            )
    else:
        if any(directory.glob("checkpoint-*")):
            raise RuntimeError(
                f"Legacy checkpoints have no run fingerprint: {directory}. "
                "Move that run aside explicitly before starting a new run."
            )
        atomic_json(path, {"fingerprint": signature, "manifest": manifest})
    return signature


def checkpoint_complete(path: Path, signature: str) -> bool:
    try:
        record = json.loads((path / "complete.json").read_text(encoding="utf-8"))
        state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
        if record["fingerprint"] != signature or record["step"] != state["global_step"]:
            return False
        required = {
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
        }
        if record.get("fp16"):
            required.add("amp_scaler.pt")
        if not required.issubset(record["files"]):
            return False
        model_files(path, tokenizer=False)
        return all(
            (path / name).is_file()
            and size > 0
            and (path / name).stat().st_size == size
            for name, size in record["files"].items()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_complete_checkpoint(directory: Path, signature: str) -> str | None:
    candidates = sorted(
        (
            item
            for item in directory.glob("checkpoint-*")
            if item.is_dir() and item.name[11:].isdigit()
        ),
        key=lambda item: int(item.name[11:]),
        reverse=True,
    )
    for candidate in candidates:
        if checkpoint_complete(candidate, signature):
            state = json.loads(
                (candidate / "trainer_state.json").read_text(encoding="utf-8")
            )
            best = state.get("best_model_checkpoint")
            if best and not checkpoint_complete(Path(best), signature):
                raise RuntimeError(
                    f"Best-model checkpoint is missing/incomplete: {best}"
                )
            return str(candidate)
        print(f"Skipping incomplete checkpoint: {candidate}", flush=True)
    if candidates:
        raise RuntimeError(
            f"No complete checkpoint remains in {directory}; refusing silent restart."
        )
    return None


class SafeCheckpointCallback(TrainerCallback):
    def __init__(self, signature: str, interval: int = 1000):
        self.signature = signature
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        # Extra saves do not trigger expensive translation evaluation mid-epoch.
        # Let the epoch-end hook evaluate FIRST at an exact epoch boundary.
        at_epoch_end = state.epoch is not None and float(state.epoch).is_integer()
        if (
            self.interval > 0
            and state.global_step % self.interval == 0
            and not at_epoch_end
        ):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        if not args.should_save:
            return control
        path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        atomic_json(
            path / "complete.json",
            {
                "fingerprint": self.signature,
                "step": state.global_step,
                "fp16": bool(args.fp16),
                "files": {
                    item.name: item.stat().st_size
                    for item in path.iterdir()
                    if item.is_file()
                    and item.name not in ("complete.json", "complete.json.tmp")
                },
            },
        )
        if not checkpoint_complete(path, self.signature):
            raise RuntimeError(f"Checkpoint was not saved completely: {path}")
        return control
