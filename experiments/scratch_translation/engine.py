from __future__ import annotations

import contextlib
import math
import os
import random
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from sacrebleu.metrics import BLEU, CHRF
from torch.nn import functional as F

from .common import (
    DIRECTIONS,
    GROUPS,
    atomic_json,
    digest,
    file_hash,
    implementation,
    read_json,
    suite_root,
    versions,
    writable,
)
from .data import Vocabulary, collate, load_prepared, normalize_text, sampling_plan
from .model import TranslationTransformer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(model):
    import hashlib

    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rng_state(include_cuda=False):
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), *state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if include_cuda else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    state_np = state["numpy"]
    np.random.set_state(
        (state_np[0], np.asarray(state_np[1], dtype=np.uint32), *state_np[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def amp_context(device, enabled):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if enabled
        else contextlib.nullcontext()
    )


def atomic_torch(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable(path.with_name(path.name + ".tmp"))
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def save_checkpoint(run, payload, keep):
    path = run / "checkpoints" / f"step_{payload['step']:08d}.pt"
    atomic_torch(path, payload)
    # The pointer is published last: an interrupted .tmp is never resumable.
    record = {
        "file": path.name,
        "sha256": file_hash(path),
        "step": payload["step"],
        "fingerprint": payload["fingerprint"],
        "best_step": payload["best_step"],
    }
    atomic_json(path.with_suffix(".json"), record)
    atomic_json(run / "latest.json", record)
    # Keep the latest N complete checkpoints plus the selected best if older.
    paths = sorted((run / "checkpoints").glob("step_*.pt"), reverse=True)
    preserve = {item.name for item in paths[:keep]} | {
        f"step_{payload['best_step']:08d}.pt"
    }
    for old in paths:
        if old.name not in preserve:
            metadata = old.with_suffix(".json")
            # Only remove this experiment's recognized, complete checkpoint pairs.
            if (
                metadata.is_file()
                and read_json(metadata).get("fingerprint") == payload["fingerprint"]
            ):
                writable(old).unlink()
                writable(metadata).unlink()


def read_checkpoint(run, step=None):
    metadata = (
        run / "latest.json"
        if step is None
        else run / "checkpoints" / f"step_{step:08d}.json"
    )
    record = read_json(metadata)
    if record["file"] != f"step_{record['step']:08d}.pt":
        raise RuntimeError("Invalid checkpoint filename.")
    path = writable(run / "checkpoints" / record["file"])
    if not path.is_file() or file_hash(path) != record["sha256"]:
        raise RuntimeError(
            f"Checkpoint is incomplete/corrupted: {path}. Preserve it for diagnosis."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["fingerprint"] != record["fingerprint"]
        or payload["step"] != record["step"]
    ):
        raise RuntimeError("Checkpoint identity does not match its manifest.")
    return payload


def learning_rate(step, settings):
    warmup = settings["warmup_steps"]
    return settings["learning_rate"] * min(step / warmup, math.sqrt(warmup / step))


def translation_loss(logits, labels, pad_id, smoothing):
    # Summed non-padding token CE, normalized over the entire accumulated batch.
    return F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=pad_id,
        label_smoothing=smoothing,
        reduction="sum",
    )


def normalize_prediction(language, text):
    # Wrong-script output is a model error to score, not a reason to crash validation.
    try:
        return normalize_text(language, text)
    except ValueError:
        return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


@torch.no_grad()
def score_model(model, vocab, rows, config, device, amp=False):
    was_training = model.training
    model.eval()
    result = {}
    try:
        for direction in DIRECTIONS:
            part = [row for row in rows if row["direction"] == direction]
            if not part:
                raise ValueError(f"Evaluation is missing {direction}.")
            print(f"  evaluating {direction}: {len(part)} samples", flush=True)
            predictions, references, loss_sum, tokens = [], [], 0.0, 0
            for start in range(0, len(part), config["evaluation"]["batch_size"]):
                batch = part[start : start + config["evaluation"]["batch_size"]]
                source, decoder, labels = collate(batch, vocab.pad, device)
                with amp_context(device, amp):
                    output = model(source, decoder)
                    loss_sum += float(translation_loss(output, labels, vocab.pad, 0.0))
                    generated = model.generate(
                        source,
                        vocab.bos,
                        vocab.eos,
                        config["evaluation"]["max_new_tokens"],
                        forbidden_ids=list(vocab.languages.values()),
                    )
                tokens += int(labels.ne(vocab.pad).sum())
                target = direction.split("-")[1]
                predictions.extend(
                    normalize_prediction(target, vocab.decode(sequence))
                    for sequence in generated.cpu().tolist()
                )
                references.extend(row["tgt_text"] for row in batch)
            bleu = BLEU(tokenize="zh" if direction.endswith("-zh") else "13a")
            chrf = CHRF(word_order=2)
            result[direction] = {
                "bleu": float(bleu.corpus_score(predictions, [references]).score),
                "chrf2": float(chrf.corpus_score(predictions, [references]).score),
                "bleu_signature": str(bleu.get_signature()),
                "chrf_signature": str(chrf.get_signature()),
                "loss": loss_sum / tokens,
                "samples": len(part),
            }
        macro = {
            key: sum(value[key] for value in result.values()) / len(result)
            for key in ("bleu", "chrf2", "loss")
        }
        if not all(
            math.isfinite(value[key])
            for value in result.values()
            for key in ("bleu", "chrf2", "loss")
        ):
            raise RuntimeError("Non-finite validation metrics.")
        return {"directions": result, "macro": macro, "decoding": "greedy"}
    finally:
        model.train(was_training)


def run_identity(config, group, prepared, device):
    return {
        "schema": 1,
        "group": group,
        "config": config,
        "prepared_fingerprint": digest(prepared),
        "implementation": implementation(),
        "versions": versions(),
        "device_type": device.type,
        "pretrained_weights": False,
    }


def train(config, group, device_name="auto", stop_after_steps=None):
    """stop_after_steps supports short smoke runs without changing the planned budget."""
    if group not in GROUPS:
        raise ValueError(group)
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This isolated training loop supports one process/GPU only.")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    root, prepared = load_prepared(config)
    run = writable(suite_root(config) / "runs" / group)
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(writable(run / ".train.lock")), timeout=0):
        identity = run_identity(config, group, prepared, device)
        fingerprint = digest(identity)
        manifest_path = run / "run_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != identity:
                raise RuntimeError(
                    "Run/data/config/code/version mismatch. Choose a NEW suite, never resume another arm."
                )
        else:
            if (run / "latest.json").exists() or any(
                (run / "checkpoints").glob("*.pt")
            ):
                raise RuntimeError(
                    "Unidentified existing checkpoints; refusing to adopt them."
                )
            atomic_json(manifest_path, identity)
        vocab = Vocabulary(root / "vocabulary.model", config["model"]["max_length"])
        # Each arm gets its own tokenizer snapshot. No model is loaded from the other arm.
        own_vocab = writable(run / "vocabulary.model")
        if not own_vocab.exists():
            own_vocab.write_bytes((root / "vocabulary.model").read_bytes())
        if file_hash(own_vocab) != prepared["files"]["vocabulary.model"]:
            raise RuntimeError("Run vocabulary was changed.")
        human = pd.read_parquet(root / "human.parquet").to_dict("records")
        teacher = (
            pd.read_parquet(root / "teacher.parquet").to_dict("records")
            if group == "human_kd"
            else []
        )
        validation = pd.read_parquet(root / "validation.parquet").to_dict("records")
        settings = config["training"]
        seed_all(config["experiment"]["seed"])
        model = TranslationTransformer(
            vocab.sp.get_piece_size(), config["model"], vocab.pad
        ).to(device)
        initial_hash = state_hash(model)
        atomic_json(
            run / "initialization.json",
            {
                "random_weights_sha256": initial_hash,
                "parameters": sum(p.numel() for p in model.parameters()),
                "seed": config["experiment"]["seed"],
                "pretrained_weights": False,
            },
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings["learning_rate"],
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=settings["weight_decay"],
        )
        amp = bool(settings["amp"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
        step, cursor, best_step, best_score, exposure_tokens, elapsed = (
            0,
            0,
            0,
            -float("inf"),
            0,
            0.0,
        )
        if (run / "latest.json").exists():
            saved = read_checkpoint(run)
            if (
                saved["fingerprint"] != fingerprint
                or saved["initial_hash"] != initial_hash
            ):
                raise RuntimeError(
                    "Checkpoint belongs to a different arm or initialization."
                )
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"])
            step, cursor = saved["step"], saved["cursor"]
            best_step, best_score = saved["best_step"], saved["best_score"]
            exposure_tokens, elapsed = saved["target_tokens"], saved["seconds"]
            print(f"Resumed {group}: step={step}, best={best_step}", flush=True)
        total = settings["max_steps"]
        stop = min(total, step + stop_after_steps) if stop_after_steps else total
        block_size = len(DIRECTIONS) * config["sampling"]["rows_per_direction"]
        active_block, plan = None, None

        def next_rows(count):
            nonlocal cursor, active_block, plan
            selected = []
            while len(selected) < count:
                block, position = divmod(cursor, block_size)
                if block != active_block:
                    plan, sampling = sampling_plan(config, group, human, teacher, block)
                    atomic_json(run / "sampling" / f"block_{block:05d}.json", sampling)
                    print(
                        f"Sampling block {block}: {len(plan):,} rows, group={group}",
                        flush=True,
                    )
                    active_block = block
                take = min(count - len(selected), block_size - position)
                selected.extend(
                    (human if kind == "human" else teacher)[index]
                    for kind, index in plan[position : position + take]
                )
                cursor += take
            return selected

        print(
            f"{group}: random-init Transformer, parameters={sum(p.numel() for p in model.parameters()):,}; "
            f"budget={total} updates, effective_batch={settings['batch_size'] * settings['gradient_accumulation_steps']}",
            flush=True,
        )
        started = time.monotonic()
        model.train()
        while step < stop:
            batches = [
                next_rows(settings["batch_size"])
                for _ in range(settings["gradient_accumulation_steps"])
            ]
            denominator = sum(
                len(row["tgt_ids"]) - 1 for batch in batches for row in batch
            )
            lr = learning_rate(step + 1, settings)
            for item in optimizer.param_groups:
                item["lr"] = lr
            before_update = rng_state(device.type == "cuda")
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for batch in batches:
                    source, decoder, labels = collate(batch, vocab.pad, device)
                    with amp_context(device, amp):
                        loss = translation_loss(
                            model(source, decoder),
                            labels,
                            vocab.pad,
                            settings["label_smoothing"],
                        )
                        scaled_loss = loss / denominator
                    if not bool(torch.isfinite(scaled_loss)):
                        raise RuntimeError(
                            "Non-finite loss; stop and resume only after diagnosis."
                        )
                    scaler.scale(scaled_loss).backward()
                    loss_sum += float(loss.detach())
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings["max_grad_norm"]
                )
                if bool(torch.isfinite(grad_norm)):
                    break
                if not amp or attempt == 7:
                    raise RuntimeError(
                        "Non-finite gradients; no optimizer update/checkpoint was committed."
                    )
                scaler.update(new_scale=scaler.get_scale() / 2)
                restore_rng(before_update)
                print(
                    f"FP16 overflow: retry SAME batch, reduced scale={scaler.get_scale()}",
                    flush=True,
                )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            exposure_tokens += denominator
            if step % settings["log_steps"] == 0 or step == 1:
                entry = {
                    "step": step,
                    "loss": loss_sum / denominator,
                    "lr": lr,
                    "examples_seen": cursor,
                    "target_tokens": exposure_tokens,
                    "seconds": elapsed + time.monotonic() - started,
                }
                atomic_json(run / "logs" / f"step_{step:08d}.json", entry)
                print(
                    f"step={step}/{total} loss={entry['loss']:.4f} lr={lr:.3g} "
                    f"examples={cursor:,} elapsed={entry['seconds'] / 60:.1f}min",
                    flush=True,
                )
            should_eval = step % settings["eval_steps"] == 0 or step == total
            if should_eval:
                scores = score_model(model, vocab, validation, config, device, amp)
                atomic_json(run / "validation" / f"step_{step:08d}.json", scores)
                if scores["macro"]["chrf2"] > best_score:
                    best_score, best_step = scores["macro"]["chrf2"], step
                print(
                    f"Validation macro chrF2={scores['macro']['chrf2']:.3f}; best_step={best_step}",
                    flush=True,
                )
            if should_eval or step % settings["checkpoint_steps"] == 0 or step == stop:
                payload = {
                    "fingerprint": fingerprint,
                    "initial_hash": initial_hash,
                    "step": step,
                    "cursor": cursor,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": rng_state(device.type == "cuda"),
                    "best_step": best_step,
                    "best_score": best_score,
                    "target_tokens": exposure_tokens,
                    "seconds": elapsed + time.monotonic() - started,
                }
                save_checkpoint(run, payload, settings["keep_checkpoints"])
                print(f"Checkpoint committed: step={step}", flush=True)
        report = {
            "status": "complete" if step == total else "paused",
            "group": group,
            "step": step,
            "max_steps": total,
            "best_step": best_step,
            "best_macro_chrf2": best_score if best_step else None,
            "examples_seen": cursor,
            "target_tokens": exposure_tokens,
            "run_fingerprint": fingerprint,
            "initial_weights_sha256": initial_hash,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        atomic_json(run / "train_report.json", report)
        return report


def evaluate(config, group, device_name="auto"):
    if group not in GROUPS:
        raise ValueError(group)
    root, prepared = load_prepared(config)
    run = suite_root(config) / "runs" / group
    with FileLock(str(writable(run / ".train.lock")), timeout=0):
        report = read_json(run / "train_report.json")
        if report["status"] != "complete":
            raise RuntimeError(
                "Finish the fixed training budget before looking at the test scores."
            )
        saved = read_checkpoint(run, report["best_step"])
        identity = read_json(run / "run_manifest.json")
        if (
            identity["config"] != config
            or identity["prepared_fingerprint"] != digest(prepared)
            or saved["fingerprint"] != digest(identity)
        ):
            raise RuntimeError("Evaluation input/run identity mismatch.")
        if (
            identity["implementation"] != implementation()
            or identity["versions"] != versions()
        ):
            raise RuntimeError(
                "Code/dependencies changed since training; preserve the original environment for evaluation."
            )
        device = torch.device(
            "cuda"
            if device_name == "auto" and torch.cuda.is_available()
            else "cpu"
            if device_name == "auto"
            else device_name
        )
        vocab = Vocabulary(run / "vocabulary.model", config["model"]["max_length"])
        if file_hash(run / "vocabulary.model") != prepared["files"]["vocabulary.model"]:
            raise RuntimeError("Run tokenizer was changed.")
        model = TranslationTransformer(
            vocab.sp.get_piece_size(), config["model"], vocab.pad
        ).to(device)
        model.load_state_dict(saved["model"])
        rows = pd.read_parquet(root / "test.parquet").to_dict("records")
        scores = score_model(
            model,
            vocab,
            rows,
            config,
            device,
            config["training"]["amp"] and device.type == "cuda",
        )
        result = {
            **scores,
            "group": group,
            "split": "flores_devtest",
            "best_step": report["best_step"],
            "evaluation_device": device.type,
            "run_fingerprint": saved["fingerprint"],
            "prepared_fingerprint": digest(prepared),
        }
        atomic_json(run / "evaluation" / "metrics.json", result)
        print(f"Evaluation complete: {run / 'evaluation/metrics.json'}", flush=True)
        return result


def compare(config):
    root = suite_root(config)
    reports = {
        group: read_json(root / "runs" / group / "evaluation/metrics.json")
        for group in GROUPS
    }
    runs = {
        group: read_json(root / "runs" / group / "run_manifest.json")
        for group in GROUPS
    }
    initial = {
        group: read_json(root / "runs" / group / "initialization.json")
        for group in GROUPS
    }
    training = {
        group: read_json(root / "runs" / group / "train_report.json")
        for group in GROUPS
    }
    for group in GROUPS:
        if (
            runs[group]["config"] != config
            or reports[group]["run_fingerprint"] != digest(runs[group])
            or training[group]["run_fingerprint"] != digest(runs[group])
            or training[group]["status"] != "complete"
            or training[group]["step"] != config["training"]["max_steps"]
        ):
            raise RuntimeError(
                "Comparison requires two verified, completed runs under this config."
            )
    left, right = reports["human_only"], reports["human_kd"]
    if (
        left["prepared_fingerprint"] != right["prepared_fingerprint"]
        or runs["human_only"]["config"] != runs["human_kd"]["config"]
        or runs["human_only"]["implementation"] != runs["human_kd"]["implementation"]
        or runs["human_only"]["versions"] != runs["human_kd"]["versions"]
        or runs["human_only"]["device_type"] != runs["human_kd"]["device_type"]
        or left["evaluation_device"] != right["evaluation_device"]
        or training["human_only"]["examples_seen"]
        != training["human_kd"]["examples_seen"]
        or initial["human_only"] != initial["human_kd"]
    ):
        raise RuntimeError(
            "Not a controlled comparison: data/config/code/versions/initialization differ."
        )
    directions = {
        direction: {
            "human_only": left["directions"][direction],
            "human_kd": right["directions"][direction],
            "delta": {
                metric: right["directions"][direction][metric]
                - left["directions"][direction][metric]
                for metric in ("bleu", "chrf2")
            },
        }
        for direction in DIRECTIONS
    }
    result = {
        "comparison": "human_kd minus human_only",
        "directions": directions,
        "macro_delta": {
            metric: right["macro"][metric] - left["macro"][metric]
            for metric in ("bleu", "chrf2")
        },
        "note": "Same updates/examples, not necessarily equal non-padding tokens. Different KD source coverage is part of the treatment.",
    }
    atomic_json(root / "comparison.json", result)
    print(result["macro_delta"], flush=True)
    return result
