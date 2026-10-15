from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import time

import numpy as np
import torch
from filelock import FileLock

from .common import (DIRECTIONS, atomic_json, digest, file_hash, implementation,
                     read_json, suite_root, total_steps, versions, writable)
from .data import Vocabulary, collate, load_data, sampling_plan
from .metrics import score_model, translation_loss
from .model import TranslationTransformer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_hash(model):
    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rng_state(include_cuda=False):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if include_cuda else []}


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable(path.with_name(path.name + ".tmp"))
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_copy(source, destination):
    destination = writable(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = writable(destination.with_name(destination.name + ".tmp"))
    temporary.write_bytes(source.read_bytes())
    temporary.replace(destination)


def save_checkpoint(run, payload, keep):
    path = run / "checkpoints" / f"step_{payload['step']:08d}.pt"
    atomic_torch(path, payload)
    record = {"file": path.name, "sha256": file_hash(path), "step": payload["step"],
              "fingerprint": payload["fingerprint"], "best_step": payload["best_step"],
              "phase": payload["phase"]}
    atomic_json(path.with_suffix(".json"), record)
    atomic_json(run / "latest.json", record)
    complete = sorted((run / "checkpoints").glob("step_*.pt"), reverse=True)
    preserve = {item.name for item in complete[:keep]}
    if payload["best_step"]:
        preserve.add(f"step_{payload['best_step']:08d}.pt")
    for old in complete:
        metadata = old.with_suffix(".json")
        if old.name not in preserve and metadata.is_file() and read_json(metadata).get("fingerprint") == payload["fingerprint"]:
            writable(old).unlink()
            writable(metadata).unlink()


def read_checkpoint(run, step=None):
    metadata = run / "latest.json" if step is None else run / "checkpoints" / f"step_{step:08d}.json"
    record = read_json(metadata)
    if record["file"] != f"step_{record['step']:08d}.pt":
        raise RuntimeError("Invalid checkpoint filename.")
    path = writable(run / "checkpoints" / record["file"])
    if not path.is_file() or file_hash(path) != record["sha256"]:
        raise RuntimeError(f"Incomplete/corrupted v2 checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["fingerprint"] != record["fingerprint"] or payload["step"] != record["step"]:
        raise RuntimeError("Checkpoint does not match its manifest.")
    return payload


def device_for(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def phase_at(config, completed_steps):
    start = 0
    for index, phase in enumerate(config["training"]["phases"]):
        end = start + phase["steps"]
        if completed_steps < end:
            return index, phase, completed_steps - start
        start = end
    return len(config["training"]["phases"]) - 1, config["training"]["phases"][-1], config["training"]["phases"][-1]["steps"]


def phase_lr(local_step, phase):
    step = local_step + 1
    warmup = phase["warmup_steps"]
    return phase["learning_rate"] * min(step / warmup, math.sqrt(warmup / step))


def run_identity(config, manifest, device):
    return {"schema": 2, "experiment": "scratch_translation_v2",
            "config": config, "frozen_prepared_fingerprint": digest(manifest),
            "frozen_prepared_files": manifest["files"], "implementation": implementation(),
            "versions": versions(), "device_type": device.type,
            "pretrained_weights": False, "v1_model_weights_used": False}


def train(config, device_name="auto", stop_after_steps=None):
    if stop_after_steps is not None and stop_after_steps < 1:
        raise ValueError("stop_after_steps must be positive.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("v2 supports one process/GPU only.")
    device = device_for(device_name)
    if device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    prepared_root, prepared, frames = load_data(config)
    run = writable(suite_root(config) / "run")
    run.mkdir(parents=True, exist_ok=True)
    with FileLock(str(writable(run / ".train.lock")), timeout=0):
        identity = run_identity(config, prepared, device)
        fingerprint = digest(identity)
        manifest_path = run / "run_manifest.json"
        if manifest_path.exists():
            if read_json(manifest_path) != identity:
                raise RuntimeError("v2 code/data/config/version changed. Use a new suite.")
        else:
            if (run / "latest.json").exists() or any((run / "checkpoints").glob("*.pt")):
                raise RuntimeError("Unidentified v2 checkpoints; refusing to adopt them.")
            atomic_json(manifest_path, identity)
        own_vocab = run / "vocabulary.model"
        if not own_vocab.exists():
            atomic_copy(prepared_root / "vocabulary.model", own_vocab)
        if file_hash(own_vocab) != prepared["files"]["vocabulary.model"]:
            raise RuntimeError("v2 tokenizer snapshot changed.")
        vocab = Vocabulary(own_vocab, config["model"]["max_length"])
        human = frames["human"].to_dict("records")
        teacher = frames["teacher"].to_dict("records")
        validation = frames["validation"].to_dict("records")
        for phase in config["training"]["phases"]:
            sampling_plan(config, phase["data"], human, teacher, 0)
        seed_all(config["experiment"]["seed"])
        model = TranslationTransformer(vocab.sp.get_piece_size(), config["model"], vocab.pad).to(device)
        initial_hash = state_hash(model)
        initialization = {"random_weights_sha256": initial_hash,
                          "parameters": sum(parameter.numel() for parameter in model.parameters()),
                          "seed": config["experiment"]["seed"], "pretrained_weights": False,
                          "v1_model_weights_used": False}
        if (run / "initialization.json").exists() and read_json(run / "initialization.json") != initialization:
            raise RuntimeError("v2 initialization identity changed.")
        atomic_json(run / "initialization.json", initialization)
        settings = config["training"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98),
                                      eps=1e-9, weight_decay=settings["weight_decay"])
        amp = bool(settings["amp"] and device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
        step, best_step, best_score = 0, 0, -float("inf")
        cursors = {"mixed": 0, "human": 0}
        target_tokens, elapsed = 0, 0.0
        if (run / "latest.json").exists():
            saved = read_checkpoint(run)
            if saved["fingerprint"] != fingerprint or saved["initial_hash"] != initial_hash:
                raise RuntimeError("Checkpoint is not from this v2 run.")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            restore_rng(saved["rng"])
            step, best_step, best_score = saved["step"], saved["best_step"], saved["best_score"]
            cursors, target_tokens, elapsed = saved["cursors"], saved["target_tokens"], saved["seconds"]
            print(f"Resumed v2 at step={step}, phase={saved['phase']}, best={best_step}", flush=True)
        total = total_steps(config)
        stop = min(total, step + stop_after_steps) if stop_after_steps else total
        block_size = len(DIRECTIONS) * config["sampling"]["rows_per_direction"]
        active_key, plan = None, None

        def next_rows(count, phase_name):
            nonlocal active_key, plan
            selected = []
            while len(selected) < count:
                block, position = divmod(cursors[phase_name], block_size)
                key = (phase_name, block)
                if key != active_key:
                    plan, sampling = sampling_plan(config, phase_name, human, teacher, block)
                    atomic_json(run / "sampling" / f"{phase_name}_block_{block:05d}.json", sampling)
                    active_key = key
                    print(f"Sampling {phase_name} block {block}: {len(plan):,} rows", flush=True)
                take = min(count - len(selected), block_size - position)
                selected.extend((human if kind == "human" else teacher)[index]
                                for kind, index in plan[position:position + take])
                cursors[phase_name] += take
            return selected

        started = time.monotonic()
        model.train()
        previous_phase = None
        while step < stop:
            _, phase, local_step = phase_at(config, step)
            phase_name = phase["data"]
            if phase["name"] != previous_phase:
                print(f"Entering phase {phase['name']}: global_step={step}, local_step={local_step}, "
                      f"data={phase_name}", flush=True)
                previous_phase = phase["name"]
            batches = [next_rows(settings["batch_size"], phase_name)
                       for _ in range(settings["gradient_accumulation_steps"])]
            denominator = sum(len(row["tgt_ids"]) - 1 for batch in batches for row in batch)
            lr = phase_lr(local_step, phase)
            for group in optimizer.param_groups:
                group["lr"] = lr
            before_update = rng_state(device.type == "cuda")
            for attempt in range(8):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for batch in batches:
                    source, decoder, labels = collate(batch, vocab.pad, device)
                    context = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
                    with context:
                        loss = translation_loss(model(source, decoder), labels, vocab.pad,
                                                settings["label_smoothing"])
                    scaled_loss = loss / denominator
                    if not bool(torch.isfinite(scaled_loss)):
                        raise RuntimeError("Non-finite loss; no update was committed.")
                    scaler.scale(scaled_loss).backward()
                    loss_sum += float(loss.detach())
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
                if bool(torch.isfinite(grad_norm)):
                    break
                if not amp or attempt == 7:
                    raise RuntimeError("Persistent non-finite gradients; no update was committed.")
                scaler.update(new_scale=scaler.get_scale() / 2)
                restore_rng(before_update)
                print(f"FP16 overflow: retry same batch at scale={scaler.get_scale()}", flush=True)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            target_tokens += denominator
            elapsed_now = elapsed + time.monotonic() - started
            if step % settings["log_steps"] == 0 or step == 1:
                log = {"step": step, "phase": phase["name"], "phase_step": local_step + 1,
                       "loss": loss_sum / denominator, "lr": lr,
                       "examples_seen": sum(cursors.values()), "target_tokens": target_tokens,
                       "seconds": elapsed_now}
                atomic_json(run / "logs" / f"step_{step:08d}.json", log)
                print(f"step={step}/{total} phase={phase['name']} loss={log['loss']:.4f} "
                      f"lr={lr:.3g} elapsed={elapsed_now / 60:.1f}min", flush=True)
            should_eval = step % settings["eval_steps"] == 0 or step == total
            if should_eval:
                scores = score_model(model, vocab, validation, config, device, "greedy", amp)
                atomic_json(run / "validation" / f"step_{step:08d}.json", scores)
                if scores["macro"]["chrf"] > best_score:
                    best_score, best_step = scores["macro"]["chrf"], step
                print(f"Validation macro chrF={scores['macro']['chrf']:.3f}; best={best_step}", flush=True)
                model.train()
            if should_eval or step % settings["checkpoint_steps"] == 0 or step == stop:
                payload = {"fingerprint": fingerprint, "initial_hash": initial_hash,
                           "step": step, "phase": phase["name"], "cursors": cursors,
                           "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "rng": rng_state(device.type == "cuda"),
                           "best_step": best_step, "best_score": best_score,
                           "target_tokens": target_tokens, "seconds": elapsed_now}
                save_checkpoint(run, payload, settings["keep_checkpoints"])
                print(f"Checkpoint committed: step={step}, phase={phase['name']}", flush=True)
        report = {"status": "complete" if step == total else "paused", "step": step,
                  "max_steps": total, "best_step": best_step,
                  "best_macro_chrf": best_score if best_step else None,
                  "phase": phase_at(config, min(step, total - 1))[1]["name"],
                  "examples_seen": sum(cursors.values()), "phase_cursors": cursors,
                  "target_tokens": target_tokens, "run_fingerprint": fingerprint,
                  **initialization}
        atomic_json(run / "train_report.json", report)
        return report


def evaluate(config, device_name="auto"):
    device = device_for(device_name)
    prepared_root, prepared, frames = load_data(config)
    run = suite_root(config) / "run"
    with FileLock(str(writable(run / ".train.lock")), timeout=0):
        report = read_json(run / "train_report.json")
        if report["status"] != "complete":
            raise RuntimeError("Finish all v2 phases before final evaluation.")
        identity = read_json(run / "run_manifest.json")
        if identity != run_identity(config, prepared, device):
            raise RuntimeError("v2 evaluation identity differs from training.")
        checkpoint = read_checkpoint(run, report["best_step"])
        vocab = Vocabulary(run / "vocabulary.model", config["model"]["max_length"])
        if file_hash(run / "vocabulary.model") != prepared["files"]["vocabulary.model"]:
            raise RuntimeError("v2 tokenizer changed.")
        model = TranslationTransformer(vocab.sp.get_piece_size(), config["model"], vocab.pad).to(device)
        model.load_state_dict(checkpoint["model"])
        scores = score_model(model, vocab, frames["test"].to_dict("records"), config, device,
                             "beam", config["training"]["amp"] and device.type == "cuda", True)
        result = {**scores, "split": "flores_devtest", "best_step": report["best_step"],
                  "run_fingerprint": checkpoint["fingerprint"],
                  "frozen_prepared_fingerprint": digest(prepared)}
        atomic_json(run / "evaluation" / "beam_metrics.json", result)
        print(f"v2 evaluation complete: {run / 'evaluation/beam_metrics.json'}", flush=True)
        return result
