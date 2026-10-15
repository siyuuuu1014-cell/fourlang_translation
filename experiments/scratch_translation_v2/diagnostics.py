"""Read-only diagnostics for the frozen v1 runs; all new output belongs to v2."""

from __future__ import annotations

import torch
from filelock import FileLock

from .common import PACKAGE, PROJECT, atomic_json, file_hash, read_json, suite_root, writable
from .data import Vocabulary, load_data
from .engine import device_for
from .metrics import score_model
from .model import TranslationTransformer

GROUPS = ("human_only", "human_kd")


def _v1_root():
    root = (PROJECT / "experiments/scratch_translation/artifacts/fourlang_scratch_v1").resolve()
    boundary = (PROJECT / "experiments/scratch_translation/artifacts").resolve()
    if not root.is_relative_to(boundary) or not root.is_dir():
        raise FileNotFoundError(f"Frozen v1 artifacts are missing: {root}")
    return root


def _load_v1(group, device, prepared):
    run = _v1_root() / "runs" / group
    report = read_json(run / "train_report.json")
    if report.get("status") != "complete" or not report.get("best_step"):
        raise RuntimeError(f"Frozen v1 {group} is not complete.")
    metadata = read_json(run / "checkpoints" / f"step_{report['best_step']:08d}.json")
    checkpoint_root = (run / "checkpoints").resolve()
    checkpoint_path = (checkpoint_root / metadata["file"]).resolve()
    if not checkpoint_path.is_relative_to(checkpoint_root):
        raise RuntimeError("Invalid frozen v1 checkpoint path.")
    if file_hash(checkpoint_path) != metadata["sha256"]:
        raise RuntimeError(f"Frozen v1 {group} checkpoint is corrupted.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["fingerprint"] != metadata["fingerprint"]:
        raise RuntimeError(f"Frozen v1 {group} checkpoint identity mismatch.")
    identity = read_json(run / "run_manifest.json")
    if identity.get("pretrained_weights") is not False:
        raise RuntimeError("v1 diagnostic accepts only the random-initialized experiment.")
    model_config = identity["config"]["model"]
    if file_hash(run / "vocabulary.model") != prepared["files"]["vocabulary.model"]:
        raise RuntimeError(f"Frozen v1 {group} tokenizer changed.")
    vocab = Vocabulary(run / "vocabulary.model", model_config["max_length"])
    model = TranslationTransformer(vocab.sp.get_piece_size(), model_config, vocab.pad).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, vocab, report, metadata


def diagnose_v1(config, device_name="auto"):
    """Select one decoder on v1 validation, then evaluate both arms on test."""
    device = device_for(device_name)
    _, prepared, frames = load_data(config)
    output = writable(suite_root(config) / "v1_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(writable(output / ".diagnostics.lock")), timeout=0):
        validation = frames["validation"].to_dict("records")
        test = frames["test"].to_dict("records")
        validation_scores = {}
        loaded = {}
        for group in GROUPS:
            model, vocab, report, metadata = _load_v1(group, device, prepared)
            loaded[group] = (model, vocab, report, metadata)
            validation_scores[group] = {}
            for decoding in ("greedy", "beam"):
                scores = score_model(model, vocab, validation, config, device, decoding,
                                     config["training"]["amp"] and device.type == "cuda", True)
                validation_scores[group][decoding] = scores
                atomic_json(output / group / f"validation_{decoding}.json", scores)
        averages = {
            decoding: sum(validation_scores[group][decoding]["macro"]["chrf"]
                          for group in GROUPS) / len(GROUPS)
            for decoding in ("greedy", "beam")
        }
        selected = max(averages, key=averages.get)
        selection = {"criterion": "mean validation macro character-chrF across both v1 arms",
                     "selected_decoder": selected, "scores": averages,
                     "same_decoder_for_both_arms": True}
        atomic_json(output / "decoder_selection.json", selection)
        final = {}
        for group, (model, vocab, report, metadata) in loaded.items():
            scores = score_model(model, vocab, test, config, device, selected,
                                 config["training"]["amp"] and device.type == "cuda", True)
            final[group] = {**scores, "group": group, "split": "flores_devtest",
                            "best_step": report["best_step"],
                            "checkpoint_sha256": metadata["sha256"],
                            "frozen_prepared_manifest": prepared["files"]}
            atomic_json(output / group / "test_selected_decoder.json", final[group])
        comparison = {"selection": selection,
                      "human_only": final["human_only"]["macro"],
                      "human_kd": final["human_kd"]["macro"],
                      "delta_human_kd_minus_human_only": {
                          key: final["human_kd"]["macro"][key] - final["human_only"]["macro"][key]
                          for key in ("bleu", "chrf", "chrf2", "loss")},
                      "writes_to_v1": False,
                      "v2_diagnostics_source": str(PACKAGE)}
        atomic_json(output / "comparison.json", comparison)
        print(f"v1 decoding diagnostic complete: {output / 'comparison.json'}", flush=True)
        return comparison
