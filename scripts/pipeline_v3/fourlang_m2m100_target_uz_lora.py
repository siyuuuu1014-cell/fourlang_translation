"""Train and gate one balanced en/zh->uz LoRA routed on M2M100 Exp4."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    set_seed,
)

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import WeightedTrainer, metrics, tokenize_rows, translate  # noqa: E402
from scripts.pipeline_v2.training_safety import atomic_json, file_sha256, fingerprint, model_signature  # noqa: E402
from scripts.pipeline_v3.fourlang_m2m100_ru_uz_lora import (  # noqa: E402
    adapter_signature,
    load_metrics,
    read_jsonl,
)
from scripts.pipeline_v3.fourlang_m2m100_student import summarize_metrics, verify_m2m100_artifact  # noqa: E402

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_target_uz_lora_v1.toml"
EXPECTED_DIRECTIONS = ("en-uz", "zh-uz")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def validate_directions(config: dict[str, Any]) -> tuple[str, ...]:
    values = tuple(str(item) for item in config["experiment"]["repair_directions"])
    if values != EXPECTED_DIRECTIONS:
        raise RuntimeError(f"This experiment is locked to {EXPECTED_DIRECTIONS}; got {values}")
    return values


def select_rows(path: Path, directions: tuple[str, ...]) -> list[dict[str, Any]]:
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    selected = []
    for index, row in enumerate(read_jsonl(path), start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing fields: {sorted(missing)}")
        if f"{row['src_lang']}-{row['tgt_lang']}" in directions:
            selected.append(row)
    counts = Counter(f"{row['src_lang']}-{row['tgt_lang']}" for row in selected)
    if set(counts) != set(directions):
        raise RuntimeError(f"Missing target-uz directions in {path}: counts={dict(counts)}")
    return selected


def balance_direction_mass(
    rows: list[dict[str, Any]], directions: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    totals: Counter[str] = Counter()
    for row in rows:
        direction = f"{row['src_lang']}-{row['tgt_lang']}"
        weight = float(row.get("weight", 1.0))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid weight for {direction}: {weight}")
        totals[direction] += weight
    if set(totals) != set(directions):
        raise RuntimeError(f"Direction mismatch while balancing: {dict(totals)}")
    target_mass = sum(totals.values()) / len(directions)
    multipliers = {direction: target_mass / totals[direction] for direction in directions}
    balanced = []
    balanced_totals: Counter[str] = Counter()
    for row in rows:
        direction = f"{row['src_lang']}-{row['tgt_lang']}"
        value = dict(row)
        value["weight"] = float(row.get("weight", 1.0)) * multipliers[direction]
        balanced.append(value)
        balanced_totals[direction] += value["weight"]
    return balanced, {
        "original_weight_sum": dict(totals),
        "normalization_multiplier": multipliers,
        "balanced_weight_sum": dict(balanced_totals),
        "target_direction_mass": target_mass,
    }


def training_manifest(config: dict[str, Any]) -> dict[str, Any]:
    base = project_path(config["base_model"]["path"])
    train = project_path(config["data"]["train"])
    validation = project_path(config["data"]["validation"])
    return {
        "schema_version": 1,
        "method": "balanced_target_language_lora_v1",
        "experiment": config["experiment"],
        "base_model": str(base.resolve()),
        "base_model_signature": model_signature(base),
        "train": str(train.resolve()),
        "train_sha256": file_sha256(train),
        "validation": str(validation.resolve()),
        "validation_sha256": file_sha256(validation),
        "lora": config["lora"],
        "training": config["training"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "datasets")
        },
    }


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    directions = validate_directions(config)
    base = project_path(config["base_model"]["path"])
    train_path = project_path(config["data"]["train"])
    validation_path = project_path(config["data"]["validation"])
    benchmark = project_path(config["benchmark"]["path"])
    baseline = project_path(config["baseline"]["exp4_metrics"])
    verify_m2m100_artifact(base)
    for item in (train_path, validation_path, benchmark, baseline):
        if not item.is_file():
            raise FileNotFoundError(item)
    train_rows = select_rows(train_path, directions)
    validation_rows = select_rows(validation_path, directions)
    _, balance = balance_direction_mass(train_rows, directions)
    frame = pd.read_parquet(benchmark, columns=["en", "zh", "uz"])
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    minimum = float(config["experiment"]["minimum_free_disk_gb"])
    if free_gb < minimum:
        raise RuntimeError(f"Need at least {minimum:g} GiB free; found {free_gb:.2f} GiB.")
    train_counts = Counter(f"{row['src_lang']}-{row['tgt_lang']}" for row in train_rows)
    validation_counts = Counter(
        f"{row['src_lang']}-{row['tgt_lang']}" for row in validation_rows
    )
    report = {
        "schema_version": 1,
        "status": "READY",
        "repair_directions": directions,
        "routing": {direction: "exp4_plus_target_uz_adapter" for direction in directions},
        "ru-uz": "existing_dedicated_adapter_unchanged",
        "all_other_directions": "exp4_base",
        "train_counts": dict(train_counts),
        "validation_counts": dict(validation_counts),
        "balance": balance,
        "benchmark_rows": len(frame),
        "free_disk_gb": free_gb,
    }
    write_json(project_path(config["outputs"]["preflight"]), report)
    print(f"M2M100_TARGET_UZ_LORA_PREFLIGHT_READY: {project_path(config['outputs']['preflight'])}", flush=True)
    return report


def train(config: dict[str, Any]) -> dict[str, Any]:
    preflight(config)
    directions = validate_directions(config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This safety-locked adapter route supports one GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter training.")
    destination = project_path(config["outputs"]["adapter"])
    manifest = training_manifest(config)
    manifest_record = {"fingerprint": fingerprint(manifest), "manifest": manifest}
    if destination.exists():
        path = destination / "route_manifest.json"
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != manifest_record:
            raise RuntimeError(f"Existing adapter differs; refusing overwrite: {destination}")
        adapter_signature(destination)
        print(f"M2M100_TARGET_UZ_LORA_TRAINING_REUSED: {destination}", flush=True)
        return json.loads(project_path(config["outputs"]["training_report"]).read_text(encoding="utf-8"))
    train_rows, balance = balance_direction_mass(
        select_rows(project_path(config["data"]["train"]), directions), directions
    )
    validation_rows, _ = balance_direction_mass(
        select_rows(project_path(config["data"]["validation"]), directions), directions
    )
    base_path = project_path(config["base_model"]["path"])
    settings = config["training"]
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    tokenizer = M2M100Tokenizer.from_pretrained(base_path, local_files_only=True)
    tokenizer.src_lang = "en"
    tokenizer.tgt_lang = "uz"
    base_model = M2M100ForConditionalGeneration.from_pretrained(
        base_path, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    lora = config["lora"]
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias="none",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    if trainable <= 0 or trainable >= total:
        raise RuntimeError(f"Invalid LoRA parameter counts: trainable={trainable}, total={total}")
    runtime_config = {"training": settings}
    train_data = tokenize_rows(train_rows, tokenizer, "m2m100", runtime_config)
    validation_data = tokenize_rows(validation_rows, tokenizer, "m2m100", runtime_config)
    checkpoint_dir = destination.parent / "checkpoints"
    if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise RuntimeError(f"Partial prior run exists; inspect before retrying: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(settings["epochs"]),
        per_device_train_batch_size=int(settings["batch_size"]),
        per_device_eval_batch_size=int(settings["batch_size"]),
        gradient_accumulation_steps=int(settings["gradient_accumulation_steps"]),
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        warmup_ratio=float(settings["warmup_ratio"]),
        eval_strategy="epoch",
        save_strategy="no",
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=int(settings["dataloader_num_workers"]),
        group_by_length=True,
        optim=str(settings["optim"]),
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.model_accepts_loss_kwargs = False
    print(
        f"M2M100_TARGET_UZ_LORA_TRAINING: rows={len(train_rows)} epochs={settings['epochs']} "
        f"trainable={trainable} total={total}",
        flush=True,
    )
    started = time.time()
    result = trainer.train()
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Partial temporary adapter exists: {temporary}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    atomic_json(temporary / "route_manifest.json", manifest_record)
    adapter_signature(temporary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(destination)
    report = {
        "schema_version": 1,
        "status": "TRAINED_NOT_PROMOTED",
        "repair_directions": directions,
        "adapter": str(destination.resolve()),
        "base_model": str(base_path.resolve()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "direction_balance": balance,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_loss": float(result.training_loss),
        "seconds": time.time() - started,
        "run_fingerprint": manifest_record["fingerprint"],
    }
    write_json(project_path(config["outputs"]["training_report"]), report)
    print(f"M2M100_TARGET_UZ_LORA_TRAINING_COMPLETE: {destination}", flush=True)
    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def load_model(config: dict[str, Any], source: str, target: str):
    route = f"{source}-{target}"
    directions = validate_directions(config)
    if route not in directions:
        raise RuntimeError(f"Target-uz adapter is not routed for {route}")
    base = project_path(config["base_model"]["path"])
    adapter = project_path(config["outputs"]["adapter"])
    adapter_signature(adapter)
    tokenizer = M2M100Tokenizer.from_pretrained(base, local_files_only=True)
    model = M2M100ForConditionalGeneration.from_pretrained(
        base,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed benchmark evaluation.")
    directions = validate_directions(config)
    destination = project_path(config["outputs"]["metrics"])
    benchmark = project_path(config["benchmark"]["path"])
    signature = fingerprint(
        {
            "base": model_signature(project_path(config["base_model"]["path"])),
            "adapter": adapter_signature(project_path(config["outputs"]["adapter"])),
            "benchmark": file_sha256(benchmark),
            "deployment": config["deployment"],
            "routes": directions,
        }
    )
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("evaluation_signature") != signature:
            raise RuntimeError(f"Existing evaluation differs; refusing overwrite: {destination}")
        print(f"M2M100_TARGET_UZ_LORA_EVALUATION_REUSED: {destination}", flush=True)
        return payload
    frame = pd.read_parquet(benchmark, columns=["en", "zh", "uz"])
    exp4 = load_metrics(project_path(config["baseline"]["exp4_metrics"]))
    values = {key: dict(value) for key, value in exp4["metrics"].items()}
    for direction in directions:
        source, target = direction.split("-")
        tokenizer, model = load_model(config, source, target)
        try:
            predictions = translate(
                tokenizer,
                model,
                "m2m100",
                source,
                target,
                frame[source].fillna("").astype(str).tolist(),
                {"training": config["training"], "deployment": config["deployment"]},
            )
            values[direction] = metrics(
                predictions, frame[target].fillna("").astype(str).tolist(), target
            )
            print(f"Target-uz evaluated {direction}: {values[direction]}", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "status": "EVALUATED_ROUTED_SYSTEM_NOT_PROMOTED",
        "evaluation_signature": signature,
        "routing": {direction: "exp4_plus_target_uz_adapter" for direction in directions},
        "non_repair_routing": "exp4_base; ru-uz keeps its separate approved adapter",
        "metrics": values,
        "summary": summarize_metrics(values),
    }
    write_json(destination, payload)
    print(f"M2M100_TARGET_UZ_LORA_EVALUATION_COMPLETE: {destination}", flush=True)
    return payload


def compare(config: dict[str, Any]) -> dict[str, Any]:
    candidate = evaluate(config)
    exp4 = load_metrics(project_path(config["baseline"]["exp4_metrics"]))
    deltas = {
        direction: candidate["metrics"][direction]["chrf2"]
        - exp4["metrics"][direction]["chrf2"]
        for direction in EXPECTED_DIRECTIONS
    }
    settings = config["selection"]
    constraints = {
        "zh_uz_recovery_from_exp4_at_least_minimum": deltas["zh-uz"]
        >= float(settings["minimum_zh_uz_recovery_from_exp4"]),
        "en_uz_regression_from_exp4_at_most_limit": -deltas["en-uz"]
        <= float(settings["maximum_en_uz_regression_from_exp4"]),
        "macro_chrf2_at_least_minimum": candidate["summary"]["macro_chrf2"]
        >= float(settings["minimum_macro_chrf2"]),
        "worst_chrf2_at_least_minimum": candidate["summary"]["worst_chrf2"]
        >= float(settings["minimum_worst_chrf2"]),
        "non_repair_directions_are_exact_exp4_routes": True,
    }
    report = {
        "schema_version": 1,
        "status": "CANDIDATE_PASSED" if all(constraints.values()) else "CANDIDATE_REJECTED",
        "candidate": candidate,
        "baseline": exp4,
        "chrf2_deltas_from_exp4": deltas,
        "constraints": constraints,
        "promotion_performed": False,
    }
    destination = project_path(config["outputs"]["comparison"])
    write_json(destination, report)
    print(f"M2M100_TARGET_UZ_LORA_COMPARISON_READY: {destination} status={report['status']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "evaluate", "compare", "run-all"))
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "preflight":
        preflight(config)
    elif args.action == "train":
        train(config)
    elif args.action == "evaluate":
        evaluate(config)
    elif args.action == "compare":
        compare(config)
    else:
        train(config)
        evaluate(config)
        compare(config)


if __name__ == "__main__":
    main()
