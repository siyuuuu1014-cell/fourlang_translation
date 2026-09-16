"""Train and regression-gate a ru->uz-only LoRA routed on top of M2M100 Exp4."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import shutil
import sys
import time
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
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    WeightedTrainer,
    metrics,
    tokenize_rows,
    translate,
)
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    atomic_json,
    file_sha256,
    fingerprint,
    model_signature,
)
from scripts.pipeline_v3.fourlang_flow import directions  # noqa: E402
from scripts.pipeline_v3.fourlang_m2m100_student import (  # noqa: E402
    summarize_metrics,
    verify_m2m100_artifact,
)

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_ru_uz_lora_v1.toml"
EXPECTED_DIRECTION = "ru-uz"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def repair_rows(path: Path) -> list[dict[str, Any]]:
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    selected = []
    for index, row in enumerate(read_jsonl(path), start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing fields: {sorted(missing)}")
        if f"{row['src_lang']}-{row['tgt_lang']}" == EXPECTED_DIRECTION:
            selected.append(row)
    if not selected:
        raise RuntimeError(f"No {EXPECTED_DIRECTION} rows in {path}")
    return selected


def adapter_signature(path: Path) -> dict[str, str]:
    required = ("adapter_config.json", "adapter_model.safetensors", "route_manifest.json")
    result = {}
    for name in required:
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            raise FileNotFoundError(f"Adapter artifact is missing or empty: {item}")
        result[name] = file_sha256(item)
    return result


def load_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("metrics", payload)
    if set(values) != set(directions()):
        raise RuntimeError(f"Expected exactly 12 directions in {path}")
    return {"metrics": values, "summary": payload.get("summary", summarize_metrics(values))}


def training_manifest(config: dict[str, Any]) -> dict[str, Any]:
    base = project_path(config["base_model"]["path"])
    train = project_path(config["data"]["train"])
    validation = project_path(config["data"]["validation"])
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "method": "direction_routed_lora_v1",
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
    if config["experiment"]["repair_direction"] != EXPECTED_DIRECTION:
        raise RuntimeError("This route is intentionally locked to ru-uz.")
    base = project_path(config["base_model"]["path"])
    verify_m2m100_artifact(base)
    train = project_path(config["data"]["train"])
    validation = project_path(config["data"]["validation"])
    benchmark = project_path(config["benchmark"]["path"])
    exp4_metrics = project_path(config["baselines"]["exp4_metrics"])
    targeted_metrics = project_path(config["baselines"]["targeted_v2_metrics"])
    for item in (train, validation, benchmark, exp4_metrics, targeted_metrics):
        if not item.is_file():
            raise FileNotFoundError(item)
    train_rows = repair_rows(train)
    validation_rows = repair_rows(validation)
    frame = pd.read_parquet(benchmark, columns=["ru", "uz"])
    if frame.empty:
        raise RuntimeError("The fixed benchmark is empty.")
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    minimum = float(config["experiment"]["minimum_free_disk_gb"])
    if free_gb < minimum:
        raise RuntimeError(f"Need at least {minimum:g} GiB free; found {free_gb:.2f} GiB.")
    report = {
        "schema_version": 1,
        "status": "READY",
        "repair_direction": EXPECTED_DIRECTION,
        "routing": {"ru-uz": "exp4_plus_adapter", "all_other_directions": "exp4_base"},
        "base_model": str(base.resolve()),
        "train": str(train.resolve()),
        "train_rows": len(train_rows),
        "validation": str(validation.resolve()),
        "validation_rows": len(validation_rows),
        "benchmark": str(benchmark.resolve()),
        "benchmark_rows": len(frame),
        "free_disk_gb": free_gb,
        "minimum_free_disk_gb": minimum,
        "planned_optimizer_steps": math.ceil(
            len(train_rows)
            / (int(config["training"]["batch_size"]) * int(config["training"]["gradient_accumulation_steps"]))
            * float(config["training"]["epochs"])
        ),
    }
    write_json(project_path(config["outputs"]["preflight"]), report)
    print(f"M2M100_RU_UZ_LORA_PREFLIGHT_READY: {project_path(config['outputs']['preflight'])}", flush=True)
    return report


def train(config: dict[str, Any]) -> dict[str, Any]:
    preflight(config)
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This safety-locked adapter route supports one GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for adapter training.")
    destination = project_path(config["outputs"]["adapter"])
    manifest = training_manifest(config)
    run_fingerprint = fingerprint(manifest)
    manifest_record = {"fingerprint": run_fingerprint, "manifest": manifest}
    if destination.exists():
        route_manifest = destination / "route_manifest.json"
        if not route_manifest.is_file() or json.loads(route_manifest.read_text(encoding="utf-8")) != manifest_record:
            raise RuntimeError(f"Existing adapter differs; refusing overwrite: {destination}")
        adapter_signature(destination)
        print(f"M2M100_RU_UZ_LORA_TRAINING_REUSED: {destination}", flush=True)
        return json.loads(project_path(config["outputs"]["training_report"]).read_text(encoding="utf-8"))

    base_path = project_path(config["base_model"]["path"])
    train_rows = repair_rows(project_path(config["data"]["train"]))
    validation_rows = repair_rows(project_path(config["data"]["validation"]))
    settings = config["training"]
    set_seed(int(config["experiment"]["seed"]))
    tokenizer = M2M100Tokenizer.from_pretrained(base_path, local_files_only=True)
    tokenizer.src_lang = "ru"
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
        seed=int(config["experiment"]["seed"]),
        data_seed=int(config["experiment"]["seed"]),
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
        f"M2M100_RU_UZ_LORA_TRAINING: rows={len(train_rows)} epochs={settings['epochs']} "
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
        "repair_direction": EXPECTED_DIRECTION,
        "adapter": str(destination.resolve()),
        "base_model": str(base_path.resolve()),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "training_loss": float(result.training_loss),
        "seconds": time.time() - started,
        "run_fingerprint": run_fingerprint,
    }
    write_json(project_path(config["outputs"]["training_report"]), report)
    print(f"M2M100_RU_UZ_LORA_TRAINING_COMPLETE: {destination}", flush=True)
    del trainer, model, base_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def load_routed_model(config: dict[str, Any], source: str, target: str):
    base_path = project_path(config["base_model"]["path"])
    tokenizer = M2M100Tokenizer.from_pretrained(base_path, local_files_only=True)
    model = M2M100ForConditionalGeneration.from_pretrained(
        base_path,
        local_files_only=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    route = f"{source}-{target}"
    if route == EXPECTED_DIRECTION:
        adapter = project_path(config["outputs"]["adapter"])
        adapter_signature(adapter)
        model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model, route == EXPECTED_DIRECTION


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed benchmark evaluation.")
    destination = project_path(config["outputs"]["metrics"])
    adapter = project_path(config["outputs"]["adapter"])
    signature = fingerprint(
        {
            "base": model_signature(project_path(config["base_model"]["path"])),
            "adapter": adapter_signature(adapter),
            "benchmark": file_sha256(project_path(config["benchmark"]["path"])),
            "deployment": config["deployment"],
            "routing": {EXPECTED_DIRECTION: "adapter", "default": "base"},
        }
    )
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("evaluation_signature") != signature:
            raise RuntimeError(f"Existing evaluation differs; refusing overwrite: {destination}")
        print(f"M2M100_RU_UZ_LORA_EVALUATION_REUSED: {destination}", flush=True)
        return payload
    exp4 = load_metrics(project_path(config["baselines"]["exp4_metrics"]))
    frame = pd.read_parquet(project_path(config["benchmark"]["path"]), columns=["ru", "uz"])
    tokenizer, model, adapter_active = load_routed_model(config, "ru", "uz")
    if not adapter_active:
        raise RuntimeError("ru-uz evaluation did not activate the adapter.")
    try:
        predictions = translate(
            tokenizer,
            model,
            "m2m100",
            "ru",
            "uz",
            frame["ru"].fillna("").astype(str).tolist(),
            config,
        )
        repaired = metrics(predictions, frame["uz"].fillna("").astype(str).tolist(), "uz")
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    values = {key: dict(value) for key, value in exp4["metrics"].items()}
    values[EXPECTED_DIRECTION] = repaired
    payload = {
        "schema_version": 1,
        "status": "EVALUATED_ROUTED_SYSTEM_NOT_PROMOTED",
        "evaluation_signature": signature,
        "routing": {EXPECTED_DIRECTION: "exp4_plus_adapter", "all_other_directions": "exp4_base"},
        "metrics": values,
        "summary": summarize_metrics(values),
        "metric_provenance": {
            EXPECTED_DIRECTION: "evaluated_live_with_adapter_on_fixed_benchmark",
            "all_other_directions": "exact_exp4_fixed_benchmark_metrics_reused_because_adapter_is_not_loaded",
        },
    }
    write_json(destination, payload)
    print(f"M2M100_RU_UZ_LORA_EVALUATION_COMPLETE: {destination}", flush=True)
    return payload


def compare(config: dict[str, Any]) -> dict[str, Any]:
    candidate = load_metrics(project_path(config["outputs"]["metrics"]))
    exp4 = load_metrics(project_path(config["baselines"]["exp4_metrics"]))
    targeted = load_metrics(project_path(config["baselines"]["targeted_v2_metrics"]))
    repair = candidate["metrics"][EXPECTED_DIRECTION]["chrf2"] - exp4["metrics"][EXPECTED_DIRECTION]["chrf2"]
    settings = config["selection"]
    constraints = {
        "ru_uz_recovery_from_exp4_at_least_minimum": repair
        >= float(settings["minimum_ru_uz_recovery_from_exp4"]),
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
        "baselines": {"exp4": exp4, "targeted_v2": targeted},
        "deltas": {
            "ru_uz_chrf2_from_exp4": repair,
            "ru_uz_chrf2_from_targeted_v2": candidate["metrics"][EXPECTED_DIRECTION]["chrf2"]
            - targeted["metrics"][EXPECTED_DIRECTION]["chrf2"],
            "macro_chrf2_from_exp4": candidate["summary"]["macro_chrf2"]
            - exp4["summary"]["macro_chrf2"],
        },
        "constraints": constraints,
        "promotion_performed": False,
    }
    destination = project_path(config["outputs"]["comparison"])
    write_json(destination, report)
    print(f"M2M100_RU_UZ_LORA_COMPARISON_READY: {destination} status={report['status']}", flush=True)
    return report


def translate_text(config: dict[str, Any], source: str, target: str, text: str) -> None:
    if source == target or source not in {"en", "zh", "uz", "ru"} or target not in {"en", "zh", "uz", "ru"}:
        raise ValueError("source and target must be different members of en, zh, uz, ru")
    tokenizer, model, adapter_active = load_routed_model(config, source, target)
    try:
        output = translate(tokenizer, model, "m2m100", source, target, [text], config)[0]
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(json.dumps({"source": source, "target": target, "adapter_active": adapter_active, "translation": output}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "train", "evaluate", "compare", "run-all", "translate"))
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--text")
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
    elif args.action == "run-all":
        train(config)
        evaluate(config)
        compare(config)
    else:
        if not args.source or not args.target or args.text is None:
            parser.error("translate requires --source, --target, and --text")
        translate_text(config, args.source, args.target, args.text)


if __name__ == "__main__":
    main()
