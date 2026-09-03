from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

try:
    from .common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        parquet_columns,
        read_json,
        write_json,
    )
except ImportError:
    from common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        parquet_columns,
        read_json,
        write_json,
    )

sys.path.insert(0, str(PROJECT_ROOT))
from src.model_utils import load_tokenizer as load_project_tokenizer  # noqa: E402


def resolve_model_reference(local: str, repo_id: str, revision: str) -> str:
    if Path(local).exists():
        return local
    return snapshot_download(repo_id=repo_id, revision=revision, local_files_only=True)


def candidate_path(candidate: dict[str, Any], source: str, target: str) -> str:
    if candidate["family"] == "marian_pair":
        local = str(candidate[f"{source}_{target}_path"])
        return resolve_model_reference(
            local,
            str(candidate[f"{source}_{target}_repo_id"]),
            str(candidate[f"{source}_{target}_revision"]),
        )
    local = str(candidate["path"])
    return resolve_model_reference(
        local, str(candidate["repo_id"]), str(candidate["revision"])
    )


def load_model(
    candidate: dict[str, Any], source: str, target: str, *, training: bool = False
):
    family = str(candidate["family"])
    path = candidate_path(candidate, source, target)
    dtype = torch.float32 if training else torch.float16
    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    if not training:
        kwargs["torch_dtype"] = dtype
    if family == "small100":
        tokenizer = load_project_tokenizer(path, "small100")
        model = M2M100ForConditionalGeneration.from_pretrained(path, **kwargs)
    elif family == "m2m100":
        tokenizer = M2M100Tokenizer.from_pretrained(path, local_files_only=True)
        model = M2M100ForConditionalGeneration.from_pretrained(path, **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, use_fast=False
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(path, **kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def prepare_inputs(
    tokenizer: Any,
    family: str,
    source: str,
    target: str,
    texts: list[str],
    max_length: int,
):
    generation: dict[str, Any] = {}
    prepared = texts
    if family == "small100":
        tokenizer.tgt_lang = target
    elif family == "m2m100":
        tokenizer.src_lang = source
        generation["forced_bos_token_id"] = tokenizer.get_lang_id(target)
    elif family == "madlad":
        prepared = [f"<2{target}> {text}" for text in texts]
    encoded = tokenizer(
        prepared,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return encoded, generation


def translate(
    tokenizer: Any,
    model: Any,
    family: str,
    source: str,
    target: str,
    texts: list[str],
    config: dict,
) -> list[str]:
    batch_size = 8
    output: list[str] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded, generation = prepare_inputs(
            tokenizer,
            family,
            source,
            target,
            batch,
            int(config["training"]["max_source_length"]),
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **encoded,
                **generation,
                do_sample=False,
                num_beams=int(config["deployment"]["num_beams"]),
                max_new_tokens=int(config["deployment"]["max_new_tokens"]),
            )
        output.extend(
            text.strip()
            for text in tokenizer.batch_decode(tokens, skip_special_tokens=True)
        )
    return output


def metrics(
    predictions: list[str], references: list[str], target: str
) -> dict[str, float]:
    tokenizer = "13a"
    return {
        "bleu": float(
            BLEU(tokenize=tokenizer).corpus_score(predictions, [references]).score
        ),
        "chrf2": float(
            CHRF(word_order=2).corpus_score(predictions, [references]).score
        ),
        "samples": len(predictions),
    }


def benchmark_frames(config: dict, group: str) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(PROJECT_ROOT / value)
        for name, value in config["benchmarks"][group].items()
    }


def evaluate_candidate(candidate: dict[str, Any], config: dict) -> dict[str, Any]:
    _, left, right, _ = pair_info(config)
    all_metrics = {}
    parameters = 0
    for source, target in ((left, right), (right, left)):
        tokenizer, model = load_model(candidate, source, target)
        parameters = max(
            parameters, sum(parameter.numel() for parameter in model.parameters())
        )
        predictions, references = [], []
        for frame in benchmark_frames(config, "selection").values():
            source_column, target_column = parquet_columns(frame, left, right)
            if source == right:
                source_column, target_column = target_column, source_column
            texts = frame[source_column].fillna("").astype(str).tolist()
            refs = frame[target_column].fillna("").astype(str).tolist()
            predictions.extend(
                translate(
                    tokenizer, model, candidate["family"], source, target, texts, config
                )
            )
            references.extend(refs)
        all_metrics[f"{source}-{target}"] = metrics(predictions, references, target)
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"status": "ok", "parameters": parameters, "metrics": all_metrics}


def bakeoff(config: dict, role: str) -> None:
    pair, _, _, _ = pair_info(config)
    results: dict[str, Any] = {}
    for candidate in commercial_candidates(config, role):
        try:
            result = evaluate_candidate(candidate, config)
            if role == "student" and result["parameters"] > int(
                config["selection"]["max_student_parameters"]
            ):
                result = {**result, "status": "ineligible_size"}
        except Exception as error:
            result = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        results[candidate["id"]] = result
    write_json(
        PROJECT_ROOT / "results" / "model_selection" / pair / f"{role}_scores.json",
        {"schema_version": 1, "role": role, "candidates": results},
    )


def load_selected(config: dict, role: str) -> dict[str, Any]:
    pair, _, _, _ = pair_info(config)
    return dict(
        read_json(
            PROJECT_ROOT
            / "results"
            / "model_selection"
            / pair
            / f"selected_{role}.json"
        )
    )


def selected_for(selection: dict[str, Any], source: str, target: str) -> dict[str, Any]:
    return dict(selection["directions"][f"{source}-{target}"]["candidate"])


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class WeightedTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)
        outputs = model(**inputs)
        if weights is None:
            loss = outputs.loss
        else:
            labels = inputs["labels"]
            logits = outputs.logits
            token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
            mask = labels.ne(-100)
            sample_loss = (token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)
            loss = (
                sample_loss * weights.to(sample_loss.device)
            ).sum() / weights.sum().clamp_min(1e-8)
        return (loss, outputs) if return_outputs else loss


def tokenize_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    family: str,
    config: dict,
    source: str | None = None,
    target: str | None = None,
) -> Dataset:
    selected = (
        rows
        if source is None
        else [
            row
            for row in rows
            if row["src_lang"] == source and row["tgt_lang"] == target
        ]
    )
    if not selected:
        raise RuntimeError(
            f"No training rows for {source or 'shared'}-{target or 'bidirectional'}."
        )
    max_source = int(config["training"]["max_source_length"])
    max_target = int(config["training"]["max_target_length"])

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        source_text = row["src_text"]
        row_source = row["src_lang"]
        row_target = row["tgt_lang"]
        if family == "small100":
            tokenizer.tgt_lang = row_target
        elif family == "m2m100":
            tokenizer.src_lang = row_source
            tokenizer.tgt_lang = row_target
        elif family == "madlad":
            source_text = f"<2{row_target}> {source_text}"
        encoded = tokenizer(source_text, truncation=True, max_length=max_source)
        encoded["labels"] = tokenizer(
            text_target=row["tgt_text"], truncation=True, max_length=max_target
        )["input_ids"]
        encoded["weight"] = float(row.get("weight", 1.0))
        return encoded

    return Dataset.from_list(selected).map(
        encode, remove_columns=list(selected[0].keys())
    )


def train_model(
    candidate: dict[str, Any],
    source_model: str,
    source: str,
    target: str,
    train_rows: list[dict],
    validation_rows: list[dict],
    destination: Path,
    config: dict,
    *,
    shared: bool = False,
) -> dict[str, Any]:
    runtime_candidate = {**candidate, "path": source_model}
    if candidate["family"] == "marian_pair":
        runtime_candidate[f"{source}_{target}_path"] = source_model
    tokenizer, model = load_model(runtime_candidate, source, target, training=True)
    train_data = tokenize_rows(
        train_rows,
        tokenizer,
        candidate["family"],
        config,
        None if shared else source,
        None if shared else target,
    )
    validation_data = tokenize_rows(
        validation_rows,
        tokenizer,
        candidate["family"],
        config,
        None if shared else source,
        None if shared else target,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        destination.parent.parent
        / "checkpoints"
        / ("shared" if shared else f"{source}_{target}")
    )
    settings = config["training"]
    seed = int(config["direction"]["seed"])
    set_seed(seed)
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
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
        gradient_checkpointing=bool(settings.get("gradient_checkpointing", False)),
    )
    trainer = WeightedTrainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=int(settings["early_stopping_patience"])
            )
        ],
    )
    started = time.time()
    result = trainer.train()
    trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))
    if candidate["family"] == "small100":
        tokenizer_source = Path(source_model) / "tokenization_small100.py"
        if not tokenizer_source.is_file():
            raise FileNotFoundError(
                f"SMaLL-100 tokenizer implementation is missing: {tokenizer_source}"
            )
        shutil.copy2(tokenizer_source, destination / tokenizer_source.name)
    return {
        "direction": "shared_bidirectional" if shared else f"{source}-{target}",
        "train_samples": len(train_data),
        "validation_samples": len(validation_data),
        "train_loss": float(result.training_loss),
        "seconds": time.time() - started,
        "model": str(destination),
        "seed": seed,
    }


def train(config: dict, experiment: str) -> None:
    pair, left, right, version = pair_info(config)
    selection = load_selected(config, "student")
    if experiment == "exp1":
        data_root = PROJECT_ROOT / "data" / "splits" / pair / version
    else:
        data_root = PROJECT_ROOT / "data" / "distillation" / pair / version
    train_rows = load_jsonl(data_root / "train.jsonl")
    validation_rows = load_jsonl(data_root / "validation.jsonl")
    root = PROJECT_ROOT / "results" / "student" / pair / experiment
    reports = []
    shared = selection["training_layout"] == "shared_bidirectional"
    if shared:
        candidate = selected_for(selection, left, right)
        source_model = (
            candidate_path(candidate, left, right)
            if experiment == "exp1"
            else str(
                PROJECT_ROOT
                / "results"
                / "student"
                / pair
                / "exp1"
                / "best_model"
                / "shared"
            )
        )
        reports.append(
            train_model(
                candidate,
                source_model,
                left,
                right,
                train_rows,
                validation_rows,
                root / "best_model" / "shared",
                config,
                shared=True,
            )
        )
    else:
        for source, target in ((left, right), (right, left)):
            candidate = selected_for(selection, source, target)
            source_model = (
                candidate_path(candidate, source, target)
                if experiment == "exp1"
                else str(
                    PROJECT_ROOT
                    / "results"
                    / "student"
                    / pair
                    / "exp1"
                    / "best_model"
                    / f"{source}_{target}"
                )
            )
            reports.append(
                train_model(
                    candidate,
                    source_model,
                    source,
                    target,
                    train_rows,
                    validation_rows,
                    root / "best_model" / f"{source}_{target}",
                    config,
                )
            )
    artifact_names = ["shared"] if shared else [f"{left}_{right}", f"{right}_{left}"]
    write_json(
        root / "model_layout.json",
        {
            "pair": pair,
            "experiment": experiment,
            "layout": selection["training_layout"],
            "directions": selection["directions"],
            "artifacts": [
                str(
                    (root / "best_model" / name / "config.json").relative_to(
                        PROJECT_ROOT
                    )
                )
                for name in artifact_names
            ],
        },
    )
    write_json(
        root / "train_report.json",
        {
            "experiment": experiment,
            "full_parameter_finetuning": True,
            "distillation": experiment == "exp2",
            "layout": selection["training_layout"],
            "models": reports,
        },
    )


def evaluate_experiment(config: dict, experiment: str) -> None:
    pair, left, right, _ = pair_info(config)
    selection = load_selected(config, "student")
    result = {}
    for source, target in ((left, right), (right, left)):
        candidate = selected_for(selection, source, target)
        model_name = (
            "shared"
            if selection["training_layout"] == "shared_bidirectional"
            else f"{source}_{target}"
        )
        path = (
            PROJECT_ROOT
            / "results"
            / "student"
            / pair
            / experiment
            / "best_model"
            / model_name
        )
        runtime = {**candidate, "path": str(path), f"{source}_{target}_path": str(path)}
        tokenizer, model = load_model(runtime, source, target)
        predictions, references, per_benchmark = [], [], {}
        for benchmark_name, frame in benchmark_frames(config, "final").items():
            source_column, target_column = parquet_columns(frame, left, right)
            if source == right:
                source_column, target_column = target_column, source_column
            texts = frame[source_column].fillna("").astype(str).tolist()
            benchmark_predictions = translate(
                tokenizer, model, candidate["family"], source, target, texts, config
            )
            benchmark_references = frame[target_column].fillna("").astype(str).tolist()
            per_benchmark[benchmark_name] = metrics(
                benchmark_predictions, benchmark_references, target
            )
            predictions.extend(benchmark_predictions)
            references.extend(benchmark_references)
        result[f"{source}-{target}"] = {
            **metrics(predictions, references, target),
            "benchmarks": per_benchmark,
        }
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        PROJECT_ROOT / "results" / "evaluation" / pair / experiment / "metrics.json",
        result,
    )


def generate_teacher(config: dict) -> None:
    pair, _, _, _ = pair_info(config)
    selection = load_selected(config, "teacher")
    rows = load_jsonl(
        PROJECT_ROOT / "data" / "pipeline_v2" / pair / "kd_candidates.jsonl"
    )
    output = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["src_lang"], row["tgt_lang"]), []).append(row)
    for (source, target), direction_rows in grouped.items():
        candidate = selected_for(selection, source, target)
        tokenizer, model = load_model(candidate, source, target)
        generated = translate(
            tokenizer,
            model,
            candidate["family"],
            source,
            target,
            [row["src_text"] for row in direction_rows],
            config,
        )
        for row, teacher_text in zip(direction_rows, generated, strict=True):
            output.append(
                {**row, "teacher_text": teacher_text, "teacher_id": candidate["id"]}
            )
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    path = PROJECT_ROOT / "data" / "pipeline_v2" / pair / "teacher_generated.parquet"
    pd.DataFrame(output).to_parquet(path, index=False)


def freeze(config: dict) -> None:
    pair, left, right, _ = pair_info(config)
    gate = read_json(
        PROJECT_ROOT / "results" / "evaluation" / pair / "promotion_gate.json"
    )
    if gate.get("status") != "PASS":
        raise RuntimeError("Refusing to freeze: promotion gate did not pass.")
    selection = load_selected(config, "student")
    destination = PROJECT_ROOT / config["deployment"]["destination"]
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Stale freeze staging directory exists: {staging}")
    staging.mkdir(parents=True)
    shared = selection["training_layout"] == "shared_bidirectional"
    if shared:
        shutil.copytree(
            PROJECT_ROOT
            / "results"
            / "student"
            / pair
            / "exp2"
            / "best_model"
            / "shared",
            staging / "shared",
        )
    else:
        for source, target in ((left, right), (right, left)):
            shutil.copytree(
                PROJECT_ROOT
                / "results"
                / "student"
                / pair
                / "exp2"
                / "best_model"
                / f"{source}_{target}",
                staging / f"{source}_{target}",
            )
    backup = None
    if destination.exists():
        backup = destination.with_name(
            destination.name + ".backup-" + str(int(time.time()))
        )
        destination.rename(backup)
    try:
        staging.rename(destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    registry_path = PROJECT_ROOT / config["deployment"]["registry"]
    registry = read_json(registry_path)
    for source, target in ((left, right), (right, left)):
        candidate = selected_for(selection, source, target)
        registry_architecture = (
            "small100"
            if candidate["family"] == "small100"
            else ("m2m100" if candidate["family"] == "m2m100" else "marian")
        )
        key = f"{source}_{target}"
        registry["models"][key] = {
            "model_name": config["deployment"]["model_name"],
            "architecture": registry_architecture,
            "path": str(
                (
                    Path(config["deployment"]["destination"])
                    / ("shared" if shared else key)
                ).as_posix()
            ),
            "source_lang": source,
            "target_lang": target,
            "status": "ready",
            "generation": {
                "num_beams": int(config["deployment"]["num_beams"]),
                "max_new_tokens": int(config["deployment"]["max_new_tokens"]),
                "do_sample": False,
            },
        }
    write_json(registry_path, registry)
    write_json(
        destination / "model_card.json",
        {
            "pair": pair,
            "student_selection": selection,
            "full_parameter_finetuning": True,
            "distillation": True,
            "promotion_gate": gate,
            "commercial_license_verified": True,
            "previous_deployment_backup": str(backup) if backup else None,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generic full-parameter seq2seq distillation workflow."
    )
    parser.add_argument(
        "action", choices=("bakeoff", "train", "evaluate", "generate_teacher", "freeze")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("student", "teacher"))
    parser.add_argument("--experiment", choices=("exp1", "exp2"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "bakeoff":
        if not args.role:
            parser.error("bakeoff requires --role")
        bakeoff(config, args.role)
    elif args.action == "train":
        if not args.experiment:
            parser.error("train requires --experiment")
        train(config, args.experiment)
    elif args.action == "evaluate":
        if not args.experiment:
            parser.error("evaluate requires --experiment")
        evaluate_experiment(config, args.experiment)
    elif args.action == "generate_teacher":
        generate_teacher(config)
    else:
        freeze(config)


if __name__ == "__main__":
    main()
