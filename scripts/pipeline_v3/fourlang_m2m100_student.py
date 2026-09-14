from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    load_model,
    metrics,
    train_model,
    translate,
)
from scripts.pipeline_v2.training_safety import file_sha256, model_files  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import directions  # noqa: E402

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_v1.toml"
KD_EXPERIMENT = "m2m100_kd_v1"
TARGETED_EXPERIMENT = "m2m100_targeted_v1"
EXPECTED_MODEL_TYPE = "m2m_100"
EXPECTED_REPO = "facebook/m2m100_418M"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def candidate(config: dict[str, Any], source_model: Path | None = None) -> dict[str, Any]:
    value = dict(config["student"])
    if value.get("family") != "m2m100" or value.get("repo_id") != EXPECTED_REPO:
        raise RuntimeError("This pipeline is locked to facebook/m2m100_418M.")
    if source_model is not None:
        value["path"] = str(source_model)
        value["require_local_artifact"] = True
    return value


def verify_m2m100_artifact(path: Path) -> dict[str, Any]:
    model_files(path)
    config_path = path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"Refusing non-M2M100 source model at {path}: "
            f"model_type={payload.get('model_type')!r}"
        )
    return {
        "path": str(path.resolve()),
        "model_type": payload["model_type"],
        "config_sha256": file_sha256(config_path),
    }


def data_paths(config: dict[str, Any], stage: str) -> tuple[Path, Path]:
    section = config["data"][stage]
    return project_path(section["train"]), project_path(section["validation"])


def validate_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, int]:
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{index} missing fields: {sorted(missing)}")
        direction = f"{row['src_lang']}-{row['tgt_lang']}"
        counts[direction] += 1
    missing_directions = sorted(set(directions()) - set(counts))
    extra_directions = sorted(set(counts) - set(directions()))
    if missing_directions or extra_directions:
        raise RuntimeError(
            f"{path} direction mismatch; missing={missing_directions}, extra={extra_directions}"
        )
    return dict(sorted(counts.items()))


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    base = project_path(config["student"]["path"])
    report: dict[str, Any] = {
        "status": "READY",
        "student": candidate(config),
        "base_model": verify_m2m100_artifact(base),
        "datasets": {},
    }
    for stage in ("kd", "targeted"):
        train_path, validation_path = data_paths(config, stage)
        if not train_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(
                f"Missing existing {stage} data: {train_path} or {validation_path}"
            )
        train_rows = read_jsonl(train_path)
        validation_rows = read_jsonl(validation_path)
        report["datasets"][stage] = {
            "train": str(train_path.resolve()),
            "train_rows": len(train_rows),
            "train_sha256": file_sha256(train_path),
            "train_directions": validate_rows(train_rows, train_path),
            "validation": str(validation_path.resolve()),
            "validation_rows": len(validation_rows),
            "validation_sha256": file_sha256(validation_path),
            "validation_directions": validate_rows(validation_rows, validation_path),
        }
    benchmark = project_path(config["benchmarks"]["flores_devtest"])
    if not benchmark.is_file():
        raise FileNotFoundError(f"Missing fixed evaluation benchmark: {benchmark}")
    report["benchmark"] = {
        "path": str(benchmark.resolve()),
        "sha256": file_sha256(benchmark),
    }
    output = project_path(config["outputs"]["root"]) / "preflight.json"
    write_json(output, report)
    print(f"M2M100_PREFLIGHT_READY: {output}", flush=True)
    return report


def output_root(config: dict[str, Any], experiment: str) -> Path:
    return project_path(config["outputs"]["root"]) / experiment


def exported_model(config: dict[str, Any], experiment: str) -> Path:
    return output_root(config, experiment) / "best_model" / "shared"


def train_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "kd":
        experiment = KD_EXPERIMENT
        source_model = project_path(config["student"]["path"])
    elif stage == "targeted":
        experiment = TARGETED_EXPERIMENT
        source_model = exported_model(config, KD_EXPERIMENT)
    else:
        raise ValueError(stage)
    verify_m2m100_artifact(source_model)
    train_path, validation_path = data_paths(config, stage)
    train_rows = read_jsonl(train_path)
    validation_rows = read_jsonl(validation_path)
    validate_rows(train_rows, train_path)
    validate_rows(validation_rows, validation_path)
    destination = exported_model(config, experiment)
    report = train_model(
        candidate(config, source_model),
        str(source_model),
        "en",
        "zh",
        train_rows,
        validation_rows,
        destination,
        config,
        experiment=experiment,
        shared=True,
    )
    payload = {
        "schema_version": 1,
        "status": "TRAINED",
        "student": candidate(config),
        "source_model": str(source_model.resolve()),
        "train_data": str(train_path.resolve()),
        "validation_data": str(validation_path.resolve()),
        "model": report,
    }
    write_json(output_root(config, experiment) / "train_report.json", payload)
    print(f"M2M100_{stage.upper()}_TRAINING_COMPLETE: {destination}", flush=True)
    return payload


def summarize_metrics(values: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = list(values.values())
    worst_direction = min(values, key=lambda key: values[key]["chrf2"])
    return {
        "macro_bleu": sum(row["bleu"] for row in rows) / len(rows),
        "macro_chrf2": sum(row["chrf2"] for row in rows) / len(rows),
        "worst_direction": worst_direction,
        "worst_chrf2": values[worst_direction]["chrf2"],
    }


def evaluate_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    experiment = KD_EXPERIMENT if stage == "kd" else TARGETED_EXPERIMENT
    model_path = exported_model(config, experiment)
    verify_m2m100_artifact(model_path)
    benchmark_path = project_path(config["benchmarks"]["flores_devtest"])
    frame = pd.read_parquet(benchmark_path)
    missing_columns = sorted(set(config["multilingual"]["languages"]) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Benchmark is missing language columns: {missing_columns}")
    runtime_candidate = candidate(config, model_path)
    tokenizer, model = load_model(runtime_candidate, "en", "zh")
    values: dict[str, dict[str, float]] = {}
    try:
        for direction in directions():
            source, target = direction.split("-")
            predictions = translate(
                tokenizer,
                model,
                "m2m100",
                source,
                target,
                frame[source].fillna("").astype(str).tolist(),
                config,
            )
            values[direction] = metrics(
                predictions,
                frame[target].fillna("").astype(str).tolist(),
                target,
            )
            print(f"Evaluated {direction}: {values[direction]}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "status": "EVALUATED",
        "experiment": experiment,
        "model": str(model_path.resolve()),
        "benchmark": str(benchmark_path.resolve()),
        "metrics": values,
        "summary": summarize_metrics(values),
    }
    destination = project_path(config["outputs"]["evaluation_root"]) / experiment / "metrics.json"
    write_json(destination, payload)
    print(f"M2M100_{stage.upper()}_EVALUATION_COMPLETE: {destination}", flush=True)
    return payload


def compare(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_root = project_path(config["outputs"]["evaluation_root"])
    paths = {
        KD_EXPERIMENT: evaluation_root / KD_EXPERIMENT / "metrics.json",
        TARGETED_EXPERIMENT: evaluation_root / TARGETED_EXPERIMENT / "metrics.json",
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    kd = payloads[KD_EXPERIMENT]
    targeted = payloads[TARGETED_EXPERIMENT]
    deltas = {
        direction: {
            metric: targeted["metrics"][direction][metric] - kd["metrics"][direction][metric]
            for metric in ("bleu", "chrf2")
        }
        for direction in directions()
    }
    report = {
        "schema_version": 1,
        "status": "COMPARED",
        "kd_summary": kd["summary"],
        "targeted_summary": targeted["summary"],
        "targeted_minus_kd": deltas,
    }
    destination = evaluation_root / "comparison.json"
    write_json(destination, report)
    print(f"M2M100_COMPARISON_READY: {destination}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fresh M2M100-418M four-language student."
    )
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "train-kd",
            "eval-kd",
            "train-targeted",
            "eval-targeted",
            "compare",
            "run-all",
        ),
    )
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "preflight":
        preflight(config)
    elif args.action == "train-kd":
        train_stage(config, "kd")
    elif args.action == "eval-kd":
        evaluate_stage(config, "kd")
    elif args.action == "train-targeted":
        train_stage(config, "targeted")
    elif args.action == "eval-targeted":
        evaluate_stage(config, "targeted")
    elif args.action == "compare":
        compare(config)
    else:
        preflight(config)
        train_stage(config, "kd")
        evaluate_stage(config, "kd")
        train_stage(config, "targeted")
        evaluate_stage(config, "targeted")
        compare(config)


if __name__ == "__main__":
    main()

