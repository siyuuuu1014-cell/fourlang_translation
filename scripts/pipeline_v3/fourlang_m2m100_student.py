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
KD_CONTINUATION_EXPERIMENT = "m2m100_kd_v2"
TARGETED_EXPERIMENT = "m2m100_targeted_v1"
HUMAN_EXPERIMENT = "m2m100_human_v1"
KD_FROM_HUMAN_EXPERIMENT = "m2m100_kd_from_human_v1"
TARGETED_FROM_HUMAN_EXPERIMENT = "m2m100_targeted_from_human_v1"
TARGETED_FROM_HUMAN_V2_EXPERIMENT = "m2m100_targeted_from_human_v2"
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


def preflight(
    config: dict[str, Any],
    stages: tuple[str, ...] = ("kd", "targeted"),
    report_name: str = "preflight.json",
) -> dict[str, Any]:
    base = project_path(config["student"]["path"])
    report: dict[str, Any] = {
        "status": "READY",
        "student": candidate(config),
        "base_model": verify_m2m100_artifact(base),
        "datasets": {},
    }
    for stage in stages:
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
    output = project_path(config["outputs"]["root"]) / report_name
    write_json(output, report)
    print(f"M2M100_PREFLIGHT_READY: {output}", flush=True)
    return report


def output_root(config: dict[str, Any], experiment: str) -> Path:
    return project_path(config["outputs"]["root"]) / experiment


def exported_model(config: dict[str, Any], experiment: str) -> Path:
    return output_root(config, experiment) / "best_model" / "shared"


def stage_plan(config: dict[str, Any], stage: str) -> tuple[str, Path, str]:
    if stage == "human":
        return HUMAN_EXPERIMENT, project_path(config["student"]["path"]), "human"
    if stage == "kd_from_human":
        return (
            KD_FROM_HUMAN_EXPERIMENT,
            exported_model(config, HUMAN_EXPERIMENT),
            "kd",
        )
    if stage == "targeted_from_human":
        return (
            TARGETED_FROM_HUMAN_EXPERIMENT,
            exported_model(config, KD_FROM_HUMAN_EXPERIMENT),
            "targeted",
        )
    if stage == "targeted_from_human_v2":
        return (
            TARGETED_FROM_HUMAN_V2_EXPERIMENT,
            exported_model(config, KD_FROM_HUMAN_EXPERIMENT),
            "targeted",
        )
    if stage == "kd":
        return KD_EXPERIMENT, project_path(config["student"]["path"]), "kd"
    if stage == "kd_v2":
        return (
            KD_CONTINUATION_EXPERIMENT,
            exported_model(config, KD_EXPERIMENT),
            "kd",
        )
    if stage == "targeted":
        return TARGETED_EXPERIMENT, exported_model(config, KD_EXPERIMENT), "targeted"
    raise ValueError(stage)


def train_stage(config: dict[str, Any], stage: str) -> dict[str, Any]:
    experiment, source_model, data_stage = stage_plan(config, stage)
    verify_m2m100_artifact(source_model)
    train_path, validation_path = data_paths(config, data_stage)
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
    experiment, _, _ = stage_plan(config, stage)
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


def compare_kd_v2(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_root = project_path(config["outputs"]["evaluation_root"])
    experiments = (KD_EXPERIMENT, TARGETED_EXPERIMENT, KD_CONTINUATION_EXPERIMENT)
    payloads = {
        name: json.loads(
            (evaluation_root / name / "metrics.json").read_text(encoding="utf-8")
        )
        for name in experiments
    }
    candidate_payload = payloads[KD_CONTINUATION_EXPERIMENT]

    def delta_from(baseline: str) -> dict[str, Any]:
        baseline_payload = payloads[baseline]
        return {
            "summary": {
                metric: candidate_payload["summary"][metric]
                - baseline_payload["summary"][metric]
                for metric in ("macro_bleu", "macro_chrf2", "worst_chrf2")
            },
            "directions": {
                direction: {
                    metric: candidate_payload["metrics"][direction][metric]
                    - baseline_payload["metrics"][direction][metric]
                    for metric in ("bleu", "chrf2")
                }
                for direction in directions()
            },
        }

    report = {
        "schema_version": 1,
        "status": "COMPARED",
        "candidate": KD_CONTINUATION_EXPERIMENT,
        "summaries": {
            name: payloads[name]["summary"] for name in experiments
        },
        "candidate_minus": {
            KD_EXPERIMENT: delta_from(KD_EXPERIMENT),
            TARGETED_EXPERIMENT: delta_from(TARGETED_EXPERIMENT),
        },
    }
    destination = evaluation_root / "kd_v2_comparison.json"
    write_json(destination, report)
    print(f"M2M100_KD_V2_COMPARISON_READY: {destination}", flush=True)
    return report


def compare_human_route(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_root = project_path(config["outputs"]["evaluation_root"])
    experiments = (
        HUMAN_EXPERIMENT,
        KD_FROM_HUMAN_EXPERIMENT,
        TARGETED_FROM_HUMAN_EXPERIMENT,
    )
    payloads = {
        name: json.loads(
            (evaluation_root / name / "metrics.json").read_text(encoding="utf-8")
        )
        for name in experiments
    }

    def delta(candidate_name: str, baseline_name: str) -> dict[str, Any]:
        candidate_payload = payloads[candidate_name]
        baseline_payload = payloads[baseline_name]
        return {
            "summary": {
                metric: candidate_payload["summary"][metric]
                - baseline_payload["summary"][metric]
                for metric in ("macro_bleu", "macro_chrf2", "worst_chrf2")
            },
            "directions": {
                direction: {
                    metric: candidate_payload["metrics"][direction][metric]
                    - baseline_payload["metrics"][direction][metric]
                    for metric in ("bleu", "chrf2")
                }
                for direction in directions()
            },
        }

    report = {
        "schema_version": 1,
        "status": "COMPARED",
        "route": list(experiments),
        "summaries": {name: payloads[name]["summary"] for name in experiments},
        "stage_deltas": {
            "kd_minus_human": delta(KD_FROM_HUMAN_EXPERIMENT, HUMAN_EXPERIMENT),
            "targeted_minus_kd": delta(
                TARGETED_FROM_HUMAN_EXPERIMENT, KD_FROM_HUMAN_EXPERIMENT
            ),
            "targeted_minus_human": delta(
                TARGETED_FROM_HUMAN_EXPERIMENT, HUMAN_EXPERIMENT
            ),
        },
    }
    destination = evaluation_root / "human_route_comparison.json"
    write_json(destination, report)
    print(f"M2M100_HUMAN_ROUTE_COMPARISON_READY: {destination}", flush=True)
    return report


def compare_targeted_from_human_v2(config: dict[str, Any]) -> dict[str, Any]:
    evaluation_root = project_path(config["outputs"]["evaluation_root"])
    experiments = (
        KD_FROM_HUMAN_EXPERIMENT,
        TARGETED_FROM_HUMAN_EXPERIMENT,
        TARGETED_FROM_HUMAN_V2_EXPERIMENT,
    )
    payloads = {
        name: json.loads(
            (evaluation_root / name / "metrics.json").read_text(encoding="utf-8")
        )
        for name in experiments
    }
    candidate_payload = payloads[TARGETED_FROM_HUMAN_V2_EXPERIMENT]

    def delta_from_payload(baseline_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "summary": {
                metric: candidate_payload["summary"][metric]
                - baseline_payload["summary"][metric]
                for metric in ("macro_bleu", "macro_chrf2", "worst_chrf2")
            },
            "directions": {
                direction: {
                    metric: candidate_payload["metrics"][direction][metric]
                    - baseline_payload["metrics"][direction][metric]
                    for metric in ("bleu", "chrf2")
                }
                for direction in directions()
            },
        }

    candidate_minus = {
        name: delta_from_payload(payloads[name]) for name in experiments[:-1]
    }
    kd_delta = candidate_minus[KD_FROM_HUMAN_EXPERIMENT]
    target_directions = ("en-uz", "zh-uz", "ru-uz")
    non_target_directions = tuple(
        direction for direction in directions() if direction not in target_directions
    )
    acceptance = {
        "macro_chrf2_gain_at_least_0_3": (
            kd_delta["summary"]["macro_chrf2"] >= 0.3
        ),
        "all_uz_target_chrf2_improved": all(
            kd_delta["directions"][direction]["chrf2"] > 0
            for direction in target_directions
        ),
        "no_non_target_chrf2_regression_below_minus_0_5": all(
            kd_delta["directions"][direction]["chrf2"] >= -0.5
            for direction in non_target_directions
        ),
        "worst_chrf2_not_lower": kd_delta["summary"]["worst_chrf2"] >= 0,
    }
    acceptance["passed"] = all(acceptance.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "COMPARED",
        "candidate": TARGETED_FROM_HUMAN_V2_EXPERIMENT,
        "summaries": {name: payloads[name]["summary"] for name in experiments},
        "candidate_minus": candidate_minus,
        "acceptance": acceptance,
    }

    nllb_path_value = config.get("baselines", {}).get("nllb_exp3_v2_metrics")
    if nllb_path_value:
        nllb_path = project_path(nllb_path_value)
        report["nllb_exp3_v2"] = {"path": str(nllb_path.resolve()), "exists": nllb_path.is_file()}
        if nllb_path.is_file():
            nllb_payload = json.loads(nllb_path.read_text(encoding="utf-8"))
            nllb_metrics = nllb_payload.get("metrics", nllb_payload)
            normalized_nllb = {
                "metrics": nllb_metrics,
                "summary": summarize_metrics(nllb_metrics),
            }
            report["nllb_exp3_v2"]["summary"] = normalized_nllb["summary"]
            report["candidate_minus_nllb_exp3_v2"] = delta_from_payload(normalized_nllb)

    destination = evaluation_root / "targeted_from_human_v2_comparison.json"
    write_json(destination, report)
    print(f"M2M100_TARGETED_FROM_HUMAN_V2_COMPARISON_READY: {destination}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a fresh M2M100-418M four-language student."
    )
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "preflight-human-route",
            "train-kd",
            "eval-kd",
            "train-targeted",
            "eval-targeted",
            "train-kd-v2",
            "eval-kd-v2",
            "compare",
            "compare-kd-v2",
            "run-kd-v2",
            "train-human",
            "eval-human",
            "train-kd-from-human",
            "eval-kd-from-human",
            "train-targeted-from-human",
            "eval-targeted-from-human",
            "train-targeted-from-human-v2",
            "eval-targeted-from-human-v2",
            "compare-targeted-from-human-v2",
            "run-targeted-from-human-v2",
            "compare-human-route",
            "run-human-route",
            "run-all",
        ),
    )
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "preflight":
        preflight(config)
    elif args.action == "preflight-human-route":
        preflight(
            config,
            stages=("human", "kd", "targeted"),
            report_name="human_route_preflight.json",
        )
    elif args.action == "train-kd":
        train_stage(config, "kd")
    elif args.action == "eval-kd":
        evaluate_stage(config, "kd")
    elif args.action == "train-targeted":
        train_stage(config, "targeted")
    elif args.action == "eval-targeted":
        evaluate_stage(config, "targeted")
    elif args.action == "train-kd-v2":
        train_stage(config, "kd_v2")
    elif args.action == "eval-kd-v2":
        evaluate_stage(config, "kd_v2")
    elif args.action == "compare":
        compare(config)
    elif args.action == "compare-kd-v2":
        compare_kd_v2(config)
    elif args.action == "run-kd-v2":
        preflight(config)
        train_stage(config, "kd_v2")
        evaluate_stage(config, "kd_v2")
        compare_kd_v2(config)
    elif args.action == "train-human":
        train_stage(config, "human")
    elif args.action == "eval-human":
        evaluate_stage(config, "human")
    elif args.action == "train-kd-from-human":
        train_stage(config, "kd_from_human")
    elif args.action == "eval-kd-from-human":
        evaluate_stage(config, "kd_from_human")
    elif args.action == "train-targeted-from-human":
        train_stage(config, "targeted_from_human")
    elif args.action == "eval-targeted-from-human":
        evaluate_stage(config, "targeted_from_human")
    elif args.action == "train-targeted-from-human-v2":
        train_stage(config, "targeted_from_human_v2")
    elif args.action == "eval-targeted-from-human-v2":
        evaluate_stage(config, "targeted_from_human_v2")
    elif args.action == "compare-targeted-from-human-v2":
        compare_targeted_from_human_v2(config)
    elif args.action == "run-targeted-from-human-v2":
        preflight(config, stages=("targeted",), report_name="targeted_from_human_v2_preflight.json")
        train_stage(config, "targeted_from_human_v2")
        evaluate_stage(config, "targeted_from_human_v2")
        compare_targeted_from_human_v2(config)
    elif args.action == "compare-human-route":
        compare_human_route(config)
    elif args.action == "run-human-route":
        preflight(
            config,
            stages=("human", "kd", "targeted"),
            report_name="human_route_preflight.json",
        )
        train_stage(config, "human")
        evaluate_stage(config, "human")
        train_stage(config, "kd_from_human")
        evaluate_stage(config, "kd_from_human")
        train_stage(config, "targeted_from_human")
        evaluate_stage(config, "targeted_from_human")
        compare_human_route(config)
    else:
        preflight(config)
        train_stage(config, "kd")
        evaluate_stage(config, "kd")
        train_stage(config, "targeted")
        evaluate_stage(config, "targeted")
        compare(config)


if __name__ == "__main__":
    main()
