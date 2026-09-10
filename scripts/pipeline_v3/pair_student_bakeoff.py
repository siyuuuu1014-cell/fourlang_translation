"""Evaluate one deployable bidirectional Student backbone per language pair.

This is a selection-only workflow. It never trains, promotes, or changes the
existing six-pair baseline artifacts.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import (  # noqa: E402
    COMMERCIAL_LICENSES,
    PROJECT_ROOT,
    load_config,
    parquet_columns,
    project_path,
    write_json,
)
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    benchmark_frames,
    candidate_path,
    load_model,
    metrics,
    translate,
)
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint  # noqa: E402

CONFIG_PATH = "configs/specialists/pair_student_bakeoff.toml"
RESULT_ROOT = Path("results/model_selection/pair_students")
PAIR_IDS = ("en_zh", "en_uz", "en_ru", "zh_uz", "zh_ru", "uz_ru")


def pair_by_id(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [item for item in config["pairs"] if item["id"] == pair_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one pair {pair_id!r}.")
    return dict(matches[0])


def eligible_candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for item in config["student_candidates"]:
        candidate = dict(item)
        if config["experiment"].get("commercial_use", False) and not (
            candidate.get("commercial_allowed") is True
            and str(candidate.get("license", "")).lower() in COMMERCIAL_LICENSES
        ):
            continue
        candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No deployment-eligible Student candidates are configured.")
    return candidates


def validate_config(config: dict[str, Any]) -> None:
    pair_ids = [item.get("id") for item in config.get("pairs", [])]
    if set(pair_ids) != set(PAIR_IDS) or len(pair_ids) != len(PAIR_IDS):
        raise ValueError(f"pairs must contain exactly {list(PAIR_IDS)}")
    candidate_ids = [item.get("id") for item in config.get("student_candidates", [])]
    if len(candidate_ids) != len(set(candidate_ids)) or not candidate_ids:
        raise ValueError("Student candidate IDs must be nonempty and unique.")
    for pair in config["pairs"]:
        languages = pair.get("languages", [])
        if len(languages) != 2 or pair["id"] != "_".join(languages):
            raise ValueError(
                f"Invalid pair/language mapping for {pair.get('id')!r}."
            )
        if not str(pair.get("benchmark", "")).strip():
            raise ValueError(f"Missing selection benchmark for {pair['id']}.")
    eligible_candidates(config)
    if int(config["selection"]["max_student_parameters"]) < 1:
        raise ValueError("max_student_parameters must be positive.")


def runtime_config(config: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    left, right = pair["languages"]
    return {
        "direction": {
            "pair": pair["id"],
            "source_lang": left,
            "target_lang": right,
            "version": "bakeoff_v1",
            "commercial_use": bool(config["experiment"].get("commercial_use")),
        },
        "benchmarks": {"selection": {"flores_dev": pair["benchmark"]}},
        "training": dict(config["training"]),
        "deployment": dict(config["deployment"]),
    }


def result_directory(pair_id: str) -> Path:
    return PROJECT_ROOT / RESULT_ROOT / pair_id


def selection_manifest(
    config: dict[str, Any], pair: dict[str, Any]
) -> dict[str, Any]:
    benchmark = project_path(pair["benchmark"])
    implementation = (
        PROJECT_ROOT / "scripts/pipeline_v3/pair_student_bakeoff.py",
        PROJECT_ROOT / "scripts/pipeline_v2/seq2seq_flow.py",
        PROJECT_ROOT / "scripts/pipeline_v2/common.py",
        PROJECT_ROOT / "scripts/pipeline_v3/language_normalization.py",
        PROJECT_ROOT / "src/model_utils.py",
    )
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "selection": config["selection"],
        "training": config["training"],
        "deployment": config["deployment"],
        "pair": pair,
        "benchmark_sha256": file_sha256(benchmark),
        "candidates": eligible_candidates(config),
        "implementation_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in implementation
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "sacrebleu", "sentencepiece")
        },
    }


def model_disk_bytes(path: str | Path) -> int:
    root = Path(path)
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def evaluate_pair_candidate(
    candidate: dict[str, Any], config: dict[str, Any], pair: dict[str, Any]
) -> dict[str, Any]:
    left, right = pair["languages"]
    run_config = runtime_config(config, pair)
    frames = benchmark_frames(run_config, "selection")
    all_metrics: dict[str, Any] = {}
    parameters = 0
    disk_bytes = 0
    for source, target in ((left, right), (right, left)):
        started_load = time.perf_counter()
        tokenizer, model = load_model(candidate, source, target)
        load_seconds = time.perf_counter() - started_load
        parameters = max(
            parameters, sum(parameter.numel() for parameter in model.parameters())
        )
        disk_bytes = max(
            disk_bytes,
            model_disk_bytes(candidate_path(candidate, source, target)),
        )
        predictions: list[str] = []
        references: list[str] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started_generation = time.perf_counter()
        for frame in frames.values():
            source_column, target_column = parquet_columns(frame, left, right)
            if source == right:
                source_column, target_column = target_column, source_column
            texts = frame[source_column].fillna("").astype(str).tolist()
            refs = frame[target_column].fillna("").astype(str).tolist()
            predictions.extend(
                translate(
                    tokenizer,
                    model,
                    candidate["family"],
                    source,
                    target,
                    texts,
                    run_config,
                )
            )
            references.extend(refs)
        generation_seconds = time.perf_counter() - started_generation
        direction_metrics = metrics(predictions, references, target)
        samples = int(direction_metrics["samples"])
        all_metrics[f"{source}-{target}"] = {
            **direction_metrics,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "seconds_per_sample": generation_seconds / max(1, samples),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            ),
        }
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "status": "ok",
        "parameters": parameters,
        "model_disk_bytes": disk_bytes,
        "metrics": all_metrics,
    }


def score_summary(result: dict[str, Any]) -> dict[str, float]:
    metrics_by_direction = list(result["metrics"].values())
    return {
        "minimum_direction_chrf2": min(
            float(item["chrf2"]) for item in metrics_by_direction
        ),
        "macro_chrf2": sum(float(item["chrf2"]) for item in metrics_by_direction)
        / len(metrics_by_direction),
        "macro_bleu": sum(float(item["bleu"]) for item in metrics_by_direction)
        / len(metrics_by_direction),
        "mean_seconds_per_sample": sum(
            float(item["seconds_per_sample"]) for item in metrics_by_direction
        )
        / len(metrics_by_direction),
    }


def load_scores(
    config: dict[str, Any], pair: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    manifest = selection_manifest(config, pair)
    signature = fingerprint(manifest)
    path = result_directory(pair["id"]) / "scores.json"
    if not path.is_file():
        return {}, manifest, signature
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("fingerprint") != signature or report.get("manifest") != manifest:
        raise RuntimeError(
            f"Existing bake-off belongs to different inputs/configuration: {path}"
        )
    return dict(report.get("candidates", {})), manifest, signature


def save_scores(
    pair_id: str,
    candidates: dict[str, Any],
    manifest: dict[str, Any],
    signature: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "pair": pair_id,
        "fingerprint": signature,
        "manifest": manifest,
        "candidates": candidates,
    }
    write_json(result_directory(pair_id) / "scores.json", payload)
    return payload


def evaluate(
    config: dict[str, Any],
    pair_id: str,
    candidate_id: str | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    validate_config(config)
    pair = pair_by_id(config, pair_id)
    configured = eligible_candidates(config)
    if candidate_id:
        configured = [item for item in configured if item["id"] == candidate_id]
        if not configured:
            raise KeyError(f"Unknown or ineligible candidate {candidate_id!r}.")
    results, manifest, signature = load_scores(config, pair)
    maximum = int(config["selection"]["max_student_parameters"])
    for candidate in configured:
        if not force and results.get(candidate["id"], {}).get("status") == "ok":
            continue
        try:
            result = evaluate_pair_candidate(candidate, config, pair)
            result["selection_metrics"] = score_summary(result)
            if int(result["parameters"]) > maximum:
                result["status"] = "ineligible_size"
        except Exception as error:
            result = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        results[candidate["id"]] = result
        save_scores(pair_id, results, manifest, signature)
    return save_scores(pair_id, results, manifest, signature)


def rank(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    validate_config(config)
    pair = pair_by_id(config, pair_id)
    results, manifest, signature = load_scores(config, pair)
    configured_ids = {item["id"] for item in eligible_candidates(config)}
    if config["selection"].get("require_all_candidates", True):
        missing = sorted(configured_ids - set(results))
        incomplete = sorted(
            candidate_id
            for candidate_id in configured_ids & set(results)
            if results[candidate_id].get("status") != "ok"
        )
        if missing or incomplete:
            raise RuntimeError(
                f"Bake-off is incomplete; missing={missing}, incomplete={incomplete}."
            )
    ranked = []
    for candidate_id, result in results.items():
        if candidate_id not in configured_ids or result.get("status") != "ok":
            continue
        values = result["selection_metrics"]
        ranked.append(
            {
                "candidate_id": candidate_id,
                **values,
                "parameters": result["parameters"],
                "model_disk_bytes": result["model_disk_bytes"],
            }
        )
    if not ranked:
        raise RuntimeError(f"No eligible completed candidates for {pair_id}.")
    ranked.sort(
        key=lambda item: (
            item["minimum_direction_chrf2"],
            item["macro_chrf2"],
            item["macro_bleu"],
            -item["mean_seconds_per_sample"],
        ),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "pair": pair_id,
        "fingerprint": signature,
        "selection_policy": (
            "one bidirectional candidate; maximize minimum direction chrF2, then "
            "macro chrF2, macro BLEU, and lower latency on FLORES dev"
        ),
        "winner": ranked[0]["candidate_id"],
        "ranking": ranked,
        "not_a_training_decision": True,
        "next_stage": "fixed-budget Exp1 pilot",
        "manifest": manifest,
    }
    write_json(result_directory(pair_id) / "ranking.json", payload)
    return payload


def status(config: dict[str, Any]) -> dict[str, Any]:
    validate_config(config)
    candidate_ids = [item["id"] for item in eligible_candidates(config)]
    rows = []
    for pair in config["pairs"]:
        try:
            results, _, _ = load_scores(config, pair)
            error = None
        except (OSError, RuntimeError) as exc:
            results, error = {}, str(exc)
        rows.append(
            {
                "pair": pair["id"],
                "benchmark": pair["benchmark"],
                "candidate_status": {
                    candidate_id: results.get(candidate_id, {}).get("status", "pending")
                    for candidate_id in candidate_ids
                },
                "ranking_ready": all(
                    results.get(candidate_id, {}).get("status") == "ok"
                    for candidate_id in candidate_ids
                ),
                "error": error,
            }
        )
    return {"experiment": config["experiment"]["id"], "pairs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("status", "evaluate", "rank", "run")
    )
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--pair", choices=PAIR_IDS)
    parser.add_argument("--candidate-id")
    parser.add_argument(
        "--force", action="store_true", help="Re-evaluate an already completed candidate."
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "status":
        payload = status(config)
    else:
        if not args.pair:
            parser.error(f"{args.action} requires --pair")
        if args.action == "evaluate":
            payload = evaluate(config, args.pair, args.candidate_id, force=args.force)
        elif args.action == "rank":
            if args.candidate_id or args.force:
                parser.error("rank does not accept --candidate-id or --force")
            payload = rank(config, args.pair)
        else:
            if args.candidate_id:
                parser.error("run does not accept --candidate-id")
            evaluate(config, args.pair, None, force=args.force)
            payload = rank(config, args.pair)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
