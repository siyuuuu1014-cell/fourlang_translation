"""Build, evaluate, and regression-gate M2M100 weight-soup candidates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import load_model, metrics, translate  # noqa: E402
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

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_soup_v1.toml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def alpha_id(alpha: float) -> str:
    return f"exp4_{int(round(alpha * 100)):03d}"


def candidate_model_path(config: dict[str, Any], alpha: float) -> Path:
    return project_path(config["outputs"]["model_root"]) / alpha_id(alpha) / "best_model" / "shared"


def candidate_metrics_path(config: dict[str, Any], alpha: float) -> Path:
    return project_path(config["outputs"]["evaluation_root"]) / alpha_id(alpha) / "metrics.json"


def read_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("metrics", payload)
    if set(values) != set(directions()):
        raise RuntimeError(f"Metrics do not contain exactly 12 directions: {path}")
    return {
        "metrics": values,
        "summary": payload.get("summary", summarize_metrics(values)),
    }


def fusion_manifest(config: dict[str, Any], alpha: float) -> dict[str, Any]:
    exp4 = project_path(config["models"]["exp4"])
    targeted = project_path(config["models"]["targeted_v2"])
    return {
        "schema_version": 1,
        "method": "linear_parameter_interpolation_v1",
        "alpha_exp4": alpha,
        "alpha_targeted_v2": 1.0 - alpha,
        "exp4": {"path": str(exp4.resolve()), "signature": model_signature(exp4)},
        "targeted_v2": {"path": str(targeted.resolve()), "signature": model_signature(targeted)},
        "implementation_sha256": file_sha256(Path(__file__)),
    }


def build_candidate(config: dict[str, Any], alpha: float) -> Path:
    if not 0.0 < alpha < 1.0:
        raise ValueError("Fusion alpha must be strictly between zero and one.")
    destination = candidate_model_path(config, alpha)
    manifest = fusion_manifest(config, alpha)
    signature = fingerprint(manifest)
    manifest_path = destination / "fusion_manifest.json"
    if destination.exists():
        if not manifest_path.is_file():
            raise RuntimeError(f"Existing candidate has no fusion manifest: {destination}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != {"fingerprint": signature, "manifest": manifest}:
            raise RuntimeError(f"Existing candidate differs; refusing overwrite: {destination}")
        verify_m2m100_artifact(destination)
        print(f"M2M100_SOUP_MODEL_REUSED: {destination}", flush=True)
        return destination

    temp = destination.with_name(destination.name + ".tmp")
    if temp.exists():
        raise RuntimeError(f"Partial temporary candidate exists; inspect it: {temp}")
    exp4_path = project_path(config["models"]["exp4"])
    targeted_path = project_path(config["models"]["targeted_v2"])
    verify_m2m100_artifact(exp4_path)
    verify_m2m100_artifact(targeted_path)
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024 ** 3)
    minimum = float(config["experiment"]["minimum_free_disk_gb"])
    if free_gb < minimum:
        raise RuntimeError(f"Need at least {minimum:g} GiB free; found {free_gb:.2f} GiB.")

    print(f"M2M100_SOUP_BUILDING: {alpha_id(alpha)}", flush=True)
    exp4_model = M2M100ForConditionalGeneration.from_pretrained(
        exp4_path, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    targeted_model = M2M100ForConditionalGeneration.from_pretrained(
        targeted_path, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float32
    )
    exp4_parameters = dict(exp4_model.named_parameters())
    targeted_parameters = dict(targeted_model.named_parameters())
    if exp4_parameters.keys() != targeted_parameters.keys():
        raise RuntimeError("Source models have different parameter keys.")
    with torch.no_grad():
        for name, parameter in exp4_parameters.items():
            other = targeted_parameters[name]
            if parameter.shape != other.shape:
                raise RuntimeError(f"Parameter shape differs: {name}")
            parameter.mul_(alpha).add_(other, alpha=1.0 - alpha)
        exp4_buffers = dict(exp4_model.named_buffers())
        targeted_buffers = dict(targeted_model.named_buffers())
        if exp4_buffers.keys() != targeted_buffers.keys():
            raise RuntimeError("Source models have different buffer keys.")
        for name, buffer in exp4_buffers.items():
            other = targeted_buffers[name]
            if buffer.shape != other.shape:
                raise RuntimeError(f"Buffer shape differs: {name}")
            if buffer.is_floating_point():
                buffer.mul_(alpha).add_(other, alpha=1.0 - alpha)
            elif not torch.equal(buffer, other):
                raise RuntimeError(f"Non-floating buffer differs: {name}")
    temp.mkdir(parents=True)
    exp4_model.save_pretrained(temp, safe_serialization=True)
    tokenizer = M2M100Tokenizer.from_pretrained(exp4_path, local_files_only=True)
    tokenizer.save_pretrained(temp)
    atomic_json(temp / "fusion_manifest.json", {"fingerprint": signature, "manifest": manifest})
    verify_m2m100_artifact(temp)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp.replace(destination)
    del (
        exp4_model,
        targeted_model,
        tokenizer,
        exp4_parameters,
        targeted_parameters,
        exp4_buffers,
        targeted_buffers,
    )
    gc.collect()
    print(f"M2M100_SOUP_MODEL_READY: {destination}", flush=True)
    return destination


def evaluate_candidate(config: dict[str, Any], alpha: float) -> dict[str, Any]:
    model_path = candidate_model_path(config, alpha)
    verify_m2m100_artifact(model_path)
    destination = candidate_metrics_path(config, alpha)
    model_fingerprint = model_signature(model_path)
    benchmark_path = project_path(config["benchmark"]["path"])
    evaluation_signature = fingerprint(
        {
            "model": model_fingerprint,
            "benchmark_sha256": file_sha256(benchmark_path),
            "decoding": config["deployment"],
        }
    )
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous.get("evaluation_signature") != evaluation_signature:
            raise RuntimeError(f"Existing evaluation differs; refusing overwrite: {destination}")
        print(f"M2M100_SOUP_EVALUATION_REUSED: {destination}", flush=True)
        return previous
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full 12-direction soup evaluation.")
    frame = pd.read_parquet(benchmark_path)
    runtime_candidate = {**config["student"], "path": str(model_path)}
    tokenizer, model = load_model(runtime_candidate, "en", "zh")
    values = {}
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
            print(f"Soup {alpha_id(alpha)} evaluated {direction}: {values[direction]}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "status": "EVALUATED",
        "candidate": alpha_id(alpha),
        "alpha_exp4": alpha,
        "model": str(model_path.resolve()),
        "evaluation_signature": evaluation_signature,
        "metrics": values,
        "summary": summarize_metrics(values),
    }
    write_json(destination, payload)
    print(f"M2M100_SOUP_EVALUATION_READY: {destination}", flush=True)
    return payload


def score_candidate(
    candidate: dict[str, Any], exp4: dict[str, Any], targeted: dict[str, Any], settings: dict[str, Any]
) -> dict[str, Any]:
    best = {
        direction: max(exp4["metrics"][direction]["chrf2"], targeted["metrics"][direction]["chrf2"])
        for direction in directions()
    }
    regressions = {
        direction: max(0.0, best[direction] - candidate["metrics"][direction]["chrf2"])
        for direction in directions()
    }
    tolerance = float(settings["regression_tolerance"])
    excess_sum = sum(max(0.0, value - tolerance) for value in regressions.values())
    maximum = max(regressions.values())
    penalty = (
        float(settings["regression_sum_penalty"]) * excess_sum
        + float(settings["maximum_regression_penalty"]) * maximum
    )
    macro = float(candidate["summary"]["macro_chrf2"])
    constraints = {
        "max_direction_regression_at_most_limit": maximum
        <= float(settings["hard_max_direction_regression"]),
        "ru_uz_recovery_from_exp4_at_least_minimum": (
            candidate["metrics"]["ru-uz"]["chrf2"] - exp4["metrics"]["ru-uz"]["chrf2"]
            >= float(settings["minimum_ru_uz_recovery_from_exp4"])
        ),
        "zh_uz_regression_at_most_limit": regressions["zh-uz"]
        <= float(settings["maximum_zh_uz_regression"]),
        "macro_chrf2_at_least_minimum": macro >= float(settings["minimum_macro_chrf2"]),
        "worst_chrf2_at_least_minimum": float(candidate["summary"]["worst_chrf2"])
        >= float(settings["minimum_worst_chrf2"]),
    }
    return {
        "selection_score": macro - penalty,
        "macro_chrf2": macro,
        "penalty": penalty,
        "regressions_from_best_existing_by_direction": regressions,
        "maximum_direction_regression": maximum,
        "constraints": constraints,
        "passed": all(constraints.values()),
    }


def compare(config: dict[str, Any]) -> dict[str, Any]:
    exp4 = read_metrics(project_path(config["metrics"]["exp4"]))
    targeted = read_metrics(project_path(config["metrics"]["targeted_v2"]))
    candidates = {}
    for alpha in config["experiment"]["alphas_exp4"]:
        value = float(alpha)
        payload = read_metrics(candidate_metrics_path(config, value))
        payload["candidate"] = alpha_id(value)
        payload["alpha_exp4"] = value
        payload["model"] = str(candidate_model_path(config, value).resolve())
        payload["selection"] = score_candidate(payload, exp4, targeted, config["selection"])
        candidates[alpha_id(value)] = payload
    passed = [value for value in candidates.values() if value["selection"]["passed"]]
    selected = max(passed, key=lambda value: value["selection"]["selection_score"]) if passed else None
    report = {
        "schema_version": 1,
        "status": "CANDIDATE_SELECTED" if selected else "NO_CANDIDATE_PASSED",
        "selection_policy": config["selection"],
        "baselines": {"exp4": exp4, "targeted_v2": targeted},
        "candidates": candidates,
        "selected": (
            {
                "candidate": selected["candidate"],
                "alpha_exp4": selected["alpha_exp4"],
                "model": selected["model"],
                "summary": selected["summary"],
                "selection": selected["selection"],
            }
            if selected
            else None
        ),
        "promotion_performed": False,
        "models_deleted": False,
    }
    destination = project_path(config["outputs"]["comparison"])
    write_json(destination, report)
    print(f"M2M100_SOUP_COMPARISON_READY: {destination} status={report['status']}", flush=True)
    return report


def run(config: dict[str, Any]) -> None:
    for alpha in config["experiment"]["alphas_exp4"]:
        value = float(alpha)
        build_candidate(config, value)
        evaluate_candidate(config, value)
    compare(config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "evaluate", "compare", "run-all"))
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "build":
        for alpha in config["experiment"]["alphas_exp4"]:
            build_candidate(config, float(alpha))
    elif args.action == "evaluate":
        for alpha in config["experiment"]["alphas_exp4"]:
            evaluate_candidate(config, float(alpha))
    elif args.action == "compare":
        compare(config)
    else:
        run(config)


if __name__ == "__main__":
    main()
