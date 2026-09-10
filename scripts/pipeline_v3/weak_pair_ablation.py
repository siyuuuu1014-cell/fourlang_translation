"""Isolated full-data and directional ablations for weak pair specialists."""

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

from scripts.pipeline_v2.common import load_config, read_json, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    load_model,
    metrics,
    train_model,
    translate,
)
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    file_sha256,
    fingerprint,
    model_files,
)
from scripts.pipeline_v3.data_safety import protect_splits  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import _read_table, normalize_rows  # noqa: E402

PROJECT_ROOT = PROJECT_ROOT_BOOTSTRAP
PAIR_IDS = ("zh_uz", "uz_ru")
TRAIN_VARIANTS = (
    "bidir_full",
    "directional_full",
    "full_weighted_60_40",
    "full_native_lr2e6",
)
BASELINE_VARIANTS = ("baseline_exp1", "baseline_exp2")
ALL_VARIANTS = BASELINE_VARIANTS + TRAIN_VARIANTS


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pair_config(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [item for item in config["pairs"] if item["id"] == pair_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one pair configuration for {pair_id!r}.")
    return dict(matches[0])


def pair_directions(pair: dict[str, Any]) -> tuple[str, str]:
    left, right = pair["languages"]
    return f"{left}-{right}", f"{right}-{left}"


def validate_direction(pair: dict[str, Any], direction: str | None) -> str:
    if direction not in pair_directions(pair):
        raise ValueError(
            f"direction must be one of {list(pair_directions(pair))}; got {direction!r}."
        )
    return str(direction)


def run_id(variant: str, direction: str | None = None) -> str:
    if variant == "directional_full":
        if not direction:
            raise ValueError("directional_full requires a direction.")
        return f"directional_full__{direction.replace('-', '_')}"
    return variant


def variant_settings(config: dict[str, Any], variant: str) -> dict[str, Any]:
    settings = config.get("variants", {}).get(variant)
    if not isinstance(settings, dict):
        raise KeyError(f"Missing variants.{variant} configuration.")
    return dict(settings)


def prepared_root(pair_id: str, variant: str, direction: str | None = None) -> Path:
    return (
        PROJECT_ROOT
        / "data/experiments/weak_pair_ablation"
        / pair_id
        / run_id(variant, direction)
    )


def artifact_root(pair_id: str, variant: str, direction: str | None = None) -> Path:
    return (
        PROJECT_ROOT
        / "results/experiments/weak_pair_ablation"
        / pair_id
        / run_id(variant, direction)
    )


def evaluation_path(pair_id: str, variant: str, direction: str | None = None) -> Path:
    return (
        PROJECT_ROOT
        / "results/evaluation/weak_pair_ablation"
        / pair_id
        / f"{run_id(variant, direction)}.json"
    )


def _candidate(config: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    candidate = {
        "id": config["experiment"]["backbone"],
        **config["model"],
    }
    if path is not None:
        candidate["path"] = str(path)
        candidate["require_local_artifact"] = True
    return candidate


def _expected_pairs(pair: dict[str, Any]) -> set[tuple[str, str]]:
    return {tuple(item.split("-")) for item in pair_directions(pair)}


def _validate_pair_frame(
    frame: pd.DataFrame, pair: dict[str, Any], label: str
) -> None:
    present = set(zip(frame["src_lang"], frame["tgt_lang"], strict=True))
    expected = _expected_pairs(pair)
    if present != expected:
        raise RuntimeError(
            f"{pair['id']} {label} directions must be {sorted(expected)}; "
            f"found {sorted(present)}."
        )


def _write_jsonl(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in frame.to_dict("records")
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _composition(frame: pd.DataFrame) -> dict[str, Any]:
    direction = frame["src_lang"].astype(str) + "-" + frame["tgt_lang"].astype(str)
    origins = frame["training_source"].fillna("unknown").astype(str)
    by_direction = Counter(direction.tolist())
    by_origin = Counter(origins.tolist())
    by_direction_origin = Counter(zip(direction.tolist(), origins.tolist(), strict=True))
    return {
        "rows": len(frame),
        "by_direction": dict(sorted(by_direction.items())),
        "by_training_source": dict(sorted(by_origin.items())),
        "by_direction_and_training_source": {
            f"{direction}|{origin}": count
            for (direction, origin), count in sorted(by_direction_origin.items())
        },
    }


def _effective_origin_mass(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    working = frame.copy()
    working["direction"] = (
        working["src_lang"].astype(str) + "-" + working["tgt_lang"].astype(str)
    )
    working["origin_group"] = working["training_source"].apply(
        lambda value: "teacher" if "teacher" in str(value).lower() else "human"
    )
    working["weight"] = working["weight"].fillna(1.0).astype(float)
    result: dict[str, dict[str, float]] = {}
    for (direction, origin), group in working.groupby(
        ["direction", "origin_group"], sort=True
    ):
        result.setdefault(str(direction), {})[str(origin)] = float(
            group["weight"].sum()
        )
    return result


def _balance_teacher_mass(
    frame: pd.DataFrame, teacher_ratio: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0.0 < teacher_ratio < 1.0:
        raise ValueError("teacher_effective_ratio must be between 0 and 1.")
    balanced = frame.copy()
    balanced["weight"] = balanced["weight"].fillna(1.0).astype(float)
    directions = balanced["src_lang"].astype(str) + "-" + balanced["tgt_lang"].astype(str)
    teacher_mask = balanced["training_source"].astype(str).str.lower().str.contains(
        "teacher"
    )
    multipliers: dict[str, float] = {}
    before = _effective_origin_mass(balanced)
    for direction in sorted(set(directions)):
        current = directions == direction
        teacher_mass = float(balanced.loc[current & teacher_mask, "weight"].sum())
        human_mass = float(balanced.loc[current & ~teacher_mask, "weight"].sum())
        if teacher_mass <= 0 or human_mass <= 0:
            raise RuntimeError(
                f"{direction} requires both Teacher KD and human replay for weighting."
            )
        multiplier = (
            teacher_ratio * human_mass / ((1.0 - teacher_ratio) * teacher_mass)
        )
        balanced.loc[current & teacher_mask, "weight"] *= multiplier
        multipliers[direction] = multiplier
    after = _effective_origin_mass(balanced)
    achieved = {
        direction: masses["teacher"] / (masses["teacher"] + masses["human"])
        for direction, masses in after.items()
    }
    return balanced, {
        "target_teacher_effective_ratio": teacher_ratio,
        "teacher_weight_multipliers": multipliers,
        "effective_mass_before": before,
        "effective_mass_after": after,
        "achieved_teacher_effective_ratio": achieved,
    }


def validate(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    configured = [item.get("id") for item in config.get("pairs", [])]
    if configured != list(PAIR_IDS):
        errors.append(f"pairs must be ordered exactly as {list(PAIR_IDS)}")
    inventory: dict[str, Any] = {}
    for pair in config.get("pairs", []):
        pair_id = str(pair.get("id"))
        languages = pair.get("languages", [])
        if len(languages) != 2 or set(pair_id.split("_")) != set(languages):
            errors.append(f"{pair_id}: languages do not match pair id")
        if pair.get("kd_quality_status") != "approved":
            errors.append(f"{pair_id}: KD data is not approved")
        if pair.get("selected_baseline_stage") not in {"exp1", "exp2"}:
            errors.append(f"{pair_id}: invalid selected_baseline_stage")
        paths = {
            key: project_path(pair.get(key, ""))
            for key in (
                "kd_train",
                "human_train",
                "validation",
                "exp1_model",
                "exp2_model",
            )
        }
        inventory[pair_id] = {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in paths.items()
        }
    report = {
        "schema_version": 1,
        "experiment": config.get("experiment", {}).get("id"),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "inventory": inventory,
    }
    write_json(
        PROJECT_ROOT / "reports/pipeline/weak_pair_ablation/validation.json", report
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def prepare(
    config: dict[str, Any],
    pair_id: str,
    variant: str,
    direction: str | None = None,
) -> dict[str, Any]:
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Cannot prepare non-training variant {variant!r}.")
    pair = pair_config(config, pair_id)
    if pair.get("kd_quality_status") != "approved":
        raise RuntimeError(f"{pair_id} KD data has not passed its quality gate.")
    if variant == "directional_full":
        direction = validate_direction(pair, direction)
    elif direction is not None:
        raise ValueError(f"{variant} does not accept --direction.")

    source_path = project_path(pair["kd_train"])
    validation_path = project_path(pair["validation"])
    human_path = project_path(pair["human_train"])
    train_frame = normalize_rows(_read_table(source_path), origin=pair["kd_train"])
    validation_frame = normalize_rows(
        _read_table(validation_path), origin=pair["validation"]
    )
    previous_train = normalize_rows(_read_table(human_path), origin=pair["human_train"])
    _validate_pair_frame(train_frame, pair, "KD data")
    _validate_pair_frame(validation_frame, pair, "validation data")

    train_frame = train_frame.drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    ).copy()
    benchmark_paths = [
        project_path(config["benchmarks"]["flores_dev"]),
        project_path(config["benchmarks"]["protected_devtest"]),
    ]
    train_frame, validation_frame, leakage = protect_splits(
        train_frame,
        validation_frame,
        [pd.read_parquet(path) for path in benchmark_paths],
        tuple(pair["languages"]),
        previous_train,
    )
    if variant == "directional_full":
        source, target = direction.split("-")
        train_frame = train_frame[
            (train_frame["src_lang"] == source)
            & (train_frame["tgt_lang"] == target)
        ].copy()
        validation_frame = validation_frame[
            (validation_frame["src_lang"] == source)
            & (validation_frame["tgt_lang"] == target)
        ].copy()
    weighting = None
    if variant == "full_weighted_60_40":
        settings = variant_settings(config, variant)
        train_frame, weighting = _balance_teacher_mass(
            train_frame, float(settings["teacher_effective_ratio"])
        )
    if train_frame.empty or validation_frame.empty:
        raise RuntimeError(f"{pair_id} {run_id(variant, direction)} has empty data.")

    output = prepared_root(pair_id, variant, direction)
    train_path = output / "train.jsonl"
    prepared_validation_path = output / "validation.jsonl"
    _write_jsonl(train_path, train_frame)
    _write_jsonl(prepared_validation_path, validation_frame)
    config_fingerprint = fingerprint(
        {
            "experiment": config["experiment"],
            "pair": pair,
            "variant": variant,
            "direction": direction,
            "training": config["training"],
            "variant_settings": variant_settings(config, variant),
        }
    )
    report = {
        "schema_version": 1,
        "pair": pair_id,
        "variant": variant,
        "direction": direction,
        "run_id": run_id(variant, direction),
        "selection": "all unique eligible rows; no resampling or repetition",
        "composition": _composition(train_frame),
        "weighting": weighting,
        "validation_rows": len(validation_frame),
        "leakage_audit": leakage,
        "config_fingerprint": config_fingerprint,
        "file_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in (
                source_path,
                validation_path,
                human_path,
                *benchmark_paths,
                train_path,
                prepared_validation_path,
            )
        },
    }
    write_json(output / "prepare_report.json", report)
    return report


def verify_prepared(
    config: dict[str, Any],
    pair_id: str,
    variant: str,
    direction: str | None = None,
) -> None:
    pair = pair_config(config, pair_id)
    root = prepared_root(pair_id, variant, direction)
    report_path = root / "prepare_report.json"
    message = f"{pair_id} {run_id(variant, direction)} data is missing or stale."
    if not report_path.is_file():
        raise RuntimeError(message)
    report = read_json(report_path)
    expected = fingerprint(
        {
            "experiment": config["experiment"],
            "pair": pair,
            "variant": variant,
            "direction": direction,
            "training": config["training"],
            "variant_settings": variant_settings(config, variant),
        }
    )
    if (
        report.get("config_fingerprint") != expected
        or report.get("leakage_audit", {}).get("protected_overlap_after") != 0
    ):
        raise RuntimeError(message)
    for name, expected_hash in report.get("file_sha256", {}).items():
        path = PROJECT_ROOT / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"{message} Changed file: {name}")


def trained_model_path(
    pair_id: str, variant: str, direction: str | None = None
) -> Path:
    root = artifact_root(pair_id, variant, direction) / "best_model"
    if variant != "directional_full":
        return root / "shared"
    checked = str(direction).replace("-", "_")
    return root / checked


def model_path(
    config: dict[str, Any],
    pair_id: str,
    variant: str,
    direction: str | None = None,
) -> Path:
    pair = pair_config(config, pair_id)
    if variant == "baseline_exp1":
        return project_path(pair["exp1_model"])
    if variant == "baseline_exp2":
        return project_path(pair["exp2_model"])
    return trained_model_path(pair_id, variant, direction)


def train(
    config: dict[str, Any],
    pair_id: str,
    variant: str,
    direction: str | None = None,
) -> dict[str, Any]:
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Cannot train baseline variant {variant!r}.")
    pair = pair_config(config, pair_id)
    if variant == "directional_full":
        direction = validate_direction(pair, direction)
        source, target = direction.split("-")
        shared = False
    else:
        if direction is not None:
            raise ValueError(f"{variant} does not accept --direction.")
        source, target = pair["languages"]
        shared = True
    verify_prepared(config, pair_id, variant, direction)
    source_model = project_path(pair["exp1_model"])
    model_files(source_model)
    prepared = prepared_root(pair_id, variant, direction)
    destination = trained_model_path(pair_id, variant, direction)
    runtime_config = {
        **config,
        "training": {
            **config["training"],
            "exp2": {
                **config["training"]["exp2"],
                "learning_rate": float(
                    variant_settings(config, variant)["learning_rate"]
                ),
            },
        },
    }
    report = train_model(
        _candidate(config),
        str(source_model),
        source,
        target,
        _read_jsonl(prepared / "train.jsonl"),
        _read_jsonl(prepared / "validation.jsonl"),
        destination,
        runtime_config,
        experiment="exp2",
        shared=shared,
    )
    payload = {
        "schema_version": 1,
        "pair": pair_id,
        "variant": variant,
        "direction": direction,
        "source_model": str(source_model),
        "model": report,
    }
    write_json(artifact_root(pair_id, variant, direction) / "train_report.json", payload)
    return payload


def evaluate(
    config: dict[str, Any],
    pair_id: str,
    variant: str,
    direction: str | None = None,
) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    if variant == "directional_full":
        direction = validate_direction(pair, direction)
        directions = (direction,)
    else:
        if direction is not None:
            raise ValueError(f"{variant} does not accept --direction.")
        directions = pair_directions(pair)
    path = model_path(config, pair_id, variant, direction)
    model_files(path)
    candidate = _candidate(config, path)
    benchmark = pd.read_parquet(project_path(config["benchmarks"]["flores_dev"]))
    tokenizer, model = load_model(candidate, *pair["languages"])
    scores: dict[str, Any] = {}
    try:
        for current in directions:
            source, target = current.split("-")
            predictions = translate(
                tokenizer,
                model,
                candidate["family"],
                source,
                target,
                benchmark[source].astype(str).tolist(),
                config,
            )
            scores[current] = metrics(
                predictions, benchmark[target].astype(str).tolist(), target
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload = {
        "schema_version": 1,
        "pair": pair_id,
        "variant": variant,
        "direction": direction,
        "benchmark": "flores_dev",
        "model_path": str(path),
        "scores": scores,
    }
    write_json(evaluation_path(pair_id, variant, direction), payload)
    return payload


def _signal(delta: float, noise_floor: float, meaningful: float) -> str:
    if delta >= meaningful:
        return "meaningful_gain"
    if delta >= noise_floor:
        return "small_gain"
    if delta > -noise_floor:
        return "flat"
    return "regression"


def compare(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    baseline_variant = f"baseline_{pair['selected_baseline_stage']}"
    baseline_path = evaluation_path(pair_id, baseline_variant)
    if not baseline_path.is_file():
        raise RuntimeError(f"Evaluate {baseline_variant} before comparing {pair_id}.")
    baseline = read_json(baseline_path)["scores"]
    noise_floor = float(config["comparison"]["noise_floor_chrf2"])
    meaningful = float(config["comparison"]["meaningful_gain_chrf2"])
    variants: list[tuple[str, str | None]] = [
        ("bidir_full", None),
        ("full_weighted_60_40", None),
        ("full_native_lr2e6", None),
    ]
    variants.extend(("directional_full", item) for item in pair_directions(pair))
    rows: list[dict[str, Any]] = []
    winners: dict[str, Any] = {}
    for direction in pair_directions(pair):
        candidates = [
            {
                "variant": baseline_variant,
                "direction": direction,
                **baseline[direction],
                "delta_bleu": 0.0,
                "delta_chrf2": 0.0,
                "signal": "baseline",
            }
        ]
        for variant, variant_direction in variants:
            path = evaluation_path(pair_id, variant, variant_direction)
            if not path.is_file():
                continue
            score = read_json(path)["scores"].get(direction)
            if score is None:
                continue
            delta_bleu = float(score["bleu"]) - float(baseline[direction]["bleu"])
            delta_chrf2 = float(score["chrf2"]) - float(
                baseline[direction]["chrf2"]
            )
            candidates.append(
                {
                    "variant": run_id(variant, variant_direction),
                    "direction": direction,
                    **score,
                    "delta_bleu": delta_bleu,
                    "delta_chrf2": delta_chrf2,
                    "signal": _signal(delta_chrf2, noise_floor, meaningful),
                }
            )
        candidates.sort(key=lambda item: (item["chrf2"], item["bleu"]), reverse=True)
        winners[direction] = candidates[0]
        rows.extend(candidates)
    report = {
        "schema_version": 1,
        "pair": pair_id,
        "benchmark": "flores_dev",
        "baseline_variant": baseline_variant,
        "thresholds": {
            "noise_floor_chrf2": noise_floor,
            "meaningful_gain_chrf2": meaningful,
        },
        "winners": winners,
        "comparisons": rows,
        "missing_runs_are_omitted": True,
    }
    write_json(
        PROJECT_ROOT
        / f"results/evaluation/weak_pair_ablation/{pair_id}/comparison.json",
        report,
    )
    return report


def status(config: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for pair in config["pairs"]:
        pair_id = pair["id"]
        for variant, direction in (
            ("baseline_exp1", None),
            ("baseline_exp2", None),
            ("bidir_full", None),
            ("full_weighted_60_40", None),
            ("full_native_lr2e6", None),
            *(('directional_full', item) for item in pair_directions(pair)),
        ):
            rows.append(
                {
                    "pair": pair_id,
                    "run_id": run_id(variant, direction),
                    "prepared": (
                        prepared_root(pair_id, variant, direction)
                        / "prepare_report.json"
                    ).is_file()
                    if variant in TRAIN_VARIANTS
                    else None,
                    "model": model_path(config, pair_id, variant, direction).is_dir(),
                    "evaluated": evaluation_path(
                        pair_id, variant, direction
                    ).is_file(),
                }
            )
    return {"experiment": config["experiment"]["id"], "runs": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weak-pair full-data and directional student ablations."
    )
    parser.add_argument(
        "action", choices=("validate", "status", "prepare", "train", "evaluate", "compare")
    )
    parser.add_argument(
        "--config", default="configs/specialists/weak_pair_ablation.toml"
    )
    parser.add_argument("--pair", choices=PAIR_IDS)
    parser.add_argument("--variant", choices=ALL_VARIANTS)
    parser.add_argument("--direction")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action in {"prepare", "train", "evaluate"}:
        if not args.pair or not args.variant:
            parser.error(f"{args.action} requires --pair and --variant")
    if args.action == "compare" and not args.pair:
        parser.error("compare requires --pair")
    if args.variant == "directional_full" and not args.direction:
        parser.error("directional_full requires --direction")
    if args.variant != "directional_full" and args.direction:
        parser.error("--direction is only valid with directional_full")

    if args.action == "validate":
        result = validate(config)
    elif args.action == "status":
        result = status(config)
    elif args.action == "prepare":
        result = prepare(config, args.pair, args.variant, args.direction)
    elif args.action == "train":
        result = train(config, args.pair, args.variant, args.direction)
    elif args.action == "evaluate":
        result = evaluate(config, args.pair, args.variant, args.direction)
    else:
        result = compare(config, args.pair)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
