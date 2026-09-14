"""Orchestrate six independent bidirectional specialists using existing training logic."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import load_config, read_json, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    candidate_path,
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
from scripts.pipeline_v3.fourlang_flow import (  # noqa: E402
    _read_table,
    normalize_rows,
    sample_coverage_first,
)

PROJECT_ROOT = PROJECT_ROOT_BOOTSTRAP
PAIR_IDS = ("en_zh", "en_uz", "en_ru", "zh_uz", "zh_ru", "uz_ru")
MODES = {"train_full", "resume_exp2", "reuse_exp2"}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def pair_config(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    matches = [item for item in config["pairs"] if item["id"] == pair_id]
    if len(matches) != 1:
        raise KeyError(f"Expected exactly one pair configuration for {pair_id!r}.")
    return dict(matches[0])


def pair_directions(pair: dict[str, Any]) -> tuple[str, str]:
    source, target = pair["languages"]
    return f"{source}-{target}", f"{target}-{source}"


def artifact_root(pair_id: str, experiment: str) -> Path:
    return PROJECT_ROOT / "results/student/pair_specialists" / pair_id / experiment


def prepared_root(pair_id: str, experiment: str) -> Path:
    return PROJECT_ROOT / "data/specialists" / pair_id / experiment


def _candidate(
    config: dict[str, Any], path: str | Path | None = None
) -> dict[str, Any]:
    candidate = {
        "id": config["experiment"]["backbone"],
        **config["model"],
    }
    if path is not None:
        candidate["path"] = str(path)
        candidate["require_local_artifact"] = True
    return candidate


def _validate_pair_frame(frame: pd.DataFrame, pair: dict[str, Any], label: str) -> None:
    expected = {tuple(direction.split("-")) for direction in pair_directions(pair)}
    present = set(zip(frame["src_lang"], frame["tgt_lang"], strict=True))
    if present != expected:
        raise RuntimeError(
            f"{pair['id']} {label} directions must be {sorted(expected)}; "
            f"found {sorted(present)}."
        )


def _sample_training_rows(
    frame: pd.DataFrame,
    pair: dict[str, Any],
    balancing: dict[str, Any],
    seed: int,
    experiment: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    ).copy()
    frame["direction"] = frame["src_lang"] + "-" + frame["tgt_lang"]
    target_rows = int(balancing["rows_per_direction"])
    teacher_ratio = float(balancing.get("teacher_ratio", 0.0))
    max_teacher_repeats = int(balancing.get("max_teacher_repeats", 3))
    parts: list[pd.DataFrame] = []
    report: dict[str, Any] = {}
    for index, direction in enumerate(pair_directions(pair)):
        pool = frame[frame["direction"] == direction]
        if pool.empty:
            raise RuntimeError(f"No training rows for {direction}.")
        if experiment == "exp1":
            selected = sample_coverage_first(pool, target_rows, seed + index)
            report[direction] = {
                "available_human_rows": len(pool),
                "selected_human_rows": len(selected),
            }
        else:
            origins = pool["training_source"].fillna("").astype(str).str.lower()
            teacher = pool[origins.str.contains("teacher")]
            human = pool[~origins.str.contains("teacher")]
            teacher_target = round(target_rows * teacher_ratio)
            human_target = target_rows - teacher_target
            if teacher.empty or human.empty:
                raise RuntimeError(
                    f"{direction} Exp2 requires both Teacher KD and human replay."
                )
            repeats = math.ceil(teacher_target / len(teacher))
            if repeats > max_teacher_repeats:
                raise RuntimeError(
                    f"{direction} needs {repeats} Teacher repeats, above the "
                    f"configured maximum {max_teacher_repeats}."
                )
            teacher_selected = sample_coverage_first(
                teacher, teacher_target, seed + index * 2
            )
            human_selected = sample_coverage_first(
                human, human_target, seed + index * 2 + 1
            )
            selected = pd.concat([teacher_selected, human_selected], ignore_index=True)
            selected = selected.sample(frac=1, random_state=seed + index).reset_index(
                drop=True
            )
            report[direction] = {
                "available_teacher_rows": len(teacher),
                "available_human_rows": len(human),
                "selected_teacher_rows": len(teacher_selected),
                "selected_human_rows": len(human_selected),
                "teacher_max_repeats": repeats,
            }
        parts.append(selected)
    return pd.concat(parts, ignore_index=True), report


def validate(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    configured_ids = [item.get("id") for item in config.get("pairs", [])]
    if configured_ids != list(PAIR_IDS):
        errors.append(f"pairs must be ordered exactly as {list(PAIR_IDS)}")
    inventory: dict[str, Any] = {}
    for pair in config.get("pairs", []):
        pair_id = str(pair.get("id"))
        mode = pair.get("mode")
        languages = pair.get("languages", [])
        if mode not in MODES:
            errors.append(f"{pair_id}: unsupported mode {mode!r}")
        if len(languages) != 2 or set(pair_id.split("_")) != set(languages):
            errors.append(f"{pair_id}: languages do not match pair id")
        files = {}
        for key in ("human_train", "validation", "kd_train"):
            path = project_path(pair.get(key, ""))
            files[key] = {"path": str(path), "exists": path.is_file()}
        if mode in {"resume_exp2", "reuse_exp2"}:
            path = project_path(pair.get("existing_exp1", ""))
            files["existing_exp1"] = {"path": str(path), "exists": path.is_dir()}
        if mode == "reuse_exp2":
            path = project_path(pair.get("existing_exp2", ""))
            files["existing_exp2"] = {"path": str(path), "exists": path.is_dir()}
        data_ready = all(
            files[key]["exists"] for key in ("human_train", "validation", "kd_train")
        )
        existing_ready = all(
            item["exists"] for key, item in files.items() if key.startswith("existing_")
        )
        inventory[pair_id] = {
            "mode": mode,
            "kd_quality_status": pair.get("kd_quality_status"),
            "files": files,
            "exp1_ready": files["human_train"]["exists"]
            and files["validation"]["exists"]
            and existing_ready,
            "exp2_ready": data_ready
            and existing_ready
            and pair.get("kd_quality_status") == "approved",
        }
    report = {
        "schema_version": 1,
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "pairs": inventory,
    }
    write_json(
        PROJECT_ROOT / "reports/pipeline/pair_specialists/readiness.json", report
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def prepare(config: dict[str, Any], pair_id: str, experiment: str) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    if experiment == "exp2" and pair.get("kd_quality_status") != "approved":
        raise RuntimeError(
            f"{pair_id} Exp2 is blocked by "
            f"kd_quality_status={pair.get('kd_quality_status')!r}."
        )
    field = "human_train" if experiment == "exp1" else "kd_train"
    source_path = project_path(pair[field])
    validation_path = project_path(pair["validation"])
    train = normalize_rows(_read_table(source_path), origin=pair[field])
    validation = normalize_rows(_read_table(validation_path), origin=pair["validation"])
    _validate_pair_frame(train, pair, field)
    _validate_pair_frame(validation, pair, "validation")
    untrusted = (
        validation["training_source"]
        .astype(str)
        .str.lower()
        .str.contains("teacher|pseudo|synthetic", regex=True)
    )
    if untrusted.any():
        raise RuntimeError(f"{pair_id} validation contains generated targets.")
    benchmark_paths = [
        project_path(config["benchmarks"][name])
        for name in ("flores_dev", "flores_devtest")
    ]
    previous_train = None
    previous_path = None
    if experiment == "exp2":
        if "existing_exp1" in pair:
            # Existing Exp1 models were trained before this branch and may have used
            # more rows than the controlled branch budget. Protect against the real
            # human input, not a newly sampled approximation of that run.
            previous_path = project_path(pair["human_train"])
        else:
            previous_path = prepared_root(pair_id, "exp1") / "train.jsonl"
            if not previous_path.is_file():
                raise FileNotFoundError(
                    f"Prepare {pair_id} Exp1 data before Exp2: {previous_path}"
                )
        previous_train = normalize_rows(_read_table(previous_path), origin="exp1")
    train, validation, leakage = protect_splits(
        train,
        validation,
        [pd.read_parquet(path) for path in benchmark_paths],
        tuple(pair["languages"]),
        previous_train,
    )
    balancing = config["balancing"][experiment]
    train, sampling = _sample_training_rows(
        train,
        pair,
        balancing,
        int(config["experiment"]["seed"]),
        experiment,
    )
    output = prepared_root(pair_id, experiment)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation)):
        (output / f"{name}.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in frame.to_dict("records")
            ),
            encoding="utf-8",
        )
    config_fingerprint = fingerprint(
        {
            "pair": pair,
            "balancing": balancing,
            "seed": config["experiment"]["seed"],
            "experiment": experiment,
        }
    )
    hash_inputs = [
        output / "train.jsonl",
        output / "validation.jsonl",
        source_path,
        validation_path,
        *benchmark_paths,
    ]
    if previous_path is not None and previous_path not in hash_inputs:
        hash_inputs.append(previous_path)
    report = {
        "schema_version": 1,
        "pair": pair_id,
        "experiment": experiment,
        "rows": len(train),
        "validation_rows": len(validation),
        "sampling": sampling,
        "leakage_audit": leakage,
        "config_fingerprint": config_fingerprint,
        "file_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in hash_inputs
        },
    }
    write_json(
        PROJECT_ROOT
        / f"reports/pipeline/pair_specialists/{pair_id}/{experiment}_data.json",
        report,
    )
    return report


def verify_prepared(config: dict[str, Any], pair_id: str, experiment: str) -> None:
    pair = pair_config(config, pair_id)
    report_path = (
        PROJECT_ROOT
        / f"reports/pipeline/pair_specialists/{pair_id}/{experiment}_data.json"
    )
    message = f"{pair_id} {experiment} data is missing, stale, or unaudited; prepare it again."
    if not report_path.is_file():
        raise RuntimeError(message)
    report = read_json(report_path)
    expected = fingerprint(
        {
            "pair": pair,
            "balancing": config["balancing"][experiment],
            "seed": config["experiment"]["seed"],
            "experiment": experiment,
        }
    )
    if (
        report.get("schema_version") != 1
        or report.get("config_fingerprint") != expected
        or report.get("leakage_audit", {}).get("protected_overlap_after") != 0
    ):
        raise RuntimeError(message)
    for name, expected_hash in report.get("file_sha256", {}).items():
        path = PROJECT_ROOT / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"{message} Changed file: {name}")


def _prepared_rows(pair_id: str, experiment: str, split: str) -> list[dict[str, Any]]:
    path = prepared_root(pair_id, experiment) / f"{split}.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _source_model(config: dict[str, Any], pair: dict[str, Any], experiment: str) -> str:
    if experiment == "exp1":
        return candidate_path(_candidate(config), *pair["languages"])
    if "existing_exp1" in pair:
        source = project_path(pair["existing_exp1"])
    else:
        source = artifact_root(pair["id"], "exp1") / "best_model/shared"
    model_files(source)
    return str(source)


def train(config: dict[str, Any], pair_id: str, experiment: str) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    if experiment == "exp2" and pair.get("kd_quality_status") != "approved":
        raise RuntimeError(f"{pair_id} has not passed its data-quality gate.")
    if pair["mode"] == "reuse_exp2":
        raise RuntimeError(
            f"{pair_id} reuses its accepted Exp2 and must not be retrained."
        )
    if experiment == "exp1" and pair["mode"] == "resume_exp2":
        raise RuntimeError(f"{pair_id} already has Exp1; continue with Exp2.")
    verify_prepared(config, pair_id, experiment)
    root = artifact_root(pair_id, experiment)
    source = _source_model(config, pair, experiment)
    report = train_model(
        _candidate(config),
        source,
        *pair["languages"],
        _prepared_rows(pair_id, experiment, "train"),
        _prepared_rows(pair_id, experiment, "validation"),
        root / "best_model/shared",
        config,
        experiment=experiment,
        shared=True,
    )
    payload = {"single_bidirectional_model": True, "model": report}
    write_json(root / "train_report.json", payload)
    write_json(
        root / "model_layout.json",
        {
            "layout": "single_bidirectional_pair_model",
            "candidate": _candidate(config),
            "directions": list(pair_directions(pair)),
            "artifacts": [
                str((root / "best_model/shared/config.json").relative_to(PROJECT_ROOT))
            ],
        },
    )
    return payload


def model_path(pair: dict[str, Any], experiment: str) -> Path:
    existing = pair.get(f"existing_{experiment}")
    if existing:
        return project_path(existing)
    return artifact_root(pair["id"], experiment) / "best_model/shared"


def evaluate(config: dict[str, Any], pair_id: str, experiment: str) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    path = model_path(pair, experiment)
    model_files(path)
    candidate = _candidate(config, path)
    benchmark = pd.read_parquet(project_path(config["benchmarks"]["flores_devtest"]))
    tokenizer, model = load_model(candidate, *pair["languages"])
    result: dict[str, Any] = {}
    try:
        for direction in pair_directions(pair):
            source, target = direction.split("-")
            predictions = translate(
                tokenizer,
                model,
                candidate["family"],
                source,
                target,
                benchmark[source].astype(str).tolist(),
                config,
            )
            result[direction] = metrics(
                predictions, benchmark[target].astype(str).tolist(), target
            )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        PROJECT_ROOT
        / f"results/evaluation/pair_specialists/{pair_id}/{experiment}.json",
        result,
    )
    return result


def gate(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    root = PROJECT_ROOT / "results/evaluation/pair_specialists" / pair_id
    baseline = read_json(root / "exp1.json")
    candidate = read_json(root / "exp2.json")
    pair = pair_config(config, pair_id)
    checks = []
    for direction in pair_directions(pair):
        passed = all(
            candidate[direction][metric] >= baseline[direction][metric]
            for metric in ("bleu", "chrf2")
        )
        checks.append(
            {
                "direction": direction,
                "passed": passed,
                "exp1": baseline[direction],
                "exp2": candidate[direction],
            }
        )
    report = {
        "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
        "directions": checks,
    }
    write_json(root / "promotion_gate.json", report)
    return report


def promote(config: dict[str, Any], pair_id: str) -> dict[str, Any]:
    pair = pair_config(config, pair_id)
    gate_path = (
        PROJECT_ROOT
        / f"results/evaluation/pair_specialists/{pair_id}/promotion_gate.json"
    )
    gate_report = read_json(gate_path)
    if gate_report.get("status") != "PASS":
        raise RuntimeError(f"Refusing to promote {pair_id} without a passing gate.")
    source = model_path(pair, "exp2")
    model_files(source)
    if pair["mode"] == "reuse_exp2":
        destination = source
        backup = None
    else:
        destination = project_path(config["deployment"]["root"]) / f"{pair_id}_v1"
        staging = destination.with_name(destination.name + ".staging")
        if staging.exists():
            raise RuntimeError(f"Stale staging path exists: {staging}")
        shutil.copytree(source, staging)
        backup = None
        if destination.exists():
            backup = destination.with_name(
                destination.name + f".backup-{int(time.time())}"
            )
            destination.rename(backup)
        staging.rename(destination)
    registry_path = project_path(config["deployment"]["registry"])
    registry = read_json(registry_path)
    model_name = f"{pair_id}_pair_specialist_v1"
    relative_destination = str(destination.relative_to(PROJECT_ROOT).as_posix())
    for direction in pair_directions(pair):
        source_lang, target_lang = direction.split("-")
        registry["models"][direction.replace("-", "_")] = {
            "model_name": model_name,
            "architecture": config["model"]["family"],
            "path": relative_destination,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "status": "ready",
            "generation": {
                "num_beams": int(config["deployment"]["num_beams"]),
                "max_new_tokens": int(config["deployment"]["max_new_tokens"]),
                "do_sample": False,
            },
        }
    write_json(registry_path, registry)
    card = {
        "model_name": model_name,
        "architecture": config["model"]["family"],
        "single_bidirectional_model": True,
        "directions": list(pair_directions(pair)),
        "source_model": str(source),
        "promotion_gate": gate_report,
        "previous_deployment_backup": str(backup) if backup else None,
    }
    write_json(destination / "model_card.json", card)
    return card


def status(config: dict[str, Any]) -> dict[str, Any]:
    readiness = validate(config)
    rows = []
    for pair in config["pairs"]:
        pair_id = pair["id"]
        rows.append(
            {
                "pair": pair_id,
                "mode": pair["mode"],
                "kd_quality_status": pair["kd_quality_status"],
                "exp1_ready": readiness["pairs"][pair_id]["exp1_ready"],
                "exp2_ready": readiness["pairs"][pair_id]["exp2_ready"],
                "exp1_model": model_path(pair, "exp1").is_dir(),
                "exp2_model": model_path(pair, "exp2").is_dir(),
            }
        )
    result = {"experiment": config["experiment"]["id"], "pairs": rows}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Six independent bidirectional specialist experiments."
    )
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "status",
            "prepare",
            "train",
            "evaluate",
            "gate",
            "promote",
        ),
    )
    parser.add_argument("--config", default="configs/specialists/six_pair.toml")
    parser.add_argument("--pair", choices=PAIR_IDS)
    parser.add_argument("--experiment", choices=("exp1", "exp2"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action in {"prepare", "train", "evaluate"}:
        if not args.pair or not args.experiment:
            parser.error(f"{args.action} requires --pair and --experiment")
    if args.action in {"gate", "promote"} and not args.pair:
        parser.error(f"{args.action} requires --pair")
    if args.action == "validate":
        print(json.dumps(validate(config), ensure_ascii=False, indent=2))
    elif args.action == "status":
        status(config)
    elif args.action == "prepare":
        print(json.dumps(prepare(config, args.pair, args.experiment), indent=2))
    elif args.action == "train":
        print(json.dumps(train(config, args.pair, args.experiment), indent=2))
    elif args.action == "evaluate":
        print(json.dumps(evaluate(config, args.pair, args.experiment), indent=2))
    elif args.action == "gate":
        print(json.dumps(gate(config, args.pair), indent=2))
    else:
        print(json.dumps(promote(config, args.pair), indent=2))


if __name__ == "__main__":
    main()
