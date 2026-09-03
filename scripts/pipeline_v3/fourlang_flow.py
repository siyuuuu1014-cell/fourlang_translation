from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from huggingface_hub import snapshot_download

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.build_benchmarks import download, read_flores  # noqa: E402
from scripts.pipeline_v2.common import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    read_json,
    write_json,
)
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    candidate_path,
    load_model,
    metrics,
    train_model,
    translate,
)

LANGUAGES = ("en", "zh", "uz", "ru")
UNORDERED_PAIRS = (
    "en_zh",
    "en_uz",
    "en_ru",
    "zh_uz",
    "zh_ru",
    "uz_ru",
)


def directions(languages: tuple[str, ...] = LANGUAGES) -> tuple[str, ...]:
    return tuple(
        f"{source}-{target}"
        for source in languages
        for target in languages
        if source != target
    )


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        return pd.read_json(path, lines=True)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported training data format: {path}")


def normalize_rows(frame: pd.DataFrame, *, origin: str) -> pd.DataFrame:
    aliases = {
        "source_lang": "src_lang",
        "target_lang": "tgt_lang",
        "source_text": "src_text",
        "target_text": "tgt_text",
        "training_weight": "weight",
    }
    frame = frame.rename(
        columns={old: new for old, new in aliases.items() if new not in frame.columns}
    ).copy()
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{origin} is missing directed columns: {sorted(missing)}")
    frame["src_lang"] = frame["src_lang"].astype(str).str.lower()
    frame["tgt_lang"] = frame["tgt_lang"].astype(str).str.lower()
    frame["src_text"] = frame["src_text"].fillna("").astype(str).str.strip()
    frame["tgt_text"] = frame["tgt_text"].fillna("").astype(str).str.strip()
    frame = frame[(frame["src_text"] != "") & (frame["tgt_text"] != "")].copy()
    if "weight" in frame.columns:
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0)
    else:
        frame["weight"] = 1.0
    frame["training_source"] = frame.get("training_source", "human_parallel")
    frame["origin"] = origin
    present = set(
        frame["src_lang"].astype(str) + "-" + frame["tgt_lang"].astype(str)
    )
    invalid = sorted(present - set(directions()))
    if invalid:
        raise ValueError(f"{origin} contains unsupported directions: {invalid}")
    return frame[
        [
            "src_lang",
            "tgt_lang",
            "src_text",
            "tgt_text",
            "weight",
            "training_source",
            "origin",
        ]
    ]


def balance_training_rows(
    frame: pd.DataFrame, *, seed: int, configured_rows: int = 0
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["direction"] = frame["src_lang"] + "-" + frame["tgt_lang"]
    counts = Counter(frame["direction"])
    missing = sorted(set(directions()) - set(counts))
    if missing:
        raise RuntimeError(f"Training data is missing directions: {missing}")
    target = configured_rows or min(counts.values())
    if target < 1:
        raise ValueError("Balanced rows per direction must be positive.")
    if any(counts[item] < target for item in directions()):
        short = {item: counts[item] for item in directions() if counts[item] < target}
        raise RuntimeError(f"Directions below configured balance target {target}: {short}")
    parts = []
    for index, direction in enumerate(directions()):
        part = frame[frame["direction"] == direction].sample(
            n=target, random_state=seed + index
        )
        parts.append(part)
    balanced = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    return balanced.drop(columns="direction"), {
        "input_rows_by_direction": dict(sorted(counts.items())),
        "balanced_rows_per_direction": target,
        "output_rows": len(balanced),
        "directions": len(directions()),
    }


def validate(config: dict[str, Any]) -> None:
    languages = tuple(config["multilingual"]["languages"])
    errors = []
    if languages != LANGUAGES:
        errors.append(f"languages must be exactly {list(LANGUAGES)}")
    pairs = [item["pair"] for item in config["pair_data"]]
    if set(pairs) != set(UNORDERED_PAIRS) or len(pairs) != len(set(pairs)):
        errors.append(f"pair_data must contain exactly {list(UNORDERED_PAIRS)}")
    candidates = {item["id"]: item for item in config["student_candidates"]}
    expected_candidates = {"small100", "m2m100_418m", "nllb_600m"}
    if set(candidates) != expected_candidates:
        errors.append(f"student candidates must be {sorted(expected_candidates)}")
    inventory = {}
    missing_data = []
    for item in config["pair_data"]:
        inventory[item["pair"]] = {}
        for field in ("train", "validation", "kd_train"):
            path = _path(item[field])
            inventory[item["pair"]][field] = {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "exists": path.is_file(),
            }
            if not path.is_file():
                missing_data.append(str(path.relative_to(PROJECT_ROOT)))
    status = "FAIL" if errors else ("WAITING_FOR_DATA" if missing_data else "PASS")
    report = {
        "schema_version": 1,
        "status": status,
        "languages": list(languages),
        "directions": list(directions(languages)),
        "single_final_model": True,
        "errors": errors,
        "missing_data": missing_data,
        "data_inventory": inventory,
    }
    write_json(PROJECT_ROOT / "reports/pipeline/fourlang/validation.json", report)
    if errors:
        raise RuntimeError("; ".join(errors))


def build_benchmarks(config: dict[str, Any]) -> None:
    benchmark = config["benchmarks"]
    payload = download(benchmark["flores_url"])
    codes = config["language_codes"]["nllb"]
    report = {"protected_from_training": True, "splits": {}}
    for split, key in (("dev", "flores_dev"), ("devtest", "flores_devtest")):
        columns = {lang: read_flores(payload, codes[lang], split) for lang in LANGUAGES}
        lengths = {len(value) for value in columns.values()}
        if len(lengths) != 1:
            raise RuntimeError(f"FLORES {split} is not aligned across four languages.")
        frame = pd.DataFrame(columns)
        path = _path(benchmark[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        report["splits"][split] = {"rows": len(frame), "path": str(path.relative_to(PROJECT_ROOT))}
    write_json(PROJECT_ROOT / "reports/pipeline/fourlang/benchmarks.json", report)


def aggregate(config: dict[str, Any], experiment: str) -> None:
    train_field = "train" if experiment == "exp1" else "kd_train"
    train_parts, validation_parts = [], []
    for item in config["pair_data"]:
        train_path = _path(item[train_field])
        validation_path = _path(item["validation"])
        if not train_path.is_file() or not validation_path.is_file():
            raise FileNotFoundError(
                f"Missing {item['pair']} input: {train_path} or {validation_path}"
            )
        train_part = normalize_rows(
            _read_table(train_path), origin=str(train_path.relative_to(PROJECT_ROOT))
        )
        expected = set(item["pair"].split("_"))
        if any(
            {row.src_lang, row.tgt_lang} != expected
            for row in train_part[["src_lang", "tgt_lang"]].itertuples(index=False)
        ):
            raise RuntimeError(f"{item['pair']} train input contains another language pair.")
        train_parts.append(train_part)
        validation = normalize_rows(
            _read_table(validation_path), origin=str(validation_path.relative_to(PROJECT_ROOT))
        )
        if any(
            {row.src_lang, row.tgt_lang} != expected
            for row in validation[["src_lang", "tgt_lang"]].itertuples(index=False)
        ):
            raise RuntimeError(
                f"{item['pair']} validation input contains another language pair."
            )
        untrusted = validation["training_source"].astype(str).str.lower().str.contains(
            "teacher|pseudo|synthetic", regex=True
        )
        if untrusted.any():
            raise RuntimeError(f"Pseudo/Teacher rows are forbidden in validation: {item['pair']}")
        validation_parts.append(validation)
    train = pd.concat(train_parts, ignore_index=True).drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    )
    train, report = balance_training_rows(
        train,
        seed=int(config["multilingual"]["seed"]),
        configured_rows=int(config["balancing"]["train_rows_per_direction"]),
    )
    validation = pd.concat(validation_parts, ignore_index=True).drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    )
    output = PROJECT_ROOT / "data/multilingual/fourlang" / experiment
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("train", train), ("validation", validation)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in frame.to_dict("records")),
            encoding="utf-8",
        )
    report["validation_rows"] = len(validation)
    report["experiment"] = experiment
    write_json(PROJECT_ROOT / f"reports/pipeline/fourlang/{experiment}_data.json", report)


def _candidate_by_id(config: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [item for item in config["student_candidates"] if item["id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(candidate_id)
    return dict(matches[0])


def prepare_models(config: dict[str, Any]) -> None:
    report = {}
    for candidate in config["student_candidates"]:
        local = Path(candidate["path"])
        snapshot_download(
            repo_id=candidate["repo_id"],
            revision=candidate["revision"],
            local_dir=local,
        )
        report[candidate["id"]] = {"path": str(local), "revision": candidate["revision"]}
    write_json(PROJECT_ROOT / "reports/pipeline/fourlang/model_inventory.json", report)


def evaluate_candidate(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_parquet(_path(config["benchmarks"]["flores_dev"]))
    tokenizer, model = load_model(candidate, "en", "zh")
    result = {}
    try:
        for direction in directions():
            source, target = direction.split("-")
            predictions = translate(
                tokenizer,
                model,
                candidate["family"],
                source,
                target,
                frame[source].astype(str).tolist(),
                config,
            )
            result[direction] = metrics(predictions, frame[target].astype(str).tolist(), target)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    values = list(result.values())
    return {
        "status": "ok",
        "metrics": result,
        "macro_chrf2": sum(item["chrf2"] for item in values) / len(values),
        "worst_chrf2": min(item["chrf2"] for item in values),
        "macro_bleu": sum(item["bleu"] for item in values) / len(values),
    }


def bakeoff(config: dict[str, Any]) -> None:
    results = {}
    for candidate in config["student_candidates"]:
        try:
            results[candidate["id"]] = evaluate_candidate(config, candidate)
        except Exception as error:
            results[candidate["id"]] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
    write_json(PROJECT_ROOT / "results/model_selection/fourlang/student_scores.json", {"candidates": results})


def select_student(config: dict[str, Any]) -> None:
    scores = read_json(PROJECT_ROOT / "results/model_selection/fourlang/student_scores.json")
    failures = {key: value for key, value in scores["candidates"].items() if value["status"] != "ok"}
    if failures:
        raise RuntimeError(f"Global bake-off is incomplete: {sorted(failures)}")
    ranked = sorted(
        scores["candidates"].items(),
        key=lambda item: (
            item[1]["worst_chrf2"], item[1]["macro_chrf2"], item[1]["macro_bleu"]
        ),
        reverse=True,
    )
    winner = ranked[0][0]
    write_json(
        PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json",
        {
            "candidate_id": winner,
            "candidate": _candidate_by_id(config, winner),
            "selection_policy": "maximize worst-direction chrF2, then macro chrF2, then macro BLEU across all 12 directions",
            "ranking": [
                {
                    "candidate_id": key,
                    "worst_chrf2": value["worst_chrf2"],
                    "macro_chrf2": value["macro_chrf2"],
                    "macro_bleu": value["macro_bleu"],
                }
                for key, value in ranked
            ],
            "training_layout": "single_four_language_model",
        },
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def train(config: dict[str, Any], experiment: str) -> None:
    selected = read_json(PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json")
    candidate = selected["candidate"]
    source_model = (
        candidate_path(candidate, "en", "zh")
        if experiment == "exp1"
        else str(PROJECT_ROOT / "results/student/fourlang/exp1/best_model/shared")
    )
    root = PROJECT_ROOT / f"results/student/fourlang/{experiment}"
    report = train_model(
        candidate,
        source_model,
        "en",
        "zh",
        _jsonl(PROJECT_ROOT / f"data/multilingual/fourlang/{experiment}/train.jsonl"),
        _jsonl(PROJECT_ROOT / f"data/multilingual/fourlang/{experiment}/validation.jsonl"),
        root / "best_model/shared",
        config,
        experiment=experiment,
        shared=True,
    )
    write_json(root / "train_report.json", {"single_model": True, "model": report})
    write_json(
        root / "model_layout.json",
        {
            "layout": "single_four_language_model",
            "candidate": candidate,
            "directions": list(directions()),
            "artifacts": [str((root / "best_model/shared/config.json").relative_to(PROJECT_ROOT))],
        },
    )


def evaluate(config: dict[str, Any], experiment: str) -> None:
    selected = read_json(PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json")
    candidate = {
        **selected["candidate"],
        "path": str(PROJECT_ROOT / f"results/student/fourlang/{experiment}/best_model/shared"),
    }
    frame = pd.read_parquet(_path(config["benchmarks"]["flores_devtest"]))
    tokenizer, model = load_model(candidate, "en", "zh")
    result = {}
    try:
        for direction in directions():
            source, target = direction.split("-")
            predictions = translate(
                tokenizer, model, candidate["family"], source, target,
                frame[source].astype(str).tolist(), config,
            )
            result[direction] = metrics(predictions, frame[target].astype(str).tolist(), target)
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(PROJECT_ROOT / f"results/evaluation/fourlang/{experiment}/metrics.json", result)


def promotion_gate(config: dict[str, Any]) -> None:
    baseline = read_json(PROJECT_ROOT / "results/evaluation/fourlang/exp1/metrics.json")
    candidate = read_json(PROJECT_ROOT / "results/evaluation/fourlang/exp2/metrics.json")
    checks = []
    for direction in directions():
        passed = all(candidate[direction][metric] >= baseline[direction][metric] for metric in ("bleu", "chrf2"))
        checks.append({"direction": direction, "passed": passed, "exp1": baseline[direction], "exp2": candidate[direction]})
    report = {"status": "PASS" if all(item["passed"] for item in checks) else "FAIL", "directions": checks}
    write_json(PROJECT_ROOT / "results/evaluation/fourlang/promotion_gate.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Exp2 did not match or beat Exp1 in every direction.")


def freeze(config: dict[str, Any]) -> None:
    gate = read_json(PROJECT_ROOT / "results/evaluation/fourlang/promotion_gate.json")
    if gate["status"] != "PASS":
        raise RuntimeError("Refusing to freeze without a passing 12-direction gate.")
    selected = read_json(PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json")
    destination = _path(config["deployment"]["destination"])
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise RuntimeError(f"Stale staging path exists: {staging}")
    shutil.copytree(PROJECT_ROOT / "results/student/fourlang/exp2/best_model/shared", staging)
    backup = None
    if destination.exists():
        backup = destination.with_name(destination.name + f".backup-{int(time.time())}")
        destination.rename(backup)
    staging.rename(destination)
    registry_path = _path(config["deployment"]["registry"])
    registry = read_json(registry_path)
    family = selected["candidate"]["family"]
    for direction in directions():
        source, target = direction.split("-")
        registry["models"][direction.replace("-", "_")] = {
            "model_name": config["deployment"]["model_name"],
            "architecture": family,
            "path": str(destination.relative_to(PROJECT_ROOT).as_posix()),
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
            "single_model": True,
            "languages": list(LANGUAGES),
            "directions": list(directions()),
            "student_selection": selected,
            "promotion_gate": gate,
            "previous_deployment_backup": str(backup) if backup else None,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="One-model four-language translation pipeline.")
    parser.add_argument(
        "action",
        choices=("validate", "benchmarks", "prepare_models", "aggregate", "bakeoff", "select", "train", "evaluate", "gate", "freeze"),
    )
    parser.add_argument("--config", default="configs/multilingual/fourlang.toml")
    parser.add_argument("--experiment", choices=("exp1", "exp2"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "validate":
        validate(config)
    elif args.action == "benchmarks":
        build_benchmarks(config)
    elif args.action == "prepare_models":
        prepare_models(config)
    elif args.action == "aggregate":
        if not args.experiment:
            parser.error("aggregate requires --experiment")
        aggregate(config, args.experiment)
    elif args.action == "bakeoff":
        bakeoff(config)
    elif args.action == "select":
        select_student(config)
    elif args.action == "train":
        if not args.experiment:
            parser.error("train requires --experiment")
        train(config, args.experiment)
    elif args.action == "evaluate":
        if not args.experiment:
            parser.error("evaluate requires --experiment")
        evaluate(config, args.experiment)
    elif args.action == "gate":
        promotion_gate(config)
    else:
        freeze(config)


if __name__ == "__main__":
    main()
