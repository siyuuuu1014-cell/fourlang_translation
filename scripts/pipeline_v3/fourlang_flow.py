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
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)
from scripts.pipeline_v3.data_safety import protect_splits  # noqa: E402
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    file_sha256, fingerprint, model_files,
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
        "training_origin": "training_source",
        "sample_weight": "weight",
        "sample_origin": "training_source",
    }
    frame = frame.rename(
        columns={old: new for old, new in aliases.items() if new not in frame.columns}
    ).copy()
    if ("src_lang" not in frame.columns or "tgt_lang" not in frame.columns) and (
        "direction" in frame.columns
    ):
        normalized_direction = (
            frame["direction"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", "-", regex=False)
        )
        directed = normalized_direction.str.extract(
            r"^(?P<src_lang>[a-z]{2})-(?P<tgt_lang>[a-z]{2})$"
        )
        invalid_direction = directed.isna().any(axis=1)
        if invalid_direction.any():
            examples = sorted(set(normalized_direction[invalid_direction]))[:5]
            raise ValueError(
                f"{origin} contains malformed direction values: {examples}"
            )
        if "src_lang" not in frame.columns:
            frame["src_lang"] = directed["src_lang"]
        if "tgt_lang" not in frame.columns:
            frame["tgt_lang"] = directed["tgt_lang"]
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{origin} is missing directed columns: {sorted(missing)}")
    frame["src_lang"] = frame["src_lang"].astype(str).str.lower()
    frame["tgt_lang"] = frame["tgt_lang"].astype(str).str.lower()
    source_before = frame["src_text"].fillna("").astype(str).tolist()
    target_before = frame["tgt_text"].fillna("").astype(str).tolist()
    frame["src_text"] = [
        normalize_language_text(language, text)
        for language, text in zip(
            frame["src_lang"], source_before, strict=True
        )
    ]
    frame["tgt_text"] = [
        normalize_language_text(language, text)
        for language, text in zip(
            frame["tgt_lang"], target_before, strict=True
        )
    ]
    normalization = Counter()
    for language, before, after in zip(
        frame["src_lang"], source_before, frame["src_text"], strict=True
    ):
        normalization[f"{language}_cells"] += 1
        normalization[f"{language}_converted"] += before.strip() != after
    for language, before, after in zip(
        frame["tgt_lang"], target_before, frame["tgt_text"], strict=True
    ):
        normalization[f"{language}_cells"] += 1
        normalization[f"{language}_converted"] += before.strip() != after
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
    result = frame[
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
    result.attrs["script_normalization"] = dict(sorted(normalization.items()))
    return result


def balance_training_rows(
    frame: pd.DataFrame,
    *,
    seed: int,
    configured_rows: int = 0,
    rows_by_direction: dict[str, int] | None = None,
    teacher_ratio: float | None = None,
    max_teacher_repeats: int = 3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.drop_duplicates(["src_lang", "tgt_lang", "src_text", "tgt_text"]).copy()
    frame["direction"] = frame["src_lang"] + "-" + frame["tgt_lang"]
    counts = Counter(frame["direction"])
    missing = sorted(set(directions()) - set(counts))
    if missing:
        raise RuntimeError(f"Training data is missing directions: {missing}")
    default_target = configured_rows or min(counts.values())
    configured_targets = rows_by_direction or {}
    invalid_targets = sorted(set(configured_targets) - set(directions()))
    if invalid_targets:
        raise ValueError(f"Unsupported direction sampling targets: {invalid_targets}")
    targets = {
        direction: int(configured_targets.get(direction, default_target))
        for direction in directions()
    }
    if any(target < 1 for target in targets.values()):
        raise ValueError("Rows per direction must be positive.")
    if teacher_ratio is not None and not 0.0 <= teacher_ratio <= 1.0:
        raise ValueError("Teacher ratio must be between zero and one.")
    if max_teacher_repeats < 1:
        raise ValueError("Maximum Teacher repeats must be positive.")

    parts = []
    sampled_with_replacement = {}
    source_mix_by_direction = {}
    for index, direction in enumerate(directions()):
        target = targets[direction]
        direction_frame = frame[frame["direction"] == direction]
        available = counts[direction]
        if teacher_ratio is None:
            replace = available < target
            part = sample_coverage_first(direction_frame, target, seed + index)
            teacher_rows = int(
                part["training_source"]
                .astype(str)
                .str.lower()
                .str.contains("teacher")
                .sum()
            )
        else:
            teacher_mask = (
                direction_frame["training_source"]
                .astype(str)
                .str.lower()
                .str.contains("teacher")
            )
            teacher_pool = direction_frame[teacher_mask]
            human_pool = direction_frame[~teacher_mask]
            desired_teacher = round(target * teacher_ratio)
            teacher_rows = min(
                desired_teacher,
                len(teacher_pool) * max_teacher_repeats,
            )
            human_rows = target - teacher_rows
            if human_rows and human_pool.empty:
                raise RuntimeError(
                    f"{direction} cannot satisfy human replay target: no human rows."
                )
            sampled_teacher = sample_coverage_first(teacher_pool, teacher_rows, seed + index * 2)
            sampled_human = sample_coverage_first(human_pool, human_rows, seed + index * 2 + 1)
            part = pd.concat([sampled_teacher, sampled_human], ignore_index=True)
            part = part.sample(frac=1, random_state=seed + index).reset_index(drop=True)
            replace = len(teacher_pool) < teacher_rows or len(human_pool) < human_rows
            source_mix_by_direction[direction] = {
                "teacher_kd": teacher_rows,
                "human_replay": human_rows,
                "available_teacher_rows": len(teacher_pool),
                "available_human_rows": len(human_pool),
                "teacher_unique_sampled": min(teacher_rows, len(teacher_pool)),
                "human_unique_sampled": min(human_rows, len(human_pool)),
                "teacher_max_repeats": (teacher_rows + len(teacher_pool) - 1) // len(teacher_pool) if len(teacher_pool) else 0,
                "human_max_repeats": (human_rows + len(human_pool) - 1) // len(human_pool) if len(human_pool) else 0,
            }
        parts.append(part)
        if replace:
            sampled_with_replacement[direction] = {
                "available_unique_rows": available,
                "sampled_rows": target,
            }
    balanced = pd.concat(parts, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    return balanced.drop(columns="direction"), {
        "sampling_strategy": "coverage_first_cycles_v1",
        "input_rows_by_direction": dict(sorted(counts.items())),
        "target_rows_by_direction": targets,
        "balanced_rows_per_direction": (
            next(iter(targets.values())) if len(set(targets.values())) == 1 else None
        ),
        "output_rows": len(balanced),
        "directions": len(directions()),
        "sampled_with_replacement": sampled_with_replacement,
        "source_mix_by_direction": source_mix_by_direction,
    }


def sample_coverage_first(pool: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """Cover every available row before repeats; multiplicities differ by at most one."""
    if size == 0:
        return pool.iloc[:0].copy()
    if size < 0 or pool.empty:
        raise ValueError("Cannot sample a positive quota from an empty pool.")
    cycles, remainder = divmod(size, len(pool))
    parts = [pool.sample(frac=1, random_state=seed + cycle) for cycle in range(cycles)]
    if remainder:
        parts.append(pool.sample(n=remainder, random_state=seed + cycles))
    return pd.concat(parts, ignore_index=True)


def validate(config: dict[str, Any]) -> None:
    languages = tuple(config["multilingual"]["languages"])
    errors = []
    if languages != LANGUAGES:
        errors.append(f"languages must be exactly {list(LANGUAGES)}")
    codes = config.get("language_codes", {}).get("nllb", {})
    if codes.get("zh") != "zho_Hans":
        errors.append("zh must use the Simplified Chinese code zho_Hans")
    if codes.get("uz") != "uzn_Latn":
        errors.append("uz must use the Latin Uzbek code uzn_Latn")
    contract = config.get("text_contract", {})
    if contract.get("zh_script") != "simplified" or not contract.get(
        "convert_traditional_to_simplified", False
    ):
        errors.append("text_contract must convert zh to Simplified Chinese")
    if contract.get("uz_script") != "latin" or not contract.get(
        "transliterate_cyrillic_to_latin", False
    ):
        errors.append("text_contract must transliterate uz to Latin script")
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
        frame = pd.DataFrame(
            {
                lang: [normalize_language_text(lang, text) for text in values]
                for lang, values in columns.items()
            }
        )
        path = _path(benchmark[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        report["splits"][split] = {"rows": len(frame), "path": str(path.relative_to(PROJECT_ROOT))}
    write_json(PROJECT_ROOT / "reports/pipeline/fourlang/benchmarks.json", report)


def aggregate(config: dict[str, Any], experiment: str) -> None:
    train_field = "train" if experiment == "exp1" else "kd_train"
    train_parts, validation_parts = [], []
    normalization_totals: Counter[str] = Counter()
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
        normalization_totals.update(train_part.attrs["script_normalization"])
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
        normalization_totals.update(validation.attrs["script_normalization"])
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
    validation = pd.concat(validation_parts, ignore_index=True).drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    )
    benchmark_paths = [_path(config["benchmarks"][name])
                       for name in ("flores_dev", "flores_devtest")]
    protection_inputs = list(benchmark_paths)
    previous_train = None
    if experiment == "exp2":
        previous_path = PROJECT_ROOT / "data/multilingual/fourlang/exp1/train.jsonl"
        previous_train = normalize_rows(_read_table(previous_path), origin="exp1_training")
        protection_inputs.append(previous_path)
    train, validation, audit = protect_splits(
        train, validation, [pd.read_parquet(path) for path in benchmark_paths],
        LANGUAGES, previous_train,
    )
    remaining_directions = set(validation["src_lang"] + "-" + validation["tgt_lang"])
    if remaining_directions != set(directions()):
        raise RuntimeError("Protected validation must retain all 12 directions.")
    balancing = config["balancing"].get(experiment, {})
    train, report = balance_training_rows(
        train,
        seed=int(config["multilingual"]["seed"]),
        configured_rows=int(balancing.get("default_rows_per_direction", 0)),
        rows_by_direction={
            str(key): int(value)
            for key, value in balancing.get("rows_by_direction", {}).items()
        },
        teacher_ratio=(
            float(balancing["teacher_ratio"])
            if "teacher_ratio" in balancing
            else None
        ),
        max_teacher_repeats=int(balancing.get("max_teacher_repeats", 3)),
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
    report["script_normalization"] = dict(sorted(normalization_totals.items()))
    report["schema_version"] = 2
    report["leakage_audit"] = audit
    report["aggregation_config_fingerprint"] = fingerprint({
        "balancing": balancing, "seed": config["multilingual"]["seed"],
        "text_contract": config.get("text_contract", {}), "pair_data": config["pair_data"],
    })
    report["file_sha256"] = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
        for path in [output / "train.jsonl", output / "validation.jsonl", *protection_inputs]
    }
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
    output_path = PROJECT_ROOT / "results/model_selection/fourlang/student_scores.json"
    results: dict[str, Any] = {}
    if output_path.exists():
        checkpoint = read_json(output_path)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("candidates"), dict):
            results = dict(checkpoint["candidates"])

    candidates = list(config["student_candidates"])
    configured_ids = {candidate["id"] for candidate in candidates}
    results = {key: value for key, value in results.items() if key in configured_ids}

    for index, candidate in enumerate(candidates, start=1):
        candidate_id = candidate["id"]
        previous = results.get(candidate_id)
        if isinstance(previous, dict) and previous.get("status") == "ok":
            print(
                f"[{index}/{len(candidates)}] skipping completed candidate: {candidate_id}",
                flush=True,
            )
            continue

        print(
            f"[{index}/{len(candidates)}] evaluating candidate: {candidate_id}",
            flush=True,
        )
        try:
            results[candidate_id] = evaluate_candidate(config, candidate)
        except Exception as error:
            results[candidate_id] = {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        write_json(output_path, {"candidates": results})
        print(
            f"[{index}/{len(candidates)}] checkpoint saved: {candidate_id} "
            f"({results[candidate_id]['status']})",
            flush=True,
        )


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


def verify_exp2_data(config: dict[str, Any]) -> None:
    message = "Exp2 data is stale or unaudited. Re-run aggregate --experiment exp2 before training."
    report_path = PROJECT_ROOT / "reports/pipeline/fourlang/exp2_data.json"
    if not report_path.is_file():
        raise RuntimeError(message)
    report = read_json(report_path)
    expected = fingerprint({
        "balancing": config["balancing"].get("exp2", {}),
        "seed": config["multilingual"]["seed"],
        "text_contract": config.get("text_contract", {}), "pair_data": config["pair_data"],
    })
    hashes = report.get("file_sha256", {})
    required = {
        "data/multilingual/fourlang/exp2/train.jsonl",
        "data/multilingual/fourlang/exp2/validation.jsonl",
        "data/multilingual/fourlang/exp1/train.jsonl",
        str(config["benchmarks"]["flores_dev"]).replace("\\", "/"),
        str(config["benchmarks"]["flores_devtest"]).replace("\\", "/"),
    }
    if (report.get("schema_version") != 2 or
        report.get("sampling_strategy") != "coverage_first_cycles_v1" or
        report.get("aggregation_config_fingerprint") != expected or
        report.get("leakage_audit", {}).get("protected_overlap_after") != 0 or
        not required.issubset(hashes)):
        raise RuntimeError(message)
    for name, expected_hash in hashes.items():
        path = PROJECT_ROOT / name
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise RuntimeError(f"{message} Changed file: {name}")


def train(config: dict[str, Any], experiment: str) -> None:
    if experiment == "exp2":
        verify_exp2_data(config)
    selected = read_json(PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json")
    candidate = selected["candidate"]
    source_model = (
        candidate_path(candidate, "en", "zh")
        if experiment == "exp1"
        else str(PROJECT_ROOT / "results/student/fourlang/exp1/best_model/shared")
    )
    if experiment == "exp2":
        model_files(Path(source_model))
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
        "require_local_artifact": True,
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
