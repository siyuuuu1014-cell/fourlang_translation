"""Build the approved, versioned Exp4 weighted dataset from frozen score shards."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config  # noqa: E402
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    atomic_json,
    file_sha256,
    fingerprint,
    model_signature,
)
from scripts.pipeline_v3.fourlang_flow import directions  # noqa: E402
from scripts.pipeline_v3.preview_m2m100_error_replay import (  # noqa: E402
    PREVIEW_STATUS,
    add_quality_flags,
    apply_provisional_weights,
    assign_bands,
    duplicate_identity,
    normalize_direction_mass,
    read_scored,
)

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_error_replay_v1.toml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    digest = file_sha256(temporary)
    temporary.replace(path)
    return digest


def validate_approval(config: dict[str, Any]) -> None:
    approval = config.get("approval", {})
    training = config["training"]["m2m100_error_replay_v1"]
    if approval.get("status") != "USER_APPROVED":
        raise RuntimeError("Formal Exp4 build is not marked USER_APPROVED.")
    for key in ("epochs", "learning_rate", "save_total_limit"):
        if float(approval[key]) != float(training[key]):
            raise RuntimeError(f"Approval and training setting differ: {key}")
    if int(training["save_total_limit"]) != 1:
        raise RuntimeError("Exp4 must retain at most one checkpoint.")


def validation_ids(path: Path) -> set[str]:
    result = set()
    for row in read_jsonl(path):
        required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"Validation row missing {sorted(missing)}")
        result.add(duplicate_identity(row))
    return result


def build(config: dict[str, Any]) -> dict[str, Any]:
    validate_approval(config)
    preview_config_path = project_path(config["preview_inputs"]["config"])
    preview_report_path = project_path(config["preview_inputs"]["report"])
    score_manifest_path = project_path(config["preview_inputs"]["score_manifest"])
    preview_config = load_config(preview_config_path)
    preview_report = read_json(preview_report_path)
    score_manifest = read_json(score_manifest_path)
    if preview_report.get("status") != PREVIEW_STATUS:
        raise RuntimeError("Approved preview report is missing or has the wrong status.")
    if score_manifest.get("status") != "SCORING_COMPLETE_PREVIEW_ONLY":
        raise RuntimeError("Difficulty scoring is incomplete.")
    expected_parameters = preview_report["provisional_parameters_requiring_user_approval"]
    if expected_parameters["difficulty_band_percentiles"] != preview_config["difficulty_bands"]:
        raise RuntimeError("Difficulty-band settings changed after preview approval.")
    if expected_parameters["weight_multipliers"] != preview_config["provisional_weights"]:
        raise RuntimeError("Weight settings changed after preview approval.")
    if expected_parameters["quality_review"] != preview_config["quality_review"]:
        raise RuntimeError("Quality settings changed after preview approval.")
    approved_ratio = float(config["approval"]["zh_uz_total_direction_mass_ratio"])
    if approved_ratio != float(preview_config["provisional_weights"]["zh_uz_direction_multiplier"]):
        raise RuntimeError("Approved zh-uz priority differs from the preview.")

    preview_output = project_path(preview_config["preview"]["output_root"])
    records = read_scored(preview_output)
    if len(records) != int(preview_report["total_rows"]):
        raise RuntimeError("Scored rows do not match the approved preview count.")
    assign_bands(records, preview_config)
    add_quality_flags(records, preview_config)
    apply_provisional_weights(records, preview_config)
    normalize_direction_mass(records, preview_config)

    train_path = project_path(config["data"]["error_replay"]["train"])
    manifest_path = project_path(config["data"]["error_replay"]["manifest"])
    validation_path = project_path(config["data"]["error_replay"]["validation"])
    source_model = project_path(config["source_model"]["path"])
    for required in (validation_path, source_model):
        if not required.exists():
            raise FileNotFoundError(required)

    free_gb = shutil.disk_usage(train_path.parent.parent).free / (1024 ** 3)
    if free_gb < 16:
        raise RuntimeError(f"Need at least 16 GiB free before building Exp4; found {free_gb:.2f} GiB.")
    held_out = validation_ids(validation_path)
    overlap = [item["record_id"] for item in records if item["duplicate_group_id"] in held_out]
    if overlap:
        raise RuntimeError(
            f"Training/validation leakage detected for {len(overlap)} physical rows; refusing build."
        )

    formal_rows = []
    direction_counts: Counter[str] = Counter()
    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    direction_weight: Counter[str] = Counter()
    flagged_records = 0
    for item in records:
        row = dict(item["row"])
        original_weight = row.get("weight")
        row["weight"] = item["proposed_weight"]
        row["exp4_error_replay"] = {
            "record_id": item["record_id"],
            "source_dataset": item["source_dataset"],
            "source_line_number": item["source_line_number"],
            "original_weight": original_weight,
            "difficulty_nll": item["difficulty_nll"],
            "difficulty_percentile_within_direction": item["difficulty_percentile_within_direction"],
            "difficulty_band": item["difficulty_band"],
            "quality_flags": item["quality_flags"],
            "duplicate_group_id": item["duplicate_group_id"],
            "duplicate_group_size": item["duplicate_group_size"],
            "proposed_weight_components": {
                "original_weight_effective": item["original_weight_effective"],
                "band_multiplier": item["band_multiplier"],
                "quality_multiplier": item["quality_multiplier"],
                "duplicate_multiplier": item["duplicate_multiplier"],
                "direction_mass_normalization_multiplier": item["direction_mass_normalization_multiplier"],
            },
        }
        formal_rows.append(row)
        direction_counts[item["direction"]] += 1
        band_counts[item["direction"]][item["difficulty_band"]] += 1
        direction_weight[item["direction"]] += item["proposed_weight"]
        flagged_records += bool(item["quality_flags"])

    ordinary = [direction_weight[value] for value in directions() if value != "zh-uz"]
    if max(ordinary) - min(ordinary) > 1e-6:
        raise RuntimeError("Ordinary direction masses are not equal after reconstruction.")
    if not math.isclose(direction_weight["zh-uz"] / ordinary[0], approved_ratio, rel_tol=1e-9):
        raise RuntimeError("zh-uz direction mass does not match approval.")

    build_signature = fingerprint(
        {
            "approval": config["approval"],
            "score_fingerprint": score_manifest["fingerprint"],
            "preview_config_sha256": file_sha256(preview_config_path),
            "preview_report_sha256": file_sha256(preview_report_path),
            "validation_sha256": file_sha256(validation_path),
            "source_model": model_signature(source_model),
        }
    )
    if train_path.exists() or manifest_path.exists():
        if not train_path.is_file() or not manifest_path.is_file():
            raise RuntimeError("Partial Exp4 output exists; inspect it before retrying.")
        existing = read_json(manifest_path)
        if (
            existing.get("build_signature") != build_signature
            or existing.get("train_sha256") != file_sha256(train_path)
        ):
            raise RuntimeError("Existing Exp4 output differs; refusing overwrite.")
        print(f"EXP4_ERROR_REPLAY_DATA_REUSED: {train_path}", flush=True)
        return existing

    train_sha256 = atomic_jsonl(train_path, formal_rows)
    manifest = {
        "schema_version": 1,
        "status": "FORMAL_TRAINING_DATA_READY_NOT_TRAINED",
        "build_signature": build_signature,
        "approval": config["approval"],
        "train": str(train_path.resolve()),
        "train_sha256": train_sha256,
        "rows": len(formal_rows),
        "rows_by_direction": dict(sorted(direction_counts.items())),
        "rows_by_direction_and_band": {key: dict(value) for key, value in sorted(band_counts.items())},
        "flagged_records": flagged_records,
        "weight_sum_by_direction": dict(sorted(direction_weight.items())),
        "validation": str(validation_path.resolve()),
        "validation_sha256": file_sha256(validation_path),
        "validation_overlap_records": 0,
        "source_model": str(source_model.resolve()),
        "source_model_signature": model_signature(source_model),
        "score_manifest": str(score_manifest_path.resolve()),
        "score_fingerprint": score_manifest["fingerprint"],
        "preview_report": str(preview_report_path.resolve()),
        "formal_training_started": False,
    }
    atomic_json(manifest_path, manifest)
    print(f"EXP4_ERROR_REPLAY_DATA_READY_NOT_TRAINED: {train_path}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    build(load_config(args.config))


if __name__ == "__main__":
    main()
