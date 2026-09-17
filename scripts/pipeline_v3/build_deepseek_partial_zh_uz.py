"""Freeze completed DeepSeek rows and build an isolated ZH-UZ training mix."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import load_config, read_json, write_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256  # noqa: E402
from scripts.pipeline_v3 import deepseek_teacher as teacher  # noqa: E402
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)

PROJECT_ROOT = PROJECT_ROOT_BOOTSTRAP
OUTPUT_ROOT = PROJECT_ROOT / "data/distillation/zh_uz/deepseek_partial_v1"
BASE_TRAIN = PROJECT_ROOT / "data/distillation/zh_uz/flores_relaxed_8k/train.jsonl"
TARGET_DIRECTION = "zh-uz"
DEEPSEEK_WEIGHT = 1.5


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported training data format: {path}")


def normalize_base_rows(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "source_lang": "src_lang",
        "target_lang": "tgt_lang",
        "source_text": "src_text",
        "target_text": "tgt_text",
        "training_weight": "weight",
        "sample_weight": "weight",
        "training_origin": "training_source",
        "sample_origin": "training_source",
    }
    result = frame.rename(
        columns={old: new for old, new in aliases.items() if new not in frame.columns}
    ).copy()
    required = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Baseline training data is missing columns: {sorted(missing)}")
    result["src_lang"] = result["src_lang"].astype(str).str.lower()
    result["tgt_lang"] = result["tgt_lang"].astype(str).str.lower()
    result["src_text"] = [
        normalize_language_text(language, text)
        for language, text in zip(
            result["src_lang"], result["src_text"].fillna("").astype(str), strict=True
        )
    ]
    result["tgt_text"] = [
        normalize_language_text(language, text)
        for language, text in zip(
            result["tgt_lang"], result["tgt_text"].fillna("").astype(str), strict=True
        )
    ]
    result = result[(result["src_text"] != "") & (result["tgt_text"] != "")].copy()
    if "weight" in result:
        result["weight"] = pd.to_numeric(result["weight"], errors="coerce").fillna(1.0)
    else:
        result["weight"] = 1.0
    if "training_source" not in result:
        result["training_source"] = "unknown_baseline"
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_key(language: str, text: str) -> tuple[str, str]:
    return language, normalize_language_text(language, text).casefold()


def audit_completed(
    completed: list[dict[str, Any]], config: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in completed:
        translation_failures = teacher.hard_check(
            raw, str(raw["teacher_text"]), config
        )
        source_failures = (
            teacher.source_filter_reasons(str(raw["src_text"]))
            if config.get("source_filter", {}).get("enabled", True)
            else []
        )
        rows.append(
            {
                **raw,
                "revised_translation_pass": not translation_failures,
                "revised_translation_failures_json": json.dumps(
                    translation_failures
                ),
                "source_filter_pass": not source_failures,
                "source_filter_failures_json": json.dumps(source_failures),
                "combined_usable": not translation_failures and not source_failures,
            }
        )
    return pd.DataFrame(rows)


def build_training_rows(
    baseline: pd.DataFrame,
    audited: pd.DataFrame,
    *,
    deepseek_weight: float = DEEPSEEK_WEIGHT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    base = baseline.copy()
    existing = {
        _source_key(str(row.src_lang), str(row.src_text))
        for row in base.itertuples()
    }
    usable = audited[
        audited["combined_usable"].astype(bool)
        & (audited["src_lang"].astype(str) == "zh")
        & (audited["tgt_lang"].astype(str) == "uz")
    ].copy()
    usable = usable.sort_values("pair_id", kind="stable")
    accepted: list[dict[str, Any]] = []
    seen = set(existing)
    rejections: Counter[str] = Counter()
    for row in usable.to_dict(orient="records"):
        key = _source_key("zh", str(row["src_text"]))
        if key in seen:
            rejections["DUPLICATE_SOURCE"] += 1
            continue
        seen.add(key)
        accepted.append(
            {
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": normalize_language_text("zh", str(row["src_text"])),
                "tgt_text": normalize_language_text("uz", str(row["teacher_text"])),
                "weight": float(deepseek_weight),
                "training_source": "teacher_kd_deepseek_partial_v1",
                "source_id": str(row["pair_id"]),
            }
        )
    combined = base.to_dict(orient="records") + accepted
    return combined, accepted, dict(sorted(rejections.items()))


def _direction_counts(frame: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, int]:
    selected = frame if mask is None else frame[mask]
    values = selected["src_lang"].astype(str) + "-" + selected["tgt_lang"].astype(str)
    return dict(sorted(Counter(values.tolist()).items()))


def build(config: dict[str, Any]) -> dict[str, Any]:
    report_path = OUTPUT_ROOT / "build_report.json"
    train_path = OUTPUT_ROOT / "train.jsonl"
    if report_path.is_file():
        report = read_json(report_path)
        expected = report.get("file_sha256", {}).get("train")
        if not train_path.is_file() or file_sha256(train_path) != expected:
            raise RuntimeError("Frozen DeepSeek partial v1 output changed; refusing reuse")
        return {**report, "reused": True}
    if OUTPUT_ROOT.exists():
        raise RuntimeError(f"Partial output exists without a valid report: {OUTPUT_ROOT}")

    input_path = teacher.project_path(config["input"]["candidates"])
    selected, source_rejections = teacher.select_for_mode(
        teacher.read_jsonl(input_path), config, True
    )
    manifest = teacher.manifest_payload(
        config, input_path, selected, True, source_rejections
    )
    checkpoint = teacher.output_root(config, True) / "generations.jsonl"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = file_sha256(checkpoint)
    completed_by_key = teacher.load_state(
        checkpoint, selected, manifest["generation_fingerprint"]
    )
    if file_sha256(checkpoint) != checkpoint_sha:
        raise RuntimeError("DeepSeek checkpoint changed while building the snapshot")
    completed = [
        completed_by_key[teacher.row_key(row)]
        for row in selected
        if teacher.row_key(row) in completed_by_key
    ]
    if not completed:
        raise RuntimeError("DeepSeek checkpoint has no completed rows")

    audited = audit_completed(completed, config)
    baseline = normalize_base_rows(_read_table(BASE_TRAIN))
    combined, accepted, duplicate_rejections = build_training_rows(
        baseline, audited
    )
    if len(accepted) < 4000:
        raise RuntimeError(
            f"Only {len(accepted)} usable unique zh-uz rows; at least 4000 required"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    reaudit_path = OUTPUT_ROOT / "reaudit.parquet"
    reaudit_jsonl = OUTPUT_ROOT / "reaudit.jsonl"
    accepted_path = OUTPUT_ROOT / "accepted_deepseek.jsonl"
    audited.to_parquet(reaudit_path, index=False)
    audited.to_json(reaudit_jsonl, orient="records", lines=True, force_ascii=False)
    _write_jsonl(accepted_path, accepted)
    _write_jsonl(train_path, combined)

    usable_mask = audited["combined_usable"].astype(bool)
    baseline_zh_uz = baseline[
        (baseline["src_lang"].astype(str) == "zh")
        & (baseline["tgt_lang"].astype(str) == "uz")
    ]
    baseline_effective_mass = float(baseline_zh_uz["weight"].sum())
    deepseek_effective_mass = float(sum(row["weight"] for row in accepted))
    report = {
        "schema_version": 1,
        "status": "FROZEN_DEEPSEEK_PARTIAL_V1_READY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "generation_fingerprint": manifest["generation_fingerprint"],
        "completed_rows": len(audited),
        "completed_by_direction": _direction_counts(audited),
        "usable_by_direction": _direction_counts(audited, usable_mask),
        "accepted_new_zh_uz_rows": len(accepted),
        "deepseek_weight": DEEPSEEK_WEIGHT,
        "effective_mass": {
            "baseline_zh_uz": baseline_effective_mass,
            "deepseek_zh_uz": deepseek_effective_mass,
            "deepseek_share": deepseek_effective_mass
            / (baseline_effective_mass + deepseek_effective_mass),
        },
        "duplicate_rejections": duplicate_rejections,
        "baseline_rows": len(baseline),
        "combined_rows": len(combined),
        "output": str(train_path),
        "checkpoint_modified": False,
        "file_sha256": {
            "baseline": file_sha256(BASE_TRAIN),
            "reaudit": file_sha256(reaudit_path),
            "accepted_deepseek": file_sha256(accepted_path),
            "train": file_sha256(train_path),
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/directions/zh_uz_deepseek_teacher_v1.toml",
    )
    args = parser.parse_args()
    result = build(load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
