from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STEP_VERSION = "18E3_V1"

VALID_WINNERS = {"OPUS", "MADLAD", "TIE", "BOTH_BAD"}
BANDS = ["B0_EXACT", "B1_LOW", "B2_MEDIUM", "B3_HIGH", "B4_VERY_HIGH"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18E3 - calibrate a conservative train-only Teacher gate "
            "from Step18E2 Qwen dual-order calibration, then project that "
            "policy onto the full Step18D 20K Teacher subset."
        )
    )
    p.add_argument("--project_root", default=None)
    p.add_argument(
        "--calibration",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/18e2_pairwise/"
            "qwen_teacher_routing_calibration_results_v1.parquet"
        ),
    )
    p.add_argument(
        "--full_scored",
        default=(
            "data/distillation/zh_en/v1/"
            "18d_disagreement/"
            "teacher_disagreement_20k_v1.parquet"
        ),
    )
    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/18e3_gate_policy"
        ),
    )
    p.add_argument("--min_compatible_percent", type=float, default=90.0)
    p.add_argument("--min_acceptable_percent", type=float, default=90.0)
    p.add_argument("--max_major_error_percent", type=float, default=5.0)
    p.add_argument("--max_both_bad_percent", type=float, default=2.5)
    p.add_argument("--min_dual_order_consistency_percent", type=float, default=75.0)
    p.add_argument(
        "--force_qwen_band",
        action="append",
        default=None,
        help="Can be passed multiple times. Default: B4_VERY_HIGH.",
    )
    p.add_argument(
        "--canonical_tiebreak_teacher",
        choices=["OPUS", "MADLAD"],
        default="OPUS",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.fillna(False)
        .map(
            lambda x: (
                x if isinstance(x, bool)
                else str(x).strip().lower() in {"1", "true", "yes"}
            )
        )
        .astype(bool)
    )


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return 0.0, 100.0
    p = successes / total
    z2 = z * z
    den = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / den
    margin = (
        z
        * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
        / den
    )
    return 100.0 * max(0.0, center - margin), 100.0 * min(1.0, center + margin)


def assign_local_bands(part: pd.DataFrame):
    out = part.copy()
    exact_mask = as_bool_series(out["opus_madlad_exact_match"])
    out["calibration_band"] = ""
    out.loc[exact_mask, "calibration_band"] = "B0_EXACT"

    nonexact = out.loc[~exact_mask].copy()
    thresholds = {"q25": None, "q50": None, "q75": None}

    if len(nonexact) == 0:
        return out, thresholds

    q25 = float(nonexact["teacher_disagreement_score"].quantile(0.25))
    q50 = float(nonexact["teacher_disagreement_score"].quantile(0.50))
    q75 = float(nonexact["teacher_disagreement_score"].quantile(0.75))
    thresholds = {"q25": q25, "q50": q50, "q75": q75}

    s = nonexact["teacher_disagreement_score"]
    out.loc[nonexact.index[s <= q25], "calibration_band"] = "B1_LOW"
    out.loc[nonexact.index[(s > q25) & (s <= q50)], "calibration_band"] = "B2_MEDIUM"
    out.loc[nonexact.index[(s > q50) & (s <= q75)], "calibration_band"] = "B3_HIGH"
    out.loc[nonexact.index[s > q75], "calibration_band"] = "B4_VERY_HIGH"

    return out, thresholds


def teacher_metrics(part: pd.DataFrame, teacher: str):
    n = len(part)
    acceptable_col = (
        "final_opus_acceptable" if teacher == "OPUS"
        else "final_madlad_acceptable"
    )
    major_col = (
        "final_opus_major_error" if teacher == "OPUS"
        else "final_madlad_major_error"
    )

    winners = part["final_winner"].astype(str)
    compatible = winners.isin({teacher, "TIE"})
    acceptable = as_bool_series(part[acceptable_col])
    major = as_bool_series(part[major_col])

    compatible_n = int(compatible.sum())
    acceptable_n = int(acceptable.sum())
    major_n = int(major.sum())

    comp_low, comp_high = wilson_interval(compatible_n, n)
    acc_low, acc_high = wilson_interval(acceptable_n, n)
    major_low, major_high = wilson_interval(major_n, n)

    return {
        "compatible_rows": compatible_n,
        "compatible_percent": 100.0 * compatible_n / n if n else 0.0,
        "compatible_wilson_low": comp_low,
        "compatible_wilson_high": comp_high,
        "acceptable_rows": acceptable_n,
        "acceptable_percent": 100.0 * acceptable_n / n if n else 0.0,
        "acceptable_wilson_low": acc_low,
        "acceptable_wilson_high": acc_high,
        "major_error_rows": major_n,
        "major_error_percent": 100.0 * major_n / n if n else 0.0,
        "major_error_wilson_low": major_low,
        "major_error_wilson_high": major_high,
    }


def candidate_is_auto_safe(metrics, both_bad_percent, dual_consistency_percent, args):
    return (
        metrics["compatible_percent"] >= args.min_compatible_percent
        and metrics["acceptable_percent"] >= args.min_acceptable_percent
        and metrics["major_error_percent"] <= args.max_major_error_percent
        and both_bad_percent <= args.max_both_bad_percent
        and dual_consistency_percent >= args.min_dual_order_consistency_percent
    )


def choose_between_safe_teachers(opus, madlad, canonical):
    opus_key = (
        -opus["major_error_percent"],
        opus["compatible_percent"],
        opus["acceptable_percent"],
    )
    mad_key = (
        -madlad["major_error_percent"],
        madlad["compatible_percent"],
        madlad["acceptable_percent"],
    )
    if opus_key > mad_key:
        return "OPUS"
    if mad_key > opus_key:
        return "MADLAD"
    return canonical


def summarize_cell(part, direction, source_dataset, band, force_qwen_bands, args):
    n = len(part)
    winners = part["final_winner"].astype(str)
    counts = winners.value_counts().to_dict()

    opus_count = int(counts.get("OPUS", 0))
    madlad_count = int(counts.get("MADLAD", 0))
    tie_count = int(counts.get("TIE", 0))
    both_bad_count = int(counts.get("BOTH_BAD", 0))
    both_bad_percent = 100.0 * both_bad_count / n if n else 0.0

    dual_consistency_percent = (
        100.0 * as_bool_series(part["dual_order_consistent"]).mean()
        if n else 0.0
    )

    opus = teacher_metrics(part, "OPUS")
    madlad = teacher_metrics(part, "MADLAD")

    opus_safe = candidate_is_auto_safe(
        opus, both_bad_percent, dual_consistency_percent, args
    )
    madlad_safe = candidate_is_auto_safe(
        madlad, both_bad_percent, dual_consistency_percent, args
    )

    if band == "B0_EXACT":
        gate_action = "AUTO_EQUAL"
        auto_teacher = args.canonical_tiebreak_teacher
        qwen_required = False
        reason = (
            "OPUS and MADLAD outputs are exact matches; canonical Teacher "
            "only determines provenance."
        )
    elif band in force_qwen_bands:
        gate_action = "QWEN_REQUIRED"
        auto_teacher = ""
        qwen_required = True
        reason = "Very-high disagreement band is forced to Qwen."
    elif opus_safe and madlad_safe:
        chosen = choose_between_safe_teachers(
            opus, madlad, args.canonical_tiebreak_teacher
        )
        gate_action = f"AUTO_{chosen}"
        auto_teacher = chosen
        qwen_required = False
        reason = (
            "Both Teachers meet safety thresholds; choose lower major-error / "
            "higher compatible / higher acceptable rate."
        )
    elif opus_safe:
        gate_action = "AUTO_OPUS"
        auto_teacher = "OPUS"
        qwen_required = False
        reason = "OPUS meets all safety thresholds."
    elif madlad_safe:
        gate_action = "AUTO_MADLAD"
        auto_teacher = "MADLAD"
        qwen_required = False
        reason = "MADLAD meets all safety thresholds."
    else:
        gate_action = "QWEN_REQUIRED"
        auto_teacher = ""
        qwen_required = True
        reason = "No Teacher meets all conservative auto-routing thresholds."

    return {
        "direction": direction,
        "source_dataset": source_dataset,
        "calibration_band": band,
        "calibration_rows": int(n),

        "opus_winner_rows": opus_count,
        "opus_winner_percent": 100.0 * opus_count / n if n else 0.0,
        "madlad_winner_rows": madlad_count,
        "madlad_winner_percent": 100.0 * madlad_count / n if n else 0.0,
        "tie_rows": tie_count,
        "tie_percent": 100.0 * tie_count / n if n else 0.0,
        "both_bad_rows": both_bad_count,
        "both_bad_percent": both_bad_percent,
        "dual_order_consistency_percent": dual_consistency_percent,

        "opus_compatible_percent": opus["compatible_percent"],
        "opus_compatible_wilson_low": opus["compatible_wilson_low"],
        "opus_acceptable_percent": opus["acceptable_percent"],
        "opus_acceptable_wilson_low": opus["acceptable_wilson_low"],
        "opus_major_error_percent": opus["major_error_percent"],
        "opus_major_error_wilson_high": opus["major_error_wilson_high"],

        "madlad_compatible_percent": madlad["compatible_percent"],
        "madlad_compatible_wilson_low": madlad["compatible_wilson_low"],
        "madlad_acceptable_percent": madlad["acceptable_percent"],
        "madlad_acceptable_wilson_low": madlad["acceptable_wilson_low"],
        "madlad_major_error_percent": madlad["major_error_percent"],
        "madlad_major_error_wilson_high": madlad["major_error_wilson_high"],

        "opus_auto_safe": bool(opus_safe),
        "madlad_auto_safe": bool(madlad_safe),

        "gate_action": gate_action,
        "auto_teacher": auto_teacher,
        "qwen_required": bool(qwen_required),
        "policy_reason": reason,
    }


def main():
    args = parse_args()

    root = (
        Path(args.project_root).resolve()
        if args.project_root else infer_project_root()
    )

    calibration_path = resolve_path(root, args.calibration)
    full_scored_path = resolve_path(root, args.full_scored)
    output_dir = resolve_path(root, args.output_dir)

    if not calibration_path.exists():
        raise FileNotFoundError(calibration_path)
    if not full_scored_path.exists():
        raise FileNotFoundError(full_scored_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    policy_csv = output_dir / "teacher_gate_policy_v1.csv"
    policy_json = output_dir / "teacher_gate_policy_v1.json"
    projection_parquet = output_dir / "teacher_gate_projection_20k_v1.parquet"
    projection_summary_csv = output_dir / "teacher_gate_projection_summary_v1.csv"
    report_json = output_dir / "teacher_gate_calibration_report_v1.json"

    outputs = [
        policy_csv,
        policy_json,
        projection_parquet,
        projection_summary_csv,
        report_json,
    ]

    if not args.overwrite and any(p.exists() for p in outputs):
        raise RuntimeError("Step18E3 outputs already exist. Use --overwrite.")

    force_qwen_bands = set(
        args.force_qwen_band if args.force_qwen_band else ["B4_VERY_HIGH"]
    )

    unknown = force_qwen_bands - set(BANDS)
    if unknown:
        raise RuntimeError(f"Unknown force_qwen_band: {sorted(unknown)}")

    print("=" * 115)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18E3 - CALIBRATE CONSERVATIVE TEACHER GATE POLICY")
    print("=" * 115)

    print("\nCalibration:")
    print(calibration_path)
    print("\nFull scored 20K:")
    print(full_scored_path)

    calibration = pd.read_parquet(calibration_path).copy()
    full = pd.read_parquet(full_scored_path).copy()

    required_cal = {
        "qwen_review_id",
        "direction",
        "source_dataset",
        "calibration_band",
        "teacher_disagreement_score",
        "final_winner",
        "dual_order_consistent",
        "final_opus_acceptable",
        "final_opus_major_error",
        "final_madlad_acceptable",
        "final_madlad_major_error",
    }
    missing = required_cal - set(calibration.columns)
    if missing:
        raise RuntimeError(f"Calibration missing columns: {sorted(missing)}")

    required_full = {
        "kd_candidate_id",
        "direction",
        "source_dataset",
        "teacher_disagreement_score",
        "opus_madlad_exact_match",
        "opus_prediction",
        "madlad_prediction",
        "human_reference",
    }
    missing_full = required_full - set(full.columns)
    if missing_full:
        raise RuntimeError(f"Full data missing columns: {sorted(missing_full)}")

    if calibration["qwen_review_id"].duplicated().any():
        raise RuntimeError("Duplicate qwen_review_id.")

    invalid_winners = (
        set(calibration["final_winner"].astype(str).unique())
        - VALID_WINNERS
    )
    if invalid_winners:
        raise RuntimeError(f"Invalid final winners: {sorted(invalid_winners)}")

    print("\nCalibration rows:", len(calibration))
    print("Full rows:", len(full))

    policy_rows = []
    observed_cells = set()

    for (direction, source_dataset, band), part in calibration.groupby(
        ["direction", "source_dataset", "calibration_band"],
        sort=True,
        dropna=False,
    ):
        key = (str(direction), str(source_dataset), str(band))
        observed_cells.add(key)
        policy_rows.append(
            summarize_cell(
                part,
                str(direction),
                str(source_dataset),
                str(band),
                force_qwen_bands,
                args,
            )
        )

    policy = (
        pd.DataFrame(policy_rows)
        .sort_values(["direction", "source_dataset", "calibration_band"])
        .reset_index(drop=True)
    )

    expected_cells = {
        (d, s, b)
        for d in ["en_zh", "zh_en"]
        for s in ["ALT", "Tatoeba"]
        for b in BANDS
    }

    missing_cells = expected_cells - observed_cells
    extra_cells = observed_cells - expected_cells

    if missing_cells:
        raise RuntimeError(f"Missing calibration cells: {sorted(missing_cells)}")
    if extra_cells:
        raise RuntimeError(f"Unexpected calibration cells: {sorted(extra_cells)}")

    # Recreate full 20K bands exactly as Step18E1.
    full_parts = []
    band_thresholds = {}

    for (direction, source_dataset), part in full.groupby(
        ["direction", "source_dataset"],
        sort=True,
    ):
        banded, thresholds = assign_local_bands(part)
        band_thresholds[f"{direction}|{source_dataset}"] = thresholds
        full_parts.append(banded)

    projected = pd.concat(full_parts, ignore_index=True)

    projected = projected.merge(
        policy[
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "gate_action",
                "auto_teacher",
                "qwen_required",
                "policy_reason",
            ]
        ],
        on=["direction", "source_dataset", "calibration_band"],
        how="left",
        validate="many_to_one",
    )

    if projected["gate_action"].isna().any():
        bad = projected.loc[
            projected["gate_action"].isna(),
            ["direction", "source_dataset", "calibration_band"],
        ].drop_duplicates()
        raise RuntimeError(
            "Projection contains unmapped cells:\n"
            + bad.to_string(index=False)
        )

    projected["routing_status"] = projected["gate_action"]
    projected["selected_teacher_pre_qwen"] = projected["auto_teacher"]
    projected["ready_for_final_kd"] = ~as_bool_series(projected["qwen_required"])

    projection_summary = (
        projected.groupby(
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "gate_action",
                "auto_teacher",
                "qwen_required",
            ],
            dropna=False,
        )
        .agg(
            rows=("kd_candidate_id", "size"),
            mean_disagreement_score=("teacher_disagreement_score", "mean"),
            min_disagreement_score=("teacher_disagreement_score", "min"),
            max_disagreement_score=("teacher_disagreement_score", "max"),
        )
        .reset_index()
        .sort_values(["direction", "source_dataset", "calibration_band"])
    )

    total_rows = len(projected)
    qwen_rows = int(as_bool_series(projected["qwen_required"]).sum())
    auto_rows = total_rows - qwen_rows

    assertions = {
        "calibration_rows_800": len(calibration) == 800,
        "calibration_review_id_unique": calibration["qwen_review_id"].is_unique,
        "all_final_winners_valid": calibration["final_winner"].astype(str).isin(VALID_WINNERS).all(),
        "policy_has_20_cells": len(policy) == 20,
        "policy_cells_unique": not policy[
            ["direction", "source_dataset", "calibration_band"]
        ].duplicated().any(),
        "full_rows_preserved": len(projected) == len(full),
        "full_candidate_id_unique": projected["kd_candidate_id"].is_unique,
        "all_full_rows_have_band": projected["calibration_band"].isin(BANDS).all(),
        "all_full_rows_have_gate_action": projected["gate_action"].notna().all(),
        "forced_qwen_bands_are_qwen_only": (
            as_bool_series(
                projected.loc[
                    projected["calibration_band"].isin(force_qwen_bands),
                    "qwen_required",
                ]
            ).all()
        ),
        "qwen_and_auto_partition_complete": qwen_rows + auto_rows == total_rows,
    }

    failed = [k for k, v in assertions.items() if not bool(v)]
    if failed:
        raise RuntimeError("STEP18E3 assertion failure:\n" + "\n".join(failed))

    policy.to_csv(policy_csv, index=False, encoding="utf-8-sig")
    projected.to_parquet(projection_parquet, index=False)
    projection_summary.to_csv(
        projection_summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    policy_payload = {
        "policy_name": "ZH_EN_TEACHER_GATE_POLICY_V1",
        "step_version": STEP_VERSION,
        "principles": {
            "human_reference_is_diagnostic_only": True,
            "B0_exact": (
                "auto-route because OPUS and MADLAD target texts are identical"
            ),
            "forced_qwen_bands": sorted(force_qwen_bands),
            "non_exact_auto_route_requires_all_safety_thresholds": True,
            "tie_is_compatible_with_either_teacher": True,
            "qwen_required_rows_are_not_final_kd_targets_yet": True,
        },
        "thresholds": {
            "min_compatible_percent": args.min_compatible_percent,
            "min_acceptable_percent": args.min_acceptable_percent,
            "max_major_error_percent": args.max_major_error_percent,
            "max_both_bad_percent": args.max_both_bad_percent,
            "min_dual_order_consistency_percent": args.min_dual_order_consistency_percent,
            "canonical_tiebreak_teacher": args.canonical_tiebreak_teacher,
        },
        "local_disagreement_thresholds": band_thresholds,
        "cells": policy.to_dict(orient="records"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_FOR_STEP_18F_SELECTIVE_QWEN",
    }

    with policy_json.open("w", encoding="utf-8") as f:
        json.dump(policy_payload, f, ensure_ascii=False, indent=2)

    report = {
        "step": "18E3",
        "step_version": STEP_VERSION,
        "inputs": {
            "calibration": str(calibration_path),
            "full_scored": str(full_scored_path),
        },
        "counts": {
            "calibration_rows": int(len(calibration)),
            "policy_cells": int(len(policy)),
            "full_projected_rows": int(total_rows),
            "auto_routed_rows": int(auto_rows),
            "qwen_required_rows": int(qwen_rows),
            "auto_routed_percent": 100.0 * auto_rows / total_rows if total_rows else 0.0,
            "qwen_required_percent": 100.0 * qwen_rows / total_rows if total_rows else 0.0,
            "gate_action": {
                str(k): int(v)
                for k, v in projected["gate_action"].value_counts().to_dict().items()
            },
            "auto_teacher": {
                str(k): int(v)
                for k, v in (
                    projected.loc[
                        ~as_bool_series(projected["qwen_required"]),
                        "auto_teacher",
                    ]
                    .replace("", "NONE")
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
        },
        "thresholds": policy_payload["thresholds"],
        "forced_qwen_bands": sorted(force_qwen_bands),
        "assertions": {k: bool(v) for k, v in assertions.items()},
        "outputs": {
            "policy_csv": str(policy_csv),
            "policy_json": str(policy_json),
            "projection": str(projection_parquet),
            "projection_summary": str(projection_summary_csv),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_STEP_18F_SELECTIVE_QWEN",
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 115)
    print("STEP 18E3 RESULT")
    print("=" * 115)

    print("\nCalibration rows:", len(calibration))
    print("Policy cells:", len(policy))

    print("\nSafety thresholds:")
    print("min_compatible_percent:", args.min_compatible_percent)
    print("min_acceptable_percent:", args.min_acceptable_percent)
    print("max_major_error_percent:", args.max_major_error_percent)
    print("max_both_bad_percent:", args.max_both_bad_percent)
    print(
        "min_dual_order_consistency_percent:",
        args.min_dual_order_consistency_percent,
    )
    print("forced_qwen_bands:", sorted(force_qwen_bands))

    print("\nPolicy:")
    display_cols = [
        "direction",
        "source_dataset",
        "calibration_band",
        "calibration_rows",
        "tie_percent",
        "opus_compatible_percent",
        "opus_acceptable_percent",
        "opus_major_error_percent",
        "madlad_compatible_percent",
        "madlad_acceptable_percent",
        "madlad_major_error_percent",
        "dual_order_consistency_percent",
        "gate_action",
    ]
    print(policy[display_cols].to_string(index=False))

    print("\n20K projection:")
    print("Rows:", total_rows)
    print("Auto-routed:", auto_rows, f"({100.0 * auto_rows / total_rows:.2f}%)")
    print("Qwen required:", qwen_rows, f"({100.0 * qwen_rows / total_rows:.2f}%)")

    print("\nGate action counts:")
    print(projected["gate_action"].value_counts().to_string())

    print("\nDirection × source × band projection:")
    print(projection_summary.to_string(index=False))

    print("\nAssertions:")
    for k, v in assertions.items():
        print(f"{k}: {bool(v)}")

    print("\nPolicy JSON:")
    print(policy_json)

    print("\nProjection:")
    print(projection_parquet)

    print("\nReport:")
    print(report_json)

    print("\nSTATUS:")
    print("READY_FOR_STEP_18F_SELECTIVE_QWEN")


if __name__ == "__main__":
    main()
