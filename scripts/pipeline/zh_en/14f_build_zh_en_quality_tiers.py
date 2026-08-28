from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Version
# ============================================================

STEP_VERSION = "14F_V1"

EXPECTED_JUDGE_PROMPT_VERSION = (
    "ZH_EN_JUDGE_V5_1_STRUCTURED_FINAL"
)

VALID_TIERS = {
    "GOLD",
    "SILVER",
    "BRONZE",
    "REJECT",
    "QUARANTINE",
}

TRAINING_WEIGHTS = {
    "GOLD": 1.0,
    "SILVER": 0.8,
    "BRONZE": 0.5,
    "REJECT": 0.0,
    "QUARANTINE": 0.0,
}


# ============================================================
# Helpers
# ============================================================

def ensure_unique_pair_id(
    df: pd.DataFrame,
    name: str,
):
    if "pair_id" not in df.columns:
        raise RuntimeError(
            f"{name} does not contain pair_id."
        )

    duplicated = df["pair_id"].duplicated()

    if duplicated.any():
        examples = (
            df.loc[
                duplicated,
                "pair_id",
            ]
            .astype(str)
            .head(20)
            .tolist()
        )

        raise RuntimeError(
            f"\nDuplicate pair_id found in {name}.\n"
            f"Count: {int(duplicated.sum())}\n"
            f"Examples: {examples}"
        )


def save_dataframe(
    df: pd.DataFrame,
    parquet_path: Path,
    csv_path: Path | None = None,
):
    parquet_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        parquet_path,
        index=False,
    )

    if csv_path is not None:
        df.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )


def normalize_string_column(
    df: pd.DataFrame,
    column: str,
):
    if column not in df.columns:
        return

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# Main
# ============================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    risk_root = (
        project_root
        / "data"
        / "pipeline"
        / "zh_en"
        / "14d_risk_routing"
    )

    auto_accept_file = (
        risk_root
        / "auto_accept_v1.parquet"
    )

    needs_qwen_file = (
        risk_root
        / "needs_qwen_v1.parquet"
    )

    full_review_file = (
        project_root
        / "data"
        / "pipeline"
        / "zh_en"
        / "14e_qwen_review"
        / "full_review"
        / "qwen_review_results_v1.parquet"
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    output_root = (
        project_root
        / "data"
        / "approved"
        / "zh_en"
        / "v1"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_file = (
        output_root
        / "zh_en_all_pairs_v1.parquet"
    )

    all_csv = (
        output_root
        / "zh_en_all_pairs_v1.csv"
    )

    approved_file = (
        output_root
        / "zh_en_approved_v1.parquet"
    )

    approved_csv = (
        output_root
        / "zh_en_approved_v1.csv"
    )

    training_file = (
        output_root
        / "zh_en_training_view_v1.parquet"
    )

    training_csv = (
        output_root
        / "zh_en_training_view_v1.csv"
    )

    gold_file = (
        output_root
        / "zh_en_gold_v1.parquet"
    )

    silver_file = (
        output_root
        / "zh_en_silver_v1.parquet"
    )

    bronze_file = (
        output_root
        / "zh_en_bronze_v1.parquet"
    )

    reject_file = (
        output_root
        / "zh_en_reject_v1.parquet"
    )

    quarantine_file = (
        output_root
        / "zh_en_quarantine_v1.parquet"
    )

    quality_report_csv = (
        output_root
        / "quality_tier_report_v1.csv"
    )

    source_report_csv = (
        output_root
        / "source_tier_report_v1.csv"
    )

    decision_report_csv = (
        output_root
        / "decision_report_v1.csv"
    )

    dataset_report_json = (
        output_root
        / "dataset_report_v1.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14F - QUALITY TIER / APPROVAL"
    )

    print(
        "=" * 110
    )

    print(
        "\nStep version:",
        STEP_VERSION
    )

    # ========================================================
    # Input existence
    # ========================================================

    for path in [
        auto_accept_file,
        needs_qwen_file,
        full_review_file,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    print(
        "\nAUTO_ACCEPT:"
    )
    print(
        auto_accept_file
    )

    print(
        "\nNEEDS_QWEN:"
    )
    print(
        needs_qwen_file
    )

    print(
        "\nQwen full review:"
    )
    print(
        full_review_file
    )

    # ========================================================
    # Load
    # ========================================================

    auto_df = pd.read_parquet(
        auto_accept_file
    )

    risk_df = pd.read_parquet(
        needs_qwen_file
    )

    review_df = pd.read_parquet(
        full_review_file
    )

    for df in [
        auto_df,
        risk_df,
        review_df,
    ]:
        normalize_string_column(
            df,
            "pair_id",
        )

        normalize_string_column(
            df,
            "source_dataset",
        )

    ensure_unique_pair_id(
        auto_df,
        "AUTO_ACCEPT",
    )

    ensure_unique_pair_id(
        risk_df,
        "NEEDS_QWEN",
    )

    ensure_unique_pair_id(
        review_df,
        "QWEN_FULL_REVIEW",
    )

    print(
        "\nInput counts:"
    )

    print(
        "AUTO_ACCEPT:",
        len(auto_df)
    )

    print(
        "NEEDS_QWEN:",
        len(risk_df)
    )

    print(
        "QWEN full review:",
        len(review_df)
    )

    # ========================================================
    # Original 14D consistency
    # ========================================================

    auto_ids = set(
        auto_df[
            "pair_id"
        ]
        .astype(str)
    )

    risk_ids = set(
        risk_df[
            "pair_id"
        ]
        .astype(str)
    )

    original_overlap = (
        auto_ids
        &
        risk_ids
    )

    if original_overlap:
        raise RuntimeError(
            "\nAUTO_ACCEPT and NEEDS_QWEN overlap.\n"
            f"Count: {len(original_overlap)}"
        )

    original_total = (
        len(auto_df)
        +
        len(risk_df)
    )

    # ========================================================
    # Validate judge result
    # ========================================================

    required_review_columns = {
        "pair_id",
        "review_group",
        "judge_label",
        "judge_parse_success",
        "judge_prompt_version",
    }

    missing = (
        required_review_columns
        -
        set(review_df.columns)
    )

    if missing:
        raise RuntimeError(
            "\nFull review missing columns:\n"
            f"{sorted(missing)}"
        )

    normalize_string_column(
        review_df,
        "review_group",
    )

    normalize_string_column(
        review_df,
        "judge_label",
    )

    normalize_string_column(
        review_df,
        "judge_prompt_version",
    )

    prompt_versions = sorted(
        set(
            review_df[
                "judge_prompt_version"
            ]
        )
    )

    print(
        "\nJudge prompt versions:"
    )

    print(
        prompt_versions
    )

    unexpected_prompt_versions = [
        value
        for value in prompt_versions
        if value
        != EXPECTED_JUDGE_PROMPT_VERSION
    ]

    if unexpected_prompt_versions:
        raise RuntimeError(
            "\nUnexpected judge prompt version.\n"
            f"Expected: "
            f"{EXPECTED_JUDGE_PROMPT_VERSION}\n"
            f"Found: "
            f"{unexpected_prompt_versions}"
        )

    # ========================================================
    # Validate review-group membership
    # ========================================================

    valid_review_groups = {
        "RISK_REVIEW",
        "AUTO_ACCEPT_AUDIT",
    }

    invalid_groups = set(
        review_df[
            "review_group"
        ]
    ) - valid_review_groups

    if invalid_groups:
        raise RuntimeError(
            "\nInvalid review_group values:\n"
            f"{sorted(invalid_groups)}"
        )

    review_risk = (
        review_df[
            review_df[
                "review_group"
            ]
            ==
            "RISK_REVIEW"
        ]
        .copy()
    )

    review_auto = (
        review_df[
            review_df[
                "review_group"
            ]
            ==
            "AUTO_ACCEPT_AUDIT"
        ]
        .copy()
    )

    review_risk_ids = set(
        review_risk[
            "pair_id"
        ]
        .astype(str)
    )

    review_auto_ids = set(
        review_auto[
            "pair_id"
        ]
        .astype(str)
    )

    # All NEEDS_QWEN must be reviewed.
    missing_risk_review = (
        risk_ids
        -
        review_risk_ids
    )

    extra_risk_review = (
        review_risk_ids
        -
        risk_ids
    )

    if missing_risk_review:
        raise RuntimeError(
            "\nSome NEEDS_QWEN rows are missing "
            "from full Qwen review.\n"
            f"Count: {len(missing_risk_review)}"
        )

    if extra_risk_review:
        raise RuntimeError(
            "\nRISK_REVIEW contains pair_ids "
            "not present in NEEDS_QWEN.\n"
            f"Count: {len(extra_risk_review)}"
        )

    # AUTO_ACCEPT_AUDIT must be subset of AUTO_ACCEPT.
    invalid_auto_audit = (
        review_auto_ids
        -
        auto_ids
    )

    if invalid_auto_audit:
        raise RuntimeError(
            "\nAUTO_ACCEPT_AUDIT contains rows "
            "not present in AUTO_ACCEPT.\n"
            f"Count: {len(invalid_auto_audit)}"
        )

    print(
        "\nReview composition:"
    )

    print(
        "RISK_REVIEW:",
        len(review_risk)
    )

    print(
        "AUTO_ACCEPT_AUDIT:",
        len(review_auto)
    )

    print(
        "Total reviewed:",
        len(review_df)
    )

    # ========================================================
    # Build reviewed tier assignments
    # ========================================================

    reviewed = (
        review_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    def assign_reviewed_tier(
        row: pd.Series,
    ) -> tuple[str, str]:

        parse_success = bool(
            row[
                "judge_parse_success"
            ]
        )

        label = str(
            row[
                "judge_label"
            ]
        ).strip()

        # Parse failures always quarantine.
        if not parse_success:
            return (
                "QUARANTINE",
                "QWEN_PARSE_FAILED",
            )

        if label == "PASS":
            return (
                "GOLD",
                "QWEN_PASS",
            )

        if label == "MINOR":
            return (
                "BRONZE",
                "QWEN_MINOR",
            )

        if label == "FAIL":
            return (
                "REJECT",
                "QWEN_FAIL",
            )

        if label == "UNCERTAIN":
            return (
                "QUARANTINE",
                "QWEN_UNCERTAIN",
            )

        raise RuntimeError(
            f"Unexpected judge_label: {label!r}"
        )

    reviewed_assignments = (
        reviewed.apply(
            assign_reviewed_tier,
            axis=1,
            result_type="expand",
        )
    )

    reviewed[
        "quality_tier"
    ] = reviewed_assignments[
        0
    ]

    reviewed[
        "quality_tier_reason"
    ] = reviewed_assignments[
        1
    ]

    reviewed[
        "quality_reviewed_by_qwen"
    ] = True

    # ========================================================
    # Build unreviewed AUTO_ACCEPT → SILVER
    # ========================================================

    audited_auto_ids = set(
        review_auto[
            "pair_id"
        ]
        .astype(str)
    )

    unreviewed_auto = (
        auto_df[
            ~auto_df[
                "pair_id"
            ]
            .astype(str)
            .isin(
                audited_auto_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    unreviewed_auto[
        "quality_tier"
    ] = "SILVER"

    unreviewed_auto[
        "quality_tier_reason"
    ] = (
        "RULE_AUTO_ACCEPT_NOT_QWEN_AUDITED"
    )

    unreviewed_auto[
        "quality_reviewed_by_qwen"
    ] = False

    # ========================================================
    # Align columns
    # ========================================================

    # reviewed already contains the original source columns
    # from 14E input, so union columns safely.
    all_columns = list(
        dict.fromkeys(
            list(
                unreviewed_auto.columns
            )
            +
            list(
                reviewed.columns
            )
        )
    )

    for column in all_columns:

        if column not in unreviewed_auto.columns:
            unreviewed_auto[
                column
            ] = pd.NA

        if column not in reviewed.columns:
            reviewed[
                column
            ] = pd.NA

    # ========================================================
    # Combine
    # ========================================================

    all_df = pd.concat(
        [
            unreviewed_auto[
                all_columns
            ],
            reviewed[
                all_columns
            ],
        ],
        ignore_index=True,
    )

    ensure_unique_pair_id(
        all_df,
        "FINAL_ALL",
    )

    # ========================================================
    # Add weights / approval
    # ========================================================

    all_df[
        "training_weight"
    ] = (
        all_df[
            "quality_tier"
        ]
        .map(
            TRAINING_WEIGHTS
        )
        .astype(float)
    )

    all_df[
        "approved_for_training"
    ] = (
        all_df[
            "quality_tier"
        ]
        .isin(
            [
                "GOLD",
                "SILVER",
                "BRONZE",
            ]
        )
    )

    all_df[
        "quality_tier_version"
    ] = STEP_VERSION

    all_df[
        "quality_created_at_utc"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    # ========================================================
    # Assertions
    # ========================================================

    tiers_found = set(
        all_df[
            "quality_tier"
        ]
    )

    invalid_tiers = (
        tiers_found
        -
        VALID_TIERS
    )

    if invalid_tiers:
        raise RuntimeError(
            f"Invalid tiers: {invalid_tiers}"
        )

    if len(all_df) != original_total:
        raise RuntimeError(
            "\nFinal row count mismatch.\n"
            f"Expected from 14D: {original_total}\n"
            f"Found: {len(all_df)}"
        )

    final_ids = set(
        all_df[
            "pair_id"
        ]
        .astype(str)
    )

    original_ids = (
        auto_ids
        |
        risk_ids
    )

    missing_final = (
        original_ids
        -
        final_ids
    )

    extra_final = (
        final_ids
        -
        original_ids
    )

    if missing_final:
        raise RuntimeError(
            "\nRows lost during 14F.\n"
            f"Count: {len(missing_final)}"
        )

    if extra_final:
        raise RuntimeError(
            "\nUnexpected extra rows in 14F.\n"
            f"Count: {len(extra_final)}"
        )

    # ========================================================
    # Sort for deterministic output
    # ========================================================

    tier_order = {
        "GOLD": 0,
        "SILVER": 1,
        "BRONZE": 2,
        "REJECT": 3,
        "QUARANTINE": 4,
    }

    all_df[
        "_tier_order"
    ] = (
        all_df[
            "quality_tier"
        ]
        .map(
            tier_order
        )
    )

    all_df = (
        all_df
        .sort_values(
            [
                "_tier_order",
                "source_dataset",
                "pair_id",
            ],
            kind="stable",
        )
        .drop(
            columns=[
                "_tier_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Tier subsets
    # ========================================================

    gold_df = (
        all_df[
            all_df[
                "quality_tier"
            ]
            ==
            "GOLD"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    silver_df = (
        all_df[
            all_df[
                "quality_tier"
            ]
            ==
            "SILVER"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    bronze_df = (
        all_df[
            all_df[
                "quality_tier"
            ]
            ==
            "BRONZE"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    reject_df = (
        all_df[
            all_df[
                "quality_tier"
            ]
            ==
            "REJECT"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    quarantine_df = (
        all_df[
            all_df[
                "quality_tier"
            ]
            ==
            "QUARANTINE"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    approved_df = (
        all_df[
            all_df[
                "approved_for_training"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Training view:
    # only approved rows and core training columns first.
    preferred_training_columns = [
        "pair_id",
        "source_dataset",
        "en",
        "zh",
        "quality_tier",
        "training_weight",
        "quality_tier_reason",
        "quality_reviewed_by_qwen",
    ]

    training_columns = [
        column
        for column in preferred_training_columns
        if column in approved_df.columns
    ]

    extra_training_columns = [
        column
        for column in approved_df.columns
        if (
            column
            not in training_columns
            and
            column
            in {
                "quality_score",
                "risk_flags",
                "route",
            }
        )
    ]

    training_df = (
        approved_df[
            training_columns
            +
            extra_training_columns
        ]
        .copy()
    )

    # ========================================================
    # Reports
    # ========================================================

    quality_report = (
        all_df[
            "quality_tier"
        ]
        .value_counts()
        .rename_axis(
            "quality_tier"
        )
        .reset_index(
            name="count"
        )
    )

    quality_report[
        "percent_of_all"
    ] = (
        quality_report[
            "count"
        ]
        /
        len(all_df)
        *
        100
    )

    quality_report[
        "training_weight"
    ] = (
        quality_report[
            "quality_tier"
        ]
        .map(
            TRAINING_WEIGHTS
        )
    )

    quality_report[
        "approved_for_training"
    ] = (
        quality_report[
            "quality_tier"
        ]
        .isin(
            [
                "GOLD",
                "SILVER",
                "BRONZE",
            ]
        )
    )

    quality_report[
        "_order"
    ] = (
        quality_report[
            "quality_tier"
        ]
        .map(
            tier_order
        )
    )

    quality_report = (
        quality_report
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    source_report = (
        all_df
        .groupby(
            [
                "source_dataset",
                "quality_tier",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    source_totals = (
        source_report
        .groupby(
            "source_dataset"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )

    source_report[
        "percent_within_source"
    ] = (
        source_report[
            "count"
        ]
        /
        source_totals
        *
        100
    )

    decision_report = (
        all_df
        .groupby(
            [
                "quality_tier_reason",
                "quality_tier",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            [
                "quality_tier",
                "count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Save artifacts
    # ========================================================

    print(
        "\nSaving artifacts..."
    )

    save_dataframe(
        all_df,
        all_file,
        all_csv,
    )

    save_dataframe(
        approved_df,
        approved_file,
        approved_csv,
    )

    save_dataframe(
        training_df,
        training_file,
        training_csv,
    )

    save_dataframe(
        gold_df,
        gold_file,
    )

    save_dataframe(
        silver_df,
        silver_file,
    )

    save_dataframe(
        bronze_df,
        bronze_file,
    )

    save_dataframe(
        reject_df,
        reject_file,
    )

    save_dataframe(
        quarantine_df,
        quarantine_file,
    )

    quality_report.to_csv(
        quality_report_csv,
        index=False,
        encoding="utf-8-sig",
    )

    source_report.to_csv(
        source_report_csv,
        index=False,
        encoding="utf-8-sig",
    )

    decision_report.to_csv(
        decision_report_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # JSON report
    # ========================================================

    tier_counts = {
        tier: int(
            (
                all_df[
                    "quality_tier"
                ]
                ==
                tier
            ).sum()
        )
        for tier in [
            "GOLD",
            "SILVER",
            "BRONZE",
            "REJECT",
            "QUARANTINE",
        ]
    }

    approved_count = int(
        all_df[
            "approved_for_training"
        ].sum()
    )

    approved_percent = (
        approved_count
        /
        len(all_df)
        *
        100
    )

    report = {
        "step": "14F",
        "step_version": STEP_VERSION,

        "judge_prompt_version": (
            EXPECTED_JUDGE_PROMPT_VERSION
        ),

        "inputs": {
            "auto_accept_rows": int(
                len(auto_df)
            ),
            "needs_qwen_rows": int(
                len(risk_df)
            ),
            "full_review_rows": int(
                len(review_df)
            ),
            "risk_review_rows": int(
                len(review_risk)
            ),
            "auto_accept_audit_rows": int(
                len(review_auto)
            ),
            "original_14d_total": int(
                original_total
            ),
        },

        "tiers": tier_counts,

        "training_weights": (
            TRAINING_WEIGHTS
        ),

        "approved": {
            "rows": approved_count,
            "percent": float(
                approved_percent
            ),
        },

        "unreviewed_auto_accept": int(
            len(unreviewed_auto)
        ),

        "assertions": {
            "auto_risk_overlap_zero": (
                len(original_overlap)
                ==
                0
            ),
            "all_needs_qwen_reviewed": (
                len(missing_risk_review)
                ==
                0
            ),
            "auto_audit_subset_valid": (
                len(invalid_auto_audit)
                ==
                0
            ),
            "final_count_matches_14d": (
                len(all_df)
                ==
                original_total
            ),
            "pair_ids_unique": bool(
                not all_df[
                    "pair_id"
                ]
                .duplicated()
                .any()
            ),
            "no_rows_lost": (
                len(missing_final)
                ==
                0
            ),
            "no_extra_rows": (
                len(extra_final)
                ==
                0
            ),
        },

        "outputs": {
            "all": str(
                all_file
            ),
            "approved": str(
                approved_file
            ),
            "training_view": str(
                training_file
            ),
            "gold": str(
                gold_file
            ),
            "silver": str(
                silver_file
            ),
            "bronze": str(
                bronze_file
            ),
            "reject": str(
                reject_file
            ),
            "quarantine": str(
                quarantine_file
            ),
        },

        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "status": (
            "QUALITY_TIERS_READY_FOR_SPLIT"
        ),
    }

    with open(
        dataset_report_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final console report
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 14F RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput:"
    )

    print(
        "AUTO_ACCEPT:",
        len(auto_df)
    )

    print(
        "NEEDS_QWEN:",
        len(risk_df)
    )

    print(
        "Total 14D:",
        original_total
    )

    print(
        "\nQwen reviewed:"
    )

    print(
        "RISK_REVIEW:",
        len(review_risk)
    )

    print(
        "AUTO_ACCEPT_AUDIT:",
        len(review_auto)
    )

    print(
        "Total:",
        len(review_df)
    )

    print(
        "\nUnaudited AUTO_ACCEPT:"
    )

    print(
        len(unreviewed_auto)
    )

    print(
        "\nQuality tiers:"
    )

    print(
        quality_report
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nApproved for training:"
    )

    print(
        approved_count,
        "/",
        len(all_df),
        f"({approved_percent:.2f}%)"
    )

    print(
        "\nTier files:"
    )

    print(
        "GOLD:",
        gold_file
    )

    print(
        "SILVER:",
        silver_file
    )

    print(
        "BRONZE:",
        bronze_file
    )

    print(
        "REJECT:",
        reject_file
    )

    print(
        "QUARANTINE:",
        quarantine_file
    )

    print(
        "\nTraining view:"
    )

    print(
        training_file
    )

    print(
        "\nDataset report:"
    )

    print(
        dataset_report_json
    )

    print(
        "\nAssertions:"
    )

    for (
        key,
        value,
    ) in report[
        "assertions"
    ].items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nSTATUS:"
    )

    print(
        "QUALITY_TIERS_READY_FOR_SPLIT"
    )


if __name__ == "__main__":
    main()