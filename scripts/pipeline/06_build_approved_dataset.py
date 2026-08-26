from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. Input files
# ============================================================

STEP04_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "04_rule_checked"
    / "parallel_rule_checked.parquet"
)

STEP05B_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "results"
    / "qwen_semantic_judge_details.parquet"
)

STEP05C_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "results"
    / "second_review"
    / "second_review_details.parquet"
)


# ============================================================
# 3. Output directory
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "approved"
    / "en_uz"
    / "v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. Output files
# ============================================================

ALL_PAIRS_FILE = (
    OUTPUT_DIR
    / "en_uz_all_pairs_v1.parquet"
)

ALL_PAIRS_CSV = (
    OUTPUT_DIR
    / "en_uz_all_pairs_v1.csv"
)


APPROVED_FILE = (
    OUTPUT_DIR
    / "en_uz_approved_v1.parquet"
)

APPROVED_CSV = (
    OUTPUT_DIR
    / "en_uz_approved_v1.csv"
)


GOLD_FILE = (
    OUTPUT_DIR
    / "en_uz_gold_v1.parquet"
)

SILVER_FILE = (
    OUTPUT_DIR
    / "en_uz_silver_v1.parquet"
)

BRONZE_FILE = (
    OUTPUT_DIR
    / "en_uz_bronze_v1.parquet"
)

REJECT_FILE = (
    OUTPUT_DIR
    / "en_uz_reject_v1.parquet"
)

QUARANTINE_FILE = (
    OUTPUT_DIR
    / "en_uz_quarantine_v1.parquet"
)


TRAINING_VIEW_FILE = (
    OUTPUT_DIR
    / "en_uz_training_view_v1.parquet"
)

TRAINING_VIEW_CSV = (
    OUTPUT_DIR
    / "en_uz_training_view_v1.csv"
)


QUALITY_REPORT_FILE = (
    OUTPUT_DIR
    / "quality_tier_report_v1.csv"
)

SOURCE_REPORT_FILE = (
    OUTPUT_DIR
    / "source_report_v1.csv"
)

DECISION_REPORT_FILE = (
    OUTPUT_DIR
    / "decision_report_v1.csv"
)

DATASET_REPORT_FILE = (
    OUTPUT_DIR
    / "dataset_report_v1.json"
)


# ============================================================
# 5. Dataset config
# ============================================================

DATASET_VERSION = "en_uz_v1"

QUALITY_WEIGHTS = {
    "GOLD": 1.0,
    "SILVER": 0.8,
    "BRONZE": 0.5,
    "REJECT": 0.0,
    "QUARANTINE": 0.0,
}


# ============================================================
# 6. Helpers
# ============================================================

def clean_string(value) -> str:
    """
    安全处理 pandas NaN / None。

    这是之前报：
        Unexpected first Qwen label: NAN

    的核心修复。
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def clean_upper(value) -> str:

    return clean_string(
        value
    ).upper()


def bool_value(value) -> bool:

    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    if isinstance(value, bool):
        return value

    return (
        str(value)
        .strip()
        .lower()
        in {
            "true",
            "1",
            "yes",
        }
    )


# ============================================================
# 7. Check files
# ============================================================

def check_input_files():

    files = {
        "Step04": STEP04_FILE,
        "Step05B": STEP05B_FILE,
        "Step05C": STEP05C_FILE,
    }

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} file not found:\n"
                f"{path}"
            )


# ============================================================
# 8. Validate unique pair id
# ============================================================

def validate_unique_pair_id(
    df: pd.DataFrame,
    name: str,
):

    if "normalized_pair_id" not in df.columns:

        raise ValueError(
            f"{name} missing normalized_pair_id"
        )

    duplicate_count = int(
        df[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise RuntimeError(
            f"{name} contains "
            f"{duplicate_count} duplicated "
            f"normalized_pair_id values."
        )


# ============================================================
# 9. Load Step04
# ============================================================

def load_step04():

    print("\nLoading Step04...")

    df = pd.read_parquet(
        STEP04_FILE
    )

    print(
        "Step04 rows:",
        len(df)
    )

    required = [
        "normalized_pair_id",
        "source_text_normalized",
        "target_text_normalized",
        "pipeline_route",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Step04 missing columns: "
            f"{missing}"
        )

    validate_unique_pair_id(
        df,
        "Step04",
    )

    print(
        "Step04 unique pair IDs:",
        df[
            "normalized_pair_id"
        ].nunique()
    )

    return df


# ============================================================
# 10. Load Step05B
# ============================================================

def load_step05b():

    print("\nLoading Step05B...")

    df = pd.read_parquet(
        STEP05B_FILE
    )

    print(
        "Step05B rows:",
        len(df)
    )

    required = [
        "normalized_pair_id",
        "review_id",
        "review_type",
        "judge_label",
        "judge_confidence",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Step05B missing columns: "
            f"{missing}"
        )

    validate_unique_pair_id(
        df,
        "Step05B",
    )

    keep_columns = [
        "normalized_pair_id",

        "review_id",
        "review_type",

        "judge_label",
        "semantic_consistent",

        "judge_omission",
        "judge_addition",
        "judge_mistranslation",

        "judge_number_error",
        "judge_time_error",
        "judge_entity_error",
        "judge_negation_error",

        "judge_confidence",
        "judge_reason",
        "judge_parse_success",
        "judge_latency",
    ]

    keep_columns = [
        col
        for col in keep_columns
        if col in df.columns
    ]

    result = df[
        keep_columns
    ].copy()

    rename_map = {
        "review_id":
            "qwen_review_id",

        "review_type":
            "qwen_review_type",

        "judge_label":
            "qwen_first_label",

        "semantic_consistent":
            "qwen_first_semantic_consistent",

        "judge_omission":
            "qwen_first_omission",

        "judge_addition":
            "qwen_first_addition",

        "judge_mistranslation":
            "qwen_first_mistranslation",

        "judge_number_error":
            "qwen_first_number_error",

        "judge_time_error":
            "qwen_first_time_error",

        "judge_entity_error":
            "qwen_first_entity_error",

        "judge_negation_error":
            "qwen_first_negation_error",

        "judge_confidence":
            "qwen_first_confidence",

        "judge_reason":
            "qwen_first_reason",

        "judge_parse_success":
            "qwen_first_parse_success",

        "judge_latency":
            "qwen_first_latency",
    }

    result = result.rename(
        columns=rename_map
    )

    return result


# ============================================================
# 11. Load Step05C
# ============================================================

def load_step05c():

    print("\nLoading Step05C...")

    df = pd.read_parquet(
        STEP05C_FILE
    )

    print(
        "Step05C rows:",
        len(df)
    )

    required = [
        "normalized_pair_id",
        "second_label",
        "resolution",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Step05C missing columns: "
            f"{missing}"
        )

    validate_unique_pair_id(
        df,
        "Step05C",
    )

    keep_columns = [
        "normalized_pair_id",

        "second_label",
        "second_semantic_consistent",

        "second_omission",
        "second_addition",
        "second_mistranslation",

        "second_number_error",
        "second_time_error",
        "second_entity_error",
        "second_negation_error",

        "second_high_risk",

        "second_confidence",
        "second_reason",

        "second_parse_success",
        "second_latency",

        "resolution",
    ]

    keep_columns = [
        col
        for col in keep_columns
        if col in df.columns
    ]

    result = df[
        keep_columns
    ].copy()

    rename_map = {
        "second_label":
            "qwen_second_label",

        "second_semantic_consistent":
            "qwen_second_semantic_consistent",

        "second_omission":
            "qwen_second_omission",

        "second_addition":
            "qwen_second_addition",

        "second_mistranslation":
            "qwen_second_mistranslation",

        "second_number_error":
            "qwen_second_number_error",

        "second_time_error":
            "qwen_second_time_error",

        "second_entity_error":
            "qwen_second_entity_error",

        "second_negation_error":
            "qwen_second_negation_error",

        "second_high_risk":
            "qwen_second_high_risk",

        "second_confidence":
            "qwen_second_confidence",

        "second_reason":
            "qwen_second_reason",

        "second_parse_success":
            "qwen_second_parse_success",

        "second_latency":
            "qwen_second_latency",

        "resolution":
            "qwen_second_resolution",
    }

    result = result.rename(
        columns=rename_map
    )

    return result


# ============================================================
# 12. Final quality decision
# ============================================================

def assign_final_quality(
    row,
):

    pipeline_route = clean_upper(
        row.get(
            "pipeline_route",
            "",
        )
    )

    first_label = clean_upper(
        row.get(
            "qwen_first_label",
            "",
        )
    )

    second_resolution = clean_upper(
        row.get(
            "qwen_second_resolution",
            "",
        )
    )

    review_type = clean_upper(
        row.get(
            "qwen_review_type",
            "",
        )
    )


    # ========================================================
    # A. Step04 hard reject
    # ========================================================

    if pipeline_route == "HARD_REJECT":

        return pd.Series({
            "final_status":
                "REJECT",

            "quality_tier":
                "REJECT",

            "training_weight":
                QUALITY_WEIGHTS[
                    "REJECT"
                ],

            "final_decision_source":
                "STEP04_HARD_REJECT",

            "semantic_reviewed":
                False,
        })


    # ========================================================
    # B. No Qwen review
    #
    # 这是 AUTO_ACCEPT 中没有抽到500条审计的数据。
    # ========================================================

    if first_label == "":

        if pipeline_route != "AUTO_ACCEPT":

            raise RuntimeError(
                "Found sample without Qwen review "
                "but pipeline_route is not AUTO_ACCEPT.\n"
                f"Pair ID: "
                f"{row['normalized_pair_id']}\n"
                f"pipeline_route: "
                f"{pipeline_route}"
            )

        return pd.Series({
            "final_status":
                "APPROVED",

            "quality_tier":
                "SILVER",

            "training_weight":
                QUALITY_WEIGHTS[
                    "SILVER"
                ],

            "final_decision_source":
                "RULE_LOW_RISK",

            "semantic_reviewed":
                False,
        })


    # ========================================================
    # C. First Qwen PASS
    # ========================================================

    if first_label == "PASS":

        return pd.Series({
            "final_status":
                "APPROVED",

            "quality_tier":
                "GOLD",

            "training_weight":
                QUALITY_WEIGHTS[
                    "GOLD"
                ],

            "final_decision_source":
                (
                    "QWEN_PASS_"
                    +
                    (
                        review_type
                        if review_type
                        else "REVIEW"
                    )
                ),

            "semantic_reviewed":
                True,
        })


    # ========================================================
    # D. First Qwen MINOR
    # ========================================================

    if first_label == "MINOR":

        return pd.Series({
            "final_status":
                "APPROVED",

            "quality_tier":
                "BRONZE",

            "training_weight":
                QUALITY_WEIGHTS[
                    "BRONZE"
                ],

            "final_decision_source":
                (
                    "QWEN_MINOR_"
                    +
                    (
                        review_type
                        if review_type
                        else "REVIEW"
                    )
                ),

            "semantic_reviewed":
                True,
        })


    # ========================================================
    # E. FAIL / UNCERTAIN
    #
    # 这两类必须经过 Step05C。
    # ========================================================

    if first_label in {
        "FAIL",
        "UNCERTAIN",
    }:

        if second_resolution == "":

            raise RuntimeError(
                "First Qwen label is "
                f"{first_label}, "
                "but Step05C resolution is missing.\n"
                f"Pair ID: "
                f"{row['normalized_pair_id']}"
            )


        # ----------------------------------------------------
        # confirmed fail
        # ----------------------------------------------------

        if (
            second_resolution
            ==
            "CONFIRMED_FAIL"
        ):

            return pd.Series({
                "final_status":
                    "REJECT",

                "quality_tier":
                    "REJECT",

                "training_weight":
                    QUALITY_WEIGHTS[
                        "REJECT"
                    ],

                "final_decision_source":
                    "QWEN_SECOND_CONFIRMED_FAIL",

                "semantic_reviewed":
                    True,
            })


        # ----------------------------------------------------
        # still uncertain
        # ----------------------------------------------------

        if (
            second_resolution
            ==
            "QUARANTINE"
        ):

            return pd.Series({
                "final_status":
                    "QUARANTINE",

                "quality_tier":
                    "QUARANTINE",

                "training_weight":
                    QUALITY_WEIGHTS[
                        "QUARANTINE"
                    ],

                "final_decision_source":
                    "QWEN_SECOND_QUARANTINE",

                "semantic_reviewed":
                    True,
            })


        # ----------------------------------------------------
        # First judge error / resolved pass
        #
        # 两轮意见冲突过，所以保守地放 SILVER，
        # 不直接提升到 GOLD。
        # ----------------------------------------------------

        if second_resolution in {
            "JUDGE_ERROR",
            "RESOLVED_PASS",
        }:

            return pd.Series({
                "final_status":
                    "APPROVED",

                "quality_tier":
                    "SILVER",

                "training_weight":
                    QUALITY_WEIGHTS[
                        "SILVER"
                    ],

                "final_decision_source":
                    (
                        "QWEN_SECOND_"
                        +
                        second_resolution
                    ),

                "semantic_reviewed":
                    True,
            })


        # ----------------------------------------------------
        # Resolved minor
        # ----------------------------------------------------

        if second_resolution in {
            "DOWNGRADED_MINOR",
            "RESOLVED_MINOR",
        }:

            return pd.Series({
                "final_status":
                    "APPROVED",

                "quality_tier":
                    "BRONZE",

                "training_weight":
                    QUALITY_WEIGHTS[
                        "BRONZE"
                    ],

                "final_decision_source":
                    (
                        "QWEN_SECOND_"
                        +
                        second_resolution
                    ),

                "semantic_reviewed":
                    True,
            })


        raise RuntimeError(
            "Unknown Step05C resolution:\n"
            f"{second_resolution}\n"
            f"Pair ID: "
            f"{row['normalized_pair_id']}"
        )


    # ========================================================
    # Unexpected label
    # ========================================================

    raise RuntimeError(
        "Unexpected first Qwen label:\n"
        f"{first_label}\n"
        f"Pair ID: "
        f"{row['normalized_pair_id']}"
    )


# ============================================================
# 13. Integrity checks
# ============================================================

def run_integrity_checks(
    master: pd.DataFrame,
):

    print(
        "\nRunning invariant checks..."
    )


    # --------------------------------------------------------
    # Pair ID unique
    # --------------------------------------------------------

    duplicate_count = int(
        master[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        raise RuntimeError(
            f"Final master contains "
            f"{duplicate_count} duplicate pairs."
        )


    # --------------------------------------------------------
    # Every row must have final decision
    # --------------------------------------------------------

    if (
        master[
            "final_status"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing final_status detected."
        )


    if (
        master[
            "quality_tier"
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Missing quality_tier detected."
        )


    # --------------------------------------------------------
    # All NEEDS_QWEN reviewed
    # --------------------------------------------------------

    needs_qwen = master[
        master[
            "pipeline_route"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        ==
        "NEEDS_QWEN"
    ]

    missing_qwen = int(
        needs_qwen[
            "qwen_first_label"
        ]
        .isna()
        .sum()
    )

    if missing_qwen > 0:

        raise RuntimeError(
            f"{missing_qwen} NEEDS_QWEN "
            f"samples have no Qwen review."
        )


    # --------------------------------------------------------
    # FAIL / UNCERTAIN must have Step05C
    # --------------------------------------------------------

    first_problem = master[
        master[
            "qwen_first_label"
        ]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(
            [
                "FAIL",
                "UNCERTAIN",
            ]
        )
    ]

    missing_second = int(
        first_problem[
            "qwen_second_resolution"
        ]
        .isna()
        .sum()
    )

    if missing_second > 0:

        raise RuntimeError(
            f"{missing_second} first-pass "
            f"FAIL/UNCERTAIN samples "
            f"have no Step05C result."
        )


    # --------------------------------------------------------
    # Benchmark leakage
    # --------------------------------------------------------

    if "benchmark_leak" in master.columns:

        leak_count = int(
            master[
                "benchmark_leak"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        if leak_count > 0:

            raise RuntimeError(
                f"Benchmark leakage detected: "
                f"{leak_count}"
            )


    # --------------------------------------------------------
    # Latin Uzbek
    # --------------------------------------------------------

    if "latin_uzbek" in master.columns:

        non_latin = int(
            (
                ~master[
                    "latin_uzbek"
                ]
                .fillna(False)
                .astype(bool)
            ).sum()
        )

        if non_latin > 0:

            raise RuntimeError(
                f"Non-Latin Uzbek remains: "
                f"{non_latin}"
            )


    print(
        "Invariant checks: PASS"
    )


# ============================================================
# 14. Main
# ============================================================

def main():

    print("=" * 100)
    print("EN-UZ PIPELINE")
    print("STEP 06 - BUILD APPROVED DATASET V1")
    print("=" * 100)


    # ========================================================
    # Check files
    # ========================================================

    check_input_files()


    # ========================================================
    # Load data
    # ========================================================

    step04 = load_step04()

    step05b = load_step05b()

    step05c = load_step05c()


    # ========================================================
    # Merge Step05B
    # ========================================================

    print(
        "\nMerging Step05B..."
    )

    master = step04.merge(
        step05b,
        on="normalized_pair_id",
        how="left",
        validate="one_to_one",
    )


    if len(master) != len(step04):

        raise RuntimeError(
            "Row count changed after "
            "Step05B merge."
        )


    # ========================================================
    # Merge Step05C
    # ========================================================

    print(
        "Merging Step05C..."
    )

    master = master.merge(
        step05c,
        on="normalized_pair_id",
        how="left",
        validate="one_to_one",
    )


    if len(master) != len(step04):

        raise RuntimeError(
            "Row count changed after "
            "Step05C merge."
        )


    # ========================================================
    # Show merged review counts
    # ========================================================

    print(
        "\nMerged first Qwen labels:"
    )

    print(
        master[
            "qwen_first_label"
        ]
        .fillna("NOT_REVIEWED")
        .value_counts()
        .to_string()
    )


    print(
        "\nMerged Step05C resolutions:"
    )

    print(
        master[
            "qwen_second_resolution"
        ]
        .fillna("NO_SECOND_REVIEW")
        .value_counts()
        .to_string()
    )


    # ========================================================
    # Assign final quality
    # ========================================================

    print(
        "\nAssigning quality tiers..."
    )

    quality_df = master.apply(
        assign_final_quality,
        axis=1,
    )


    master = pd.concat(
        [
            master.reset_index(
                drop=True
            ),

            quality_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    # ========================================================
    # Dataset metadata
    # ========================================================

    master[
        "dataset_version"
    ] = DATASET_VERSION


    master[
        "pair_direction"
    ] = "en_uz"


    master[
        "approved"
    ] = (
        master[
            "final_status"
        ]
        ==
        "APPROVED"
    )


    # ========================================================
    # Integrity checks
    # ========================================================

    run_integrity_checks(
        master
    )


    # ========================================================
    # Split quality tiers
    # ========================================================

    gold_df = (
        master[
            master[
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
        master[
            master[
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
        master[
            master[
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
        master[
            master[
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
        master[
            master[
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
        master[
            master[
                "final_status"
            ]
            ==
            "APPROVED"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Final count check
    # ========================================================

    tier_sum = (
        len(gold_df)
        +
        len(silver_df)
        +
        len(bronze_df)
        +
        len(reject_df)
        +
        len(quarantine_df)
    )


    if tier_sum != len(master):

        raise RuntimeError(
            "Quality tier count mismatch.\n"
            f"Tier sum: {tier_sum}\n"
            f"Master: {len(master)}"
        )


    approved_sum = (
        len(gold_df)
        +
        len(silver_df)
        +
        len(bronze_df)
    )


    if approved_sum != len(
        approved_df
    ):

        raise RuntimeError(
            "Approved count mismatch."
        )


    # ========================================================
    # Training view
    #
    # 注意：
    # 现在仍然保留 pair-level。
    #
    # Step07 切分完成后，
    # 才展开为：
    #
    # EN -> UZ
    # UZ -> EN
    # ========================================================

    training_columns = [
        "normalized_pair_id",

        "src_lang",
        "tgt_lang",

        "source_text_normalized",
        "target_text_normalized",

        "quality_tier",
        "training_weight",

        "data_source",
        "data_sources",

        "source_families",
        "source_family_count",

        "occurrence_count",

        "risk_flags",
        "risk_count",

        "quality_score",

        "semantic_reviewed",

        "final_decision_source",

        "dataset_version",
    ]


    training_columns = [
        col
        for col in training_columns
        if col in approved_df.columns
    ]


    training_view = (
        approved_df[
            training_columns
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Save
    # ========================================================

    print(
        "\nSaving datasets..."
    )


    master.to_parquet(
        ALL_PAIRS_FILE,
        index=False,
    )

    master.to_csv(
        ALL_PAIRS_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    approved_df.to_parquet(
        APPROVED_FILE,
        index=False,
    )

    approved_df.to_csv(
        APPROVED_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    gold_df.to_parquet(
        GOLD_FILE,
        index=False,
    )

    silver_df.to_parquet(
        SILVER_FILE,
        index=False,
    )

    bronze_df.to_parquet(
        BRONZE_FILE,
        index=False,
    )

    reject_df.to_parquet(
        REJECT_FILE,
        index=False,
    )

    quarantine_df.to_parquet(
        QUARANTINE_FILE,
        index=False,
    )


    training_view.to_parquet(
        TRAINING_VIEW_FILE,
        index=False,
    )

    training_view.to_csv(
        TRAINING_VIEW_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Quality tier report
    # ========================================================

    quality_report = (
        master[
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
        "percent"
    ] = (
        quality_report[
            "count"
        ]
        /
        len(master)
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
            QUALITY_WEIGHTS
        )
    )


    quality_report.to_csv(
        QUALITY_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Source report
    # ========================================================

    if "data_source" in approved_df.columns:

        source_report = (
            approved_df[
                [
                    "data_source",
                    "quality_tier",
                ]
            ]
            .value_counts()
            .rename(
                "count"
            )
            .reset_index()
        )

    else:

        source_report = (
            pd.DataFrame()
        )


    source_report.to_csv(
        SOURCE_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Decision report
    # ========================================================

    decision_report = (
        master[
            "final_decision_source"
        ]
        .value_counts()
        .rename_axis(
            "decision"
        )
        .reset_index(
            name="count"
        )
    )


    decision_report[
        "percent"
    ] = (
        decision_report[
            "count"
        ]
        /
        len(master)
        *
        100
    )


    decision_report.to_csv(
        DECISION_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Semantic review stats
    # ========================================================

    semantic_reviewed_count = int(
        master[
            "semantic_reviewed"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )


    not_reviewed_count = (
        len(master)
        -
        semantic_reviewed_count
    )


    # ========================================================
    # Qwen label distribution
    # ========================================================

    qwen_first_distribution = (
        master[
            "qwen_first_label"
        ]
        .fillna(
            "NOT_REVIEWED"
        )
        .value_counts()
        .to_dict()
    )


    qwen_second_distribution = (
        master[
            "qwen_second_resolution"
        ]
        .fillna(
            "NO_SECOND_REVIEW"
        )
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # Dataset report
    # ========================================================

    report = {

        "dataset_name":
            "EN-UZ Approved Dataset",

        "dataset_version":
            DATASET_VERSION,

        "pair_level":
            True,

        "input_pairs":
            len(master),

        "unique_pairs":
            int(
                master[
                    "normalized_pair_id"
                ].nunique()
            ),

        "approved_pairs":
            len(
                approved_df
            ),

        "approved_rate_percent":
            float(
                len(
                    approved_df
                )
                /
                len(master)
                *
                100
            ),

        "quality_tiers": {
            "GOLD":
                len(
                    gold_df
                ),

            "SILVER":
                len(
                    silver_df
                ),

            "BRONZE":
                len(
                    bronze_df
                ),

            "REJECT":
                len(
                    reject_df
                ),

            "QUARANTINE":
                len(
                    quarantine_df
                ),
        },

        "quality_weights":
            QUALITY_WEIGHTS,

        "semantic_review": {
            "reviewed":
                semantic_reviewed_count,

            "not_individually_reviewed":
                not_reviewed_count,
        },

        "step05b_distribution": {
            str(key):
                int(value)

            for key, value
            in qwen_first_distribution.items()
        },

        "step05c_distribution": {
            str(key):
                int(value)

            for key, value
            in qwen_second_distribution.items()
        },

        "final_decision_distribution": {
            str(
                row["decision"]
            ):
                int(
                    row["count"]
                )

            for _, row
            in decision_report.iterrows()
        },

        "benchmark_leak":
            0,

        "synthetic_data_included":
            False,

        "training_policy": {
            "exp1":
                "GOLD + SILVER",

            "exp2":
                "GOLD + SILVER + BRONZE",

            "exp3":
                "GOLD + SILVER + BRONZE "
                "with quality weighting",

            "exp4":
                "Add MADLAD synthetic data",
        },
    }


    with open(
        DATASET_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Final terminal output
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STEP 06 COMPLETE")
    print("=" * 110)


    print(
        "Input pairs:",
        len(master)
    )

    print(
        "Unique pairs:",
        master[
            "normalized_pair_id"
        ].nunique()
    )


    print("\nQuality tiers:")

    print(
        quality_report
        .round(2)
        .to_string(
            index=False
        )
    )


    print("\nFinal:")

    print(
        "GOLD       :",
        len(
            gold_df
        )
    )

    print(
        "SILVER     :",
        len(
            silver_df
        )
    )

    print(
        "BRONZE     :",
        len(
            bronze_df
        )
    )

    print(
        "REJECT     :",
        len(
            reject_df
        )
    )

    print(
        "QUARANTINE :",
        len(
            quarantine_df
        )
    )


    print(
        "\nAPPROVED:",
        len(
            approved_df
        )
    )

    print(
        "Approved rate:",
        f"{len(approved_df) / len(master) * 100:.2f}%"
    )


    print(
        "\nSemantic reviewed:",
        semantic_reviewed_count
    )

    print(
        "Not individually reviewed:",
        not_reviewed_count
    )


    print(
        "\nFirst Qwen label distribution:"
    )

    print(
        master[
            "qwen_first_label"
        ]
        .fillna(
            "NOT_REVIEWED"
        )
        .value_counts()
        .to_string()
    )


    print(
        "\nSecond review distribution:"
    )

    print(
        master[
            "qwen_second_resolution"
        ]
        .fillna(
            "NO_SECOND_REVIEW"
        )
        .value_counts()
        .to_string()
    )


    print(
        "\nDecision source:"
    )

    print(
        master[
            "final_decision_source"
        ]
        .value_counts()
        .to_string()
    )


    if (
        "data_source"
        in approved_df.columns
    ):

        print(
            "\nApproved source distribution:"
        )

        print(
            approved_df[
                "data_source"
            ]
            .value_counts()
            .to_string()
        )


    print("\nFiles:")

    print(
        "All pairs:"
    )

    print(
        ALL_PAIRS_FILE
    )


    print(
        "\nApproved:"
    )

    print(
        APPROVED_FILE
    )


    print(
        "\nTraining view:"
    )

    print(
        TRAINING_VIEW_FILE
    )


    print(
        "\nDataset report:"
    )

    print(
        DATASET_REPORT_FILE
    )


    print("\nEN-UZ Dataset V1 ready.")


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()