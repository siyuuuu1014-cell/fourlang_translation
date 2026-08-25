from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Step04
# 65,897 条完整主表
# ============================================================

STEP04_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "04_rule_checked"
    / "parallel_rule_checked.parquet"
)


# ============================================================
# Step05B
# 4,079 条 Qwen 第一次语义审核结果
# ============================================================

STEP05B_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "results"
    / "qwen_semantic_judge_details.parquet"
)


# ============================================================
# Step05C
# 495 条 FAIL / UNCERTAIN 二审结果
# ============================================================

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
# 2. Output
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


# 完整主表
ALL_PAIRS_FILE = (
    OUTPUT_DIR
    / "en_uz_all_pairs_v1.parquet"
)

ALL_PAIRS_CSV = (
    OUTPUT_DIR
    / "en_uz_all_pairs_v1.csv"
)


# 最终可训练数据
APPROVED_FILE = (
    OUTPUT_DIR
    / "en_uz_approved_v1.parquet"
)

APPROVED_CSV = (
    OUTPUT_DIR
    / "en_uz_approved_v1.csv"
)


# 不同质量层
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


# 不进入训练
REJECT_FILE = (
    OUTPUT_DIR
    / "en_uz_reject_v1.parquet"
)

QUARANTINE_FILE = (
    OUTPUT_DIR
    / "en_uz_quarantine_v1.parquet"
)


# 精简训练视图
TRAINING_VIEW_FILE = (
    OUTPUT_DIR
    / "en_uz_training_view_v1.parquet"
)

TRAINING_VIEW_CSV = (
    OUTPUT_DIR
    / "en_uz_training_view_v1.csv"
)


# Report
REPORT_FILE = (
    OUTPUT_DIR
    / "dataset_report_v1.json"
)

TIER_REPORT_FILE = (
    OUTPUT_DIR
    / "quality_tier_report_v1.csv"
)

SOURCE_REPORT_FILE = (
    OUTPUT_DIR
    / "source_report_v1.csv"
)


# ============================================================
# 3. Quality policy
# ============================================================

QUALITY_WEIGHTS = {
    "GOLD": 1.0,
    "SILVER": 0.8,
    "BRONZE": 0.5,
    "REJECT": 0.0,
    "QUARANTINE": 0.0,
}


# ============================================================
# 4. Input validation
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
# 5. Load data
# ============================================================

def load_data():

    print("\nLoading Step04...")

    step04 = pd.read_parquet(
        STEP04_FILE
    )

    print(
        "Step04 rows:",
        len(step04)
    )


    print("\nLoading Step05B...")

    step05b = pd.read_parquet(
        STEP05B_FILE
    )

    print(
        "Step05B rows:",
        len(step05b)
    )


    print("\nLoading Step05C...")

    step05c = pd.read_parquet(
        STEP05C_FILE
    )

    print(
        "Step05C rows:",
        len(step05c)
    )


    return (
        step04,
        step05b,
        step05c,
    )


# ============================================================
# 6. Validate Step04
# ============================================================

def validate_step04(
    df: pd.DataFrame,
):

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


    duplicated = (
        df[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )


    if duplicated > 0:

        raise RuntimeError(
            "Step04 normalized_pair_id "
            f"contains {duplicated} duplicates."
        )


    print(
        "Step04 unique pair IDs:",
        df[
            "normalized_pair_id"
        ].nunique()
    )


# ============================================================
# 7. Prepare Step05B fields
# ============================================================

def prepare_step05b(
    df: pd.DataFrame,
):

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


    duplicated = (
        df[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )


    if duplicated > 0:

        raise RuntimeError(
            "Step05B contains duplicated "
            f"normalized_pair_id: {duplicated}"
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
# 8. Prepare Step05C fields
# ============================================================

def prepare_step05c(
    df: pd.DataFrame,
):

    required = [
        "normalized_pair_id",
        "resolution",
        "second_label",
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


    duplicated = (
        df[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )


    if duplicated > 0:

        raise RuntimeError(
            "Step05C contains duplicated "
            f"normalized_pair_id: {duplicated}"
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
# 9. Assign final quality
# ============================================================

def assign_final_quality(
    row,
):

    pipeline_route = str(
        row.get(
            "pipeline_route",
            "",
        )
    ).strip()


    first_label = str(
        row.get(
            "qwen_first_label",
            "",
        )
    ).upper().strip()


    second_resolution = str(
        row.get(
            "qwen_second_resolution",
            "",
        )
    ).upper().strip()


    review_type = str(
        row.get(
            "qwen_review_type",
            "",
        )
    ).strip()


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
                0.0,

            "final_decision_source":
                "STEP04_HARD_REJECT",

            "semantic_reviewed":
                False,
        })


    # ========================================================
    # B. No Qwen review
    #
    # AUTO_ACCEPT 中没有被抽进500条 audit 的数据
    # ========================================================

    if not first_label:

        if pipeline_route != "AUTO_ACCEPT":

            raise RuntimeError(
                "Found unreviewed sample that is "
                "not AUTO_ACCEPT:\n"
                f"{row['normalized_pair_id']}"
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
    # E. First FAIL / UNCERTAIN
    #
    # 必须存在 Step05C 二审
    # ========================================================

    if first_label in {
        "FAIL",
        "UNCERTAIN",
    }:

        if not second_resolution:

            raise RuntimeError(
                "FAIL/UNCERTAIN sample "
                "has no Step05C resolution:\n"
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
                    0.0,

                "final_decision_source":
                    "QWEN_SECOND_CONFIRMED_FAIL",

                "semantic_reviewed":
                    True,
            })


        # ----------------------------------------------------
        # uncertain after two reviews
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
                    0.0,

                "final_decision_source":
                    "QWEN_SECOND_QUARANTINE",

                "semantic_reviewed":
                    True,
            })


        # ----------------------------------------------------
        # First judge was wrong / second resolves PASS
        #
        # 不提升到 GOLD。
        # 放 SILVER 更保守。
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
        # Minor after second review
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
            "Unknown Step05C resolution: "
            f"{second_resolution}"
        )


    raise RuntimeError(
        "Unexpected first Qwen label: "
        f"{first_label}"
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 100)
    print("EN-UZ PIPELINE")
    print("STEP 06 - BUILD APPROVED DATASET V1")
    print("=" * 100)


    # ========================================================
    # Files
    # ========================================================

    check_input_files()


    # ========================================================
    # Load
    # ========================================================

    (
        step04,
        step05b,
        step05c,
    ) = load_data()


    # ========================================================
    # Validate
    # ========================================================

    validate_step04(
        step04
    )


    step05b_small = (
        prepare_step05b(
            step05b
        )
    )


    step05c_small = (
        prepare_step05c(
            step05c
        )
    )


    # ========================================================
    # Merge Step04 + 05B
    # ========================================================

    print(
        "\nMerging Step05B..."
    )


    master = step04.merge(

        step05b_small,

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

        step05c_small,

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
    # Basic final metadata
    # ========================================================

    master[
        "dataset_version"
    ] = "en_uz_v1"


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
    # 11. Final invariant checks
    # ========================================================

    print(
        "\nRunning invariant checks..."
    )


    # --------------------------------------------------------
    # Total rows
    # --------------------------------------------------------

    if len(master) != len(step04):

        raise RuntimeError(
            "Final row count mismatch."
        )


    # --------------------------------------------------------
    # Pair IDs unique
    # --------------------------------------------------------

    duplicated = (
        master[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )


    if duplicated > 0:

        raise RuntimeError(
            f"Final dataset contains "
            f"{duplicated} duplicate pair IDs."
        )


    # --------------------------------------------------------
    # No missing quality tiers
    # --------------------------------------------------------

    if master[
        "quality_tier"
    ].isna().any():

        raise RuntimeError(
            "Missing quality_tier detected."
        )


    # --------------------------------------------------------
    # NEEDS_QWEN must be reviewed
    # --------------------------------------------------------

    needs_qwen = master[
        master[
            "pipeline_route"
        ]
        ==
        "NEEDS_QWEN"
    ]


    missing_review = (
        needs_qwen[
            "qwen_first_label"
        ]
        .isna()
        .sum()
    )


    if missing_review > 0:

        raise RuntimeError(
            f"{missing_review} NEEDS_QWEN rows "
            f"were not reviewed."
        )


    # --------------------------------------------------------
    # First FAIL/UNCERTAIN must have second review
    # --------------------------------------------------------

    first_problem = master[
        master[
            "qwen_first_label"
        ]
        .fillna("")
        .isin(
            [
                "FAIL",
                "UNCERTAIN",
            ]
        )
    ]


    missing_second = (
        first_problem[
            "qwen_second_resolution"
        ]
        .isna()
        .sum()
    )


    if missing_second > 0:

        raise RuntimeError(
            f"{missing_second} first-pass "
            f"FAIL/UNCERTAIN rows "
            f"have no second review."
        )


    # --------------------------------------------------------
    # Benchmark leakage should already be zero
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
    # Cyrillic should already be removed
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
                f"Non-Latin Uzbek rows "
                f"remain: {non_latin}"
            )


    # ========================================================
    # 12. Split tiers
    # ========================================================

    gold_df = master[
        master[
            "quality_tier"
        ]
        ==
        "GOLD"
    ].copy()


    silver_df = master[
        master[
            "quality_tier"
        ]
        ==
        "SILVER"
    ].copy()


    bronze_df = master[
        master[
            "quality_tier"
        ]
        ==
        "BRONZE"
    ].copy()


    reject_df = master[
        master[
            "quality_tier"
        ]
        ==
        "REJECT"
    ].copy()


    quarantine_df = master[
        master[
            "quality_tier"
        ]
        ==
        "QUARANTINE"
    ].copy()


    approved_df = master[
        master[
            "final_status"
        ]
        ==
        "APPROVED"
    ].copy()


    # ========================================================
    # 13. Count integrity
    # ========================================================

    final_sum = (

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


    if final_sum != len(master):

        raise RuntimeError(
            "Quality tier counts do not "
            "sum to total."
        )


    # ========================================================
    # 14. Training view
    #
    # 只保留 Student 训练真正需要的关键字段。
    #
    # 当前仍然是 pair-level。
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
        "quality_score",

        "semantic_reviewed",

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
    # 15. Save
    # ========================================================

    print(
        "\nSaving dataset..."
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
    # 16. Tier report
    # ========================================================

    tier_report = (

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


    tier_report[
        "percent"
    ] = (

        tier_report[
            "count"
        ]
        /
        len(master)
        *
        100
    )


    tier_report[
        "training_weight"
    ] = (
        tier_report[
            "quality_tier"
        ]
        .map(
            QUALITY_WEIGHTS
        )
    )


    tier_report.to_csv(
        TIER_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 17. Source report
    # ========================================================

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


    source_report.to_csv(
        SOURCE_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 18. Decision-source report
    # ========================================================

    decision_distribution = (

        master[
            "final_decision_source"
        ]
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # 19. Qwen statistics
    # ========================================================

    reviewed_count = int(
        master[
            "semantic_reviewed"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )


    unreviewed_count = (
        len(master)
        -
        reviewed_count
    )


    # ========================================================
    # 20. Report
    # ========================================================

    report = {

        "dataset_name":
            "EN-UZ Approved Dataset",

        "dataset_version":
            "v1",

        "pair_level":
            True,

        "input_rows":
            len(master),

        "unique_pairs":
            int(
                master[
                    "normalized_pair_id"
                ].nunique()
            ),

        "approved_rows":
            len(
                approved_df
            ),

        "rejected_rows":
            len(
                reject_df
            ),

        "quarantine_rows":
            len(
                quarantine_df
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
                reviewed_count,

            "not_individually_reviewed":
                unreviewed_count,
        },

        "decision_distribution":
            {
                str(key):
                    int(value)

                for key, value
                in decision_distribution.items()
            },

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

        "benchmark_leak":
            0,

        "training_policy": {

            "baseline_exp1":
                "GOLD + SILVER",

            "baseline_exp2":
                "GOLD + SILVER + BRONZE",

            "weighted_exp":
                {
                    "GOLD":
                        1.0,

                    "SILVER":
                        0.8,

                    "BRONZE":
                        0.5,
                },

            "synthetic_data":
                "NOT INCLUDED IN V1",
        },
    }


    with open(
        REPORT_FILE,
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
    # 21. Terminal output
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STEP 06 COMPLETE")
    print("=" * 110)


    print(
        "Input pairs:",
        len(
            master
        )
    )

    print(
        "Unique pairs:",
        master[
            "normalized_pair_id"
        ].nunique()
    )


    print("\nQuality tiers:")

    print(
        tier_report
        .round(2)
        .to_string(
            index=False
        )
    )


    print("\nFinal dataset:")

    print(
        "GOLD:",
        len(
            gold_df
        )
    )

    print(
        "SILVER:",
        len(
            silver_df
        )
    )

    print(
        "BRONZE:",
        len(
            bronze_df
        )
    )

    print(
        "REJECT:",
        len(
            reject_df
        )
    )

    print(
        "QUARANTINE:",
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
        reviewed_count
    )

    print(
        "Not individually reviewed:",
        unreviewed_count
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
        "\nReport:"
    )

    print(
        REPORT_FILE
    )


    print("\nDataset V1 ready.")


if __name__ == "__main__":

    main()