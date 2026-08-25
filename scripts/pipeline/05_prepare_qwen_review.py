from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STEP04_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "04_rule_checked"
)

NEEDS_QWEN_FILE = (
    STEP04_DIR
    / "needs_qwen.parquet"
)

AUTO_ACCEPT_FILE = (
    STEP04_DIR
    / "auto_accept.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REVIEW_FILE = (
    OUTPUT_DIR
    / "qwen_review_input.parquet"
)

REVIEW_CSV = (
    OUTPUT_DIR
    / "qwen_review_input.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "auto_accept_audit.parquet"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "review_prepare_report.json"
)


# ============================================================
# 2. Config
# ============================================================

SEED = 2026

AUTO_ACCEPT_AUDIT_SIZE = 500


# ============================================================
# 3. Build stratified audit sample
# ============================================================

def build_audit_sample(
    df: pd.DataFrame,
    target_size: int,
) -> pd.DataFrame:

    if len(df) <= target_size:

        return df.copy()


    # --------------------------------------------------------
    # 长度分桶
    #
    # 避免500条全是短句。
    # --------------------------------------------------------

    df = df.copy()

    df["length_bucket"] = pd.cut(
        df["source_word_count"],
        bins=[
            -1,
            5,
            10,
            20,
            40,
            float("inf"),
        ],
        labels=[
            "very_short",
            "short",
            "medium",
            "long",
            "very_long",
        ],
    )


    # --------------------------------------------------------
    # 使用 data_source + length_bucket 分层
    # --------------------------------------------------------

    strata = (
        df.groupby(
            [
                "data_source",
                "length_bucket",
            ],
            observed=True,
        )
    )


    samples = []


    for _, part in strata:

        proportion = (
            len(part)
            /
            len(df)
        )

        n = max(
            1,
            round(
                target_size
                *
                proportion
            ),
        )

        n = min(
            n,
            len(part),
        )


        samples.append(
            part.sample(
                n=n,
                random_state=SEED,
            )
        )


    audit_df = pd.concat(
        samples,
        ignore_index=True,
    )


    # --------------------------------------------------------
    # 如果分层后超过目标数量，再统一抽样
    # --------------------------------------------------------

    if len(audit_df) > target_size:

        audit_df = audit_df.sample(
            n=target_size,
            random_state=SEED,
        )


    # --------------------------------------------------------
    # 如果不足目标数量，从剩余数据补齐
    # --------------------------------------------------------

    elif len(audit_df) < target_size:

        used_ids = set(
            audit_df[
                "normalized_pair_id"
            ]
        )

        remaining = df[
            ~df[
                "normalized_pair_id"
            ].isin(
                used_ids
            )
        ]


        need = (
            target_size
            -
            len(audit_df)
        )


        if len(remaining) >= need:

            extra = remaining.sample(
                n=need,
                random_state=SEED + 1,
            )

            audit_df = pd.concat(
                [
                    audit_df,
                    extra,
                ],
                ignore_index=True,
            )


    return audit_df.reset_index(
        drop=True
    )


# ============================================================
# 4. Main
# ============================================================

def main():

    print("=" * 100)
    print("EN-UZ PIPELINE")
    print("STEP 05A - PREPARE QWEN REVIEW")
    print("=" * 100)


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not NEEDS_QWEN_FILE.exists():

        raise FileNotFoundError(
            NEEDS_QWEN_FILE
        )


    if not AUTO_ACCEPT_FILE.exists():

        raise FileNotFoundError(
            AUTO_ACCEPT_FILE
        )


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    needs_qwen_df = pd.read_parquet(
        NEEDS_QWEN_FILE
    )

    auto_accept_df = pd.read_parquet(
        AUTO_ACCEPT_FILE
    )


    print(
        "\nNEEDS_QWEN:",
        len(needs_qwen_df)
    )

    print(
        "AUTO_ACCEPT:",
        len(auto_accept_df)
    )


    # ========================================================
    # 5. Full NEEDS_QWEN review set
    # ========================================================

    review_df = (
        needs_qwen_df
        .copy()
        .reset_index(
            drop=True
        )
    )


    review_df[
        "review_type"
    ] = "RISK_REVIEW"


    # ========================================================
    # 6. AUTO_ACCEPT audit sample
    # ========================================================

    audit_df = build_audit_sample(
        auto_accept_df,
        AUTO_ACCEPT_AUDIT_SIZE,
    )


    audit_df[
        "review_type"
    ] = "AUTO_ACCEPT_AUDIT"


    print(
        "\nAUTO_ACCEPT audit sample:",
        len(audit_df)
    )


    # ========================================================
    # 7. Combine
    # ========================================================

    combined_df = pd.concat(
        [
            review_df,
            audit_df,
        ],
        ignore_index=True,
    )


    combined_df.insert(
        0,
        "review_id",
        [
            f"review_{i:06d}"
            for i in range(
                1,
                len(combined_df) + 1,
            )
        ],
    )


    # ========================================================
    # 8. Safety dedup
    # ========================================================

    before = len(
        combined_df
    )

    combined_df = (
        combined_df
        .drop_duplicates(
            subset=[
                "normalized_pair_id",
                "review_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    duplicate_removed = (
        before
        -
        len(combined_df)
    )


    # ========================================================
    # 9. Save
    # ========================================================

    combined_df.to_parquet(
        REVIEW_FILE,
        index=False,
    )


    combined_df.to_csv(
        REVIEW_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    audit_df.to_parquet(
        AUDIT_FILE,
        index=False,
    )


    # ========================================================
    # 10. Report
    # ========================================================

    report = {

        "pipeline":
            "en_uz",

        "step":
            "05_prepare_qwen_review",

        "needs_qwen_total":
            len(
                needs_qwen_df
            ),

        "auto_accept_total":
            len(
                auto_accept_df
            ),

        "audit_sample":
            len(
                audit_df
            ),

        "combined_review_rows":
            len(
                combined_df
            ),

        "duplicates_removed":
            duplicate_removed,

        "review_type_distribution":
            combined_df[
                "review_type"
            ]
            .value_counts()
            .to_dict(),
    }


    with open(
        REPORT_FILE,
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
    # 11. Terminal
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 05A COMPLETE")
    print("=" * 100)


    print(
        "Risk review:",
        len(
            review_df
        )
    )

    print(
        "AUTO_ACCEPT audit:",
        len(
            audit_df
        )
    )

    print(
        "Total Qwen reviews:",
        len(
            combined_df
        )
    )


    print(
        "\nReview type distribution:"
    )

    print(
        combined_df[
            "review_type"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nRisk flags in RISK_REVIEW:"
    )


    flags = (
        review_df[
            "risk_flags"
        ]
        .fillna("")
        .str.get_dummies(
            sep="|"
        )
        .sum()
        .sort_values(
            ascending=False
        )
    )


    print(
        flags.to_string()
    )


    print(
        "\nAUTO_ACCEPT audit source distribution:"
    )

    print(
        audit_df[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nFiles:"
    )

    print(
        REVIEW_FILE
    )

    print(
        REVIEW_CSV
    )

    print(
        AUDIT_FILE
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()