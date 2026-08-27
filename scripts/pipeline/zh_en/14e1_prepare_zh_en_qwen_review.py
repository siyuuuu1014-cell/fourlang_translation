from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

STEP_VERSION = "14E1_V1"

AUTO_AUDIT_SIZE = 500

SEED = 2026

RISK_REVIEW_LABEL = "RISK_REVIEW"

AUTO_AUDIT_LABEL = "AUTO_ACCEPT_AUDIT"


# ============================================================
# Args
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14E-1 - Prepare ZH-EN "
            "Qwen3-8B quality-review input."
        )
    )

    parser.add_argument(
        "--auto_audit_size",
        type=int,
        default=AUTO_AUDIT_SIZE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Stable audit hash
#
# We use deterministic SHA256 ordering rather than depending
# entirely on pandas sampling implementation.
# ============================================================


def stable_audit_hash(
    pair_id: str,
    seed: int,
) -> str:

    value = (
        f"{seed}\n"
        f"{pair_id}"
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Proportional allocation
#
# Example:
#
# AUTO_ACCEPT
# ALT       15046
# Tatoeba   47770
#
# Audit 500
#
# roughly:
# ALT       ~120
# Tatoeba   ~380
#
# Exact values are computed from actual input.
# ============================================================


def allocate_stratified_sample_sizes(
    df: pd.DataFrame,
    total_sample_size: int,
    stratum_column: str,
) -> dict[str, int]:

    if total_sample_size <= 0:

        raise ValueError(
            "total_sample_size must be > 0"
        )

    if total_sample_size > len(df):

        raise ValueError(
            "\nRequested audit sample is larger "
            "than AUTO_ACCEPT pool.\n"
            f"Requested: {total_sample_size}\n"
            f"Available: {len(df)}"
        )

    counts = (
        df[
            stratum_column
        ]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    total_rows = int(
        counts.sum()
    )

    raw_allocations = {}

    floor_allocations = {}

    fractional_parts = {}

    for (
        stratum,
        count,
    ) in counts.items():

        exact = (
            total_sample_size
            *
            count
            /
            total_rows
        )

        floor_value = int(
            math.floor(
                exact
            )
        )

        raw_allocations[
            stratum
        ] = exact

        floor_allocations[
            stratum
        ] = floor_value

        fractional_parts[
            stratum
        ] = (
            exact
            -
            floor_value
        )

    allocated = sum(
        floor_allocations.values()
    )

    remaining = (
        total_sample_size
        -
        allocated
    )

    # Largest remainder method
    remainder_order = sorted(
        fractional_parts.keys(),
        key=lambda key: (
            -fractional_parts[key],
            key,
        ),
    )

    allocations = dict(
        floor_allocations
    )

    for stratum in (
        remainder_order[
            :remaining
        ]
    ):

        allocations[
            stratum
        ] += 1

    # Never allocate more than available.
    for stratum in allocations:

        allocations[
            stratum
        ] = min(
            allocations[
                stratum
            ],
            int(
                counts[
                    stratum
                ]
            ),
        )

    # In the extremely unlikely case that clipping causes
    # shortage, fill from remaining strata deterministically.
    current_total = sum(
        allocations.values()
    )

    shortage = (
        total_sample_size
        -
        current_total
    )

    if shortage > 0:

        for (
            stratum,
            count,
        ) in counts.items():

            if shortage <= 0:
                break

            available_extra = (
                int(
                    count
                )
                -
                allocations[
                    stratum
                ]
            )

            if available_extra <= 0:
                continue

            add = min(
                shortage,
                available_extra,
            )

            allocations[
                stratum
            ] += add

            shortage -= add

    if (
        sum(
            allocations.values()
        )
        !=
        total_sample_size
    ):

        raise RuntimeError(
            "\nFailed to allocate audit sample.\n"
            f"Expected: {total_sample_size}\n"
            f"Allocated: {sum(allocations.values())}"
        )

    return allocations


# ============================================================
# Deterministic stratified audit sampling
# ============================================================


def build_auto_accept_audit(
    auto_df: pd.DataFrame,
    total_sample_size: int,
    seed: int,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:

    allocations = (
        allocate_stratified_sample_sizes(
            df=auto_df,
            total_sample_size=
                total_sample_size,
            stratum_column=
                "source_dataset",
        )
    )

    sampled_parts = []

    for (
        source,
        sample_size,
    ) in allocations.items():

        part = (
            auto_df[
                auto_df[
                    "source_dataset"
                ]
                .astype(str)
                ==
                str(
                    source
                )
            ]
            .copy()
        )

        part[
            "_audit_hash"
        ] = [

            stable_audit_hash(
                pair_id=
                    pair_id,
                seed=
                    seed,
            )

            for pair_id
            in (
                part[
                    "pair_id"
                ]
                .astype(str)
            )
        ]

        part = (
            part
            .sort_values(
                [
                    "_audit_hash",
                    "pair_id",
                ]
            )
            .head(
                sample_size
            )
            .copy()
        )

        part[
            "audit_stratum"
        ] = (
            str(
                source
            )
        )

        sampled_parts.append(
            part
        )

    audit_df = (
        pd.concat(
            sampled_parts,
            ignore_index=True,
        )
    )

    audit_df = (
        audit_df
        .sort_values(
            [
                "source_dataset",
                "_audit_hash",
                "pair_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            audit_df
        )
        !=
        total_sample_size
    ):

        raise RuntimeError(
            "\nAUTO_ACCEPT audit size mismatch.\n"
            f"Expected: {total_sample_size}\n"
            f"Found: {len(audit_df)}"
        )

    return (
        audit_df,
        allocations,
    )


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_dir = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14d_risk_routing"
    )

    risk_file = (
        input_dir
        /
        "needs_qwen_v1.parquet"
    )

    auto_file = (
        input_dir
        /
        "auto_accept_v1.parquet"
    )

    routing_report_file = (
        input_dir
        /
        "risk_routing_report_v1.json"
    )

    output_dir = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14e_qwen_review"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_input_file = (
        output_dir
        /
        "qwen_review_input_v1.parquet"
    )

    review_input_csv = (
        output_dir
        /
        "qwen_review_input_v1.csv"
    )

    auto_audit_file = (
        output_dir
        /
        "auto_accept_audit_500_v1.parquet"
    )

    auto_audit_csv = (
        output_dir
        /
        "auto_accept_audit_500_v1.csv"
    )

    risk_snapshot_file = (
        output_dir
        /
        "risk_review_snapshot_v1.parquet"
    )

    composition_file = (
        output_dir
        /
        "review_composition_v1.csv"
    )

    report_file = (
        output_dir
        /
        "review_prepare_report_v1.json"
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14E-1 - PREPARE "
        "QWEN3-8B QUALITY REVIEW INPUT"
    )

    print(
        "=" * 110
    )

    print(
        "\nVersion:"
    )

    print(
        STEP_VERSION
    )

    print(
        "\nRisk input:"
    )

    print(
        risk_file
    )

    print(
        "\nAUTO_ACCEPT input:"
    )

    print(
        auto_file
    )

    # ========================================================
    # Required inputs
    # ========================================================

    for path in [

        risk_file,
        auto_file,
        routing_report_file,

    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if (
        review_input_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{review_input_file}\n\n"
            "Use --overwrite if you "
            "intentionally want to rebuild."
        )

    # ========================================================
    # Verify routing version
    # ========================================================

    with open(
        routing_report_file,
        "r",
        encoding="utf-8",
    ) as f:

        routing_report = (
            json.load(
                f
            )
        )

    routing_version = (
        routing_report
        .get(
            "routing_version",
            ""
        )
    )

    print(
        "\nRouting version:"
    )

    print(
        routing_version
    )

    if (
        "V4"
        not in
        str(
            routing_version
        )
    ):

        raise RuntimeError(
            "\nStep 14E-1 expects the frozen "
            "14D V4 routing output.\n"
            f"Found routing_version: "
            f"{routing_version}"
        )

    # ========================================================
    # Load
    # ========================================================

    risk_df = pd.read_parquet(
        risk_file
    )

    auto_df = pd.read_parquet(
        auto_file
    )

    required_columns = {

        "pair_id",

        "en",
        "zh",

        "source_dataset",

        "route",

        "risk_flags",

        "quality_score",

        "normalized_pair_hash",
    }

    for (
        name,
        df,
    ) in [

        (
            "NEEDS_QWEN",
            risk_df,
        ),

        (
            "AUTO_ACCEPT",
            auto_df,
        ),

    ]:

        missing = (
            required_columns
            -
            set(
                df.columns
            )
        )

        if missing:

            raise RuntimeError(
                f"\n{name} missing columns:\n"
                f"{sorted(missing)}"
            )

    # ========================================================
    # Basic routing integrity
    # ========================================================

    if not (
        risk_df[
            "route"
        ]
        .eq(
            "NEEDS_QWEN"
        )
        .all()
    ):

        raise RuntimeError(
            "Risk file contains non-NEEDS_QWEN rows."
        )

    if not (
        auto_df[
            "route"
        ]
        .eq(
            "AUTO_ACCEPT"
        )
        .all()
    ):

        raise RuntimeError(
            "AUTO file contains non-AUTO_ACCEPT rows."
        )

    if (
        risk_df[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate normalized pairs in risk input."
        )

    if (
        auto_df[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate normalized pairs in AUTO input."
        )

    risk_hashes = set(
        risk_df[
            "normalized_pair_hash"
        ]
        .astype(str)
    )

    auto_hashes = set(
        auto_df[
            "normalized_pair_hash"
        ]
        .astype(str)
    )

    overlap = (
        risk_hashes
        &
        auto_hashes
    )

    if overlap:

        raise RuntimeError(
            "\nRisk/AUTO partition overlap found.\n"
            f"Overlap count: {len(overlap)}"
        )

    print(
        "\nInput counts:"
    )

    print(
        "NEEDS_QWEN:",
        len(
            risk_df
        )
    )

    print(
        "AUTO_ACCEPT:",
        len(
            auto_df
        )
    )

    # ========================================================
    # Freeze complete risk set
    # ========================================================

    risk_review = (
        risk_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    risk_review[
        "review_group"
    ] = (
        RISK_REVIEW_LABEL
    )

    risk_review[
        "audit_stratum"
    ] = ""

    risk_review[
        "_audit_hash"
    ] = ""

    # ========================================================
    # AUTO_ACCEPT audit sample
    # ========================================================

    (
        auto_audit,
        audit_allocations,
    ) = (
        build_auto_accept_audit(

            auto_df=
                auto_df,

            total_sample_size=
                args
                .auto_audit_size,

            seed=
                args
                .seed,
        )
    )

    auto_audit[
        "review_group"
    ] = (
        AUTO_AUDIT_LABEL
    )

    # ========================================================
    # Combine
    # ========================================================

    review_df = (
        pd.concat(
            [
                risk_review,
                auto_audit,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    # ========================================================
    # Stable review IDs
    # ========================================================

    review_df = (
        review_df
        .sort_values(
            [
                "review_group",
                "source_dataset",
                "pair_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    review_df[
        "review_id"
    ] = [

        f"zh_en_review_{i:05d}"

        for i
        in range(
            len(
                review_df
            )
        )
    ]

    # ========================================================
    # Review metadata
    # ========================================================

    review_df[
        "review_status"
    ] = (
        "PENDING"
    )

    review_df[
        "review_label"
    ] = ""

    review_df[
        "review_reason"
    ] = ""

    review_df[
        "review_semantic_equivalent"
    ] = pd.NA

    review_df[
        "review_major_error"
    ] = pd.NA

    review_df[
        "review_minor_error"
    ] = pd.NA

    # ========================================================
    # Integrity
    # ========================================================

    expected_review_rows = (
        len(
            risk_review
        )
        +
        args
        .auto_audit_size
    )

    if (
        len(
            review_df
        )
        !=
        expected_review_rows
    ):

        raise RuntimeError(
            "\nReview input size mismatch.\n"
            f"Expected: {expected_review_rows}\n"
            f"Found: {len(review_df)}"
        )

    if (
        review_df[
            "review_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate review_id found."
        )

    if (
        review_df[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate normalized pair in review input."
        )

    actual_risk_count = int(
        (
            review_df[
                "review_group"
            ]
            ==
            RISK_REVIEW_LABEL
        )
        .sum()
    )

    actual_audit_count = int(
        (
            review_df[
                "review_group"
            ]
            ==
            AUTO_AUDIT_LABEL
        )
        .sum()
    )

    if (
        actual_risk_count
        !=
        len(
            risk_df
        )
    ):

        raise RuntimeError(
            "Not all risk rows were preserved."
        )

    if (
        actual_audit_count
        !=
        args.auto_audit_size
    ):

        raise RuntimeError(
            "AUTO audit count mismatch."
        )

    # ========================================================
    # Composition report
    # ========================================================

    composition = (
        review_df
        .groupby(
            [
                "review_group",
                "source_dataset",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    group_totals = (
        composition
        .groupby(
            "review_group"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )

    composition[
        "percent_within_group"
    ] = (
        composition[
            "count"
        ]
        /
        group_totals
        *
        100
    )

    # ========================================================
    # Save
    # ========================================================

    risk_review.to_parquet(
        risk_snapshot_file,
        index=False,
    )

    auto_audit.to_parquet(
        auto_audit_file,
        index=False,
    )

    auto_audit.to_csv(
        auto_audit_csv,
        index=False,
        encoding="utf-8-sig",
    )

    review_df.to_parquet(
        review_input_file,
        index=False,
    )

    review_df.to_csv(
        review_input_csv,
        index=False,
        encoding="utf-8-sig",
    )

    composition.to_csv(
        composition_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Distribution diagnostics
    # ========================================================

    risk_source_distribution = {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value,
        )
        in (
            risk_df[
                "source_dataset"
            ]
            .value_counts()
            .items()
        )
    }

    auto_pool_distribution = {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value,
        )
        in (
            auto_df[
                "source_dataset"
            ]
            .value_counts()
            .items()
        )
    }

    auto_audit_distribution = {

        str(
            key
        ):
            int(
                value
            )

        for (
            key,
            value,
        )
        in (
            auto_audit[
                "source_dataset"
            ]
            .value_counts()
            .items()
        )
    }

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "14E-1",

        "version":
            STEP_VERSION,

        "pipeline":
            "zh_en_exp1_v1",

        "routing_version":
            routing_version,

        "input": {

            "needs_qwen_file":
                str(
                    risk_file
                ),

            "auto_accept_file":
                str(
                    auto_file
                ),

            "needs_qwen_rows":
                int(
                    len(
                        risk_df
                    )
                ),

            "auto_accept_rows":
                int(
                    len(
                        auto_df
                    )
                ),
        },

        "review_policy": {

            "risk_review":
                (
                    "All NEEDS_QWEN rows are reviewed."
                ),

            "auto_accept_audit":
                (
                    "Deterministic proportional stratified "
                    "sample by source_dataset."
                ),

            "auto_audit_size":
                int(
                    args
                    .auto_audit_size
                ),

            "seed":
                int(
                    args
                    .seed
                ),

            "sampling_method":
                (
                    "SHA256(seed + pair_id) stable ordering "
                    "within source strata."
                ),
        },

        "audit_allocations":
            {

                str(
                    key
                ):
                    int(
                        value
                    )

                for (
                    key,
                    value,
                )
                in (
                    audit_allocations
                    .items()
                )
            },

        "source_distributions": {

            "risk_review":
                risk_source_distribution,

            "auto_accept_pool":
                auto_pool_distribution,

            "auto_accept_audit":
                auto_audit_distribution,
        },

        "final_review": {

            "risk_review_rows":
                int(
                    actual_risk_count
                ),

            "auto_audit_rows":
                int(
                    actual_audit_count
                ),

            "total_review_rows":
                int(
                    len(
                        review_df
                    )
                ),
        },

        "assertions": {

            "all_risk_rows_preserved":
                True,

            "auto_audit_exact_size":
                True,

            "risk_auto_partition_overlap":
                0,

            "review_duplicate_pair_hash":
                0,

            "review_duplicate_id":
                0,
        },

        "outputs": {

            "review_input_parquet":
                str(
                    review_input_file
                ),

            "review_input_csv":
                str(
                    review_input_csv
                ),

            "auto_audit_parquet":
                str(
                    auto_audit_file
                ),

            "auto_audit_csv":
                str(
                    auto_audit_csv
                ),

            "risk_snapshot":
                str(
                    risk_snapshot_file
                ),

            "composition":
                str(
                    composition_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "QWEN_REVIEW_INPUT_READY",
    }

    with open(
        report_file,
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
    # Console
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 14E-1 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nNEEDS_QWEN input:",
        len(
            risk_df
        )
    )

    print(
        "AUTO_ACCEPT pool:",
        len(
            auto_df
        )
    )

    print(
        "\nAUTO audit requested:",
        args.auto_audit_size
    )

    print(
        "AUTO audit seed:",
        args.seed
    )

    print(
        "\nAUTO audit allocation:"
    )

    for (
        source,
        count,
    ) in (
        audit_allocations
        .items()
    ):

        print(
            f"{source}: {count}"
        )

    print(
        "\nFinal review composition:"
    )

    print(
        composition
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nRisk review rows:",
        actual_risk_count
    )

    print(
        "AUTO audit rows:",
        actual_audit_count
    )

    print(
        "Total review rows:",
        len(
            review_df
        )
    )

    print(
        "\nReview input:"
    )

    print(
        review_input_file
    )

    print(
        "\nAUTO audit:"
    )

    print(
        auto_audit_file
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    print(
        "\nSTATUS:"
    )

    print(
        "QWEN_REVIEW_INPUT_READY"
    )


if __name__ == "__main__":

    main()