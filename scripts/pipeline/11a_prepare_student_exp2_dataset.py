from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


SEED = 42


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 11A - Prepare Student Exp2 "
            "Human Replay + Clean Teacher KD dataset."
        )
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--include_teacher_reference_matches",
        action="store_true",
        help=(
            "Include KD rows where teacher prediction "
            "is identical to human reference. "
            "Default: exclude them for Exp2 V1."
        ),
    )

    return parser.parse_args()


def sha1_text(text: str) -> str:

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


def detect_reference_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "real_reference",
        "target_text",
        "target",
        "reference",
    ]

    for col in candidates:

        if col in df.columns:
            return col

    raise RuntimeError(
        "Cannot find human reference column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_direction(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def make_sample_id(
    origin: str,
    direction: str,
    source: str,
    target: str,
) -> str:

    raw = (
        f"{origin}\u241f"
        f"{direction}\u241f"
        f"{source}\u241f"
        f"{target}"
    )

    return sha1_text(raw)


def main():

    args = parse_args()

    # ============================================================
    # Project paths
    # ============================================================

    if args.project_root:

        project_root = Path(
            args.project_root
        ).resolve()

    else:

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

    human_path = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10a_candidates"
        / "distillation_candidates_full_v1.parquet"
    )

    kd_path = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10d_distillation_dataset"
        / "distillation_teacher_targets_v1.parquet"
    )

    output_dir = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "11a_exp2_training"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    human_output = (
        output_dir
        / "exp2_human_replay_v1.parquet"
    )

    kd_output = (
        output_dir
        / "exp2_teacher_kd_v1.parquet"
    )

    combined_output = (
        output_dir
        / "exp2_train_combined_v1.parquet"
    )

    combined_csv = (
        output_dir
        / "exp2_train_combined_v1.csv"
    )

    report_path = (
        output_dir
        / "11a_report_v1.json"
    )

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 11A - PREPARE STUDENT EXP2 TRAINING DATA")
    print("=" * 100)

    print("\nHuman replay:")
    print(human_path)

    print("\nTeacher KD:")
    print(kd_path)

    if not human_path.exists():

        raise FileNotFoundError(
            human_path
        )

    if not kd_path.exists():

        raise FileNotFoundError(
            kd_path
        )

    # ============================================================
    # HUMAN REPLAY
    # ============================================================

    human_raw = pd.read_parquet(
        human_path
    ).copy()

    print(
        "\nHuman source rows:",
        len(human_raw),
    )

    required_human = [
        "direction",
        "source_text",
    ]

    missing = [
        col
        for col in required_human
        if col not in human_raw.columns
    ]

    if missing:

        raise RuntimeError(
            f"Human dataset missing columns: {missing}"
        )

    reference_col = (
        detect_reference_column(
            human_raw
        )
    )

    print(
        "Human target column:",
        reference_col,
    )

    human = pd.DataFrame()

    human["direction"] = (
        normalize_direction(
            human_raw[
                "direction"
            ]
        )
    )

    human["source_text"] = (
        human_raw[
            "source_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    human["target_text"] = (
        human_raw[
            reference_col
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    human["sample_origin"] = (
        "HUMAN_REPLAY"
    )

    human["target_origin"] = (
        "HUMAN_REFERENCE"
    )

    # Preserve original data quality tier.

    if "quality_tier" in human_raw.columns:

        human["quality_tier"] = (
            human_raw[
                "quality_tier"
            ]
        )

    else:

        human["quality_tier"] = (
            "UNKNOWN"
        )

    # Preserve original weighting if available.

    if "training_weight" in human_raw.columns:

        human["sample_weight"] = (
            pd.to_numeric(
                human_raw[
                    "training_weight"
                ],
                errors="coerce",
            )
            .fillna(1.0)
            .astype(float)
        )

    else:

        human["sample_weight"] = (
            1.0
        )

    if "data_source" in human_raw.columns:

        human["data_source"] = (
            human_raw[
                "data_source"
            ]
        )

    else:

        human["data_source"] = (
            "UNKNOWN"
        )

    for col in [
        "candidate_id",
        "normalized_pair_id",
        "split_group_id",
        "pair_fingerprint",
    ]:

        if col in human_raw.columns:

            human[col] = (
                human_raw[col]
            )

    human[
        "teacher_usefulness"
    ] = pd.NA

    human[
        "teacher_confidence"
    ] = pd.NA

    human[
        "teacher_equals_reference"
    ] = False

    # ============================================================
    # HUMAN VALIDATION
    # ============================================================

    human_empty_source = int(
        human[
            "source_text"
        ]
        .eq("")
        .sum()
    )

    human_empty_target = int(
        human[
            "target_text"
        ]
        .eq("")
        .sum()
    )

    human_invalid_direction = int(
        ~human[
            "direction"
        ]
        .isin(
            {
                "en_uz",
                "uz_en",
            }
        )
    ).sum()

    if human_empty_source > 0:

        raise RuntimeError(
            f"Human empty sources: "
            f"{human_empty_source}"
        )

    if human_empty_target > 0:

        raise RuntimeError(
            f"Human empty targets: "
            f"{human_empty_target}"
        )

    if human_invalid_direction > 0:

        raise RuntimeError(
            f"Human invalid direction rows: "
            f"{human_invalid_direction}"
        )

    # ============================================================
    # TEACHER KD
    # ============================================================

    kd_raw = pd.read_parquet(
        kd_path
    ).copy()

    print(
        "\n10D KD rows:",
        len(kd_raw),
    )

    required_kd = [
        "candidate_id",
        "direction",
        "source_text",
        "distillation_target",
        "distillation_weight",
        "teacher_label",
        "teacher_usefulness",
        "teacher_confidence",
        "teacher_equals_reference",
    ]

    missing = [
        col
        for col in required_kd
        if col not in kd_raw.columns
    ]

    if missing:

        raise RuntimeError(
            f"KD dataset missing columns: {missing}"
        )

    # 10D should already guarantee these.
    # We verify again here.

    if not (
        kd_raw[
            "teacher_label"
        ]
        .eq("PASS")
        .all()
    ):

        raise RuntimeError(
            "KD contains non-PASS rows."
        )

    if not (
        kd_raw[
            "teacher_usefulness"
        ]
        .isin(
            {
                "HIGH",
                "MEDIUM",
            }
        )
        .all()
    ):

        raise RuntimeError(
            "KD contains invalid usefulness."
        )

    kd_reference_matches = int(
        kd_raw[
            "teacher_equals_reference"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    print(
        "Teacher == reference:",
        kd_reference_matches,
    )

    # ============================================================
    # Exp2 V1 policy:
    # teacher must provide an alternate target.
    # ============================================================

    if (
        args
        .include_teacher_reference_matches
    ):

        kd_selected = (
            kd_raw.copy()
        )

        kd_policy = (
            "ALL_CLEAN_TEACHER"
        )

    else:

        kd_selected = (
            kd_raw[
                ~kd_raw[
                    "teacher_equals_reference"
                ]
                .fillna(False)
                .astype(bool)
            ]
            .copy()
        )

        kd_policy = (
            "ALTERNATE_TEACHER_ONLY"
        )

    print(
        "Selected KD rows:",
        len(kd_selected),
    )

    kd = pd.DataFrame()

    kd["direction"] = (
        normalize_direction(
            kd_selected[
                "direction"
            ]
        )
    )

    kd["source_text"] = (
        kd_selected[
            "source_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    kd["target_text"] = (
        kd_selected[
            "distillation_target"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    kd["sample_origin"] = (
        "TEACHER_KD"
    )

    kd["target_origin"] = (
        "MADLAD_QWEN_CLEAN"
    )

    kd["quality_tier"] = (
        "TEACHER_CLEAN"
    )

    kd["sample_weight"] = (
        pd.to_numeric(
            kd_selected[
                "distillation_weight"
            ],
            errors="coerce",
        )
        .fillna(1.0)
        .astype(float)
    )

    if "data_source" in kd_selected.columns:

        kd["data_source"] = (
            kd_selected[
                "data_source"
            ]
        )

    else:

        kd["data_source"] = (
            "MADLAD"
        )

    kd[
        "candidate_id"
    ] = kd_selected[
        "candidate_id"
    ]

    for col in [
        "normalized_pair_id",
        "split_group_id",
        "pair_fingerprint",
    ]:

        if col in kd_selected.columns:

            kd[col] = (
                kd_selected[col]
            )

    kd[
        "teacher_usefulness"
    ] = kd_selected[
        "teacher_usefulness"
    ]

    kd[
        "teacher_confidence"
    ] = kd_selected[
        "teacher_confidence"
    ]

    kd[
        "teacher_equals_reference"
    ] = kd_selected[
        "teacher_equals_reference"
    ].astype(bool)

    # ============================================================
    # KD VALIDATION
    # ============================================================

    kd_empty_source = int(
        kd[
            "source_text"
        ]
        .eq("")
        .sum()
    )

    kd_empty_target = int(
        kd[
            "target_text"
        ]
        .eq("")
        .sum()
    )

    kd_invalid_direction = int(
        (
            ~kd[
                "direction"
            ]
            .isin(
                {
                    "en_uz",
                    "uz_en",
                }
            )
        )
        .sum()
    )

    if kd_empty_source:

        raise RuntimeError(
            f"KD empty source: "
            f"{kd_empty_source}"
        )

    if kd_empty_target:

        raise RuntimeError(
            f"KD empty target: "
            f"{kd_empty_target}"
        )

    if kd_invalid_direction:

        raise RuntimeError(
            f"KD invalid direction: "
            f"{kd_invalid_direction}"
        )

    # ============================================================
    # Align schemas
    # ============================================================

    preferred_columns = [
        "direction",
        "source_text",
        "target_text",
        "sample_origin",
        "target_origin",
        "sample_weight",
        "quality_tier",
        "data_source",
        "candidate_id",
        "normalized_pair_id",
        "split_group_id",
        "pair_fingerprint",
        "teacher_usefulness",
        "teacher_confidence",
        "teacher_equals_reference",
    ]

    for col in preferred_columns:

        if col not in human.columns:
            human[col] = pd.NA

        if col not in kd.columns:
            kd[col] = pd.NA

    human = human[
        preferred_columns
    ].copy()

    kd = kd[
        preferred_columns
    ].copy()

    # ============================================================
    # Stable sample IDs
    # ============================================================

    human["exp2_sample_id"] = [
        make_sample_id(
            origin=row.sample_origin,
            direction=row.direction,
            source=row.source_text,
            target=row.target_text,
        )

        for row
        in human.itertuples(
            index=False
        )
    ]

    kd["exp2_sample_id"] = [
        make_sample_id(
            origin=row.sample_origin,
            direction=row.direction,
            source=row.source_text,
            target=row.target_text,
        )

        for row
        in kd.itertuples(
            index=False
        )
    ]

    # ============================================================
    # Combine
    # ============================================================

    combined = pd.concat(
        [
            human,
            kd,
        ],
        ignore_index=True,
    )

    # We intentionally allow:
    #
    # same source + different target
    #
    # because this is exactly the alternate-target
    # distillation strategy.
    #
    # But exact origin/direction/source/target duplicates
    # must not exist.

    exact_duplicate_mask = (
        combined
        .duplicated(
            subset=[
                "exp2_sample_id"
            ],
            keep="first",
        )
    )

    exact_duplicates = int(
        exact_duplicate_mask.sum()
    )

    if exact_duplicates > 0:

        raise RuntimeError(
            "Unexpected exact Exp2 duplicates: "
            f"{exact_duplicates}"
        )

    # ============================================================
    # Deterministic shuffle
    # ============================================================

    combined = (
        combined
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(
            drop=True
        )
    )

    # ============================================================
    # Final statistics
    # ============================================================

    origin_counts = (
        combined[
            "sample_origin"
        ]
        .value_counts()
    )

    direction_counts = (
        combined[
            "direction"
        ]
        .value_counts()
    )

    origin_direction = (
        pd.crosstab(
            combined[
                "sample_origin"
            ],
            combined[
                "direction"
            ],
        )
    )

    kd_usefulness = (
        kd[
            "teacher_usefulness"
        ]
        .value_counts()
    )

    kd_weight = (
        kd[
            "sample_weight"
        ]
        .value_counts()
        .sort_index()
    )

    kd_fraction = (
        len(kd)
        /
        len(combined)
        if len(combined)
        else 0
    )

    assertions = {
        "human_non_empty":
            bool(
                len(human) > 0
            ),

        "kd_non_empty":
            bool(
                len(kd) > 0
            ),

        "no_empty_source":
            bool(
                ~combined[
                    "source_text"
                ]
                .eq("")
                .any()
            ),

        "no_empty_target":
            bool(
                ~combined[
                    "target_text"
                ]
                .eq("")
                .any()
            ),

        "valid_directions":
            bool(
                combined[
                    "direction"
                ]
                .isin(
                    {
                        "en_uz",
                        "uz_en",
                    }
                )
                .all()
            ),

        "no_exact_sample_duplicates":
            bool(
                ~combined[
                    "exp2_sample_id"
                ]
                .duplicated()
                .any()
            ),

        "kd_only_high_medium":
            bool(
                kd[
                    "teacher_usefulness"
                ]
                .isin(
                    {
                        "HIGH",
                        "MEDIUM",
                    }
                )
                .all()
            ),

        "kd_reference_matches_excluded":
            (
                True
                if args
                .include_teacher_reference_matches
                else bool(
                    ~kd[
                        "teacher_equals_reference"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .any()
                )
            ),
    }

    status = (
        "READY_FOR_EXP2_TRAINING"
        if all(
            assertions.values()
        )
        else
        "CHECK_REQUIRED"
    )

    # ============================================================
    # Save
    # ============================================================

    human.to_parquet(
        human_output,
        index=False,
    )

    kd.to_parquet(
        kd_output,
        index=False,
    )

    combined.to_parquet(
        combined_output,
        index=False,
    )

    combined.to_csv(
        combined_csv,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "step":
            "11A",

        "version":
            "v1",

        "seed":
            SEED,

        "human_source":
            str(human_path),

        "teacher_source":
            str(kd_path),

        "kd_policy":
            kd_policy,

        "include_teacher_reference_matches":
            bool(
                args
                .include_teacher_reference_matches
            ),

        "human_rows":
            int(
                len(human)
            ),

        "teacher_10d_rows":
            int(
                len(kd_raw)
            ),

        "teacher_reference_matches_10d":
            kd_reference_matches,

        "teacher_selected_rows":
            int(
                len(kd)
            ),

        "combined_rows":
            int(
                len(combined)
            ),

        "kd_fraction":
            float(
                kd_fraction
            ),

        "origin_distribution": {
            str(k): int(v)
            for k, v
            in origin_counts.items()
        },

        "direction_distribution": {
            str(k): int(v)
            for k, v
            in direction_counts.items()
        },

        "kd_usefulness_distribution": {
            str(k): int(v)
            for k, v
            in kd_usefulness.items()
        },

        "kd_weight_distribution": {
            str(k): int(v)
            for k, v
            in kd_weight.items()
        },

        "exact_duplicates":
            exact_duplicates,

        "assertions":
            assertions,

        "status":
            status,

        "outputs": {
            "human":
                str(human_output),

            "teacher":
                str(kd_output),

            "combined":
                str(combined_output),

            "combined_csv":
                str(combined_csv),
        },
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ============================================================
    # Console
    # ============================================================

    print("\n")
    print("=" * 100)
    print("STEP 11A RESULT")
    print("=" * 100)

    print(
        "\nHuman replay:",
        len(human),
    )

    print(
        "Teacher 10D:",
        len(kd_raw),
    )

    print(
        "Teacher == reference in 10D:",
        kd_reference_matches,
    )

    print(
        "Teacher selected:",
        len(kd),
    )

    print(
        "\nCombined:",
        len(combined),
    )

    print(
        "KD fraction:",
        f"{kd_fraction * 100:.2f}%",
    )

    print(
        "\nSample origin:"
    )

    print(
        origin_counts.to_string()
    )

    print(
        "\nDirection:"
    )

    print(
        direction_counts.to_string()
    )

    print(
        "\nOrigin × direction:"
    )

    print(
        origin_direction
    )

    print(
        "\nKD usefulness:"
    )

    print(
        kd_usefulness.to_string()
    )

    print(
        "\nKD weight:"
    )

    print(
        kd_weight.to_string()
    )

    print(
        "\nAssertions:"
    )

    for key, value in (
        assertions.items()
    ):

        print(
            f"{key:40s}: {value}"
        )

    print("\n")
    print("=" * 100)
    print(
        "STATUS:",
        status,
    )
    print("=" * 100)

    print(
        "\nCombined dataset:"
    )

    print(
        combined_output
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()