from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STEP_VERSION = "18H_V1"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18H - build ZH<->EN Exp2 training data by combining "
            "train-only Human Replay with the audited Step18G KD dataset."
        )
    )

    p.add_argument("--project_root", default=None)

    p.add_argument(
        "--human_replay_source",
        default=(
            "data/distillation/zh_en/v1/"
            "18b_opus_generation/"
            "opus_teacher_predictions_train_only_v1.parquet"
        ),
        help=(
            "Train-only directed pool from Step18B. Only its human source/target "
            "fields are used; OPUS predictions are ignored."
        ),
    )

    p.add_argument(
        "--final_kd",
        default=(
            "data/distillation/zh_en/v1/"
            "18g_final_kd/"
            "final_kd_dataset_v1.parquet"
        ),
    )

    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18h_exp2_training"
        ),
    )

    p.add_argument(
        "--keep_teacher_equals_reference",
        action="store_true",
        help=(
            "By default KD rows whose KD target exactly equals the human "
            "reference are excluded to avoid duplicate supervision."
        ),
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


def first_existing(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def clean_text(value) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .replace("\u00a0", " ")
        .split()
    ).strip()


def main():
    args = parse_args()

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else infer_project_root()
    )

    human_path = resolve_path(
        root,
        args.human_replay_source,
    )

    kd_path = resolve_path(
        root,
        args.final_kd,
    )

    output_dir = resolve_path(
        root,
        args.output_dir,
    )

    if not human_path.exists():
        raise FileNotFoundError(
            human_path
        )

    if not kd_path.exists():
        raise FileNotFoundError(
            kd_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_path = (
        output_dir
        / "exp2_train_combined_v1.parquet"
    )

    human_out = (
        output_dir
        / "exp2_human_replay_v1.parquet"
    )

    kd_out = (
        output_dir
        / "exp2_kd_selected_v1.parquet"
    )

    report_path = (
        output_dir
        / "exp2_training_report_v1.json"
    )

    outputs = [
        combined_path,
        human_out,
        kd_out,
        report_path,
    ]

    if (
        not args.overwrite
        and
        any(p.exists() for p in outputs)
    ):
        raise RuntimeError(
            "Step18H outputs already exist. Use --overwrite."
        )

    print("=" * 115)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18H - BUILD EXP2 HUMAN REPLAY + KD TRAINING DATA")
    print("=" * 115)

    print("\nHuman replay source:")
    print(human_path)

    print("\nFinal KD:")
    print(kd_path)

    human_raw = pd.read_parquet(
        human_path
    ).copy()

    kd_raw = pd.read_parquet(
        kd_path
    ).copy()

    # --------------------------------------------------------
    # Human Replay
    # --------------------------------------------------------

    human_cols = set(
        human_raw.columns
    )

    direction_col = first_existing(
        human_cols,
        [
            "direction",
        ],
    )

    source_col = first_existing(
        human_cols,
        [
            "source_text",
            "source",
            "src_text",
        ],
    )

    target_col = first_existing(
        human_cols,
        [
            "target_text",
            "human_reference",
            "reference",
            "target",
            "tgt_text",
        ],
    )

    candidate_id_col = first_existing(
        human_cols,
        [
            "kd_candidate_id",
            "sample_id",
            "pair_id",
        ],
    )

    missing_human = []

    if not direction_col:
        missing_human.append(
            "direction"
        )

    if not source_col:
        missing_human.append(
            "source_text"
        )

    if not target_col:
        missing_human.append(
            "target_text/human_reference"
        )

    if missing_human:
        raise RuntimeError(
            "Human replay source missing required fields: "
            f"{missing_human}"
        )

    human = pd.DataFrame(
        {
            "training_id": (
                [
                    f"HUMAN_{i:07d}"
                    for i in range(
                        len(human_raw)
                    )
                ]
            ),
            "direction": (
                human_raw[
                    direction_col
                ]
                .astype(str)
                .str.strip()
            ),
            "source_text": (
                human_raw[
                    source_col
                ]
                .map(
                    clean_text
                )
            ),
            "target_text": (
                human_raw[
                    target_col
                ]
                .map(
                    clean_text
                )
            ),
            "training_origin": (
                "HUMAN_REPLAY"
            ),
            "training_weight": (
                1.0
            ),
            "selected_teacher": (
                ""
            ),
            "target_origin": (
                "HUMAN_REFERENCE"
            ),
        }
    )

    if candidate_id_col:
        human[
            "source_candidate_id"
        ] = (
            human_raw[
                candidate_id_col
            ]
            .astype(str)
        )

    else:
        human[
            "source_candidate_id"
        ] = ""

    # Preserve useful metadata where available.
    for col in [
        "source_dataset",
        "quality_tier",
        "quality_score",
        "risk_flags",
        "pair_id",
        "split_group_id",
    ]:
        if col in human_raw.columns:
            human[col] = (
                human_raw[col]
                .reset_index(
                    drop=True
                )
            )

    # --------------------------------------------------------
    # KD
    # --------------------------------------------------------

    required_kd = {
        "kd_candidate_id",
        "direction",
        "source_text",
        "kd_target",
        "selected_teacher",
        "target_origin",
        "teacher_equals_human_reference",
        "approved_for_kd",
    }

    missing_kd = (
        required_kd
        -
        set(
            kd_raw.columns
        )
    )

    if missing_kd:
        raise RuntimeError(
            "Final KD missing columns: "
            f"{sorted(missing_kd)}"
        )

    kd_selected = kd_raw.loc[
        kd_raw[
            "approved_for_kd"
        ]
        .fillna(False)
        .astype(bool)
    ].copy()

    teacher_equals_reference_count = int(
        kd_selected[
            "teacher_equals_human_reference"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    if not args.keep_teacher_equals_reference:
        kd_selected = kd_selected.loc[
            ~kd_selected[
                "teacher_equals_human_reference"
            ]
            .fillna(False)
            .astype(bool)
        ].copy()

    kd = pd.DataFrame(
        {
            "training_id": (
                kd_selected[
                    "kd_candidate_id"
                ]
                .astype(str)
                .map(
                    lambda x: (
                        "KD_"
                        +
                        x
                    )
                )
            ),
            "direction": (
                kd_selected[
                    "direction"
                ]
                .astype(str)
                .str.strip()
            ),
            "source_text": (
                kd_selected[
                    "source_text"
                ]
                .map(
                    clean_text
                )
            ),
            "target_text": (
                kd_selected[
                    "kd_target"
                ]
                .map(
                    clean_text
                )
            ),
            "training_origin": (
                "TEACHER_KD"
            ),
            "training_weight": (
                1.0
            ),
            "selected_teacher": (
                kd_selected[
                    "selected_teacher"
                ]
                .astype(str)
            ),
            "target_origin": (
                kd_selected[
                    "target_origin"
                ]
                .astype(str)
            ),
            "source_candidate_id": (
                kd_selected[
                    "kd_candidate_id"
                ]
                .astype(str)
            ),
        }
    )

    for col in [
        "source_dataset",
        "quality_tier",
        "quality_score",
        "risk_flags",
        "pair_id",
        "split_group_id",
        "teacher_disagreement_score",
        "decision_source_final",
        "qwen_winner",
        "calibration_band",
    ]:
        if col in kd_selected.columns:
            kd[col] = (
                kd_selected[col]
                .reset_index(
                    drop=True
                )
            )

    # --------------------------------------------------------
    # Remove KD rows that exactly duplicate Human Replay supervision.
    #
    # IMPORTANT:
    # - Human Replay must remain unchanged so Exp2 preserves the Exp1
    #   human-data distribution.
    # - Step18G already guarantees KD-internal source-target uniqueness.
    # - teacher_equals_human_reference only detects equality with the
    #   *same candidate's* human reference. A KD target can still match
    #   another Human Replay row with the same source-target pair.
    # --------------------------------------------------------

    exact_key_cols = [
        "direction",
        "source_text",
        "target_text",
    ]

    human_internal_duplicate_rows = int(
        human[
            exact_key_cols
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    kd_internal_duplicate_rows_before_cross_filter = int(
        kd[
            exact_key_cols
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    human_exact_keys = (
        human[
            exact_key_cols
        ]
        .drop_duplicates()
        .assign(
            _human_exact_overlap=True
        )
    )

    kd = kd.merge(
        human_exact_keys,
        on=exact_key_cols,
        how="left",
        validate="many_to_one",
    )

    kd_cross_human_duplicate_mask = (
        kd[
            "_human_exact_overlap"
        ]
        .fillna(False)
        .astype(bool)
    )

    kd_cross_human_duplicate_rows = int(
        kd_cross_human_duplicate_mask.sum()
    )

    kd = kd.loc[
        ~kd_cross_human_duplicate_mask
    ].copy()

    kd = kd.drop(
        columns=[
            "_human_exact_overlap"
        ]
    )

    kd_internal_duplicate_rows_after_cross_filter = int(
        kd[
            exact_key_cols
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    print(
        "\\nExact-supervision overlap audit:"
    )

    print(
        "Human internal duplicate rows (preserved):",
        human_internal_duplicate_rows,
    )

    print(
        "KD internal duplicate rows before cross filter:",
        kd_internal_duplicate_rows_before_cross_filter,
    )

    print(
        "KD rows exactly overlapping Human Replay (excluded):",
        kd_cross_human_duplicate_rows,
    )

    print(
        "KD internal duplicate rows after cross filter:",
        kd_internal_duplicate_rows_after_cross_filter,
    )

    # --------------------------------------------------------
    # Assertions before combining
    # --------------------------------------------------------

    valid_directions = {
        "en_zh",
        "zh_en",
    }

    assertions = {
        "human_nonempty": (
            len(human)
            >
            0
        ),
        "kd_nonempty": (
            len(kd)
            >
            0
        ),
        "human_valid_directions": (
            human[
                "direction"
            ]
            .isin(
                valid_directions
            )
            .all()
        ),
        "kd_valid_directions": (
            kd[
                "direction"
            ]
            .isin(
                valid_directions
            )
            .all()
        ),
        "human_no_empty_source": (
            (
                human[
                    "source_text"
                ]
                !=
                ""
            )
            .all()
        ),
        "human_no_empty_target": (
            (
                human[
                    "target_text"
                ]
                !=
                ""
            )
            .all()
        ),
        "kd_no_empty_source": (
            (
                kd[
                    "source_text"
                ]
                !=
                ""
            )
            .all()
        ),
        "kd_no_empty_target": (
            (
                kd[
                    "target_text"
                ]
                !=
                ""
            )
            .all()
        ),
        "human_weight_all_one": (
            (
                human[
                    "training_weight"
                ]
                ==
                1.0
            )
            .all()
        ),
        "kd_weight_all_one": (
            (
                kd[
                    "training_weight"
                ]
                ==
                1.0
            )
            .all()
        ),
        "kd_internal_exact_duplicates_zero": (
            kd_internal_duplicate_rows_after_cross_filter
            ==
            0
        ),
        "kd_cross_human_exact_duplicates_removed": (
            not (
                kd[
                    exact_key_cols
                ]
                .merge(
                    human[
                        exact_key_cols
                    ]
                    .drop_duplicates(),
                    on=exact_key_cols,
                    how="inner",
                )
                .shape[
                    0
                ]
                >
                0
            )
        ),
        "kd_reference_matches_excluded": (
            (
                args.keep_teacher_equals_reference
            )
            or
            (
                len(kd)
                ==
                len(
                    kd_raw.loc[
                        kd_raw[
                            "approved_for_kd"
                        ]
                        .fillna(False)
                        .astype(bool)
                    ]
                )
                -
                teacher_equals_reference_count
            )
        ),
    }

    failed = [
        key
        for key, value
        in assertions.items()
        if not bool(value)
    ]

    if failed:
        raise RuntimeError(
            "STEP18H pre-combine assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    # Align all metadata columns.
    all_columns = list(
        dict.fromkeys(
            list(human.columns)
            +
            list(kd.columns)
        )
    )

    for col in all_columns:
        if col not in human.columns:
            human[col] = (
                ""
            )

        if col not in kd.columns:
            kd[col] = (
                ""
            )

    human = human[
        all_columns
    ].copy()

    kd = kd[
        all_columns
    ].copy()

    combined = pd.concat(
        [
            human,
            kd,
        ],
        ignore_index=True,
    )

    # Human Replay is intentionally preserved exactly as Exp1 saw it.
    # Therefore Human-internal duplicates, if any, are allowed and reported.
    # What is forbidden is:
    #   1. KD-internal exact duplicates
    #   2. exact Human-vs-KD supervision overlap
    #
    # The cross-origin overlap was already removed above.

    human_keys_after = (
        human[
            exact_key_cols
        ]
        .drop_duplicates()
    )

    kd_keys_after = (
        kd[
            exact_key_cols
        ]
        .drop_duplicates()
    )

    cross_origin_exact_duplicate_rows = int(
        human_keys_after.merge(
            kd_keys_after,
            on=exact_key_cols,
            how="inner",
        ).shape[
            0
        ]
    )

    assertions.update(
        {
            "combined_training_id_unique": (
                combined[
                    "training_id"
                ]
                .is_unique
            ),
            "combined_no_empty_source": (
                (
                    combined[
                        "source_text"
                    ]
                    !=
                    ""
                )
                .all()
            ),
            "combined_no_empty_target": (
                (
                    combined[
                        "target_text"
                    ]
                    !=
                    ""
                )
                .all()
            ),
            "combined_valid_directions": (
                combined[
                    "direction"
                ]
                .isin(
                    valid_directions
                )
                .all()
            ),
            "combined_no_cross_origin_exact_duplicates": (
                cross_origin_exact_duplicate_rows
                ==
                0
            ),
            "combined_kd_internal_exact_duplicates_zero": (
                kd_internal_duplicate_rows_after_cross_filter
                ==
                0
            ),
            "combined_weight_all_one": (
                (
                    combined[
                        "training_weight"
                    ]
                    ==
                    1.0
                )
                .all()
            ),
        }
    )

    failed = [
        key
        for key, value
        in assertions.items()
        if not bool(value)
    ]

    if failed:
        raise RuntimeError(
            "STEP18H assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    human.to_parquet(
        human_out,
        index=False,
    )

    kd.to_parquet(
        kd_out,
        index=False,
    )

    combined.to_parquet(
        combined_path,
        index=False,
    )

    human_rows = len(
        human
    )

    kd_rows = len(
        kd
    )

    total_rows = len(
        combined
    )

    kd_fraction = (
        kd_rows
        /
        total_rows
        if total_rows
        else 0.0
    )

    report = {
        "step": "18H",
        "step_version": STEP_VERSION,
        "inputs": {
            "human_replay_source": str(
                human_path
            ),
            "final_kd": str(
                kd_path
            ),
        },
        "policy": {
            "human_training_weight": (
                1.0
            ),
            "kd_training_weight": (
                1.0
            ),
            "exclude_teacher_equals_human_reference": (
                not args.keep_teacher_equals_reference
            ),
            "reason": (
                "Preserve a clean Exp1-vs-Exp2 comparison and avoid "
                "duplicating identical human and KD targets."
            ),
        },
        "counts": {
            "human_replay_rows": int(
                human_rows
            ),
            "final_kd_input_rows": int(
                len(
                    kd_raw
                )
            ),
            "teacher_equals_human_reference_rows": int(
                teacher_equals_reference_count
            ),
            "human_internal_duplicate_rows_preserved": int(
                human_internal_duplicate_rows
            ),
            "kd_internal_duplicate_rows_before_cross_filter": int(
                kd_internal_duplicate_rows_before_cross_filter
            ),
            "kd_cross_human_exact_duplicate_rows_excluded": int(
                kd_cross_human_duplicate_rows
            ),
            "kd_internal_duplicate_rows_after_cross_filter": int(
                kd_internal_duplicate_rows_after_cross_filter
            ),
            "selected_kd_rows": int(
                kd_rows
            ),
            "combined_rows": int(
                total_rows
            ),
            "kd_fraction_percent": (
                100.0
                *
                kd_fraction
            ),
            "origin": {
                str(k): int(v)
                for k, v in combined[
                    "training_origin"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "direction": {
                str(k): int(v)
                for k, v in combined[
                    "direction"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "origin_direction": {
                f"{origin}|{direction}": int(
                    count
                )
                for (
                    origin,
                    direction,
                ), count
                in combined.groupby(
                    [
                        "training_origin",
                        "direction",
                    ]
                )
                .size()
                .to_dict()
                .items()
            },
            "kd_selected_teacher": {
                str(k): int(v)
                for k, v in kd[
                    "selected_teacher"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "kd_target_origin": {
                str(k): int(v)
                for k, v in kd[
                    "target_origin"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },
        "assertions": {
            key: bool(value)
            for key, value in assertions.items()
        },
        "outputs": {
            "human_replay": str(
                human_out
            ),
            "kd_selected": str(
                kd_out
            ),
            "combined_training": str(
                combined_path
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "READY_FOR_STEP_19_EXP2_TRAINING"
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print("\n" + "=" * 115)
    print("STEP 18H RESULT")
    print("=" * 115)

    print("\nHuman Replay rows:", human_rows)
    print("Final KD input rows:", len(kd_raw))
    print(
        "Teacher == human reference:",
        teacher_equals_reference_count,
    )
    print(
        "Human internal duplicate rows preserved:",
        human_internal_duplicate_rows,
    )
    print(
        "KD rows overlapping Human Replay excluded:",
        kd_cross_human_duplicate_rows,
    )
    print("Selected KD rows:", kd_rows)
    print("Combined rows:", total_rows)
    print(
        "KD fraction:",
        f"{100.0 * kd_fraction:.2f}%",
    )

    print("\nOrigin:")
    print(
        combined[
            "training_origin"
        ]
        .value_counts()
        .to_string()
    )

    print("\nDirection:")
    print(
        combined[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print("\nOrigin × direction:")
    print(
        combined.groupby(
            [
                "training_origin",
                "direction",
            ]
        )
        .size()
        .to_string()
    )

    print("\nKD selected Teacher:")
    print(
        kd[
            "selected_teacher"
        ]
        .value_counts()
        .to_string()
    )

    print("\nKD target origin:")
    print(
        kd[
            "target_origin"
        ]
        .value_counts()
        .to_string()
    )

    print("\nAssertions:")

    for key, value in assertions.items():
        print(
            f"{key}: {bool(value)}"
        )

    print("\nCombined training:")
    print(
        combined_path
    )

    print("\nReport:")
    print(
        report_path
    )

    print("\nSTATUS:")
    print(
        "READY_FOR_STEP_19_EXP2_TRAINING"
    )


if __name__ == "__main__":
    main()
