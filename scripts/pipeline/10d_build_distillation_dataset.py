from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


ERROR_COLS = [
    "omission",
    "addition",
    "mistranslation",
    "number_error",
    "time_error",
    "entity_error",
    "negation_error",
]

ALLOWED_USEFULNESS = {
    "HIGH",
    "MEDIUM",
}

USEFULNESS_WEIGHT = {
    "HIGH": 1.0,
    "MEDIUM": 0.8,
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Step 10D - Build clean EN-UZ distillation dataset V1."
    )

    parser.add_argument(
        "--project_root",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Step10C full qwen results parquet.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def normalize_text(text: str) -> str:

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalized_compare(text: str) -> str:

    text = normalize_text(text)

    return text.casefold()


def sha1_text(text: str) -> str:

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()


def contains_cyrillic(text: str) -> bool:

    if not isinstance(text, str):
        return False

    return bool(
        re.search(
            r"[\u0400-\u04FF]",
            text,
        )
    )


def load_optional_parquet(
    path: Path,
) -> pd.DataFrame | None:

    if not path.exists():
        return None

    return pd.read_parquet(path)


def collect_frozen_texts(
    project_root: Path,
):

    """
    Build frozen evaluation text sets.

    We deliberately check both individual source-side
    sentences and bilingual normalized pairs.
    """

    frozen_texts = set()
    frozen_pairs = set()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    validation_path = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "validation_exp1_bidirectional_v1.parquet"
    )

    if validation_path.exists():

        df = pd.read_parquet(
            validation_path
        )

        for _, row in df.iterrows():

            source = normalized_compare(
                row.get(
                    "source_text",
                    ""
                )
            )

            target = normalized_compare(
                row.get(
                    "target_text",
                    row.get(
                        "real_reference",
                        ""
                    )
                )
            )

            if source:
                frozen_texts.add(
                    source
                )

            if target:
                frozen_texts.add(
                    target
                )

            if source and target:
                frozen_pairs.add(
                    (
                        source,
                        target,
                    )
                )

    # --------------------------------------------------
    # Pair-level validation fallback
    # --------------------------------------------------

    possible_validation_files = [
        (
            project_root
            / "data"
            / "splits"
            / "en_uz"
            / "v1"
            / "validation_exp1_pairs_v1.parquet"
        ),
        (
            project_root
            / "data"
            / "splits"
            / "en_uz"
            / "v1"
            / "validation_exp1_v1.parquet"
        ),
    ]

    for path in possible_validation_files:

        if not path.exists():
            continue

        df = pd.read_parquet(
            path
        )

        for _, row in df.iterrows():

            en = normalized_compare(
                row.get("en", "")
            )

            uz = normalized_compare(
                row.get("uz", "")
            )

            if en:
                frozen_texts.add(en)

            if uz:
                frozen_texts.add(uz)

            if en and uz:

                frozen_pairs.add(
                    (en, uz)
                )

                frozen_pairs.add(
                    (uz, en)
                )

    # --------------------------------------------------
    # Tatoeba benchmark
    # --------------------------------------------------

    benchmark_path = (
        project_root
        / "data"
        / "benchmark"
        / "en_uz"
        / "tatoeba_en_uz_500.csv"
    )

    if benchmark_path.exists():

        df = pd.read_csv(
            benchmark_path
        )

        for _, row in df.iterrows():

            en = normalized_compare(
                row.get("en", "")
            )

            uz = normalized_compare(
                row.get("uz", "")
            )

            if en:
                frozen_texts.add(en)

            if uz:
                frozen_texts.add(uz)

            if en and uz:

                frozen_pairs.add(
                    (en, uz)
                )

                frozen_pairs.add(
                    (uz, en)
                )

    # --------------------------------------------------
    # Challenge
    # --------------------------------------------------

    challenge_path = (
        project_root
        / "data"
        / "benchmark"
        / "en_uz"
        / "challenge_v1.csv"
    )

    if challenge_path.exists():

        df = pd.read_csv(
            challenge_path
        )

        for _, row in df.iterrows():

            en = normalized_compare(
                row.get("en", "")
            )

            uz = normalized_compare(
                row.get("uz", "")
            )

            if en:
                frozen_texts.add(en)

            if uz:
                frozen_texts.add(uz)

            if en and uz:

                frozen_pairs.add(
                    (en, uz)
                )

                frozen_pairs.add(
                    (uz, en)
                )

    return frozen_texts, frozen_pairs


def ensure_columns(
    df: pd.DataFrame,
):

    required = [
        "candidate_id",
        "direction",
        "source_text",
        "real_reference",
        "teacher_prediction",
        "teacher_label",
        "teacher_usefulness",
        "teacher_confidence",
    ] + ERROR_COLS

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )


def boolify(series: pd.Series) -> pd.Series:

    if series.dtype == bool:
        return series.fillna(False)

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "y",
            }
        )
    )


def main():

    args = parse_args()

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

    if args.input:

        input_path = Path(
            args.input
        )

    else:

        input_path = (
            project_root
            / "data"
            / "distillation"
            / "en_uz"
            / "v1"
            / "10c_qwen_quality_gate"
            / "full_20k"
            / "qwen_judge_results.parquet"
        )

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        )

    else:

        output_dir = (
            project_root
            / "data"
            / "distillation"
            / "en_uz"
            / "v1"
            / "10d_distillation_dataset"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_parquet = (
        output_dir
        / "distillation_teacher_targets_v1.parquet"
    )

    output_csv = (
        output_dir
        / "distillation_teacher_targets_v1.csv"
    )

    rejected_parquet = (
        output_dir
        / "distillation_rejected_v1.parquet"
    )

    report_path = (
        output_dir
        / "10d_report_v1.json"
    )

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 10D - BUILD DISTILLATION DATASET V1")
    print("=" * 100)

    print(
        "\nInput:",
        input_path,
    )

    print(
        "Output:",
        output_dir,
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    # ==================================================
    # Load
    # ==================================================

    df = pd.read_parquet(
        input_path
    ).copy()

    ensure_columns(df)

    print(
        "\nInput rows:",
        len(df),
    )

    # Normalize judge columns

    df["teacher_label"] = (
        df["teacher_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["teacher_usefulness"] = (
        df["teacher_usefulness"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for col in ERROR_COLS:
        df[col] = boolify(
            df[col]
        )

    # ==================================================
    # Audit fields
    # ==================================================

    df["_source_norm"] = (
        df["source_text"]
        .map(normalized_compare)
    )

    df["_reference_norm"] = (
        df["real_reference"]
        .map(normalized_compare)
    )

    df["_teacher_norm"] = (
        df["teacher_prediction"]
        .map(normalized_compare)
    )

    df["teacher_is_empty"] = (
        df["_teacher_norm"].eq("")
    )

    df["source_is_empty"] = (
        df["_source_norm"].eq("")
    )

    df["reference_is_empty"] = (
        df["_reference_norm"].eq("")
    )

    df["teacher_equals_source"] = (
        df["_teacher_norm"]
        ==
        df["_source_norm"]
    )

    df["teacher_equals_reference"] = (
        df["_teacher_norm"]
        ==
        df["_reference_norm"]
    )

    df["has_any_error_flag"] = (
        df[ERROR_COLS]
        .any(axis=1)
    )

    df["teacher_has_cyrillic"] = (
        df["teacher_prediction"]
        .fillna("")
        .astype(str)
        .map(contains_cyrillic)
    )

    # ==================================================
    # Frozen leakage sets
    # ==================================================

    (
        frozen_texts,
        frozen_pairs,
    ) = collect_frozen_texts(
        project_root
    )

    print(
        "\nFrozen text entries:",
        len(frozen_texts),
    )

    print(
        "Frozen pair entries:",
        len(frozen_pairs),
    )

    df["source_leakage"] = (
        df["_source_norm"]
        .isin(frozen_texts)
    )

    df["source_teacher_pair_leakage"] = [
        (
            source,
            teacher,
        )
        in frozen_pairs

        for source, teacher
        in zip(
            df["_source_norm"],
            df["_teacher_norm"],
        )
    ]

    df["source_reference_pair_leakage"] = [
        (
            source,
            reference,
        )
        in frozen_pairs

        for source, reference
        in zip(
            df["_source_norm"],
            df["_reference_norm"],
        )
    ]

    df["any_leakage"] = (
        df[
            [
                "source_leakage",
                "source_teacher_pair_leakage",
                "source_reference_pair_leakage",
            ]
        ]
        .any(axis=1)
    )

    # ==================================================
    # Rejection reasons
    # ==================================================

    rejection_reason = pd.Series(
        "",
        index=df.index,
        dtype="object",
    )

    def reject(mask, reason):

        nonlocal rejection_reason

        target = (
            mask
            &
            rejection_reason.eq("")
        )

        rejection_reason.loc[
            target
        ] = reason

    # Order is important: first failure wins.

    reject(
        df["teacher_label"] != "PASS",
        "NOT_PASS",
    )

    reject(
        ~df[
            "teacher_usefulness"
        ].isin(ALLOWED_USEFULNESS),
        "USEFULNESS_NOT_HIGH_MEDIUM",
    )

    reject(
        df["has_any_error_flag"],
        "HAS_ERROR_FLAG",
    )

    reject(
        df["source_is_empty"],
        "EMPTY_SOURCE",
    )

    reject(
        df["reference_is_empty"],
        "EMPTY_REFERENCE",
    )

    reject(
        df["teacher_is_empty"],
        "EMPTY_TEACHER",
    )

    reject(
        df["teacher_equals_source"],
        "TEACHER_EQUALS_SOURCE",
    )

    reject(
        df["any_leakage"],
        "FROZEN_EVAL_LEAKAGE",
    )

    # EN->UZ teacher should already be normalized Latin Uzbek.
    reject(
        (
            df["direction"].eq(
                "en_uz"
            )
            &
            df[
                "teacher_has_cyrillic"
            ]
        ),
        "EN_UZ_TEACHER_HAS_CYRILLIC",
    )

    df["rejection_reason"] = (
        rejection_reason
    )

    eligible_mask = (
        df["rejection_reason"]
        .eq("")
    )

    eligible = (
        df[
            eligible_mask
        ]
        .copy()
    )

    rejected = (
        df[
            ~eligible_mask
        ]
        .copy()
    )

    print(
        "\nBefore dedup eligible:",
        len(eligible),
    )

    # ==================================================
    # Dedup clean targets
    # ==================================================

    eligible[
        "distillation_pair_key"
    ] = (
        eligible["direction"]
        .astype(str)
        + "\u241f"
        + eligible["_source_norm"]
        + "\u241f"
        + eligible["_teacher_norm"]
    )

    eligible[
        "distillation_pair_id"
    ] = (
        eligible[
            "distillation_pair_key"
        ]
        .map(sha1_text)
    )

    duplicate_mask = (
        eligible
        .duplicated(
            subset=[
                "distillation_pair_id"
            ],
            keep="first",
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    duplicate_rows = (
        eligible[
            duplicate_mask
        ]
        .copy()
    )

    if not duplicate_rows.empty:

        duplicate_rows[
            "rejection_reason"
        ] = "DUPLICATE_SOURCE_TEACHER"

        rejected = pd.concat(
            [
                rejected,
                duplicate_rows,
            ],
            ignore_index=True,
        )

    eligible = (
        eligible[
            ~duplicate_mask
        ]
        .copy()
    )

    # ==================================================
    # Distillation metadata
    # ==================================================

    eligible[
        "distillation_weight"
    ] = (
        eligible[
            "teacher_usefulness"
        ]
        .map(
            USEFULNESS_WEIGHT
        )
        .astype(float)
    )

    eligible[
        "distillation_target"
    ] = eligible[
        "teacher_prediction"
    ]

    eligible[
        "target_origin"
    ] = "MADLAD_QWEN_CLEAN"

    eligible[
        "distillation_version"
    ] = "en_uz_kd_v1"

    # Preserve exact reference matches rather than dropping them.
    # They are still correct teacher supervision, and the flag lets
    # Step11 perform ablations later.

    # ==================================================
    # Sort deterministically
    # ==================================================

    sort_cols = [
        col
        for col in [
            "direction",
            "teacher_usefulness",
            "candidate_id",
        ]
        if col in eligible.columns
    ]

    eligible = (
        eligible
        .sort_values(
            sort_cols
        )
        .reset_index(
            drop=True
        )
    )

    # ==================================================
    # Final validation
    # ==================================================

    final_assertions = {
        "all_pass":
            bool(
                eligible[
                    "teacher_label"
                ]
                .eq("PASS")
                .all()
            ),

        "all_high_medium":
            bool(
                eligible[
                    "teacher_usefulness"
                ]
                .isin(
                    ALLOWED_USEFULNESS
                )
                .all()
            ),

        "no_error_flags":
            bool(
                ~eligible[
                    ERROR_COLS
                ]
                .any(axis=1)
                .any()
            ),

        "no_empty_teacher":
            bool(
                ~eligible[
                    "teacher_is_empty"
                ]
                .any()
            ),

        "no_teacher_source_copy":
            bool(
                ~eligible[
                    "teacher_equals_source"
                ]
                .any()
            ),

        "no_frozen_leakage":
            bool(
                ~eligible[
                    "any_leakage"
                ]
                .any()
            ),

        "no_duplicate_distillation_pair":
            bool(
                ~eligible[
                    "distillation_pair_id"
                ]
                .duplicated()
                .any()
            ),

        "no_en_uz_cyrillic":
            bool(
                ~(
                    eligible[
                        "direction"
                    ].eq("en_uz")
                    &
                    eligible[
                        "teacher_has_cyrillic"
                    ]
                )
                .any()
            ),
    }

    status = (
        "READY_FOR_STUDENT_EXP2"
        if all(
            final_assertions.values()
        )
        else
        "CHECK_REQUIRED"
    )

    # ==================================================
    # Save
    # ==================================================

    internal_cols = [
        "_source_norm",
        "_reference_norm",
        "_teacher_norm",
        "distillation_pair_key",
    ]

    output_df = (
        eligible
        .drop(
            columns=[
                col
                for col in internal_cols
                if col in eligible.columns
            ]
        )
        .copy()
    )

    rejected_output = (
        rejected
        .drop(
            columns=[
                col
                for col in internal_cols
                if col in rejected.columns
            ],
            errors="ignore",
        )
        .copy()
    )

    output_df.to_parquet(
        output_parquet,
        index=False,
    )

    output_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    rejected_output.to_parquet(
        rejected_parquet,
        index=False,
    )

    # ==================================================
    # Report
    # ==================================================

    label_distribution = (
        df[
            "teacher_label"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    usefulness_distribution = (
        df[
            "teacher_usefulness"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    final_direction = (
        output_df[
            "direction"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    final_usefulness = (
        output_df[
            "teacher_usefulness"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    final_direction_usefulness = (
        output_df
        .groupby(
            [
                "direction",
                "teacher_usefulness",
            ]
        )
        .size()
        .to_dict()
    )

    rejection_distribution = (
        rejected_output[
            "rejection_reason"
        ]
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    # JSON cannot directly serialize tuple dictionary keys.
    final_direction_usefulness_json = {
        f"{k[0]}__{k[1]}": int(v)
        for k, v
        in final_direction_usefulness.items()
    }

    report = {
        "step": "10D",
        "version": "v1",

        "input_file":
            str(input_path),

        "input_rows":
            int(len(df)),

        "label_distribution": {
            str(k): int(v)
            for k, v
            in label_distribution.items()
        },

        "usefulness_distribution": {
            str(k): int(v)
            for k, v
            in usefulness_distribution.items()
        },

        "policy": {
            "teacher_label":
                "PASS",

            "allowed_usefulness":
                sorted(
                    ALLOWED_USEFULNESS
                ),

            "error_flags_required_false":
                ERROR_COLS,

            "high_weight":
                1.0,

            "medium_weight":
                0.8,

            "exclude_teacher_equals_source":
                True,

            "exclude_frozen_eval_leakage":
                True,

            "exclude_en_uz_cyrillic":
                True,

            "teacher_equals_reference":
                "KEPT_AND_FLAGGED",
        },

        "before_dedup_eligible":
            int(
                len(eligible)
                + duplicate_count
            ),

        "duplicates_removed":
            duplicate_count,

        "final_rows":
            int(
                len(output_df)
            ),

        "final_direction_distribution": {
            str(k): int(v)
            for k, v
            in final_direction.items()
        },

        "final_usefulness_distribution": {
            str(k): int(v)
            for k, v
            in final_usefulness.items()
        },

        "final_direction_usefulness":
            final_direction_usefulness_json,

        "teacher_equals_reference_final":
            int(
                output_df[
                    "teacher_equals_reference"
                ].sum()
            ),

        "rejection_distribution": {
            str(k): int(v)
            for k, v
            in rejection_distribution.items()
        },

        "frozen_text_entries":
            int(
                len(frozen_texts)
            ),

        "frozen_pair_entries":
            int(
                len(frozen_pairs)
            ),

        "assertions":
            final_assertions,

        "status":
            status,

        "output_parquet":
            str(output_parquet),

        "output_csv":
            str(output_csv),

        "rejected_parquet":
            str(rejected_parquet),
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

    # ==================================================
    # Console report
    # ==================================================

    print("\n")
    print("=" * 100)
    print("STEP 10D RESULT")
    print("=" * 100)

    print(
        "\nInput rows:",
        len(df),
    )

    print(
        "\nFinal distillation targets:",
        len(output_df),
    )

    print(
        "\nDirection:"
    )

    print(
        output_df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nUsefulness:"
    )

    print(
        output_df[
            "teacher_usefulness"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDirection × usefulness:"
    )

    print(
        pd.crosstab(
            output_df[
                "direction"
            ],
            output_df[
                "teacher_usefulness"
            ],
        )
    )

    print(
        "\nTeacher == human reference:",
        int(
            output_df[
                "teacher_equals_reference"
            ].sum()
        ),
    )

    print(
        "\nDuplicates removed:",
        duplicate_count,
    )

    print(
        "\nRejected:"
    )

    print(
        rejected_output[
            "rejection_reason"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nAssertions:"
    )

    for key, value in (
        final_assertions.items()
    ):

        print(
            f"{key:35s}: {value}"
        )

    print("\n")
    print("=" * 100)
    print(
        "STATUS:",
        status,
    )
    print("=" * 100)

    print(
        "\nDataset:"
    )

    print(
        output_parquet
    )

    print(
        "\nCSV:"
    )

    print(
        output_csv
    )

    print(
        "\nRejected audit:"
    )

    print(
        rejected_parquet
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )


if __name__ == "__main__":
    main()