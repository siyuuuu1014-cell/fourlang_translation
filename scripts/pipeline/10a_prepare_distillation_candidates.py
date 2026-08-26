from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ============================================================
# Constants
# ============================================================

PILOT_SIZE = 20_000
PILOT_PER_DIRECTION = 10_000

VALID_DIRECTIONS = {
    "en_uz": ("en", "uz"),
    "uz_en": ("uz", "en"),
}

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 10A - Prepare EN-UZ distillation candidates."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    return parser.parse_args()


# ============================================================
# Text helpers
# ============================================================

def canonical_text(text: str) -> str:
    """
    Used ONLY for leakage detection.

    - Unicode NFKC
    - apostrophe normalization
    - whitespace normalization
    - casefold
    """
    text = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    )

    for old in [
        "’",
        "‘",
        "`",
        "ʻ",
        "ʼ",
        "ʹ",
        "՚",
        "´",
    ]:
        text = text.replace(old, "'")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    return text.strip().casefold()


def contains_cyrillic(text: str) -> bool:
    return bool(
        CYRILLIC_RE.search(
            str(text or "")
        )
    )


def word_count(text: str) -> int:
    text = str(text or "").strip()

    if not text:
        return 0

    return len(
        text.split()
    )


def get_length_bucket(
    words: int,
) -> str:

    if words <= 5:
        return "very_short"

    if words <= 10:
        return "short"

    if words <= 20:
        return "medium"

    if words <= 40:
        return "long"

    return "very_long"


# ============================================================
# Stable IDs
# ============================================================

def stable_hash(
    *parts,
    prefix="",
    length=32,
):

    payload = "|".join(
        str(x)
        for x in parts
    )

    value = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:length]

    return prefix + value


# ============================================================
# Language-pair reconstruction
# ============================================================

def get_en_uz(
    direction: str,
    source: str,
    target: str,
):

    if direction == "en_uz":
        return source, target

    if direction == "uz_en":
        return target, source

    return "", ""


# ============================================================
# Exclusion set builders
# ============================================================

def build_from_bidirectional(
    df: pd.DataFrame,
):

    required = {
        "direction",
        "source_text",
        "target_text",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    pairs = set()
    en_sentences = set()
    uz_sentences = set()

    for row in df.itertuples(
        index=False
    ):

        direction = str(
            getattr(
                row,
                "direction",
            )
        )

        source = str(
            getattr(
                row,
                "source_text",
            )
        )

        target = str(
            getattr(
                row,
                "target_text",
            )
        )

        en, uz = get_en_uz(
            direction,
            source,
            target,
        )

        en = canonical_text(en)
        uz = canonical_text(uz)

        if not en or not uz:
            continue

        pairs.add(
            (
                en,
                uz,
            )
        )

        en_sentences.add(
            en
        )

        uz_sentences.add(
            uz
        )

    return {
        "pairs": pairs,
        "en": en_sentences,
        "uz": uz_sentences,
    }


def build_from_pair_table(
    df: pd.DataFrame,
):

    if not {
        "en",
        "uz",
    }.issubset(
        df.columns
    ):

        raise RuntimeError(
            "Benchmark must contain "
            "'en' and 'uz' columns."
        )

    pairs = set()
    en_sentences = set()
    uz_sentences = set()

    for en, uz in zip(
        df["en"],
        df["uz"],
    ):

        en = canonical_text(en)
        uz = canonical_text(uz)

        if not en or not uz:
            continue

        pairs.add(
            (
                en,
                uz,
            )
        )

        en_sentences.add(
            en
        )

        uz_sentences.add(
            uz
        )

    return {
        "pairs": pairs,
        "en": en_sentences,
        "uz": uz_sentences,
    }


# ============================================================
# Leakage flags
# ============================================================

def add_leak_flags(
    df: pd.DataFrame,
    name: str,
    exclusion: dict,
):

    pair_col = f"leak_{name}_pair"
    en_col = f"leak_{name}_en"
    uz_col = f"leak_{name}_uz"
    any_col = f"leak_{name}"

    df[pair_col] = [
        (
            en,
            uz,
        )
        in exclusion["pairs"]

        for en, uz
        in zip(
            df["_en_norm"],
            df["_uz_norm"],
        )
    ]

    df[en_col] = (
        df["_en_norm"]
        .isin(
            exclusion["en"]
        )
    )

    df[uz_col] = (
        df["_uz_norm"]
        .isin(
            exclusion["uz"]
        )
    )

    df[any_col] = (
        df[
            [
                pair_col,
                en_col,
                uz_col,
            ]
        ]
        .any(
            axis=1
        )
    )

    return df


# ============================================================
# Exact proportional stratified sample
# ============================================================

def stratified_sample(
    df: pd.DataFrame,
    target_n: int,
    seed: int,
):

    strata_cols = [
        "quality_tier",
        "data_source",
        "length_bucket",
    ]

    if len(df) < target_n:
        raise RuntimeError(
            f"Not enough candidates. "
            f"Need {target_n}, "
            f"available {len(df)}."
        )

    work = df.copy()

    for col in strata_cols:
        work[col] = (
            work[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    counts = (
        work
        .groupby(
            strata_cols,
            dropna=False,
        )
        .size()
        .reset_index(
            name="available"
        )
    )

    total = int(
        counts[
            "available"
        ].sum()
    )

    counts["raw_quota"] = (
        counts["available"]
        /
        total
        *
        target_n
    )

    counts["take"] = (
        counts[
            "raw_quota"
        ]
        .apply(
            math.floor
        )
        .astype(int)
    )

    counts["fraction"] = (
        counts["raw_quota"]
        -
        counts["take"]
    )

    remaining = (
        target_n
        -
        int(
            counts[
                "take"
            ].sum()
        )
    )

    # largest remainder allocation
    while remaining > 0:

        eligible = (
            counts[
                counts[
                    "take"
                ]
                <
                counts[
                    "available"
                ]
            ]
            .sort_values(
                [
                    "fraction",
                    "available",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        if eligible.empty:
            raise RuntimeError(
                "Cannot allocate exact stratified sample."
            )

        for idx in eligible.index:

            if remaining <= 0:
                break

            counts.at[
                idx,
                "take",
            ] += 1

            remaining -= 1

    selected_parts = []

    for row in counts.itertuples(
        index=False
    ):

        take = int(
            row.take
        )

        if take <= 0:
            continue

        mask = (
            work[
                "quality_tier"
            ].eq(
                row.quality_tier
            )
            &
            work[
                "data_source"
            ].eq(
                row.data_source
            )
            &
            work[
                "length_bucket"
            ].eq(
                row.length_bucket
            )
        )

        group = (
            work[
                mask
            ]
            .copy()
        )

        # deterministic pseudo-random ordering
        group["_sample_key"] = (
            group[
                "candidate_id"
            ]
            .map(
                lambda cid: stable_hash(
                    seed,
                    cid,
                    length=64,
                )
            )
        )

        group = (
            group
            .sort_values(
                [
                    "_sample_key",
                    "candidate_id",
                ]
            )
            .head(
                take
            )
            .drop(
                columns=[
                    "_sample_key"
                ]
            )
        )

        selected_parts.append(
            group
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    if len(selected) != target_n:

        raise RuntimeError(
            f"Pilot selection error: "
            f"expected {target_n}, "
            f"got {len(selected)}."
        )

    if (
        selected[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate_id "
            "in selected pilot."
        )

    return selected


# ============================================================
# Distribution report
# ============================================================

def build_distribution(
    full_df,
    pilot_df,
):

    rows = []

    dimensions = [
        "direction",
        "quality_tier",
        "data_source",
        "length_bucket",
    ]

    for scope, df in [
        (
            "FULL_READY",
            full_df,
        ),
        (
            "PILOT",
            pilot_df,
        ),
    ]:

        total = len(df)

        for dimension in dimensions:

            counts = (
                df[
                    dimension
                ]
                .fillna(
                    "UNKNOWN"
                )
                .astype(str)
                .value_counts()
            )

            for value, count in counts.items():

                rows.append(
                    {
                        "scope":
                            scope,

                        "dimension":
                            dimension,

                        "value":
                            value,

                        "count":
                            int(
                                count
                            ),

                        "percent":
                            (
                                count
                                /
                                total
                                *
                                100
                            )
                            if total
                            else 0.0,
                    }
                )

    return pd.DataFrame(
        rows
    )


def count_dict(
    series,
):

    return {
        str(k): int(v)

        for k, v
        in series
        .value_counts(
            dropna=False
        )
        .items()
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # ========================================================
    # Input paths
    # ========================================================

    train_file = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "train_exp1_bidirectional_v1.parquet"
    )

    validation_file = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "validation_exp1_bidirectional_v1.parquet"
    )

    benchmark_file = (
        project_root
        / "data"
        / "benchmark"
        / "en_uz"
        / "tatoeba_en_uz_500.csv"
    )

    challenge_file = (
        project_root
        / "data"
        / "benchmark"
        / "en_uz"
        / "challenge_v1.csv"
    )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10a_candidates"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_parquet = (
        output_dir
        / "distillation_candidates_full_v1.parquet"
    )

    full_csv = (
        output_dir
        / "distillation_candidates_full_v1.csv"
    )

    pilot_parquet = (
        output_dir
        / "distillation_pilot_20k_v1.parquet"
    )

    pilot_csv = (
        output_dir
        / "distillation_pilot_20k_v1.csv"
    )

    excluded_parquet = (
        output_dir
        / "excluded_candidates_v1.parquet"
    )

    distribution_csv = (
        output_dir
        / "candidate_distribution_v1.csv"
    )

    report_file = (
        output_dir
        / "10a_report_v1.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 110)
    print("EN-UZ STUDENT PIPELINE")
    print(
        "STEP 10A - "
        "PREPARE DISTILLATION CANDIDATES V1"
    )
    print("=" * 110)

    # ========================================================
    # Input checks
    # ========================================================

    for path in [
        train_file,
        validation_file,
        benchmark_file,
        challenge_file,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                path
            )

    train = pd.read_parquet(
        train_file
    )

    validation = pd.read_parquet(
        validation_file
    )

    benchmark = pd.read_csv(
        benchmark_file
    )

    challenge = pd.read_csv(
        challenge_file
    )

    print("\nInput rows:")
    print(
        "Train      :",
        len(train)
    )
    print(
        "Validation :",
        len(validation)
    )
    print(
        "Benchmark  :",
        len(benchmark)
    )
    print(
        "Challenge  :",
        len(challenge)
    )

    # ========================================================
    # Required training columns
    # ========================================================

    required = {
        "normalized_pair_id",
        "direction",
        "source_text",
        "target_text",
    }

    missing = (
        required
        -
        set(
            train.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing train columns: "
            f"{sorted(missing)}"
        )

    # ========================================================
    # Permanent test exclusion sets
    # ========================================================

    print(
        "\nBuilding permanent leakage guards..."
    )

    validation_exclusion = (
        build_from_bidirectional(
            validation
        )
    )

    benchmark_exclusion = (
        build_from_pair_table(
            benchmark
        )
    )

    challenge_exclusion = (
        build_from_pair_table(
            challenge
        )
    )

    # ========================================================
    # Candidate dataframe
    # ========================================================

    df = pd.DataFrame(
        index=train.index
    )

    if "sample_id" in train.columns:

        df[
            "source_sample_id"
        ] = (
            train[
                "sample_id"
            ]
            .fillna("")
            .astype(str)
        )

    else:

        df[
            "source_sample_id"
        ] = [
            f"train_{i:07d}"
            for i
            in range(
                len(train)
            )
        ]

    df[
        "normalized_pair_id"
    ] = (
        train[
            "normalized_pair_id"
        ]
        .fillna("")
        .astype(str)
    )

    if (
        "split_group_id"
        in train.columns
    ):

        df[
            "split_group_id"
        ] = (
            train[
                "split_group_id"
            ]
            .fillna("")
            .astype(str)
        )

    else:

        df[
            "split_group_id"
        ] = (
            df[
                "normalized_pair_id"
            ]
        )

    df[
        "direction"
    ] = (
        train[
            "direction"
        ]
        .fillna("")
        .astype(str)
    )

    df[
        "source_text"
    ] = (
        train[
            "source_text"
        ]
        .fillna("")
        .astype(str)
    )

    df[
        "real_reference"
    ] = (
        train[
            "target_text"
        ]
        .fillna("")
        .astype(str)
    )

    # ========================================================
    # Direction metadata
    # ========================================================

    df[
        "src_lang"
    ] = (
        df[
            "direction"
        ]
        .map(
            {
                "en_uz": "en",
                "uz_en": "uz",
            }
        )
        .fillna("")
    )

    df[
        "tgt_lang"
    ] = (
        df[
            "direction"
        ]
        .map(
            {
                "en_uz": "uz",
                "uz_en": "en",
            }
        )
        .fillna("")
    )

    # ========================================================
    # Optional metadata
    # ========================================================

    if (
        "quality_tier"
        in train.columns
    ):

        df[
            "quality_tier"
        ] = (
            train[
                "quality_tier"
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
        )

    else:

        df[
            "quality_tier"
        ] = "UNKNOWN"

    if (
        "data_source"
        in train.columns
    ):

        df[
            "data_source"
        ] = (
            train[
                "data_source"
            ]
            .fillna(
                "UNKNOWN"
            )
            .astype(str)
        )

    else:

        df[
            "data_source"
        ] = "UNKNOWN"

    default_weight = (
        df[
            "quality_tier"
        ]
        .map(
            {
                "GOLD": 1.0,
                "SILVER": 0.8,
                "BRONZE": 0.5,
            }
        )
        .fillna(
            1.0
        )
    )

    if (
        "training_weight"
        in train.columns
    ):

        parsed_weight = pd.to_numeric(
            train[
                "training_weight"
            ],
            errors="coerce",
        )

        df[
            "training_weight"
        ] = (
            parsed_weight
            .fillna(
                default_weight
            )
        )

    else:

        df[
            "training_weight"
        ] = (
            default_weight
        )

    # ========================================================
    # Teacher input
    # ========================================================

    df[
        "teacher_input"
    ] = [
        (
            f"<2uz> {source}"
            if direction
            ==
            "en_uz"

            else
            f"<2en> {source}"
            if direction
            ==
            "uz_en"

            else ""
        )

        for direction, source
        in zip(
            df[
                "direction"
            ],
            df[
                "source_text"
            ],
        )
    ]

    # ========================================================
    # Stable candidate ID
    # ========================================================

    df[
        "candidate_id"
    ] = [
        stable_hash(
            pair_id,
            direction,
            prefix="kd_",
        )

        for pair_id, direction
        in zip(
            df[
                "normalized_pair_id"
            ],
            df[
                "direction"
            ],
        )
    ]

    # ========================================================
    # Reconstruct EN / UZ
    # ========================================================

    en_values = []
    uz_values = []

    for (
        direction,
        source,
        target,
    ) in zip(
        df[
            "direction"
        ],
        df[
            "source_text"
        ],
        df[
            "real_reference"
        ],
    ):

        en, uz = get_en_uz(
            direction,
            source,
            target,
        )

        en_values.append(
            en
        )

        uz_values.append(
            uz
        )

    df["_en"] = en_values
    df["_uz"] = uz_values

    df[
        "_en_norm"
    ] = (
        df[
            "_en"
        ]
        .map(
            canonical_text
        )
    )

    df[
        "_uz_norm"
    ] = (
        df[
            "_uz"
        ]
        .map(
            canonical_text
        )
    )

    df[
        "pair_fingerprint"
    ] = [
        stable_hash(
            en,
            uz,
            prefix="pair_",
        )

        for en, uz
        in zip(
            df[
                "_en_norm"
            ],
            df[
                "_uz_norm"
            ],
        )
    ]

    # ========================================================
    # Length
    # ========================================================

    df[
        "source_word_count"
    ] = (
        df[
            "source_text"
        ]
        .map(
            word_count
        )
    )

    df[
        "target_word_count"
    ] = (
        df[
            "real_reference"
        ]
        .map(
            word_count
        )
    )

    df[
        "source_char_count"
    ] = (
        df[
            "source_text"
        ]
        .str.len()
    )

    df[
        "target_char_count"
    ] = (
        df[
            "real_reference"
        ]
        .str.len()
    )

    df[
        "length_bucket"
    ] = (
        df[
            "source_word_count"
        ]
        .map(
            get_length_bucket
        )
    )

    # ========================================================
    # Leakage checks
    # ========================================================

    df = add_leak_flags(
        df,
        "validation",
        validation_exclusion,
    )

    df = add_leak_flags(
        df,
        "benchmark",
        benchmark_exclusion,
    )

    df = add_leak_flags(
        df,
        "challenge",
        challenge_exclusion,
    )

    # ========================================================
    # Validity checks
    # ========================================================

    df[
        "empty_source"
    ] = (
        df[
            "source_text"
        ]
        .str.strip()
        .eq("")
    )

    df[
        "empty_reference"
    ] = (
        df[
            "real_reference"
        ]
        .str.strip()
        .eq("")
    )

    df[
        "missing_pair_id"
    ] = (
        df[
            "normalized_pair_id"
        ]
        .str.strip()
        .eq("")
    )

    df[
        "unsupported_direction"
    ] = ~(
        df[
            "direction"
        ]
        .isin(
            VALID_DIRECTIONS
        )
    )

    df[
        "cyrillic_uzbek"
    ] = (
        df[
            "_uz"
        ]
        .map(
            contains_cyrillic
        )
    )

    df[
        "duplicate_candidate"
    ] = (
        df[
            "candidate_id"
        ]
        .duplicated(
            keep="first"
        )
    )

    # ========================================================
    # Status
    # ========================================================

    status = []
    reasons = []

    for row in df.itertuples(
        index=False
    ):

        row_reasons = []

        hard_flags = [
            "empty_source",
            "empty_reference",
            "missing_pair_id",
            "unsupported_direction",
            "duplicate_candidate",
            "leak_validation",
            "leak_benchmark",
            "leak_challenge",
        ]

        for flag in hard_flags:

            if bool(
                getattr(
                    row,
                    flag,
                )
            ):

                row_reasons.append(
                    flag.upper()
                )

        if row_reasons:

            row_status = (
                "EXCLUDED"
            )

        elif bool(
            row.cyrillic_uzbek
        ):

            row_status = (
                "REVIEW"
            )

            row_reasons.append(
                "CYRILLIC_UZBEK"
            )

        else:

            row_status = (
                "READY"
            )

        status.append(
            row_status
        )

        reasons.append(
            ";".join(
                row_reasons
            )
        )

    df[
        "candidate_status"
    ] = status

    df[
        "exclusion_reason"
    ] = reasons

    df[
        "pilot_selected"
    ] = False

    # ========================================================
    # Audit
    # ========================================================

    print("\n")
    print("=" * 110)
    print("LEAKAGE / VALIDITY AUDIT")
    print("=" * 110)

    for name in [
        "validation",
        "benchmark",
        "challenge",
    ]:

        print(
            f"{name:10s} "
            f"any={int(df[f'leak_{name}'].sum())} "
            f"pair={int(df[f'leak_{name}_pair'].sum())} "
            f"EN={int(df[f'leak_{name}_en'].sum())} "
            f"UZ={int(df[f'leak_{name}_uz'].sum())}"
        )

    print(
        "Empty source       :",
        int(
            df[
                "empty_source"
            ].sum()
        )
    )

    print(
        "Empty reference    :",
        int(
            df[
                "empty_reference"
            ].sum()
        )
    )

    print(
        "Cyrillic Uzbek     :",
        int(
            df[
                "cyrillic_uzbek"
            ].sum()
        )
    )

    print(
        "Duplicate candidate:",
        int(
            df[
                "duplicate_candidate"
            ].sum()
        )
    )

    # ========================================================
    # Full READY pool
    # ========================================================

    full_df = (
        df[
            df[
                "candidate_status"
            ]
            ==
            "READY"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    excluded_df = (
        df[
            df[
                "candidate_status"
            ]
            !=
            "READY"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Strict safety verification
    # ========================================================

    if (
        full_df[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate IDs remain in READY pool."
        )

    for name in [
        "validation",
        "benchmark",
        "challenge",
    ]:

        if (
            full_df[
                f"leak_{name}"
            ]
            .any()
        ):

            raise RuntimeError(
                f"{name} leakage remains."
            )

    if (
        full_df[
            "cyrillic_uzbek"
        ]
        .any()
    ):

        raise RuntimeError(
            "Cyrillic Uzbek remains "
            "in READY pool."
        )

    # ========================================================
    # 20K Pilot
    # ========================================================

    en_uz_pool = (
        full_df[
            full_df[
                "direction"
            ]
            ==
            "en_uz"
        ]
        .copy()
    )

    uz_en_pool = (
        full_df[
            full_df[
                "direction"
            ]
            ==
            "uz_en"
        ]
        .copy()
    )

    print(
        "\nSelecting "
        "10,000 EN->UZ..."
    )

    pilot_en_uz = (
        stratified_sample(
            en_uz_pool,
            PILOT_PER_DIRECTION,
            args.seed,
        )
    )

    print(
        "Selecting "
        "10,000 UZ->EN..."
    )

    pilot_uz_en = (
        stratified_sample(
            uz_en_pool,
            PILOT_PER_DIRECTION,
            args.seed + 1,
        )
    )

    pilot_df = (
        pd.concat(
            [
                pilot_en_uz,
                pilot_uz_en,
            ],
            ignore_index=True,
        )
        .sort_values(
            [
                "direction",
                "candidate_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pilot_df[
        "pilot_selected"
    ] = True

    selected_ids = set(
        pilot_df[
            "candidate_id"
        ]
    )

    full_df[
        "pilot_selected"
    ] = (
        full_df[
            "candidate_id"
        ]
        .isin(
            selected_ids
        )
    )

    # ========================================================
    # Final pilot verification
    # ========================================================

    if len(
        pilot_df
    ) != PILOT_SIZE:

        raise RuntimeError(
            "Pilot size is not 20,000."
        )

    counts = count_dict(
        pilot_df[
            "direction"
        ]
    )

    if counts != {
        "en_uz": 10_000,
        "uz_en": 10_000,
    }:

        raise RuntimeError(
            f"Direction imbalance: {counts}"
        )

    if (
        pilot_df[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate IDs in pilot."
        )

    for name in [
        "validation",
        "benchmark",
        "challenge",
    ]:

        if (
            pilot_df[
                f"leak_{name}"
            ]
            .any()
        ):

            raise RuntimeError(
                f"{name} leakage in pilot."
            )

    if (
        pilot_df[
            "cyrillic_uzbek"
        ]
        .any()
    ):

        raise RuntimeError(
            "Cyrillic Uzbek in pilot."
        )

    # ========================================================
    # Public output columns
    # ========================================================

    output_columns = [
        "candidate_id",
        "source_sample_id",
        "normalized_pair_id",
        "split_group_id",
        "pair_fingerprint",

        "direction",
        "src_lang",
        "tgt_lang",

        "source_text",
        "real_reference",
        "teacher_input",

        "quality_tier",
        "training_weight",
        "data_source",

        "source_word_count",
        "target_word_count",
        "source_char_count",
        "target_char_count",
        "length_bucket",

        "pilot_selected",

        "leak_validation",
        "leak_validation_pair",
        "leak_validation_en",
        "leak_validation_uz",

        "leak_benchmark",
        "leak_benchmark_pair",
        "leak_benchmark_en",
        "leak_benchmark_uz",

        "leak_challenge",
        "leak_challenge_pair",
        "leak_challenge_en",
        "leak_challenge_uz",

        "cyrillic_uzbek",

        "candidate_status",
        "exclusion_reason",
    ]

    full_out = (
        full_df[
            output_columns
        ]
        .copy()
    )

    pilot_out = (
        pilot_df[
            output_columns
        ]
        .copy()
    )

    excluded_out = (
        excluded_df[
            output_columns
        ]
        .copy()
    )

    # ========================================================
    # Distribution
    # ========================================================

    distribution = (
        build_distribution(
            full_out,
            pilot_out,
        )
    )

    # ========================================================
    # Save
    # ========================================================

    print(
        "\nSaving outputs..."
    )

    full_out.to_parquet(
        full_parquet,
        index=False,
    )

    full_out.to_csv(
        full_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pilot_out.to_parquet(
        pilot_parquet,
        index=False,
    )

    pilot_out.to_csv(
        pilot_csv,
        index=False,
        encoding="utf-8-sig",
    )

    excluded_out.to_parquet(
        excluded_parquet,
        index=False,
    )

    distribution.to_csv(
        distribution_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Final status
    # ========================================================

    pilot_leakage = {
        name:
            int(
                pilot_out[
                    f"leak_{name}"
                ].sum()
            )

        for name in [
            "validation",
            "benchmark",
            "challenge",
        ]
    }

    ready = (
        len(
            pilot_out
        )
        ==
        20_000

        and
        count_dict(
            pilot_out[
                "direction"
            ]
        )
        ==
        {
            "en_uz": 10_000,
            "uz_en": 10_000,
        }

        and
        all(
            value == 0
            for value
            in pilot_leakage.values()
        )

        and
        int(
            pilot_out[
                "cyrillic_uzbek"
            ].sum()
        )
        ==
        0

        and
        not pilot_out[
            "candidate_id"
        ].duplicated().any()
    )

    final_status = (
        "READY_FOR_MADLAD"
        if ready
        else
        "NOT_READY"
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "10A",

        "version":
            "v1",

        "seed":
            args.seed,

        "inputs": {

            "train":
                str(
                    train_file
                ),

            "validation":
                str(
                    validation_file
                ),

            "benchmark":
                str(
                    benchmark_file
                ),

            "challenge":
                str(
                    challenge_file
                ),

            "train_rows":
                len(
                    train
                ),

            "validation_rows":
                len(
                    validation
                ),

            "benchmark_pairs":
                len(
                    benchmark
                ),

            "challenge_pairs":
                len(
                    challenge
                ),
        },

        "raw_leakage": {

            name: {

                "any":
                    int(
                        df[
                            f"leak_{name}"
                        ].sum()
                    ),

                "pair":
                    int(
                        df[
                            f"leak_{name}_pair"
                        ].sum()
                    ),

                "english":
                    int(
                        df[
                            f"leak_{name}_en"
                        ].sum()
                    ),

                "uzbek":
                    int(
                        df[
                            f"leak_{name}_uz"
                        ].sum()
                    ),
            }

            for name
            in [
                "validation",
                "benchmark",
                "challenge",
            ]
        },

        "candidate_status": (
            count_dict(
                df[
                    "candidate_status"
                ]
            )
        ),

        "full_ready": {

            "total":
                len(
                    full_out
                ),

            "direction":
                count_dict(
                    full_out[
                        "direction"
                    ]
                ),

            "quality":
                count_dict(
                    full_out[
                        "quality_tier"
                    ]
                ),

            "data_source":
                count_dict(
                    full_out[
                        "data_source"
                    ]
                ),

            "length_bucket":
                count_dict(
                    full_out[
                        "length_bucket"
                    ]
                ),
        },

        "pilot": {

            "total":
                len(
                    pilot_out
                ),

            "direction":
                count_dict(
                    pilot_out[
                        "direction"
                    ]
                ),

            "quality":
                count_dict(
                    pilot_out[
                        "quality_tier"
                    ]
                ),

            "data_source":
                count_dict(
                    pilot_out[
                        "data_source"
                    ]
                ),

            "length_bucket":
                count_dict(
                    pilot_out[
                        "length_bucket"
                    ]
                ),

            "leakage":
                pilot_leakage,

            "cyrillic":
                int(
                    pilot_out[
                        "cyrillic_uzbek"
                    ].sum()
                ),
        },

        "status":
            final_status,
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
    # Console summary
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STEP 10A COMPLETE")
    print("=" * 110)

    print(
        "\nInput training samples:",
        len(
            train
        )
    )

    print(
        "\nRaw leakage hits "
        "(automatically excluded):"
    )

    print(
        "Validation:",
        int(
            df[
                "leak_validation"
            ].sum()
        )
    )

    print(
        "Benchmark :",
        int(
            df[
                "leak_benchmark"
            ].sum()
        )
    )

    print(
        "Challenge :",
        int(
            df[
                "leak_challenge"
            ].sum()
        )
    )

    print(
        "\nCyrillic Uzbek:",
        int(
            df[
                "cyrillic_uzbek"
            ].sum()
        )
    )

    print(
        "Duplicate candidates:",
        int(
            df[
                "duplicate_candidate"
            ].sum()
        )
    )

    print("\n")
    print("=" * 110)
    print("FULL READY CANDIDATE POOL")
    print("=" * 110)

    print(
        "Total:",
        len(
            full_out
        )
    )

    print(
        "\nDirection:"
    )

    print(
        full_out[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nQuality:"
    )

    print(
        full_out[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nData source:"
    )

    print(
        full_out[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLength:"
    )

    print(
        full_out[
            "length_bucket"
        ]
        .value_counts()
        .to_string()
    )

    print("\n")
    print("=" * 110)
    print("DISTILLATION PILOT V1")
    print("=" * 110)

    print(
        "Total:",
        len(
            pilot_out
        )
    )

    print(
        "\nDirection:"
    )

    print(
        pilot_out[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nQuality:"
    )

    print(
        pilot_out[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nData source:"
    )

    print(
        pilot_out[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLength:"
    )

    print(
        pilot_out[
            "length_bucket"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLeakage:"
    )

    print(
        "Validation:",
        int(
            pilot_out[
                "leak_validation"
            ].sum()
        )
    )

    print(
        "Benchmark :",
        int(
            pilot_out[
                "leak_benchmark"
            ].sum()
        )
    )

    print(
        "Challenge :",
        int(
            pilot_out[
                "leak_challenge"
            ].sum()
        )
    )

    print(
        "Cyrillic  :",
        int(
            pilot_out[
                "cyrillic_uzbek"
            ].sum()
        )
    )

    print(
        "Duplicate IDs:",
        int(
            pilot_out[
                "candidate_id"
            ]
            .duplicated()
            .sum()
        )
    )

    print("\nSTATUS:", final_status)

    print(
        "\nFull pool:"
    )

    print(
        full_parquet
    )

    print(
        "\nPilot:"
    )

    print(
        pilot_parquet
    )

    print(
        "\nExcluded:"
    )

    print(
        excluded_parquet
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    if not ready:

        raise RuntimeError(
            "STEP 10A is NOT_READY. "
            "Do not run STEP 10B."
        )


if __name__ == "__main__":
    main()