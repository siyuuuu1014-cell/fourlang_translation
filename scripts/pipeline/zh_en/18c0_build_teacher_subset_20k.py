from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STEP_VERSION = "18C0_V1"
DEFAULT_SEED = 2026


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 18C0 - build a deterministic stratified 10k-pair / "
            "20k-directed train-only Teacher subset for ZH<->EN distillation, "
            "and reuse any already-generated MADLAD checkpoint rows."
        )
    )

    parser.add_argument("--project_root", default=None)

    parser.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/"
            "18b_opus_generation/"
            "opus_teacher_predictions_train_only_v1.parquet"
        ),
    )

    parser.add_argument(
        "--pair_target",
        type=int,
        default=10000,
        help="Number of unique bilingual pairs to sample. Directed rows = 2x.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18c0_teacher_subset"
        ),
    )

    parser.add_argument(
        "--existing_madlad_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18c_madlad_generation"
        ),
        help=(
            "Existing full-run MADLAD checkpoint directory. "
            "Matching selected IDs will be reused."
        ),
    )

    parser.add_argument(
        "--subset_madlad_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18c1_madlad_generation_20k"
        ),
        help=(
            "Output directory that will receive subset-only seeded checkpoints "
            "compatible with the existing Step18C generator."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def stable_int(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def source_length_units(
    text: str,
    source_lang: str,
) -> int:
    text = str(text).strip()

    if source_lang == "en":
        return max(
            1,
            len(
                [
                    x
                    for x in text.split()
                    if x.strip()
                ]
            ),
        )

    if source_lang == "zh":
        # Count non-whitespace characters. For Chinese this is a stable,
        # language-appropriate proxy for source length.
        return max(
            1,
            sum(
                1
                for ch in text
                if not ch.isspace()
            ),
        )

    raise ValueError(
        f"Unsupported source_lang: {source_lang}"
    )


def add_quintile_bucket(
    series: pd.Series,
    labels: list[str],
) -> pd.Series:
    # Percentile-rank based bucket avoids qcut duplicate-edge problems.
    pct = (
        series.rank(
            method="first",
            pct=True,
        )
        .clip(
            lower=1e-12,
            upper=1.0,
        )
    )

    return pd.cut(
        pct,
        bins=[
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        labels=labels,
        include_lowest=True,
    ).astype(str)


def allocate_quotas(
    counts: pd.Series,
    target: int,
) -> dict[str, int]:
    """
    Deterministic proportional allocation with:
    - at least one sample from each non-empty stratum when feasible
    - largest-remainder fill
    - never exceeds source stratum size
    """
    counts = counts.astype(int)

    total = int(counts.sum())

    if target <= 0:
        raise ValueError("target must be > 0")

    if target > total:
        raise ValueError(
            f"target={target} exceeds total={total}"
        )

    keys = list(counts.index)

    ideal = {
        key: target * int(counts[key]) / total
        for key in keys
    }

    quotas = {}

    enforce_min_one = (
        target >= len(keys)
    )

    for key in keys:
        base = int(
            math.floor(
                ideal[key]
            )
        )

        if enforce_min_one:
            base = max(
                1,
                base,
            )

        quotas[key] = min(
            int(counts[key]),
            base,
        )

    current = sum(
        quotas.values()
    )

    # If minimum-one caused overshoot, reduce from the weakest residual
    # among strata that still have quota > 1.
    while current > target:
        candidates = [
            key
            for key in keys
            if quotas[key] > 1
        ]

        if not candidates:
            raise RuntimeError(
                "Unable to reduce quotas to requested target."
            )

        key = min(
            candidates,
            key=lambda k: (
                ideal[k] - math.floor(ideal[k]),
                ideal[k],
                str(k),
            ),
        )

        quotas[key] -= 1
        current -= 1

    # Largest-remainder fill while respecting capacity.
    while current < target:
        candidates = [
            key
            for key in keys
            if quotas[key] < int(counts[key])
        ]

        if not candidates:
            raise RuntimeError(
                "Unable to allocate enough samples."
            )

        key = max(
            candidates,
            key=lambda k: (
                ideal[k] - quotas[k],
                int(counts[k]) - quotas[k],
                str(k),
            ),
        )

        quotas[key] += 1
        current += 1

    return quotas


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return pd.DataFrame(
        rows
    )


def write_jsonl(
    df: pd.DataFrame,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in df.to_dict(
            orient="records"
        ):
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_json(
    obj: dict,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    args = parse_args()

    project_root = (
        Path(
            args.project_root
        ).resolve()
        if args.project_root
        else infer_project_root()
    )

    input_path = resolve_path(
        project_root,
        args.input,
    )

    output_dir = resolve_path(
        project_root,
        args.output_dir,
    )

    existing_madlad_dir = resolve_path(
        project_root,
        args.existing_madlad_dir,
    )

    subset_madlad_dir = resolve_path(
        project_root,
        args.subset_madlad_dir,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_madlad_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subset_output = (
        output_dir
        /
        "teacher_candidates_20k_v1.parquet"
    )

    pair_output = (
        output_dir
        /
        "selected_teacher_pairs_10k_v1.parquet"
    )

    manifest_output = (
        output_dir
        /
        "teacher_subset_manifest_v1.csv"
    )

    report_output = (
        output_dir
        /
        "teacher_subset_report_v1.json"
    )

    seeded_manifest_output = (
        output_dir
        /
        "reused_madlad_checkpoint_manifest_v1.csv"
    )

    subset_en_zh_checkpoint = (
        subset_madlad_dir
        /
        "checkpoint_madlad_en_zh_v1.jsonl"
    )

    subset_zh_en_checkpoint = (
        subset_madlad_dir
        /
        "checkpoint_madlad_zh_en_v1.jsonl"
    )

    outputs = [
        subset_output,
        pair_output,
        manifest_output,
        report_output,
        seeded_manifest_output,
        subset_en_zh_checkpoint,
        subset_zh_en_checkpoint,
    ]

    if not args.overwrite:
        existing = [
            p
            for p in outputs
            if p.exists()
        ]

        if existing:
            raise RuntimeError(
                "Step18C0 outputs already exist:\n"
                +
                "\n".join(
                    str(p)
                    for p in existing
                )
                +
                "\nUse --overwrite to rebuild."
            )

    else:
        for p in outputs:
            if p.exists():
                p.unlink()

    print(
        "=" * 110
    )

    print(
        "ZH-EN DISTILLATION PIPELINE"
    )

    print(
        "STEP 18C0 - BUILD STRATIFIED 20K TEACHER SUBSET"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput:"
    )

    print(
        input_path
    )

    df = pd.read_parquet(
        input_path
    ).copy()

    required = {
        "kd_candidate_id",
        "pair_id",
        "sample_id",
        "direction",
        "source_dataset",
        "quality_tier",
        "source_lang",
        "source_text",
        "human_reference",
        "opus_prediction",
        "allowed_for_kd_training",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    if (
        df[
            "kd_candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate kd_candidate_id in Step18B input."
        )

    if (
        df[
            "pair_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Missing pair_id."
        )

    if not (
        df[
            "allowed_for_kd_training"
        ]
        .fillna(False)
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Input contains rows not allowed for KD."
        )

    direction_counts = (
        df[
            "direction"
        ]
        .value_counts()
        .to_dict()
    )

    if set(direction_counts) != {
        "en_zh",
        "zh_en",
    }:
        raise RuntimeError(
            f"Unexpected directions: {direction_counts}"
        )

    pair_size = (
        df.groupby(
            "pair_id"
        )
        .size()
    )

    if not (
        pair_size
        ==
        2
    ).all():
        raise RuntimeError(
            "Every pair must have exactly two directed rows."
        )

    pair_direction_n = (
        df.groupby(
            "pair_id"
        )[
            "direction"
        ]
        .nunique()
    )

    if not (
        pair_direction_n
        ==
        2
    ).all():
        raise RuntimeError(
            "Every pair must contain both en_zh and zh_en."
        )

    total_pairs = int(
        df[
            "pair_id"
        ]
        .nunique()
    )

    if args.pair_target > total_pairs:
        raise RuntimeError(
            f"pair_target={args.pair_target} > total_pairs={total_pairs}"
        )

    print(
        "\nFull train-only pool:"
    )

    print(
        "Unique pairs:",
        total_pairs,
    )

    print(
        "Directed rows:",
        len(df),
    )

    print(
        "Target pairs:",
        args.pair_target,
    )

    print(
        "Target directed rows:",
        args.pair_target * 2,
    )

    # --------------------------------------------------------
    # Pair-level table
    # --------------------------------------------------------

    pair_records = []

    for pair_id, part in df.groupby(
        "pair_id",
        sort=False,
    ):
        if len(part) != 2:
            raise RuntimeError(
                f"Pair {pair_id} does not have exactly 2 rows."
            )

        en_zh = part.loc[
            part[
                "direction"
            ]
            ==
            "en_zh"
        ]

        zh_en = part.loc[
            part[
                "direction"
            ]
            ==
            "zh_en"
        ]

        if (
            len(en_zh)
            !=
            1
            or
            len(zh_en)
            !=
            1
        ):
            raise RuntimeError(
                f"Invalid direction structure for pair {pair_id}"
            )

        en_row = en_zh.iloc[0]
        zh_row = zh_en.iloc[0]

        source_dataset_values = set(
            part[
                "source_dataset"
            ]
            .astype(str)
            .tolist()
        )

        quality_values = set(
            part[
                "quality_tier"
            ]
            .astype(str)
            .tolist()
        )

        if len(
            source_dataset_values
        ) != 1:
            raise RuntimeError(
                f"source_dataset differs within pair {pair_id}"
            )

        if len(
            quality_values
        ) != 1:
            raise RuntimeError(
                f"quality_tier differs within pair {pair_id}"
            )

        pair_records.append(
            {
                "pair_id": str(
                    pair_id
                ),
                "source_dataset": str(
                    en_row[
                        "source_dataset"
                    ]
                ),
                "quality_tier": str(
                    en_row[
                        "quality_tier"
                    ]
                ),
                "en_source_units": source_length_units(
                    en_row[
                        "source_text"
                    ],
                    "en",
                ),
                "zh_source_units": source_length_units(
                    zh_row[
                        "source_text"
                    ],
                    "zh",
                ),
            }
        )

    pairs = pd.DataFrame(
        pair_records
    )

    length_labels = [
        "Q1_shortest",
        "Q2_short",
        "Q3_medium",
        "Q4_long",
        "Q5_longest",
    ]

    pairs[
        "en_length_bucket"
    ] = add_quintile_bucket(
        pairs[
            "en_source_units"
        ],
        length_labels,
    )

    pairs[
        "zh_length_bucket"
    ] = add_quintile_bucket(
        pairs[
            "zh_source_units"
        ],
        length_labels,
    )

    pairs[
        "sampling_stratum"
    ] = (
        pairs[
            "source_dataset"
        ].astype(str)
        +
        "|"
        +
        pairs[
            "quality_tier"
        ].astype(str)
        +
        "|EN:"
        +
        pairs[
            "en_length_bucket"
        ].astype(str)
        +
        "|ZH:"
        +
        pairs[
            "zh_length_bucket"
        ].astype(str)
    )

    stratum_counts = (
        pairs[
            "sampling_stratum"
        ]
        .value_counts()
        .sort_index()
    )

    quotas = allocate_quotas(
        stratum_counts,
        args.pair_target,
    )

    selected_parts = []

    for stratum in sorted(
        quotas
    ):
        quota = int(
            quotas[
                stratum
            ]
        )

        part = (
            pairs.loc[
                pairs[
                    "sampling_stratum"
                ]
                ==
                stratum
            ]
            .copy()
        )

        random_state = (
            args.seed
            +
            stable_int(
                stratum
            )
            %
            1_000_000_000
        )

        sampled = part.sample(
            n=quota,
            replace=False,
            random_state=random_state,
        )

        selected_parts.append(
            sampled
        )

    selected_pairs = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # Deterministic final ordering for reproducibility.
    selected_pairs[
        "_stable_order"
    ] = selected_pairs[
        "pair_id"
    ].map(
        lambda x: stable_int(
            f"{args.seed}|{x}"
        )
    )

    selected_pairs = (
        selected_pairs.sort_values(
            "_stable_order"
        )
        .drop(
            columns=[
                "_stable_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(selected_pairs)
        !=
        args.pair_target
    ):
        raise RuntimeError(
            "Selected pair count mismatch."
        )

    if not (
        selected_pairs[
            "pair_id"
        ]
        .is_unique
    ):
        raise RuntimeError(
            "Duplicate pair_id in selected subset."
        )

    selected_pair_ids = set(
        selected_pairs[
            "pair_id"
        ]
        .astype(str)
        .tolist()
    )

    subset = (
        df.loc[
            df[
                "pair_id"
            ]
            .astype(str)
            .isin(
                selected_pair_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # Attach sampling metadata.
    subset = subset.merge(
        selected_pairs[
            [
                "pair_id",
                "en_source_units",
                "zh_source_units",
                "en_length_bucket",
                "zh_length_bucket",
                "sampling_stratum",
            ]
        ],
        on="pair_id",
        how="left",
        validate="many_to_one",
    )

    expected_directed = (
        args.pair_target
        *
        2
    )

    subset_direction_counts = (
        subset[
            "direction"
        ]
        .value_counts()
        .to_dict()
    )

    assertions = {
        "selected_pair_count_exact": (
            len(
                selected_pairs
            )
            ==
            args.pair_target
        ),
        "selected_pair_ids_unique": (
            selected_pairs[
                "pair_id"
            ]
            .is_unique
        ),
        "directed_row_count_exact": (
            len(
                subset
            )
            ==
            expected_directed
        ),
        "en_zh_exactly_pair_target": (
            int(
                subset_direction_counts.get(
                    "en_zh",
                    0,
                )
            )
            ==
            args.pair_target
        ),
        "zh_en_exactly_pair_target": (
            int(
                subset_direction_counts.get(
                    "zh_en",
                    0,
                )
            )
            ==
            args.pair_target
        ),
        "candidate_ids_unique": (
            subset[
                "kd_candidate_id"
            ]
            .is_unique
        ),
        "two_rows_per_pair": (
            subset.groupby(
                "pair_id"
            )
            .size()
            .eq(
                2
            )
            .all()
        ),
        "no_empty_source": (
            subset[
                "source_text"
            ]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_empty_human_reference": (
            subset[
                "human_reference"
            ]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_empty_opus_prediction": (
            subset[
                "opus_prediction"
            ]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
    }

    failed = [
        key
        for key, value
        in assertions.items()
        if not bool(
            value
        )
    ]

    if failed:
        raise RuntimeError(
            "Step18C0 assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    full_manifest = (
        pairs.groupby(
            [
                "source_dataset",
                "quality_tier",
                "en_length_bucket",
                "zh_length_bucket",
                "sampling_stratum",
            ],
            dropna=False,
        )
        .agg(
            full_pairs=(
                "pair_id",
                "size",
            )
        )
        .reset_index()
    )

    selected_manifest = (
        selected_pairs.groupby(
            [
                "source_dataset",
                "quality_tier",
                "en_length_bucket",
                "zh_length_bucket",
                "sampling_stratum",
            ],
            dropna=False,
        )
        .agg(
            selected_pairs=(
                "pair_id",
                "size",
            )
        )
        .reset_index()
    )

    manifest = (
        full_manifest.merge(
            selected_manifest,
            on=[
                "source_dataset",
                "quality_tier",
                "en_length_bucket",
                "zh_length_bucket",
                "sampling_stratum",
            ],
            how="left",
        )
        .fillna(
            {
                "selected_pairs": 0,
            }
        )
    )

    manifest[
        "selected_pairs"
    ] = manifest[
        "selected_pairs"
    ].astype(int)

    manifest[
        "selection_rate"
    ] = (
        manifest[
            "selected_pairs"
        ]
        /
        manifest[
            "full_pairs"
        ]
    )

    # --------------------------------------------------------
    # Reuse already generated MADLAD checkpoint rows
    # --------------------------------------------------------

    reuse_rows = []

    for direction in [
        "en_zh",
        "zh_en",
    ]:
        existing_checkpoint = (
            existing_madlad_dir
            /
            f"checkpoint_madlad_{direction}_v1.jsonl"
        )

        subset_checkpoint = (
            subset_madlad_dir
            /
            f"checkpoint_madlad_{direction}_v1.jsonl"
        )

        direction_ids = set(
            subset.loc[
                subset[
                    "direction"
                ]
                ==
                direction,
                "kd_candidate_id",
            ]
            .astype(str)
            .tolist()
        )

        existing = read_jsonl(
            existing_checkpoint
        )

        if len(
            existing
        ):
            if (
                existing[
                    "kd_candidate_id"
                ]
                .duplicated()
                .any()
            ):
                raise RuntimeError(
                    f"Duplicate IDs in existing checkpoint: {existing_checkpoint}"
                )

            reused = (
                existing.loc[
                    existing[
                        "kd_candidate_id"
                    ]
                    .astype(str)
                    .isin(
                        direction_ids
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

        else:
            reused = pd.DataFrame()

        if len(
            reused
        ):
            write_jsonl(
                reused,
                subset_checkpoint,
            )

        reuse_rows.append(
            {
                "direction": direction,
                "existing_checkpoint": str(
                    existing_checkpoint
                ),
                "existing_checkpoint_rows": int(
                    len(
                        existing
                    )
                ),
                "selected_direction_rows": int(
                    len(
                        direction_ids
                    )
                ),
                "reused_rows": int(
                    len(
                        reused
                    )
                ),
                "remaining_to_generate": int(
                    len(
                        direction_ids
                    )
                    -
                    len(
                        reused
                    )
                ),
                "subset_checkpoint": str(
                    subset_checkpoint
                ),
            }
        )

    reuse_manifest = pd.DataFrame(
        reuse_rows
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    selected_pairs.to_parquet(
        pair_output,
        index=False,
    )

    subset.to_parquet(
        subset_output,
        index=False,
    )

    manifest.to_csv(
        manifest_output,
        index=False,
        encoding="utf-8-sig",
    )

    reuse_manifest.to_csv(
        seeded_manifest_output,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "step": "18C0",
        "step_version": STEP_VERSION,
        "seed": int(
            args.seed
        ),
        "strategy": (
            "pair-level deterministic proportional stratified sampling; "
            "each selected pair contributes both en_zh and zh_en directions"
        ),
        "stratification": [
            "source_dataset",
            "quality_tier",
            "en_length_quintile",
            "zh_length_quintile",
        ],
        "counts": {
            "full_pairs": int(
                total_pairs
            ),
            "full_directed_rows": int(
                len(
                    df
                )
            ),
            "selected_pairs": int(
                len(
                    selected_pairs
                )
            ),
            "selected_directed_rows": int(
                len(
                    subset
                )
            ),
            "en_zh_rows": int(
                subset_direction_counts[
                    "en_zh"
                ]
            ),
            "zh_en_rows": int(
                subset_direction_counts[
                    "zh_en"
                ]
            ),
            "teacher_fraction_of_full_directed": float(
                len(
                    subset
                )
                /
                len(
                    df
                )
            ),
        },
        "full_pair_distribution": {
            "source_dataset": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in pairs[
                    "source_dataset"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "quality_tier": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in pairs[
                    "quality_tier"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },
        "selected_pair_distribution": {
            "source_dataset": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in selected_pairs[
                    "source_dataset"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "quality_tier": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in selected_pairs[
                    "quality_tier"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "en_length_bucket": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in selected_pairs[
                    "en_length_bucket"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "zh_length_bucket": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in selected_pairs[
                    "zh_length_bucket"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },
        "checkpoint_reuse": reuse_rows,
        "assertions": {
            key: bool(
                value
            )
            for key, value
            in assertions.items()
        },
        "outputs": {
            "selected_pairs": str(
                pair_output
            ),
            "teacher_subset": str(
                subset_output
            ),
            "sampling_manifest": str(
                manifest_output
            ),
            "checkpoint_reuse_manifest": str(
                seeded_manifest_output
            ),
            "subset_madlad_dir": str(
                subset_madlad_dir
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18C1_MADLAD_20K"
        ),
    }

    save_json(
        report,
        report_output,
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "STEP 18C0 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nFull pairs:",
        total_pairs,
    )

    print(
        "Selected pairs:",
        len(
            selected_pairs
        ),
    )

    print(
        "Selected directed rows:",
        len(
            subset
        ),
    )

    print(
        "\nDirection:"
    )

    print(
        subset[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSelected source dataset:"
    )

    print(
        selected_pairs[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSelected quality tier:"
    )

    print(
        selected_pairs[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nEN length bucket:"
    )

    print(
        selected_pairs[
            "en_length_bucket"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nZH length bucket:"
    )

    print(
        selected_pairs[
            "zh_length_bucket"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nMADLAD checkpoint reuse:"
    )

    print(
        reuse_manifest.to_string(
            index=False
        )
    )

    print(
        "\nAssertions:"
    )

    for key, value in assertions.items():
        print(
            f"{key}: {bool(value)}"
        )

    print(
        "\nTeacher subset:"
    )

    print(
        subset_output
    )

    print(
        "\nSubset MADLAD output directory:"
    )

    print(
        subset_madlad_dir
    )

    print(
        "\nReport:"
    )

    print(
        report_output
    )

    print(
        "\nSTATUS:"
    )

    print(
        "READY_FOR_STEP_18C1_MADLAD_20K"
    )


if __name__ == "__main__":
    main()
