from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STEP_VERSION = "15A_V1"

SEED = 2026
VALIDATION_RATIO = 0.05


# ============================================================
# Normalization used ONLY for leakage/group detection
# ============================================================

def normalize_for_group(
    text: str,
) -> str:

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = text.strip().lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def stable_hash(
    text: str,
    seed: int = SEED,
) -> str:

    value = (
        f"{seed}\n{text}"
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Union-Find
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}
        self.rank = {}

    def add(
        self,
        x: str,
    ):

        if x not in self.parent:

            self.parent[x] = x
            self.rank[x] = 0

    def find(
        self,
        x: str,
    ) -> str:

        parent = self.parent[x]

        if parent != x:

            self.parent[x] = self.find(
                parent
            )

        return self.parent[x]

    def union(
        self,
        a: str,
        b: str,
    ):

        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        rank_a = self.rank[ra]
        rank_b = self.rank[rb]

        if rank_a < rank_b:

            self.parent[ra] = rb

        elif rank_a > rank_b:

            self.parent[rb] = ra

        else:

            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# Protected benchmark loader
# ============================================================

def collect_protected_texts(
    project_root: Path,
):

    benchmark_root = (
        project_root
        / "data"
        / "benchmark"
        / "zh_en"
    )

    files = [
        benchmark_root
        / "flores_plus_zh_en_dev_v1.parquet",

        benchmark_root
        / "flores_plus_zh_en_devtest_v1.parquet",

        benchmark_root
        / "tatoeba_zh_en_test_v1.parquet",
    ]

    protected_en = set()
    protected_zh = set()
    protected_pairs = set()

    loaded = {}

    for path in files:

        if not path.exists():

            print(
                "WARNING: benchmark file not found:",
                path
            )

            continue

        df = pd.read_parquet(
            path
        )

        loaded[
            path.name
        ] = len(df)

        if (
            "en" not in df.columns
            or
            "zh" not in df.columns
        ):

            raise RuntimeError(
                f"Benchmark missing en/zh columns: {path}"
            )

        for en, zh in zip(
            df["en"],
            df["zh"],
        ):

            en_norm = normalize_for_group(
                en
            )

            zh_norm = normalize_for_group(
                zh
            )

            protected_en.add(
                en_norm
            )

            protected_zh.add(
                zh_norm
            )

            protected_pairs.add(
                (
                    en_norm,
                    zh_norm,
                )
            )

    return (
        protected_en,
        protected_zh,
        protected_pairs,
        loaded,
    )


# ============================================================
# Build connected groups
# ============================================================

def assign_connected_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df["_en_norm"] = (
        df["en"]
        .map(
            normalize_for_group
        )
    )

    df["_zh_norm"] = (
        df["zh"]
        .map(
            normalize_for_group
        )
    )

    uf = UnionFind()

    for en, zh in zip(
        df["_en_norm"],
        df["_zh_norm"],
    ):

        en_node = (
            "EN::"
            + en
        )

        zh_node = (
            "ZH::"
            + zh
        )

        uf.union(
            en_node,
            zh_node,
        )

    group_ids = []

    for en in df["_en_norm"]:

        root = uf.find(
            "EN::"
            + en
        )

        group_id = (
            "zh_en_group_"
            + hashlib.sha1(
                root.encode(
                    "utf-8"
                )
            ).hexdigest()[:16]
        )

        group_ids.append(
            group_id
        )

    df["split_group_id"] = (
        group_ids
    )

    return df


# ============================================================
# Group split
# ============================================================

def split_groups(
    df: pd.DataFrame,
):

    group_stats = (
        df
        .groupby(
            "split_group_id",
            as_index=False,
        )
        .agg(
            rows=(
                "pair_id",
                "size",
            )
        )
    )

    group_stats[
        "_hash"
    ] = (
        group_stats[
            "split_group_id"
        ]
        .map(
            stable_hash
        )
    )

    group_stats = (
        group_stats
        .sort_values(
            "_hash"
        )
        .reset_index(
            drop=True
        )
    )

    target_val_rows = int(
        round(
            len(df)
            *
            VALIDATION_RATIO
        )
    )

    validation_groups = set()
    current_val_rows = 0

    # Deterministic selection until ~5%.
    for _, row in (
        group_stats.iterrows()
    ):

        if (
            current_val_rows
            >=
            target_val_rows
        ):

            break

        group_id = str(
            row[
                "split_group_id"
            ]
        )

        validation_groups.add(
            group_id
        )

        current_val_rows += int(
            row[
                "rows"
            ]
        )

    validation = (
        df[
            df[
                "split_group_id"
            ]
            .isin(
                validation_groups
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    train = (
        df[
            ~df[
                "split_group_id"
            ]
            .isin(
                validation_groups
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return (
        train,
        validation,
        group_stats,
    )


# ============================================================
# Directed samples
# ============================================================

def build_bidirectional(
    df: pd.DataFrame,
) -> pd.DataFrame:

    base_columns = [
        "pair_id",
        "source_dataset",
        "quality_tier",
        "training_weight",
        "split_group_id",
    ]

    optional_columns = [
        "quality_score",
        "risk_flags",
    ]

    carry_columns = [
        col
        for col in (
            base_columns
            +
            optional_columns
        )
        if col in df.columns
    ]

    en_zh = (
        df[
            carry_columns
        ]
        .copy()
    )

    en_zh[
        "direction"
    ] = "en_zh"

    en_zh[
        "source_lang"
    ] = "en"

    en_zh[
        "target_lang"
    ] = "zh"

    en_zh[
        "source_text"
    ] = (
        df["en"]
        .astype(str)
        .values
    )

    en_zh[
        "target_text"
    ] = (
        df["zh"]
        .astype(str)
        .values
    )

    zh_en = (
        df[
            carry_columns
        ]
        .copy()
    )

    zh_en[
        "direction"
    ] = "zh_en"

    zh_en[
        "source_lang"
    ] = "zh"

    zh_en[
        "target_lang"
    ] = "en"

    zh_en[
        "source_text"
    ] = (
        df["zh"]
        .astype(str)
        .values
    )

    zh_en[
        "target_text"
    ] = (
        df["en"]
        .astype(str)
        .values
    )

    directed = pd.concat(
        [
            en_zh,
            zh_en,
        ],
        ignore_index=True,
    )

    directed[
        "sample_id"
    ] = [
        f"zh_en_directed_{i:07d}"
        for i
        in range(
            len(directed)
        )
    ]

    return directed


# ============================================================
# Leakage checks
# ============================================================

def leakage_between(
    train: pd.DataFrame,
    validation: pd.DataFrame,
):

    train_pairs = set(
        zip(
            train["_en_norm"],
            train["_zh_norm"],
        )
    )

    val_pairs = set(
        zip(
            validation["_en_norm"],
            validation["_zh_norm"],
        )
    )

    pair_overlap = (
        train_pairs
        &
        val_pairs
    )

    en_overlap = (
        set(
            train["_en_norm"]
        )
        &
        set(
            validation["_en_norm"]
        )
    )

    zh_overlap = (
        set(
            train["_zh_norm"]
        )
        &
        set(
            validation["_zh_norm"]
        )
    )

    group_overlap = (
        set(
            train[
                "split_group_id"
            ]
        )
        &
        set(
            validation[
                "split_group_id"
            ]
        )
    )

    return {
        "pair_overlap": len(
            pair_overlap
        ),
        "english_overlap": len(
            en_overlap
        ),
        "chinese_overlap": len(
            zh_overlap
        ),
        "group_overlap": len(
            group_overlap
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_file = (
        project_root
        / "data"
        / "approved"
        / "zh_en"
        / "v1"
        / "zh_en_approved_v1.parquet"
    )

    output_root = (
        project_root
        / "data"
        / "splits"
        / "zh_en"
        / "v1"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 15A - FROZEN GROUP TRAIN / VALIDATION SPLIT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput:"
    )

    print(
        input_file
    )

    if not input_file.exists():

        raise FileNotFoundError(
            input_file
        )

    df = pd.read_parquet(
        input_file
    )

    required = {
        "pair_id",
        "en",
        "zh",
        "source_dataset",
        "quality_tier",
        "training_weight",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    if (
        df["pair_id"]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate pair_id in approved input."
        )

    print(
        "\nApproved pairs:",
        len(df)
    )

    print(
        "\nQuality tiers:"
    )

    print(
        df[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSources:"
    )

    print(
        df[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Frozen benchmark safety
    # ========================================================

    (
        protected_en,
        protected_zh,
        protected_pairs,
        benchmark_files,
    ) = collect_protected_texts(
        project_root
    )

    print(
        "\nProtected benchmark files:"
    )

    for key, value in (
        benchmark_files.items()
    ):

        print(
            key,
            ":",
            value
        )

    en_norm = (
        df["en"]
        .map(
            normalize_for_group
        )
    )

    zh_norm = (
        df["zh"]
        .map(
            normalize_for_group
        )
    )

    benchmark_pair_hits = 0
    benchmark_en_hits = 0
    benchmark_zh_hits = 0

    for en, zh in zip(
        en_norm,
        zh_norm,
    ):

        if (
            en,
            zh,
        ) in protected_pairs:

            benchmark_pair_hits += 1

        if en in protected_en:

            benchmark_en_hits += 1

        if zh in protected_zh:

            benchmark_zh_hits += 1

    print(
        "\nProtected benchmark leakage:"
    )

    print(
        "Pair:",
        benchmark_pair_hits
    )

    print(
        "EN:",
        benchmark_en_hits
    )

    print(
        "ZH:",
        benchmark_zh_hits
    )

    if any(
        [
            benchmark_pair_hits,
            benchmark_en_hits,
            benchmark_zh_hits,
        ]
    ):

        raise RuntimeError(
            "Protected benchmark leakage detected."
        )

    # ========================================================
    # Connected components
    # ========================================================

    print(
        "\nBuilding connected components..."
    )

    grouped = assign_connected_groups(
        df
    )

    group_count = int(
        grouped[
            "split_group_id"
        ]
        .nunique()
    )

    largest_group = int(
        grouped[
            "split_group_id"
        ]
        .value_counts()
        .max()
    )

    print(
        "Connected groups:",
        group_count
    )

    print(
        "Largest group:",
        largest_group
    )

    # ========================================================
    # Split
    # ========================================================

    (
        train,
        validation,
        group_report,
    ) = split_groups(
        grouped
    )

    leakage = leakage_between(
        train,
        validation,
    )

    print(
        "\nSplit:"
    )

    print(
        "Train:",
        len(train)
    )

    print(
        "Validation:",
        len(validation)
    )

    print(
        "Validation percent:",
        f"{len(validation) / len(grouped) * 100:.3f}%"
    )

    print(
        "\nTrain / validation leakage:"
    )

    for key, value in (
        leakage.items()
    ):

        print(
            key,
            ":",
            value
        )

    if any(
        leakage.values()
    ):

        raise RuntimeError(
            "Train/validation leakage detected."
        )

    # ========================================================
    # Directed datasets
    # ========================================================

    train_directed = (
        build_bidirectional(
            train
        )
    )

    validation_directed = (
        build_bidirectional(
            validation
        )
    )

    # ========================================================
    # Save pair-level
    # ========================================================

    train_pair_file = (
        output_root
        / "train_pairs_v1.parquet"
    )

    validation_pair_file = (
        output_root
        / "validation_pairs_v1.parquet"
    )

    train_csv = (
        output_root
        / "train_pairs_v1.csv"
    )

    validation_csv = (
        output_root
        / "validation_pairs_v1.csv"
    )

    train.drop(
        columns=[
            "_en_norm",
            "_zh_norm",
        ],
        errors="ignore",
    ).to_parquet(
        train_pair_file,
        index=False,
    )

    validation.drop(
        columns=[
            "_en_norm",
            "_zh_norm",
        ],
        errors="ignore",
    ).to_parquet(
        validation_pair_file,
        index=False,
    )

    train.drop(
        columns=[
            "_en_norm",
            "_zh_norm",
        ],
        errors="ignore",
    ).to_csv(
        train_csv,
        index=False,
        encoding="utf-8-sig",
    )

    validation.drop(
        columns=[
            "_en_norm",
            "_zh_norm",
        ],
        errors="ignore",
    ).to_csv(
        validation_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Save bidirectional
    # ========================================================

    train_directed_file = (
        output_root
        / "train_bidirectional_v1.parquet"
    )

    validation_directed_file = (
        output_root
        / "validation_bidirectional_v1.parquet"
    )

    train_directed_csv = (
        output_root
        / "train_bidirectional_v1.csv"
    )

    validation_directed_csv = (
        output_root
        / "validation_bidirectional_v1.csv"
    )

    train_directed.to_parquet(
        train_directed_file,
        index=False,
    )

    validation_directed.to_parquet(
        validation_directed_file,
        index=False,
    )

    train_directed.to_csv(
        train_directed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    validation_directed.to_csv(
        validation_directed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Group report
    # ========================================================

    group_to_split = {}

    for gid in (
        train[
            "split_group_id"
        ]
        .unique()
    ):

        group_to_split[
            gid
        ] = "train"

    for gid in (
        validation[
            "split_group_id"
        ]
        .unique()
    ):

        group_to_split[
            gid
        ] = "validation"

    group_report[
        "split"
    ] = (
        group_report[
            "split_group_id"
        ]
        .map(
            group_to_split
        )
    )

    group_report.drop(
        columns=[
            "_hash"
        ],
        errors="ignore",
    ).to_csv(
        output_root
        / "group_report_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest_rows = []

    for split_name, part in [
        (
            "train",
            train,
        ),
        (
            "validation",
            validation,
        ),
    ]:

        tier_counts = (
            part[
                "quality_tier"
            ]
            .value_counts()
            .to_dict()
        )

        source_counts = (
            part[
                "source_dataset"
            ]
            .value_counts()
            .to_dict()
        )

        manifest_rows.append(
            {
                "split": split_name,
                "pairs": len(part),
                "directed_samples": (
                    len(part)
                    *
                    2
                ),
                "groups": int(
                    part[
                        "split_group_id"
                    ]
                    .nunique()
                ),
                "gold": int(
                    tier_counts.get(
                        "GOLD",
                        0,
                    )
                ),
                "silver": int(
                    tier_counts.get(
                        "SILVER",
                        0,
                    )
                ),
                "bronze": int(
                    tier_counts.get(
                        "BRONZE",
                        0,
                    )
                ),
                "alt": int(
                    source_counts.get(
                        "ALT",
                        0,
                    )
                ),
                "tatoeba": int(
                    source_counts.get(
                        "Tatoeba",
                        0,
                    )
                ),
            }
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    manifest.to_csv(
        output_root
        / "split_manifest_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # JSON report
    # ========================================================

    report = {
        "step": "15A",
        "step_version": STEP_VERSION,
        "seed": SEED,
        "validation_ratio_target": (
            VALIDATION_RATIO
        ),

        "input_pairs": int(
            len(df)
        ),

        "connected_groups": (
            group_count
        ),

        "largest_group": (
            largest_group
        ),

        "train_pairs": int(
            len(train)
        ),

        "validation_pairs": int(
            len(validation)
        ),

        "train_directed_samples": int(
            len(train_directed)
        ),

        "validation_directed_samples": int(
            len(validation_directed)
        ),

        "protected_benchmarks": (
            benchmark_files
        ),

        "protected_leakage": {
            "pair": int(
                benchmark_pair_hits
            ),
            "english": int(
                benchmark_en_hits
            ),
            "chinese": int(
                benchmark_zh_hits
            ),
        },

        "train_validation_leakage": (
            leakage
        ),

        "assertions": {
            "pair_count_preserved": (
                len(train)
                +
                len(validation)
                ==
                len(df)
            ),
            "pair_id_unique_train": bool(
                not train[
                    "pair_id"
                ]
                .duplicated()
                .any()
            ),
            "pair_id_unique_validation": bool(
                not validation[
                    "pair_id"
                ]
                .duplicated()
                .any()
            ),
            "pair_overlap_zero": (
                leakage[
                    "pair_overlap"
                ]
                ==
                0
            ),
            "english_overlap_zero": (
                leakage[
                    "english_overlap"
                ]
                ==
                0
            ),
            "chinese_overlap_zero": (
                leakage[
                    "chinese_overlap"
                ]
                ==
                0
            ),
            "group_overlap_zero": (
                leakage[
                    "group_overlap"
                ]
                ==
                0
            ),
            "benchmark_pair_leakage_zero": (
                benchmark_pair_hits
                ==
                0
            ),
            "benchmark_en_leakage_zero": (
                benchmark_en_hits
                ==
                0
            ),
            "benchmark_zh_leakage_zero": (
                benchmark_zh_hits
                ==
                0
            ),
        },

        "outputs": {
            "train_pairs": str(
                train_pair_file
            ),
            "validation_pairs": str(
                validation_pair_file
            ),
            "train_bidirectional": str(
                train_directed_file
            ),
            "validation_bidirectional": str(
                validation_directed_file
            ),
        },

        "created_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),

        "status": (
            "ZH_EN_FROZEN_SPLIT_READY"
        ),
    }

    with open(
        output_root
        / "split_report_v1.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 15A RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput pairs:",
        len(df)
    )

    print(
        "Connected groups:",
        group_count
    )

    print(
        "Largest group:",
        largest_group
    )

    print(
        "\nTrain pairs:",
        len(train)
    )

    print(
        "Validation pairs:",
        len(validation)
    )

    print(
        "\nTrain directed:",
        len(train_directed)
    )

    print(
        "Validation directed:",
        len(validation_directed)
    )

    print(
        "\nManifest:"
    )

    print(
        manifest.to_string(
            index=False
        )
    )

    print(
        "\nLeakage:"
    )

    for key, value in (
        leakage.items()
    ):

        print(
            key,
            ":",
            value
        )

    print(
        "\nAssertions:"
    )

    for key, value in (
        report[
            "assertions"
        ].items()
    ):

        print(
            key,
            ":",
            value
        )

    print(
        "\nOutput:"
    )

    print(
        output_root
    )

    print(
        "\nSTATUS:"
    )

    print(
        "ZH_EN_FROZEN_SPLIT_READY"
    )


if __name__ == "__main__":
    main()