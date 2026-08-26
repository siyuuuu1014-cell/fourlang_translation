from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# 1. Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# 2. Input
# ============================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "approved"
    / "en_uz"
    / "v1"
    / "en_uz_approved_v1.parquet"
)


# ============================================================
# 3. Output
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "en_uz"
    / "v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------
# Unified split
# ------------------------------------------------------------

SPLIT_MANIFEST_FILE = (
    OUTPUT_DIR
    / "split_manifest_v1.parquet"
)

SPLIT_MANIFEST_CSV = (
    OUTPUT_DIR
    / "split_manifest_v1.csv"
)


# ------------------------------------------------------------
# All approved pairs
# ------------------------------------------------------------

TRAIN_PAIRS_FILE = (
    OUTPUT_DIR
    / "train_pairs_v1.parquet"
)

VALIDATION_PAIRS_FILE = (
    OUTPUT_DIR
    / "validation_pairs_v1.parquet"
)


# ------------------------------------------------------------
# Exp1:
#
# GOLD + SILVER only
# ------------------------------------------------------------

TRAIN_EXP1_FILE = (
    OUTPUT_DIR
    / "train_exp1_gold_silver_v1.parquet"
)

VALIDATION_EXP1_FILE = (
    OUTPUT_DIR
    / "validation_exp1_gold_silver_v1.parquet"
)


# ------------------------------------------------------------
# Exp1 bidirectional
# ------------------------------------------------------------

TRAIN_EXP1_BIDIRECTIONAL_FILE = (
    OUTPUT_DIR
    / "train_exp1_bidirectional_v1.parquet"
)

TRAIN_EXP1_BIDIRECTIONAL_CSV = (
    OUTPUT_DIR
    / "train_exp1_bidirectional_v1.csv"
)

VALIDATION_EXP1_BIDIRECTIONAL_FILE = (
    OUTPUT_DIR
    / "validation_exp1_bidirectional_v1.parquet"
)

VALIDATION_EXP1_BIDIRECTIONAL_CSV = (
    OUTPUT_DIR
    / "validation_exp1_bidirectional_v1.csv"
)


# ------------------------------------------------------------
# Reports
# ------------------------------------------------------------

GROUP_REPORT_FILE = (
    OUTPUT_DIR
    / "group_report_v1.csv"
)

SPLIT_REPORT_FILE = (
    OUTPUT_DIR
    / "split_report_v1.json"
)


# ============================================================
# 4. Config
# ============================================================

SEED = 2026

VALIDATION_RATIO = 0.05

# 随机生成多套 group-level split，
# 选择 source/tier/length 分布最接近总体的一套。
N_SPLIT_TRIALS = 500


EXP1_TIERS = {
    "GOLD",
    "SILVER",
}


# ============================================================
# 5. Canonical text
#
# 比 normalized_text 再保守一步。
#
# 用于防止：
#
# Hello.
# Hello
#
# 被划入不同 split。
# ============================================================

def canonical_split_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.lower()

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
    )

    # 标点不作为 split 差异
    text = re.sub(
        r"[^\w']+",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# 6. Union Find
#
# 用于建立：
#
# EN1 ↔ UZ1
# EN1 ↔ UZ2
# EN2 ↔ UZ2
#
# 三者必须进入同一个 split。
# ============================================================

class UnionFind:

    def __init__(self):

        self.parent = {}
        self.rank = {}


    def add(
        self,
        item,
    ):

        if item not in self.parent:

            self.parent[item] = item
            self.rank[item] = 0


    def find(
        self,
        item,
    ):

        parent = self.parent[item]

        if parent != item:

            self.parent[item] = self.find(
                parent
            )

        return self.parent[item]


    def union(
        self,
        a,
        b,
    ):

        self.add(a)
        self.add(b)

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return


        rank_a = self.rank[
            root_a
        ]

        rank_b = self.rank[
            root_b
        ]


        if rank_a < rank_b:

            self.parent[
                root_a
            ] = root_b


        elif rank_a > rank_b:

            self.parent[
                root_b
            ] = root_a


        else:

            self.parent[
                root_b
            ] = root_a

            self.rank[
                root_a
            ] += 1


# ============================================================
# 7. Build connected split groups
# ============================================================

def build_split_groups(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\nBuilding shared-text groups..."
    )


    df = df.copy()


    df[
        "en_split_key"
    ] = (
        df[
            "source_text_normalized"
        ]
        .astype(str)
        .map(
            canonical_split_text
        )
    )


    df[
        "uz_split_key"
    ] = (
        df[
            "target_text_normalized"
        ]
        .astype(str)
        .map(
            canonical_split_text
        )
    )


    if (
        df[
            "en_split_key"
        ]
        ==
        ""
    ).any():

        raise RuntimeError(
            "Empty EN split key detected."
        )


    if (
        df[
            "uz_split_key"
        ]
        ==
        ""
    ).any():

        raise RuntimeError(
            "Empty UZ split key detected."
        )


    uf = UnionFind()


    # --------------------------------------------------------
    # Bipartite graph
    #
    # E:<english>
    # U:<uzbek>
    # --------------------------------------------------------

    for en_key, uz_key in zip(
        df[
            "en_split_key"
        ],
        df[
            "uz_split_key"
        ],
    ):

        en_node = (
            "E:"
            +
            en_key
        )

        uz_node = (
            "U:"
            +
            uz_key
        )

        uf.union(
            en_node,
            uz_node,
        )


    # --------------------------------------------------------
    # Root for every pair
    # --------------------------------------------------------

    roots = []


    for en_key in df[
        "en_split_key"
    ]:

        root = uf.find(
            "E:"
            +
            en_key
        )

        roots.append(
            root
        )


    df[
        "_component_root"
    ] = roots


    # --------------------------------------------------------
    # Stable group ID
    #
    # 不直接保存 UnionFind root，
    # 因为 root 会受到构建顺序影响。
    #
    # 每个 component 使用最小 pair_id
    # 生成稳定 group ID。
    # --------------------------------------------------------

    group_min_pair = (
        df.groupby(
            "_component_root"
        )[
            "normalized_pair_id"
        ]
        .transform(
            "min"
        )
    )


    df[
        "split_group_id"
    ] = (
        "group_"
        +
        group_min_pair
        .astype(str)
        .str[:20]
    )


    df = df.drop(
        columns=[
            "_component_root"
        ]
    )


    return df


# ============================================================
# 8. Length bucket
# ============================================================

def add_length_bucket(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()


    if (
        "source_word_count"
        in df.columns
    ):

        lengths = (
            df[
                "source_word_count"
            ]
            .fillna(0)
            .astype(float)
        )


    else:

        lengths = (
            df[
                "source_text_normalized"
            ]
            .astype(str)
            .str.split()
            .map(len)
        )


    df[
        "length_bucket"
    ] = pd.cut(

        lengths,

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

        include_lowest=True,
    )


    df[
        "length_bucket"
    ] = (
        df[
            "length_bucket"
        ]
        .astype(str)
    )


    return df


# ============================================================
# 9. Build group matrices
#
# 用于寻找：
#
# size
# tier
# source
# length
#
# 都比较均衡的 Validation。
# ============================================================

def build_group_statistics(
    df: pd.DataFrame,
):

    group_sizes = (
        df.groupby(
            "split_group_id"
        )
        .size()
        .rename(
            "group_size"
        )
    )


    tier_table = pd.crosstab(

        df[
            "split_group_id"
        ],

        df[
            "quality_tier"
        ],
    )


    source_table = pd.crosstab(

        df[
            "split_group_id"
        ],

        df[
            "data_source"
        ],
    )


    length_table = pd.crosstab(

        df[
            "split_group_id"
        ],

        df[
            "length_bucket"
        ],
    )


    group_ids = (
        group_sizes
        .index
        .tolist()
    )


    tier_table = tier_table.reindex(
        group_ids,
        fill_value=0,
    )


    source_table = source_table.reindex(
        group_ids,
        fill_value=0,
    )


    length_table = length_table.reindex(
        group_ids,
        fill_value=0,
    )


    return {
        "group_ids":
            np.array(
                group_ids,
                dtype=object,
            ),

        "sizes":
            group_sizes
            .reindex(
                group_ids
            )
            .to_numpy(
                dtype=np.int64
            ),

        "tier_columns":
            tier_table.columns.tolist(),

        "tier_matrix":
            tier_table.to_numpy(
                dtype=np.int64
            ),

        "source_columns":
            source_table.columns.tolist(),

        "source_matrix":
            source_table.to_numpy(
                dtype=np.int64
            ),

        "length_columns":
            length_table.columns.tolist(),

        "length_matrix":
            length_table.to_numpy(
                dtype=np.int64
            ),
    }


# ============================================================
# 10. Distribution error
# ============================================================

def distribution_error(
    selected_counts: np.ndarray,
    total_counts: np.ndarray,
    selected_total: int,
    total: int,
) -> float:

    if (
        selected_total <= 0
        or
        total <= 0
    ):

        return 999.0


    overall_prop = (
        total_counts
        /
        total
    )


    selected_prop = (
        selected_counts
        /
        selected_total
    )


    return float(
        np.abs(
            overall_prop
            -
            selected_prop
        ).mean()
    )


# ============================================================
# 11. Search best group split
# ============================================================

def choose_validation_groups(
    stats,
    total_pairs: int,
):

    group_ids = stats[
        "group_ids"
    ]

    sizes = stats[
        "sizes"
    ]

    tier_matrix = stats[
        "tier_matrix"
    ]

    source_matrix = stats[
        "source_matrix"
    ]

    length_matrix = stats[
        "length_matrix"
    ]


    n_groups = len(
        group_ids
    )


    target_validation = int(
        round(
            total_pairs
            *
            VALIDATION_RATIO
        )
    )


    total_tiers = (
        tier_matrix.sum(
            axis=0
        )
    )

    total_sources = (
        source_matrix.sum(
            axis=0
        )
    )

    total_lengths = (
        length_matrix.sum(
            axis=0
        )
    )


    rng = np.random.default_rng(
        SEED
    )


    best = None


    print(
        "\nSearching best group-level split..."
    )

    print(
        "Trials:",
        N_SPLIT_TRIALS
    )

    print(
        "Target validation pairs:",
        target_validation
    )


    for trial in range(
        N_SPLIT_TRIALS
    ):

        permutation = (
            rng.permutation(
                n_groups
            )
        )


        ordered_sizes = (
            sizes[
                permutation
            ]
        )


        cumulative = (
            np.cumsum(
                ordered_sizes
            )
        )


        boundary = int(
            np.searchsorted(
                cumulative,
                target_validation,
                side="left",
            )
        )


        # 比较 cutoff 前后两种方案，
        # 哪一种更接近目标数量。
        candidate_counts = []


        if boundary > 0:

            candidate_counts.append(
                boundary
            )


        if boundary < n_groups:

            candidate_counts.append(
                boundary + 1
            )


        if not candidate_counts:

            continue


        for take_n in (
            candidate_counts
        ):

            selected_indices = (
                permutation[
                    :take_n
                ]
            )


            validation_size = int(
                sizes[
                    selected_indices
                ].sum()
            )


            selected_tiers = (
                tier_matrix[
                    selected_indices
                ]
                .sum(
                    axis=0
                )
            )


            selected_sources = (
                source_matrix[
                    selected_indices
                ]
                .sum(
                    axis=0
                )
            )


            selected_lengths = (
                length_matrix[
                    selected_indices
                ]
                .sum(
                    axis=0
                )
            )


            size_error = (
                abs(
                    validation_size
                    -
                    target_validation
                )
                /
                max(
                    target_validation,
                    1,
                )
            )


            tier_error = (
                distribution_error(
                    selected_tiers,
                    total_tiers,
                    validation_size,
                    total_pairs,
                )
            )


            source_error = (
                distribution_error(
                    selected_sources,
                    total_sources,
                    validation_size,
                    total_pairs,
                )
            )


            length_error = (
                distribution_error(
                    selected_lengths,
                    total_lengths,
                    validation_size,
                    total_pairs,
                )
            )


            # ------------------------------------------------
            # Objective
            #
            # size 最重要，
            # 然后 tier，
            # 然后 source，
            # 最后 length。
            # ------------------------------------------------

            objective = (

                5.0
                *
                size_error

                +

                2.0
                *
                tier_error

                +

                1.5
                *
                source_error

                +

                0.5
                *
                length_error
            )


            if (
                best is None
                or
                objective
                <
                best[
                    "objective"
                ]
            ):

                best = {

                    "objective":
                        float(
                            objective
                        ),

                    "trial":
                        trial,

                    "validation_size":
                        validation_size,

                    "group_indices":
                        selected_indices.copy(),

                    "size_error":
                        float(
                            size_error
                        ),

                    "tier_error":
                        float(
                            tier_error
                        ),

                    "source_error":
                        float(
                            source_error
                        ),

                    "length_error":
                        float(
                            length_error
                        ),
                }


    if best is None:

        raise RuntimeError(
            "Failed to find validation split."
        )


    validation_groups = set(
        group_ids[
            best[
                "group_indices"
            ]
        ]
        .tolist()
    )


    print(
        "\nBest split:"
    )

    print(
        "Trial:",
        best[
            "trial"
        ]
    )

    print(
        "Validation pairs:",
        best[
            "validation_size"
        ]
    )

    print(
        "Objective:",
        round(
            best[
                "objective"
            ],
            6,
        )
    )

    print(
        "Size error:",
        round(
            best[
                "size_error"
            ],
            6,
        )
    )

    print(
        "Tier error:",
        round(
            best[
                "tier_error"
            ],
            6,
        )
    )

    print(
        "Source error:",
        round(
            best[
                "source_error"
            ],
            6,
        )
    )

    print(
        "Length error:",
        round(
            best[
                "length_error"
            ],
            6,
        )
    )


    return (
        validation_groups,
        best,
    )


# ============================================================
# 12. Bidirectional expansion
# ============================================================

def build_bidirectional(
    df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:

    # --------------------------------------------------------
    # EN -> UZ
    # --------------------------------------------------------

    en_uz = pd.DataFrame({

        "sample_id":
            df[
                "normalized_pair_id"
            ]
            .astype(str)
            .map(
                lambda x:
                    f"{x}_en_uz"
            ),

        "normalized_pair_id":
            df[
                "normalized_pair_id"
            ],

        "split":
            split_name,

        "direction":
            "en_uz",

        "src_lang":
            "en",

        "tgt_lang":
            "uz",

        "source_text":
            df[
                "source_text_normalized"
            ],

        "target_text":
            df[
                "target_text_normalized"
            ],

        "quality_tier":
            df[
                "quality_tier"
            ],

        "training_weight":
            df[
                "training_weight"
            ],

        "data_source":
            df[
                "data_source"
            ],

        "split_group_id":
            df[
                "split_group_id"
            ],
    })


    # --------------------------------------------------------
    # UZ -> EN
    # --------------------------------------------------------

    uz_en = pd.DataFrame({

        "sample_id":
            df[
                "normalized_pair_id"
            ]
            .astype(str)
            .map(
                lambda x:
                    f"{x}_uz_en"
            ),

        "normalized_pair_id":
            df[
                "normalized_pair_id"
            ],

        "split":
            split_name,

        "direction":
            "uz_en",

        "src_lang":
            "uz",

        "tgt_lang":
            "en",

        "source_text":
            df[
                "target_text_normalized"
            ],

        "target_text":
            df[
                "source_text_normalized"
            ],

        "quality_tier":
            df[
                "quality_tier"
            ],

        "training_weight":
            df[
                "training_weight"
            ],

        "data_source":
            df[
                "data_source"
            ],

        "split_group_id":
            df[
                "split_group_id"
            ],
    })


    result = pd.concat(
        [
            en_uz,
            uz_en,
        ],
        ignore_index=True,
    )


    return result


# ============================================================
# 13. Verify leakage
# ============================================================

def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
):

    print(
        "\nRunning leakage checks..."
    )


    # --------------------------------------------------------
    # Pair overlap
    # --------------------------------------------------------

    pair_overlap = (
        set(
            train_df[
                "normalized_pair_id"
            ]
        )
        &
        set(
            val_df[
                "normalized_pair_id"
            ]
        )
    )


    if pair_overlap:

        raise RuntimeError(
            "Pair-level leakage detected: "
            f"{len(pair_overlap)}"
        )


    # --------------------------------------------------------
    # Group overlap
    # --------------------------------------------------------

    group_overlap = (
        set(
            train_df[
                "split_group_id"
            ]
        )
        &
        set(
            val_df[
                "split_group_id"
            ]
        )
    )


    if group_overlap:

        raise RuntimeError(
            "Group leakage detected: "
            f"{len(group_overlap)}"
        )


    # --------------------------------------------------------
    # English overlap
    # --------------------------------------------------------

    en_overlap = (
        set(
            train_df[
                "en_split_key"
            ]
        )
        &
        set(
            val_df[
                "en_split_key"
            ]
        )
    )


    if en_overlap:

        raise RuntimeError(
            "English text leakage detected: "
            f"{len(en_overlap)}"
        )


    # --------------------------------------------------------
    # Uzbek overlap
    # --------------------------------------------------------

    uz_overlap = (
        set(
            train_df[
                "uz_split_key"
            ]
        )
        &
        set(
            val_df[
                "uz_split_key"
            ]
        )
    )


    if uz_overlap:

        raise RuntimeError(
            "Uzbek text leakage detected: "
            f"{len(uz_overlap)}"
        )


    print(
        "Pair overlap    : 0"
    )

    print(
        "Group overlap   : 0"
    )

    print(
        "English overlap : 0"
    )

    print(
        "Uzbek overlap   : 0"
    )

    print(
        "Leakage checks  : PASS"
    )


# ============================================================
# 14. Distribution report
# ============================================================

def distribution_table(
    df: pd.DataFrame,
    column: str,
):

    counts = (
        df[
            column
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            column
        )
        .reset_index(
            name="count"
        )
    )


    counts[
        "percent"
    ] = (
        counts[
            "count"
        ]
        /
        len(df)
        *
        100
    )


    return counts


# ============================================================
# 15. Main
# ============================================================

def main():

    print("=" * 110)
    print("EN-UZ PIPELINE")
    print("STEP 07 - TRAIN / VALIDATION SPLIT V1")
    print("=" * 110)


    # ========================================================
    # Input
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Approved Dataset not found:\n"
            f"{INPUT_FILE}"
        )


    print(
        "\nLoading:"
    )

    print(
        INPUT_FILE
    )


    df = pd.read_parquet(
        INPUT_FILE
    )


    print(
        "\nApproved pairs:",
        len(df)
    )


    # ========================================================
    # Required columns
    # ========================================================

    required_columns = [
        "normalized_pair_id",
        "source_text_normalized",
        "target_text_normalized",
        "quality_tier",
        "training_weight",
        "data_source",
    ]


    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            f"Missing columns: "
            f"{missing}"
        )


    # ========================================================
    # Check approved only
    # ========================================================

    if "final_status" in df.columns:

        invalid_status = (
            df[
                "final_status"
            ]
            .fillna("")
            .astype(str)
            .str.upper()
            !=
            "APPROVED"
        )


        if invalid_status.any():

            raise RuntimeError(
                "Input contains non-approved rows."
            )


    # ========================================================
    # Pair uniqueness
    # ========================================================

    duplicate_pairs = int(
        df[
            "normalized_pair_id"
        ]
        .duplicated()
        .sum()
    )


    if duplicate_pairs > 0:

        raise RuntimeError(
            f"Duplicate approved pairs: "
            f"{duplicate_pairs}"
        )


    # ========================================================
    # Benchmark safety
    # ========================================================

    if "benchmark_leak" in df.columns:

        benchmark_leaks = int(
            df[
                "benchmark_leak"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        )


        if benchmark_leaks > 0:

            raise RuntimeError(
                f"Benchmark leakage found: "
                f"{benchmark_leaks}"
            )


    # ========================================================
    # Add grouping metadata
    # ========================================================

    df = build_split_groups(
        df
    )


    df = add_length_bucket(
        df
    )


    # ========================================================
    # Group stats
    # ========================================================

    group_size_report = (
        df.groupby(
            "split_group_id"
        )
        .agg(

            pair_count=(
                "normalized_pair_id",
                "size",
            ),

            gold_count=(
                "quality_tier",
                lambda x:
                    int(
                        (
                            x
                            ==
                            "GOLD"
                        ).sum()
                    ),
            ),

            silver_count=(
                "quality_tier",
                lambda x:
                    int(
                        (
                            x
                            ==
                            "SILVER"
                        ).sum()
                    ),
            ),

            bronze_count=(
                "quality_tier",
                lambda x:
                    int(
                        (
                            x
                            ==
                            "BRONZE"
                        ).sum()
                    ),
            ),
        )
        .reset_index()
        .sort_values(
            "pair_count",
            ascending=False,
        )
    )


    group_size_report.to_csv(
        GROUP_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "\nSplit groups:",
        df[
            "split_group_id"
        ].nunique()
    )


    print(
        "Largest group:",
        int(
            group_size_report[
                "pair_count"
            ].max()
        )
    )


    print(
        "Groups with >1 pair:",
        int(
            (
                group_size_report[
                    "pair_count"
                ]
                >
                1
            ).sum()
        )
    )


    # ========================================================
    # Search split
    # ========================================================

    stats = build_group_statistics(
        df
    )


    (
        validation_groups,
        split_meta,
    ) = choose_validation_groups(
        stats,
        total_pairs=len(df),
    )


    # ========================================================
    # Assign split
    # ========================================================

    df[
        "split"
    ] = np.where(

        df[
            "split_group_id"
        ]
        .isin(
            validation_groups
        ),

        "validation",

        "train",
    )


    # ========================================================
    # Pair-level datasets
    # ========================================================

    train_pairs = (
        df[
            df[
                "split"
            ]
            ==
            "train"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    validation_pairs = (
        df[
            df[
                "split"
            ]
            ==
            "validation"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Leakage check
    # ========================================================

    verify_no_leakage(
        train_pairs,
        validation_pairs,
    )


    # ========================================================
    # Total integrity
    # ========================================================

    if (
        len(train_pairs)
        +
        len(validation_pairs)
        !=
        len(df)
    ):

        raise RuntimeError(
            "Train + Validation != total."
        )


    # ========================================================
    # Exp1 = GOLD + SILVER
    # ========================================================

    train_exp1 = (
        train_pairs[
            train_pairs[
                "quality_tier"
            ]
            .isin(
                EXP1_TIERS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    validation_exp1 = (
        validation_pairs[
            validation_pairs[
                "quality_tier"
            ]
            .isin(
                EXP1_TIERS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Bidirectional expansion
    # ========================================================

    train_bidirectional = (
        build_bidirectional(
            train_exp1,
            split_name="train",
        )
    )


    validation_bidirectional = (
        build_bidirectional(
            validation_exp1,
            split_name="validation",
        )
    )


    # ========================================================
    # Bidirectional integrity
    # ========================================================

    if (
        len(
            train_bidirectional
        )
        !=
        len(
            train_exp1
        )
        *
        2
    ):

        raise RuntimeError(
            "Train bidirectional count invalid."
        )


    if (
        len(
            validation_bidirectional
        )
        !=
        len(
            validation_exp1
        )
        *
        2
    ):

        raise RuntimeError(
            "Validation bidirectional "
            "count invalid."
        )


    # ========================================================
    # Manifest
    # ========================================================

    manifest_columns = [
        "normalized_pair_id",
        "split_group_id",
        "split",
        "quality_tier",
        "data_source",
        "en_split_key",
        "uz_split_key",
        "length_bucket",
    ]


    manifest = (
        df[
            manifest_columns
        ]
        .copy()
        .sort_values(
            "normalized_pair_id"
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Save
    # ========================================================

    print(
        "\nSaving split files..."
    )


    manifest.to_parquet(
        SPLIT_MANIFEST_FILE,
        index=False,
    )


    manifest.to_csv(
        SPLIT_MANIFEST_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    train_pairs.to_parquet(
        TRAIN_PAIRS_FILE,
        index=False,
    )


    validation_pairs.to_parquet(
        VALIDATION_PAIRS_FILE,
        index=False,
    )


    train_exp1.to_parquet(
        TRAIN_EXP1_FILE,
        index=False,
    )


    validation_exp1.to_parquet(
        VALIDATION_EXP1_FILE,
        index=False,
    )


    train_bidirectional.to_parquet(
        TRAIN_EXP1_BIDIRECTIONAL_FILE,
        index=False,
    )


    train_bidirectional.to_csv(
        TRAIN_EXP1_BIDIRECTIONAL_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    validation_bidirectional.to_parquet(
        VALIDATION_EXP1_BIDIRECTIONAL_FILE,
        index=False,
    )


    validation_bidirectional.to_csv(
        VALIDATION_EXP1_BIDIRECTIONAL_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Distribution reports
    # ========================================================

    train_tier = (
        distribution_table(
            train_pairs,
            "quality_tier",
        )
    )


    val_tier = (
        distribution_table(
            validation_pairs,
            "quality_tier",
        )
    )


    train_source = (
        distribution_table(
            train_pairs,
            "data_source",
        )
    )


    val_source = (
        distribution_table(
            validation_pairs,
            "data_source",
        )
    )


    # ========================================================
    # Report
    # ========================================================

    report = {

        "dataset":
            "en_uz_v1",

        "seed":
            SEED,

        "split_method":
            (
                "connected-component "
                "group-level split"
            ),

        "validation_ratio_target":
            VALIDATION_RATIO,

        "split_trials":
            N_SPLIT_TRIALS,

        "all_approved_pairs":
            len(df),

        "split_groups":
            int(
                df[
                    "split_group_id"
                ].nunique()
            ),

        "largest_group":
            int(
                group_size_report[
                    "pair_count"
                ].max()
            ),

        "train_pairs":
            len(
                train_pairs
            ),

        "validation_pairs":
            len(
                validation_pairs
            ),

        "validation_ratio_actual":
            float(
                len(
                    validation_pairs
                )
                /
                len(df)
            ),

        "exp1": {

            "tiers":
                [
                    "GOLD",
                    "SILVER",
                ],

            "train_pairs":
                len(
                    train_exp1
                ),

            "validation_pairs":
                len(
                    validation_exp1
                ),

            "train_bidirectional_samples":
                len(
                    train_bidirectional
                ),

            "validation_bidirectional_samples":
                len(
                    validation_bidirectional
                ),
        },

        "split_quality": {

            "objective":
                split_meta[
                    "objective"
                ],

            "size_error":
                split_meta[
                    "size_error"
                ],

            "tier_error":
                split_meta[
                    "tier_error"
                ],

            "source_error":
                split_meta[
                    "source_error"
                ],

            "length_error":
                split_meta[
                    "length_error"
                ],
        },

        "leakage_checks": {

            "pair_overlap":
                0,

            "group_overlap":
                0,

            "english_overlap":
                0,

            "uzbek_overlap":
                0,

            "benchmark_leak":
                0,
        },

        "test_policy": {

            "internal_test_split":
                False,

            "benchmark":
                "existing 429-pair benchmark",

            "challenge_benchmark":
                "existing 300-pair challenge benchmark",
        },

        "future_experiments": {

            "exp1":
                "GOLD + SILVER",

            "exp2":
                (
                    "GOLD + SILVER + BRONZE "
                    "using same split_manifest"
                ),

            "exp3":
                (
                    "quality weighted using "
                    "same split_manifest"
                ),

            "exp4":
                (
                    "MADLAD synthetic using "
                    "same validation split"
                ),
        },

        "train_quality_distribution":
            train_tier.to_dict(
                orient="records"
            ),

        "validation_quality_distribution":
            val_tier.to_dict(
                orient="records"
            ),

        "train_source_distribution":
            train_source.to_dict(
                orient="records"
            ),

        "validation_source_distribution":
            val_source.to_dict(
                orient="records"
            ),
    }


    with open(
        SPLIT_REPORT_FILE,
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
    # Terminal report
    # ========================================================

    print("\n")
    print("=" * 110)
    print("STEP 07 COMPLETE")
    print("=" * 110)


    print(
        "All approved pairs:",
        len(df)
    )


    print(
        "\nPAIR-LEVEL SPLIT"
    )

    print(
        "Train:",
        len(
            train_pairs
        ),
        f"({len(train_pairs) / len(df) * 100:.2f}%)"
    )

    print(
        "Validation:",
        len(
            validation_pairs
        ),
        f"({len(validation_pairs) / len(df) * 100:.2f}%)"
    )


    print(
        "\nEXP1 - GOLD + SILVER"
    )

    print(
        "Train pairs:",
        len(
            train_exp1
        )
    )

    print(
        "Validation pairs:",
        len(
            validation_exp1
        )
    )


    print(
        "\nEXP1 BIDIRECTIONAL"
    )

    print(
        "Train samples:",
        len(
            train_bidirectional
        )
    )

    print(
        "Validation samples:",
        len(
            validation_bidirectional
        )
    )


    print(
        "\nTrain quality:"
    )

    print(
        train_exp1[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nValidation quality:"
    )

    print(
        validation_exp1[
            "quality_tier"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nTrain sources:"
    )

    print(
        train_exp1[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nValidation sources:"
    )

    print(
        validation_exp1[
            "data_source"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nLeakage checks:"
    )

    print(
        "pair     : 0"
    )

    print(
        "group    : 0"
    )

    print(
        "English  : 0"
    )

    print(
        "Uzbek    : 0"
    )


    print(
        "\nImportant files:"
    )

    print(
        "Split manifest:"
    )

    print(
        SPLIT_MANIFEST_FILE
    )


    print(
        "\nExp1 Train:"
    )

    print(
        TRAIN_EXP1_BIDIRECTIONAL_FILE
    )


    print(
        "\nExp1 Validation:"
    )

    print(
        VALIDATION_EXP1_BIDIRECTIONAL_FILE
    )


    print(
        "\nReport:"
    )

    print(
        SPLIT_REPORT_FILE
    )


    print(
        "\nEN-UZ Train/Validation V1 ready."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()