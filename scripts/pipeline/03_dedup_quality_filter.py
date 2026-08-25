from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "02_normalized"
    / "parallel_normalized.parquet"
)

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "en_uz"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "03_filtered"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "parallel_filtered.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "parallel_filtered.csv"
)

REJECTED_FILE = (
    OUTPUT_DIR
    / "parallel_rejected.parquet"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "filter_report.json"
)


# ============================================================
# 2. Conservative thresholds
#
# 这里故意设置得比较宽。
# Step03 只排除明显异常，不做激进过滤。
# ============================================================

MIN_WORDS = 1
MAX_WORDS = 80

MIN_CHARS = 2
MAX_CHARS = 600

MIN_CHAR_RATIO = 0.25
MAX_CHAR_RATIO = 4.0


# ============================================================
# 3. Normalization
#
# 必须和 Step02 的规则兼容，
# Benchmark 也使用相同方式进行比较。
# ============================================================

UZ_APOSTROPHES = {
    "’": "'",
    "‘": "'",
    "ʻ": "'",
    "ʼ": "'",
    "`": "'",
    "´": "'",
    "ʹ": "'",
}


def normalize_spaces(text: str) -> str:

    text = (
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_english(text: str) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = (
        text
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )

    return normalize_spaces(text)


def normalize_uzbek(text: str) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    for old, new in UZ_APOSTROPHES.items():

        text = text.replace(
            old,
            new,
        )

    text = (
        text
        .replace("“", '"')
        .replace("”", '"')
    )

    return normalize_spaces(text)


# ============================================================
# 4. Stable normalized pair ID
# ============================================================

def make_normalized_pair_id(
    source_text: str,
    target_text: str,
) -> str:

    payload = (
        source_text.lower()
        +
        "\u241f"
        +
        target_text.lower()
    )

    return hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()


# ============================================================
# 5. Benchmark column detection
# ============================================================

EN_COLUMNS = [
    "en",
    "english",
    "source_text",
    "source_en",
    "src_en",
]

UZ_COLUMNS = [
    "uz",
    "uzbek",
    "target_text",
    "target_uz",
    "tgt_uz",
]


def find_column(
    columns,
    candidates,
):

    column_map = {
        str(col).strip().lower(): col
        for col in columns
    }

    for candidate in candidates:

        if candidate in column_map:

            return column_map[candidate]

    return None


# ============================================================
# 6. Read benchmark file
# ============================================================

def read_benchmark_file(
    path: Path,
):

    suffix = path.suffix.lower()

    try:

        if suffix == ".csv":

            df = pd.read_csv(
                path,
                low_memory=False,
            )

        elif suffix == ".tsv":

            df = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
            )

        elif suffix == ".parquet":

            df = pd.read_parquet(
                path
            )

        else:

            return None

    except Exception as exc:

        print(
            "[BENCHMARK READ ERROR]",
            path,
            repr(exc),
        )

        return None


    en_col = find_column(
        df.columns,
        EN_COLUMNS,
    )

    uz_col = find_column(
        df.columns,
        UZ_COLUMNS,
    )

    if (
        en_col is None
        or
        uz_col is None
    ):

        return None


    result = df[
        [
            en_col,
            uz_col,
        ]
    ].copy()

    result.columns = [
        "en",
        "uz",
    ]

    return result


# ============================================================
# 7. Load all benchmark pair IDs
# ============================================================

def load_benchmark_pair_ids():

    benchmark_ids = set()

    benchmark_files = []


    if not BENCHMARK_DIR.exists():

        return (
            benchmark_ids,
            benchmark_files,
        )


    for path in BENCHMARK_DIR.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower()
            in {
                ".csv",
                ".tsv",
                ".parquet",
            }
        ):

            benchmark_files.append(
                path
            )


    for path in sorted(
        benchmark_files
    ):

        benchmark_df = (
            read_benchmark_file(
                path
            )
        )

        if benchmark_df is None:

            continue


        for en, uz in zip(
            benchmark_df["en"],
            benchmark_df["uz"],
        ):

            if (
                pd.isna(en)
                or
                pd.isna(uz)
            ):

                continue


            en_norm = normalize_english(
                en
            )

            uz_norm = normalize_uzbek(
                uz
            )


            if (
                not en_norm
                or
                not uz_norm
            ):

                continue


            benchmark_ids.add(
                make_normalized_pair_id(
                    en_norm,
                    uz_norm,
                )
            )


    return (
        benchmark_ids,
        benchmark_files,
    )


# ============================================================
# 8. Aggregate sources
# ============================================================

def join_unique(values):

    values = sorted(
        {
            str(x)
            for x in values
            if pd.notna(x)
        }
    )

    return "|".join(values)


# ============================================================
# 9. Main
# ============================================================

def main():

    print("=" * 100)
    print("EN-UZ PIPELINE")
    print("STEP 03 - DEDUP + CONSENSUS + QUALITY FILTER")
    print("=" * 100)


    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到输入文件：\n"
            f"{INPUT_FILE}"
        )


    # ========================================================
    # Load
    # ========================================================

    print("\nLoading normalized data...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        "Input rows:",
        len(df)
    )


    # ========================================================
    # 10. Build normalized pair ID
    # ========================================================

    print(
        "\nGenerating normalized pair IDs..."
    )

    df[
        "normalized_pair_id"
    ] = [

        make_normalized_pair_id(
            src,
            tgt,
        )

        for src, tgt in zip(
            df[
                "source_text_normalized"
            ],
            df[
                "target_text_normalized"
            ],
        )
    ]


    unique_before = (
        df[
            "normalized_pair_id"
        ].nunique()
    )


    print(
        "Unique normalized pairs:",
        unique_before
    )


    # ========================================================
    # 11. Aggregate duplicate information
    # ========================================================

    print(
        "\nAggregating duplicate/source consensus..."
    )


    group_stats = (

        df.groupby(
            "normalized_pair_id",
            sort=False,
        )

        .agg(

            occurrence_count=(
                "normalized_pair_id",
                "size",
            ),

            source_count=(
                "data_source",
                "nunique",
            ),

            data_sources=(
                "data_source",
                join_unique,
            ),

            source_files=(
                "source_file",
                join_unique,
            ),

            source_file_count=(
                "source_file",
                "nunique",
            ),
        )

        .reset_index()
    )


    # ========================================================
    # 12. Keep one representative row per normalized pair
    # ========================================================

    dedup_df = (

        df
        .drop_duplicates(
            subset=[
                "normalized_pair_id",
            ],
            keep="first",
        )
        .copy()
    )


    dedup_df = dedup_df.merge(
        group_stats,
        on="normalized_pair_id",
        how="left",
    )


    # ========================================================
    # 13. Consensus indicators
    # ========================================================

    dedup_df[
        "multi_source_consensus"
    ] = (
        dedup_df[
            "source_count"
        ]
        >=
        2
    )


    dedup_df[
        "repeated_pair"
    ] = (
        dedup_df[
            "occurrence_count"
        ]
        >=
        2
    )


    print(
        "\nAfter normalized dedup:",
        len(dedup_df)
    )

    print(
        "Multi-source pairs:",
        int(
            dedup_df[
                "multi_source_consensus"
            ].sum()
        )
    )

    print(
        "Repeated pairs:",
        int(
            dedup_df[
                "repeated_pair"
            ].sum()
        )
    )


    # ========================================================
    # 14. Benchmark leakage
    # ========================================================

    print(
        "\nLoading benchmark registry..."
    )


    (
        benchmark_pair_ids,
        benchmark_files,
    ) = load_benchmark_pair_ids()


    print(
        "Benchmark files:",
        len(benchmark_files)
    )

    print(
        "Benchmark unique pairs:",
        len(benchmark_pair_ids)
    )


    dedup_df[
        "benchmark_leak"
    ] = (
        dedup_df[
            "normalized_pair_id"
        ]
        .isin(
            benchmark_pair_ids
        )
    )


    print(
        "Benchmark leaks:",
        int(
            dedup_df[
                "benchmark_leak"
            ].sum()
        )
    )


    # ========================================================
    # 15. Basic quality flags
    # ========================================================

    dedup_df[
        "valid_word_length"
    ] = (

        dedup_df[
            "source_word_count"
        ]
        .between(
            MIN_WORDS,
            MAX_WORDS,
        )

        &

        dedup_df[
            "target_word_count"
        ]
        .between(
            MIN_WORDS,
            MAX_WORDS,
        )
    )


    dedup_df[
        "valid_char_length"
    ] = (

        dedup_df[
            "source_char_count"
        ]
        .between(
            MIN_CHARS,
            MAX_CHARS,
        )

        &

        dedup_df[
            "target_char_count"
        ]
        .between(
            MIN_CHARS,
            MAX_CHARS,
        )
    )


    dedup_df[
        "valid_length_ratio"
    ] = (
        dedup_df[
            "length_ratio"
        ]
        .between(
            MIN_CHAR_RATIO,
            MAX_CHAR_RATIO,
        )
    )


    # ========================================================
    # 16. Cyrillic policy
    #
    # 当前 Student 目标是 Latin Uzbek。
    #
    # 不删除原文件，
    # 但这批样本暂时不进入下一阶段。
    # ========================================================

    dedup_df[
        "latin_uzbek"
    ] = (
        ~dedup_df[
            "uz_has_cyrillic"
        ]
    )


    # ========================================================
    # 17. Rejection reasons
    # ========================================================

    def build_reject_reason(row):

        reasons = []

        if row[
            "normalized_empty"
        ]:

            reasons.append(
                "EMPTY"
            )

        if not row[
            "valid_word_length"
        ]:

            reasons.append(
                "WORD_LENGTH"
            )

        if not row[
            "valid_char_length"
        ]:

            reasons.append(
                "CHAR_LENGTH"
            )

        if not row[
            "valid_length_ratio"
        ]:

            reasons.append(
                "LENGTH_RATIO"
            )

        if not row[
            "latin_uzbek"
        ]:

            reasons.append(
                "CYRILLIC_UZ"
            )

        if row[
            "benchmark_leak"
        ]:

            reasons.append(
                "BENCHMARK_LEAK"
            )

        return "|".join(
            reasons
        )


    print(
        "\nBuilding reject reasons..."
    )


    dedup_df[
        "reject_reason"
    ] = dedup_df.apply(
        build_reject_reason,
        axis=1,
    )


    dedup_df[
        "step03_pass"
    ] = (
        dedup_df[
            "reject_reason"
        ]
        ==
        ""
    )


    # ========================================================
    # 18. Split accepted/rejected
    # ========================================================

    accepted_df = (
        dedup_df[
            dedup_df[
                "step03_pass"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    rejected_df = (
        dedup_df[
            ~dedup_df[
                "step03_pass"
            ]
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 19. Save
    # ========================================================

    print(
        "\nSaving accepted data..."
    )

    accepted_df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )


    accepted_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    rejected_df.to_parquet(
        REJECTED_FILE,
        index=False,
    )


    # ========================================================
    # 20. Rejection statistics
    # ========================================================

    rejection_stats = {

        "normalized_empty":
            int(
                dedup_df[
                    "normalized_empty"
                ].sum()
            ),

        "invalid_word_length":
            int(
                (
                    ~dedup_df[
                        "valid_word_length"
                    ]
                ).sum()
            ),

        "invalid_char_length":
            int(
                (
                    ~dedup_df[
                        "valid_char_length"
                    ]
                ).sum()
            ),

        "invalid_length_ratio":
            int(
                (
                    ~dedup_df[
                        "valid_length_ratio"
                    ]
                ).sum()
            ),

        "cyrillic_uz":
            int(
                (
                    ~dedup_df[
                        "latin_uzbek"
                    ]
                ).sum()
            ),

        "benchmark_leak":
            int(
                dedup_df[
                    "benchmark_leak"
                ].sum()
            ),
    }


    # ========================================================
    # 21. Report
    # ========================================================

    report = {

        "pipeline":
            "en_uz",

        "step":
            "03_dedup_quality_filter",

        "input_rows":
            len(df),

        "unique_normalized_pairs":
            int(
                unique_before
            ),

        "duplicates_removed":
            int(
                len(df)
                -
                len(dedup_df)
            ),

        "multi_source_pairs":
            int(
                dedup_df[
                    "multi_source_consensus"
                ].sum()
            ),

        "repeated_pairs":
            int(
                dedup_df[
                    "repeated_pair"
                ].sum()
            ),

        "benchmark_files":
            [
                str(
                    p.relative_to(
                        PROJECT_ROOT
                    )
                )
                for p in benchmark_files
            ],

        "benchmark_unique_pairs":
            len(
                benchmark_pair_ids
            ),

        "rejection_stats":
            rejection_stats,

        "accepted_rows":
            len(
                accepted_df
            ),

        "rejected_rows":
            len(
                rejected_df
            ),

        "accepted_rate_percent":
            float(
                len(accepted_df)
                /
                len(dedup_df)
                *
                100
            ),
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
    # 22. Terminal report
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 03 COMPLETE")
    print("=" * 100)


    print(
        "Input rows:",
        len(df)
    )

    print(
        "Unique normalized pairs:",
        len(dedup_df)
    )

    print(
        "Duplicates removed:",
        len(df)
        -
        len(dedup_df)
    )


    print("\nConsensus:")

    print(
        "Repeated pairs:",
        int(
            dedup_df[
                "repeated_pair"
            ].sum()
        )
    )

    print(
        "Multi-source pairs:",
        int(
            dedup_df[
                "multi_source_consensus"
            ].sum()
        )
    )


    print("\nRejected reasons:")

    for key, value in (
        rejection_stats.items()
    ):

        print(
            f"{key:<24}: "
            f"{value}"
        )


    print("\nFinal:")

    print(
        "Accepted:",
        len(accepted_df)
    )

    print(
        "Rejected:",
        len(rejected_df)
    )

    print(
        "Accepted rate:",
        f"{len(accepted_df) / len(dedup_df) * 100:.2f}%"
    )


    print(
        "\nSource consensus distribution:"
    )

    print(
        accepted_df[
            "source_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    print("\nFiles:")

    print(
        OUTPUT_PARQUET
    )

    print(
        OUTPUT_CSV
    )

    print(
        REJECTED_FILE
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()