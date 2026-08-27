from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from opencc import OpenCC


# ============================================================
# Constants
# ============================================================

CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)

WHITESPACE_RE = re.compile(
    r"\s+"
)

ZERO_WIDTH_CHARS = {
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM
}

QUOTE_MAP = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u2032": "'",

    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2033": '"',
})

DASH_MAP = str.maketrans({
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
})


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14B - Normalize collected "
            "ZH-EN human parallel corpus."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Generic cleanup
# ============================================================

def remove_zero_width(
    text: str,
) -> str:

    for char in ZERO_WIDTH_CHARS:
        text = text.replace(
            char,
            "",
        )

    return text


def remove_control_characters(
    text: str,
) -> tuple[str, int]:

    output = []

    removed = 0

    for char in text:

        category = (
            unicodedata
            .category(
                char
            )
        )

        # Unicode category Cc:
        # control characters
        #
        # Keep no control characters here.
        # Newlines/tabs are already sentence-level noise.
        if category == "Cc":

            removed += 1

            continue

        output.append(
            char
        )

    return (
        "".join(
            output
        ),
        removed,
    )


def normalize_common(
    text: str,
) -> tuple[str, dict]:

    raw = (
        ""
        if text is None
        else str(text)
    )

    stats = {

        "had_replacement_char":
            "\ufffd" in raw,

        "zero_width_removed":
            0,

        "control_chars_removed":
            0,
    }

    text = unicodedata.normalize(
        "NFKC",
        raw,
    )

    for char in ZERO_WIDTH_CHARS:

        count = text.count(
            char
        )

        if count:

            stats[
                "zero_width_removed"
            ] += count

            text = text.replace(
                char,
                "",
            )

    (
        text,
        control_removed,
    ) = remove_control_characters(
        text
    )

    stats[
        "control_chars_removed"
    ] = control_removed

    # NBSP
    text = text.replace(
        "\u00a0",
        " ",
    )

    text = (
        text
        .translate(
            QUOTE_MAP
        )
        .translate(
            DASH_MAP
        )
    )

    text = WHITESPACE_RE.sub(
        " ",
        text,
    )

    text = text.strip()

    return (
        text,
        stats,
    )


# ============================================================
# English normalization
# ============================================================

def normalize_english(
    text: str,
) -> tuple[str, dict]:

    (
        text,
        common_stats,
    ) = normalize_common(
        text
    )

    return (
        text,
        common_stats,
    )


# ============================================================
# Chinese normalization
# ============================================================

def normalize_chinese(
    text: str,
    opencc: OpenCC,
) -> tuple[str, dict]:

    (
        text,
        common_stats,
    ) = normalize_common(
        text
    )

    before_t2s = text

    simplified = (
        opencc.convert(
            text
        )
    )

    simplified = WHITESPACE_RE.sub(
        " ",
        simplified,
    ).strip()

    common_stats[
        "t2s_changed"
    ] = (
        simplified
        !=
        before_t2s
    )

    common_stats[
        "before_t2s"
    ] = before_t2s

    return (
        simplified,
        common_stats,
    )


# ============================================================
# Statistics
# ============================================================

def count_cjk(
    text: str,
) -> int:

    return len(
        CJK_RE.findall(
            str(text)
        )
    )


def count_latin(
    text: str,
) -> int:

    return len(
        LATIN_RE.findall(
            str(text)
        )
    )


def count_english_words(
    text: str,
) -> int:

    return len(
        [
            token
            for token
            in str(text).split()
            if token
        ]
    )


def normalized_en_key(
    text: str,
) -> str:

    # English comparison can safely be
    # case-insensitive for leakage protection.
    return (
        str(text)
        .strip()
        .casefold()
    )


def normalized_zh_key(
    text: str,
) -> str:

    return (
        str(text)
        .strip()
    )


def pair_hash(
    en: str,
    zh: str,
) -> str:

    value = (
        normalized_en_key(
            en
        )
        +
        "\n"
        +
        normalized_zh_key(
            zh
        )
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Protected evaluation sets
#
# IMPORTANT:
# Apply EXACT SAME normalization pipeline
# as training corpus.
# ============================================================

def load_protected_evaluation(
    project_root: Path,
    opencc: OpenCC,
):

    benchmark_dir = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
    )

    protected_files = {

        "flores_dev":
            benchmark_dir
            /
            "flores_plus_zh_en_dev_v1.parquet",

        "flores_devtest":
            benchmark_dir
            /
            "flores_plus_zh_en_devtest_v1.parquet",

        "tatoeba_frozen":
            benchmark_dir
            /
            "tatoeba_zh_en_test_v1.parquet",
    }

    protected_pairs = set()
    protected_en = set()
    protected_zh = set()

    reports = {}

    for (
        name,
        path,
    ) in protected_files.items():

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        df = pd.read_parquet(
            path
        )

        if (
            "en" not in df.columns
            or
            "zh" not in df.columns
        ):

            raise RuntimeError(
                f"{name} missing en/zh columns."
            )

        normalized_count = 0

        for (
            en_raw,
            zh_raw,
        ) in zip(
            df["en"],
            df["zh"],
        ):

            (
                en,
                _,
            ) = normalize_english(
                en_raw
            )

            (
                zh,
                _,
            ) = normalize_chinese(
                zh_raw,
                opencc,
            )

            en_key = (
                normalized_en_key(
                    en
                )
            )

            zh_key = (
                normalized_zh_key(
                    zh
                )
            )

            protected_pairs.add(
                (
                    en_key,
                    zh_key,
                )
            )

            protected_en.add(
                en_key
            )

            protected_zh.add(
                zh_key
            )

            normalized_count += 1

        reports[
            name
        ] = {

            "file":
                str(
                    path
                ),

            "rows":
                int(
                    len(df)
                ),

            "normalized_rows":
                int(
                    normalized_count
                ),
        }

    return (
        protected_pairs,
        protected_en,
        protected_zh,
        reports,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # script:
    # scripts/pipeline/zh_en/14b_...
    #
    # parents[3] = project root

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    input_file = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14a_collected"
        /
        "parallel_collected_v1.parquet"
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
        "14b_normalized"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalized_file = (
        output_dir
        /
        "parallel_normalized_v1.parquet"
    )

    normalized_csv = (
        output_dir
        /
        "parallel_normalized_v1.csv"
    )

    rejected_file = (
        output_dir
        /
        "post_normalization_leakage_rejected_v1.parquet"
    )

    source_stats_file = (
        output_dir
        /
        "normalization_stats_by_source_v1.csv"
    )

    report_file = (
        output_dir
        /
        "normalization_report_v1.json"
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14B - NORMALIZE "
        "ZH-EN HUMAN PARALLEL CORPUS"
    )

    print(
        "=" * 110
    )

    print(
        "\nProject root:"
    )

    print(
        project_root
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

    if (
        normalized_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{normalized_file}\n\n"
            "Use --overwrite if you "
            "intentionally want to rebuild."
        )

    # ========================================================
    # OpenCC
    # ========================================================

    print(
        "\nLoading OpenCC t2s..."
    )

    opencc = OpenCC(
        "t2s"
    )

    print(
        "OpenCC loaded."
    )

    # ========================================================
    # Load corpus
    # ========================================================

    df = pd.read_parquet(
        input_file
    )

    required_columns = {

        "pair_id",
        "en",
        "zh",
        "source_dataset",
    }

    missing = (
        required_columns
        -
        set(
            df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Input missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "\nInput rows:",
        len(df)
    )

    print(
        "\nSource distribution:"
    )

    print(
        df[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Preserve pre-normalization text
    # ========================================================

    df[
        "en_before_normalization"
    ] = df[
        "en"
    ].astype(str)

    df[
        "zh_before_normalization"
    ] = df[
        "zh"
    ].astype(str)

    # ========================================================
    # Normalize
    # ========================================================

    normalized_en = []
    normalized_zh = []

    en_replacement_char = []
    zh_replacement_char = []

    en_zero_width_removed = []
    zh_zero_width_removed = []

    en_control_removed = []
    zh_control_removed = []

    zh_t2s_changed = []

    for (
        en_raw,
        zh_raw,
    ) in zip(
        df[
            "en_before_normalization"
        ],
        df[
            "zh_before_normalization"
        ],
    ):

        (
            en,
            en_stats,
        ) = normalize_english(
            en_raw
        )

        (
            zh,
            zh_stats,
        ) = normalize_chinese(
            zh_raw,
            opencc,
        )

        normalized_en.append(
            en
        )

        normalized_zh.append(
            zh
        )

        en_replacement_char.append(
            bool(
                en_stats[
                    "had_replacement_char"
                ]
            )
        )

        zh_replacement_char.append(
            bool(
                zh_stats[
                    "had_replacement_char"
                ]
            )
        )

        en_zero_width_removed.append(
            int(
                en_stats[
                    "zero_width_removed"
                ]
            )
        )

        zh_zero_width_removed.append(
            int(
                zh_stats[
                    "zero_width_removed"
                ]
            )
        )

        en_control_removed.append(
            int(
                en_stats[
                    "control_chars_removed"
                ]
            )
        )

        zh_control_removed.append(
            int(
                zh_stats[
                    "control_chars_removed"
                ]
            )
        )

        zh_t2s_changed.append(
            bool(
                zh_stats[
                    "t2s_changed"
                ]
            )
        )

    df["en"] = (
        normalized_en
    )

    df["zh"] = (
        normalized_zh
    )

    df[
        "en_had_replacement_char"
    ] = (
        en_replacement_char
    )

    df[
        "zh_had_replacement_char"
    ] = (
        zh_replacement_char
    )

    df[
        "en_zero_width_removed"
    ] = (
        en_zero_width_removed
    )

    df[
        "zh_zero_width_removed"
    ] = (
        zh_zero_width_removed
    )

    df[
        "en_control_chars_removed"
    ] = (
        en_control_removed
    )

    df[
        "zh_control_chars_removed"
    ] = (
        zh_control_removed
    )

    df[
        "zh_t2s_changed"
    ] = (
        zh_t2s_changed
    )

    # ========================================================
    # Text statistics
    # ========================================================

    df[
        "en_chars"
    ] = (
        df["en"]
        .astype(str)
        .str.len()
    )

    df[
        "zh_chars"
    ] = (
        df["zh"]
        .astype(str)
        .str.len()
    )

    df[
        "en_words"
    ] = [
        count_english_words(
            text
        )
        for text
        in df["en"]
    ]

    df[
        "en_latin_chars"
    ] = [
        count_latin(
            text
        )
        for text
        in df["en"]
    ]

    df[
        "en_cjk_chars"
    ] = [
        count_cjk(
            text
        )
        for text
        in df["en"]
    ]

    df[
        "zh_cjk_chars"
    ] = [
        count_cjk(
            text
        )
        for text
        in df["zh"]
    ]

    df[
        "zh_latin_chars"
    ] = [
        count_latin(
            text
        )
        for text
        in df["zh"]
    ]

    df[
        "normalized_pair_hash"
    ] = [
        pair_hash(
            en,
            zh,
        )
        for (
            en,
            zh,
        )
        in zip(
            df["en"],
            df["zh"],
        )
    ]

    # ========================================================
    # Empty diagnostics
    #
    # Do NOT remove here.
    # Step 14C owns hard filtering.
    # ========================================================

    empty_en = int(
        df[
            "en"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    empty_zh = int(
        df[
            "zh"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    # ========================================================
    # Duplicate diagnostics after normalization
    #
    # Do NOT remove here.
    # Step 14C owns dedup.
    # ========================================================

    duplicate_pair_rows = int(
        df[
            "normalized_pair_hash"
        ]
        .duplicated(
            keep=False
        )
        .sum()
    )

    unique_pair_duplicate_count = int(
        df[
            "normalized_pair_hash"
        ]
        .duplicated(
            keep="first"
        )
        .sum()
    )

    # ========================================================
    # Load protected benchmark
    # after EXACT SAME normalization.
    # ========================================================

    (
        protected_pairs,
        protected_en,
        protected_zh,
        protected_reports,
    ) = (
        load_protected_evaluation(
            project_root,
            opencc,
        )
    )

    print(
        "\nProtected normalized evaluation:"
    )

    print(
        "Pairs:",
        len(
            protected_pairs
        )
    )

    print(
        "English:",
        len(
            protected_en
        )
    )

    print(
        "Chinese:",
        len(
            protected_zh
        )
    )

    # ========================================================
    # Post-normalization leakage check
    # ========================================================

    accepted_rows = []
    rejected_rows = []

    pair_hits = 0
    en_hits = 0
    zh_hits = 0

    new_pair_hits = 0

    for _, row in df.iterrows():

        en_key = (
            normalized_en_key(
                row["en"]
            )
        )

        zh_key = (
            normalized_zh_key(
                row["zh"]
            )
        )

        pair_hit = (
            (
                en_key,
                zh_key,
            )
            in
            protected_pairs
        )

        en_hit = (
            en_key
            in
            protected_en
        )

        zh_hit = (
            zh_key
            in
            protected_zh
        )

        if pair_hit:
            pair_hits += 1

        if en_hit:
            en_hits += 1

        if zh_hit:
            zh_hits += 1

        if (
            pair_hit
            or
            en_hit
            or
            zh_hit
        ):

            rejected = (
                row.to_dict()
            )

            rejected[
                "post_norm_pair_hit"
            ] = bool(
                pair_hit
            )

            rejected[
                "post_norm_en_hit"
            ] = bool(
                en_hit
            )

            rejected[
                "post_norm_zh_hit"
            ] = bool(
                zh_hit
            )

            rejected_rows.append(
                rejected
            )

        else:

            accepted_rows.append(
                row.to_dict()
            )

    clean_df = pd.DataFrame(
        accepted_rows
    )

    leakage_rejected_df = (
        pd.DataFrame(
            rejected_rows
        )
    )

    # ========================================================
    # Final leakage assertions
    # ========================================================

    final_pair_hits = 0
    final_en_hits = 0
    final_zh_hits = 0

    for (
        en,
        zh,
    ) in zip(
        clean_df["en"],
        clean_df["zh"],
    ):

        en_key = (
            normalized_en_key(
                en
            )
        )

        zh_key = (
            normalized_zh_key(
                zh
            )
        )

        if (
            en_key,
            zh_key,
        ) in protected_pairs:

            final_pair_hits += 1

        if en_key in protected_en:

            final_en_hits += 1

        if zh_key in protected_zh:

            final_zh_hits += 1

    assert (
        final_pair_hits == 0
    )

    assert (
        final_en_hits == 0
    )

    assert (
        final_zh_hits == 0
    )

    # ========================================================
    # Source statistics
    # ========================================================

    source_stats = []

    for (
        source_name,
        part,
    ) in df.groupby(
        "source_dataset"
    ):

        source_stats.append({

            "source_dataset":
                source_name,

            "rows":
                int(
                    len(part)
                ),

            "zh_t2s_changed":
                int(
                    part[
                        "zh_t2s_changed"
                    ]
                    .sum()
                ),

            "zh_t2s_changed_percent":
                float(
                    part[
                        "zh_t2s_changed"
                    ]
                    .mean()
                    *
                    100
                ),

            "en_replacement_char_rows":
                int(
                    part[
                        "en_had_replacement_char"
                    ]
                    .sum()
                ),

            "zh_replacement_char_rows":
                int(
                    part[
                        "zh_had_replacement_char"
                    ]
                    .sum()
                ),

            "avg_en_words":
                float(
                    part[
                        "en_words"
                    ]
                    .mean()
                ),

            "avg_en_chars":
                float(
                    part[
                        "en_chars"
                    ]
                    .mean()
                ),

            "avg_zh_chars":
                float(
                    part[
                        "zh_chars"
                    ]
                    .mean()
                ),

            "avg_zh_cjk_chars":
                float(
                    part[
                        "zh_cjk_chars"
                    ]
                    .mean()
                ),
        })

    source_stats_df = (
        pd.DataFrame(
            source_stats
        )
    )

    # ========================================================
    # Global statistics
    # ========================================================

    changed_t2s = int(
        df[
            "zh_t2s_changed"
        ]
        .sum()
    )

    replacement_en = int(
        df[
            "en_had_replacement_char"
        ]
        .sum()
    )

    replacement_zh = int(
        df[
            "zh_had_replacement_char"
        ]
        .sum()
    )

    total_zero_width_removed = int(
        df[
            "en_zero_width_removed"
        ]
        .sum()
        +
        df[
            "zh_zero_width_removed"
        ]
        .sum()
    )

    total_control_removed = int(
        df[
            "en_control_chars_removed"
        ]
        .sum()
        +
        df[
            "zh_control_chars_removed"
        ]
        .sum()
    )

    # ========================================================
    # Save
    # ========================================================

    clean_df.to_parquet(
        normalized_file,
        index=False,
    )

    clean_df.to_csv(
        normalized_csv,
        index=False,
        encoding="utf-8-sig",
    )

    if len(
        leakage_rejected_df
    ) > 0:

        leakage_rejected_df.to_parquet(
            rejected_file,
            index=False,
        )

    source_stats_df.to_csv(
        source_stats_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "14B",

        "pipeline":
            "zh_en_exp1_v1",

        "input_file":
            str(
                input_file
            ),

        "input_rows":
            int(
                len(df)
            ),

        "normalization_policy": {

            "unicode":
                "NFKC",

            "whitespace":
                "collapse + strip",

            "zero_width":
                "remove",

            "control_characters":
                "remove",

            "quotes":
                "normalize selected Unicode quotes",

            "dashes":
                "normalize selected Unicode dashes",

            "chinese_script":
                "OpenCC t2s",

            "lowercase_english":
                False,

            "quality_filtering":
                False,

            "deduplication":
                (
                    "diagnostic only; "
                    "actual dedup in Step 14C"
                ),
        },

        "normalization_stats": {

            "zh_t2s_changed_rows":
                int(
                    changed_t2s
                ),

            "zh_t2s_changed_percent":
                float(
                    changed_t2s
                    /
                    max(
                        len(df),
                        1,
                    )
                    *
                    100
                ),

            "empty_en_after_normalization":
                int(
                    empty_en
                ),

            "empty_zh_after_normalization":
                int(
                    empty_zh
                ),

            "english_replacement_char_rows":
                int(
                    replacement_en
                ),

            "chinese_replacement_char_rows":
                int(
                    replacement_zh
                ),

            "zero_width_characters_removed":
                int(
                    total_zero_width_removed
                ),

            "control_characters_removed":
                int(
                    total_control_removed
                ),

            "normalized_duplicate_rows":
                int(
                    duplicate_pair_rows
                ),

            "normalized_duplicates_beyond_first":
                int(
                    unique_pair_duplicate_count
                ),

            "average_en_words":
                float(
                    df[
                        "en_words"
                    ]
                    .mean()
                ),

            "average_en_chars":
                float(
                    df[
                        "en_chars"
                    ]
                    .mean()
                ),

            "average_zh_chars":
                float(
                    df[
                        "zh_chars"
                    ]
                    .mean()
                ),

            "average_zh_cjk_chars":
                float(
                    df[
                        "zh_cjk_chars"
                    ]
                    .mean()
                ),
        },

        "protected_evaluation":
            protected_reports,

        "post_normalization_leakage": {

            "pair_hits":
                int(
                    pair_hits
                ),

            "english_sentence_hits":
                int(
                    en_hits
                ),

            "chinese_sentence_hits":
                int(
                    zh_hits
                ),

            "rejected_rows":
                int(
                    len(
                        leakage_rejected_df
                    )
                ),
        },

        "final_rows":
            int(
                len(
                    clean_df
                )
            ),

        "final_assertions": {

            "protected_pair_leakage":
                int(
                    final_pair_hits
                ),

            "protected_en_leakage":
                int(
                    final_en_hits
                ),

            "protected_zh_leakage":
                int(
                    final_zh_hits
                ),
        },

        "source_stats":
            source_stats,

        "outputs": {

            "normalized_parquet":
                str(
                    normalized_file
                ),

            "normalized_csv":
                str(
                    normalized_csv
                ),

            "post_norm_leakage_rejected":
                (
                    str(
                        rejected_file
                    )
                    if len(
                        leakage_rejected_df
                    ) > 0
                    else None
                ),

            "source_stats":
                str(
                    source_stats_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "NORMALIZED_READY_FOR_HARD_FILTER",
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
        "STEP 14B RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput rows:",
        len(df)
    )

    print(
        "Final rows:",
        len(
            clean_df
        )
    )

    print()

    print(
        "Chinese t2s changed:",
        changed_t2s,
        f"({changed_t2s / max(len(df), 1) * 100:.2f}%)"
    )

    print()

    print(
        "Empty EN after normalization:",
        empty_en
    )

    print(
        "Empty ZH after normalization:",
        empty_zh
    )

    print()

    print(
        "Replacement-char rows:"
    )

    print(
        "EN:",
        replacement_en
    )

    print(
        "ZH:",
        replacement_zh
    )

    print()

    print(
        "Normalized duplicate rows:",
        duplicate_pair_rows
    )

    print(
        "Duplicates beyond first:",
        unique_pair_duplicate_count
    )

    print()

    print(
        "Average EN words:",
        f"{df['en_words'].mean():.2f}"
    )

    print(
        "Average EN chars:",
        f"{df['en_chars'].mean():.2f}"
    )

    print(
        "Average ZH chars:",
        f"{df['zh_chars'].mean():.2f}"
    )

    print(
        "Average ZH CJK chars:",
        f"{df['zh_cjk_chars'].mean():.2f}"
    )

    print("\n")
    print(
        "Post-normalization protected leakage:"
    )

    print(
        "Pair:",
        pair_hits
    )

    print(
        "EN  :",
        en_hits
    )

    print(
        "ZH  :",
        zh_hits
    )

    print(
        "Rejected rows:",
        len(
            leakage_rejected_df
        )
    )

    print("\n")
    print(
        "Final protected leakage:"
    )

    print(
        "Pair:",
        final_pair_hits
    )

    print(
        "EN  :",
        final_en_hits
    )

    print(
        "ZH  :",
        final_zh_hits
    )

    print("\n")
    print(
        "Source statistics:"
    )

    print(
        source_stats_df
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "Output:"
    )

    print(
        normalized_file
    )

    print()

    print(
        "Report:"
    )

    print(
        report_file
    )

    print()

    print(
        "STATUS:"
    )

    print(
        "NORMALIZED_READY_FOR_HARD_FILTER"
    )


if __name__ == "__main__":

    main()