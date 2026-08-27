from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Thresholds
#
# These are intentionally BROAD.
# Step 14C removes only obviously bad data.
# Borderline semantic-quality cases are kept for Step 14D.
# ============================================================

MIN_EN_WORDS = 1
MAX_EN_WORDS = 100

MIN_EN_CHARS = 1
MAX_EN_CHARS = 700

MIN_ZH_CJK = 1
MAX_ZH_CJK = 250

# Script purity.
#
# English should mostly be Latin.
# Chinese may legitimately contain English names,
# acronyms, model names, URLs, etc., so threshold is looser.
MIN_EN_LATIN_RATIO = 0.70
MIN_ZH_CJK_RATIO = 0.45

# Approximate cross-language length relationship:
#
# Chinese CJK characters / English words
#
# Deliberately broad. This is only a hard anomaly gate.
MIN_ZH_CJK_PER_EN_WORD = 0.20
MAX_ZH_CJK_PER_EN_WORD = 8.00

# Short examples get a wider ratio exemption.
SHORT_EN_WORD_THRESHOLD = 3

# Reject pathological repeated characters.
MAX_IDENTICAL_CHAR_RUN = 8

# Very obvious markup.
HTML_RE = re.compile(
    r"<\s*/?\s*(?:html|body|script|style|div|span|table|tr|td|p|br|iframe)\b",
    flags=re.IGNORECASE,
)

CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)

REPEATED_CHAR_RE = re.compile(
    rf"(.)\1{{{MAX_IDENTICAL_CHAR_RUN},}}"
)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14C - Hard filter normalized "
            "ZH-EN human parallel corpus."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Helpers
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


def english_word_count(
    text: str,
) -> int:

    return len(
        [
            token
            for token in str(text).split()
            if token
        ]
    )


def script_ratios(
    en: str,
    zh: str,
) -> tuple[float, float]:

    en_latin = count_latin(en)
    en_cjk = count_cjk(en)

    zh_latin = count_latin(zh)
    zh_cjk = count_cjk(zh)

    en_units = (
        en_latin
        +
        en_cjk
    )

    zh_units = (
        zh_latin
        +
        zh_cjk
    )

    en_latin_ratio = (
        en_latin
        /
        max(
            en_units,
            1,
        )
    )

    zh_cjk_ratio = (
        zh_cjk
        /
        max(
            zh_units,
            1,
        )
    )

    return (
        en_latin_ratio,
        zh_cjk_ratio,
    )


def contains_html(
    text: str,
) -> bool:

    return bool(
        HTML_RE.search(
            str(text)
        )
    )


def excessive_repetition(
    text: str,
) -> bool:

    return bool(
        REPEATED_CHAR_RE.search(
            str(text)
        )
    )


# ============================================================
# Determine ONE primary rejection reason
#
# Ordering matters:
# earlier rules have priority.
# ============================================================

def get_rejection_reason(
    row: pd.Series,
) -> str | None:

    en = str(
        row["en"]
    ).strip()

    zh = str(
        row["zh"]
    ).strip()

    # --------------------------------------------------------
    # Empty
    # --------------------------------------------------------

    if not en:
        return "EMPTY_EN"

    if not zh:
        return "EMPTY_ZH"

    # --------------------------------------------------------
    # Unicode replacement char
    # --------------------------------------------------------

    if "\ufffd" in en:
        return "REPLACEMENT_CHAR_EN"

    if "\ufffd" in zh:
        return "REPLACEMENT_CHAR_ZH"

    # --------------------------------------------------------
    # Exact same text
    # --------------------------------------------------------

    if (
        en.casefold()
        ==
        zh.casefold()
    ):
        return "SOURCE_TARGET_SAME"

    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    if contains_html(en):
        return "HTML_EN"

    if contains_html(zh):
        return "HTML_ZH"

    # --------------------------------------------------------
    # Repetition
    # --------------------------------------------------------

    if excessive_repetition(en):
        return "EXCESSIVE_REPETITION_EN"

    if excessive_repetition(zh):
        return "EXCESSIVE_REPETITION_ZH"

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    en_words = int(
        row["en_words"]
    )

    en_chars = int(
        row["en_chars"]
    )

    zh_cjk = int(
        row["zh_cjk_chars"]
    )

    if en_words < MIN_EN_WORDS:
        return "EN_TOO_SHORT"

    if en_words > MAX_EN_WORDS:
        return "EN_TOO_LONG"

    if en_chars < MIN_EN_CHARS:
        return "EN_CHAR_TOO_SHORT"

    if en_chars > MAX_EN_CHARS:
        return "EN_CHAR_TOO_LONG"

    if zh_cjk < MIN_ZH_CJK:
        return "ZH_TOO_SHORT"

    if zh_cjk > MAX_ZH_CJK:
        return "ZH_TOO_LONG"

    # --------------------------------------------------------
    # Script ratio
    # --------------------------------------------------------

    (
        en_latin_ratio,
        zh_cjk_ratio,
    ) = script_ratios(
        en,
        zh,
    )

    if (
        en_latin_ratio
        <
        MIN_EN_LATIN_RATIO
    ):
        return "EN_LOW_LATIN_RATIO"

    if (
        zh_cjk_ratio
        <
        MIN_ZH_CJK_RATIO
    ):
        return "ZH_LOW_CJK_RATIO"

    # --------------------------------------------------------
    # Very broad bilingual length ratio
    #
    # Do not use for very short sentences.
    # e.g. Fine! <-> 好吧
    # --------------------------------------------------------

    if (
        en_words
        >
        SHORT_EN_WORD_THRESHOLD
    ):

        ratio = (
            zh_cjk
            /
            max(
                en_words,
                1,
            )
        )

        if (
            ratio
            <
            MIN_ZH_CJK_PER_EN_WORD
        ):
            return "LENGTH_RATIO_TOO_LOW"

        if (
            ratio
            >
            MAX_ZH_CJK_PER_EN_WORD
        ):
            return "LENGTH_RATIO_TOO_HIGH"

    return None


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

    input_file = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14b_normalized"
        /
        "parallel_normalized_v1.parquet"
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
        "14c_filtered"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    accepted_file = (
        output_dir
        /
        "parallel_hard_filtered_v1.parquet"
    )

    accepted_csv = (
        output_dir
        /
        "parallel_hard_filtered_v1.csv"
    )

    rejected_file = (
        output_dir
        /
        "hard_rejected_v1.parquet"
    )

    rejected_csv = (
        output_dir
        /
        "hard_rejected_v1.csv"
    )

    duplicate_file = (
        output_dir
        /
        "normalized_duplicates_removed_v1.parquet"
    )

    source_report_file = (
        output_dir
        /
        "hard_filter_by_source_v1.csv"
    )

    reason_report_file = (
        output_dir
        /
        "hard_filter_reason_report_v1.csv"
    )

    report_file = (
        output_dir
        /
        "hard_filter_report_v1.json"
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14C - HARD FILTER "
        "ZH-EN HUMAN PARALLEL CORPUS"
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

    if (
        accepted_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{accepted_file}\n"
            "Use --overwrite to rebuild."
        )

    # ========================================================
    # Load
    # ========================================================

    df = pd.read_parquet(
        input_file
    )

    required = {

        "pair_id",

        "en",
        "zh",

        "source_dataset",

        "en_words",
        "en_chars",

        "zh_chars",
        "zh_cjk_chars",

        "en_latin_chars",
        "en_cjk_chars",

        "zh_latin_chars",

        "normalized_pair_hash",
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
            "Missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "\nInput rows:",
        len(df)
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
    # Step 1:
    # normalized exact dedup
    # ========================================================

    duplicated_mask = (
        df[
            "normalized_pair_hash"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_group_rows = int(
        duplicated_mask.sum()
    )

    # Prefer ALT when the exact same normalized pair exists,
    # because ALT is a more curated parallel corpus.
    #
    # This affects only exact normalized duplicates.
    source_priority = {

        "ALT":
            0,

        "Tatoeba":
            1,
    }

    df[
        "_source_priority"
    ] = (
        df[
            "source_dataset"
        ]
        .map(
            source_priority
        )
        .fillna(
            99
        )
        .astype(int)
    )

    df = (
        df
        .sort_values(
            [
                "normalized_pair_hash",
                "_source_priority",
                "pair_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicate_removed_mask = (
        df[
            "normalized_pair_hash"
        ]
        .duplicated(
            keep="first"
        )
    )

    duplicates_removed = (
        df[
            duplicate_removed_mask
        ]
        .copy()
    )

    deduped = (
        df[
            ~duplicate_removed_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    deduped = deduped.drop(
        columns=[
            "_source_priority"
        ],
        errors="ignore",
    )

    duplicates_removed = (
        duplicates_removed.drop(
            columns=[
                "_source_priority"
            ],
            errors="ignore",
        )
    )

    print("\n")
    print(
        "NORMALIZED DEDUP"
    )

    print(
        "Duplicate group rows:",
        duplicate_group_rows
    )

    print(
        "Duplicates removed:",
        len(
            duplicates_removed
        )
    )

    print(
        "Rows after dedup:",
        len(
            deduped
        )
    )

    # ========================================================
    # Step 2:
    # Hard rejection
    # ========================================================

    reasons = []

    for _, row in deduped.iterrows():

        reasons.append(
            get_rejection_reason(
                row
            )
        )

    deduped[
        "hard_reject_reason"
    ] = reasons

    reject_mask = (
        deduped[
            "hard_reject_reason"
        ]
        .notna()
    )

    rejected = (
        deduped[
            reject_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    accepted = (
        deduped[
            ~reject_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    accepted = accepted.drop(
        columns=[
            "hard_reject_reason"
        ],
        errors="ignore",
    )

    # ========================================================
    # Add diagnostics to accepted
    # ========================================================

    accepted[
        "en_latin_ratio"
    ] = [

        script_ratios(
            en,
            zh,
        )[0]

        for (
            en,
            zh,
        )
        in zip(
            accepted["en"],
            accepted["zh"],
        )
    ]

    accepted[
        "zh_cjk_ratio"
    ] = [

        script_ratios(
            en,
            zh,
        )[1]

        for (
            en,
            zh,
        )
        in zip(
            accepted["en"],
            accepted["zh"],
        )
    ]

    accepted[
        "zh_cjk_per_en_word"
    ] = (

        accepted[
            "zh_cjk_chars"
        ]

        /

        accepted[
            "en_words"
        ]
        .clip(
            lower=1
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    assert not (
        accepted[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
    )

    assert (
        accepted["en"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    assert (
        accepted["zh"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    # ========================================================
    # Reason report
    # ========================================================

    reason_counts = (
        rejected[
            "hard_reject_reason"
        ]
        .value_counts()
    )

    reason_report = (
        reason_counts
        .rename_axis(
            "reason"
        )
        .reset_index(
            name="count"
        )
    )

    if len(
        reason_report
    ) > 0:

        reason_report[
            "percent_of_input"
        ] = (

            reason_report[
                "count"
            ]

            /

            len(df)

            *
            100
        )

    # ========================================================
    # Source report
    # ========================================================

    source_rows = []

    all_sources = sorted(
        set(
            df[
                "source_dataset"
            ]
            .astype(str)
        )
    )

    for source in all_sources:

        input_count = int(
            (
                df[
                    "source_dataset"
                ]
                ==
                source
            )
            .sum()
        )

        accepted_count = int(
            (
                accepted[
                    "source_dataset"
                ]
                ==
                source
            )
            .sum()
        )

        rejected_count = int(
            (
                rejected[
                    "source_dataset"
                ]
                ==
                source
            )
            .sum()
        )

        duplicate_count = int(
            (
                duplicates_removed[
                    "source_dataset"
                ]
                ==
                source
            )
            .sum()
        )

        source_rows.append({

            "source_dataset":
                source,

            "input_rows":
                input_count,

            "accepted_rows":
                accepted_count,

            "hard_rejected_rows":
                rejected_count,

            "normalized_duplicates_removed":
                duplicate_count,

            "accepted_percent":
                (
                    accepted_count
                    /
                    max(
                        input_count,
                        1,
                    )
                    *
                    100
                ),
        })

    source_report = pd.DataFrame(
        source_rows
    )

    # ========================================================
    # Save
    # ========================================================

    accepted.to_parquet(
        accepted_file,
        index=False,
    )

    accepted.to_csv(
        accepted_csv,
        index=False,
        encoding="utf-8-sig",
    )

    if len(
        rejected
    ) > 0:

        rejected.to_parquet(
            rejected_file,
            index=False,
        )

        rejected.to_csv(
            rejected_csv,
            index=False,
            encoding="utf-8-sig",
        )

    if len(
        duplicates_removed
    ) > 0:

        duplicates_removed.to_parquet(
            duplicate_file,
            index=False,
        )

    source_report.to_csv(
        source_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    reason_report.to_csv(
        reason_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "14C",

        "pipeline":
            "zh_en_exp1_v1",

        "input_rows":
            int(
                len(df)
            ),

        "thresholds": {

            "min_en_words":
                MIN_EN_WORDS,

            "max_en_words":
                MAX_EN_WORDS,

            "max_en_chars":
                MAX_EN_CHARS,

            "min_zh_cjk":
                MIN_ZH_CJK,

            "max_zh_cjk":
                MAX_ZH_CJK,

            "min_en_latin_ratio":
                MIN_EN_LATIN_RATIO,

            "min_zh_cjk_ratio":
                MIN_ZH_CJK_RATIO,

            "min_zh_cjk_per_en_word":
                MIN_ZH_CJK_PER_EN_WORD,

            "max_zh_cjk_per_en_word":
                MAX_ZH_CJK_PER_EN_WORD,

            "short_en_word_ratio_exemption":
                SHORT_EN_WORD_THRESHOLD,

            "max_identical_char_run":
                MAX_IDENTICAL_CHAR_RUN,
        },

        "normalized_dedup": {

            "duplicate_group_rows":
                int(
                    duplicate_group_rows
                ),

            "duplicates_removed":
                int(
                    len(
                        duplicates_removed
                    )
                ),

            "rows_after_dedup":
                int(
                    len(
                        deduped
                    )
                ),
        },

        "hard_filter": {

            "accepted":
                int(
                    len(
                        accepted
                    )
                ),

            "rejected":
                int(
                    len(
                        rejected
                    )
                ),

            "accepted_percent":
                float(
                    len(
                        accepted
                    )
                    /
                    max(
                        len(df),
                        1,
                    )
                    *
                    100
                ),

            "rejection_reasons":
                {

                    str(key):
                        int(value)

                    for (
                        key,
                        value,
                    )
                    in (
                        reason_counts
                        .items()
                    )
                },
        },

        "source_report":
            source_rows,

        "final_assertions": {

            "no_empty_en":
                True,

            "no_empty_zh":
                True,

            "no_normalized_pair_duplicates":
                True,
        },

        "outputs": {

            "accepted":
                str(
                    accepted_file
                ),

            "rejected":
                (
                    str(
                        rejected_file
                    )
                    if len(
                        rejected
                    ) > 0
                    else None
                ),

            "duplicates_removed":
                (
                    str(
                        duplicate_file
                    )
                    if len(
                        duplicates_removed
                    ) > 0
                    else None
                ),

            "source_report":
                str(
                    source_report_file
                ),

            "reason_report":
                str(
                    reason_report_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "HARD_FILTER_READY_FOR_RISK_ROUTING",
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
        "STEP 14C RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput rows:",
        len(df)
    )

    print(
        "Duplicates removed:",
        len(
            duplicates_removed
        )
    )

    print(
        "Hard rejected:",
        len(
            rejected
        )
    )

    print(
        "Accepted:",
        len(
            accepted
        )
    )

    print(
        "Accepted rate:",
        f"{len(accepted) / max(len(df), 1) * 100:.2f}%"
    )

    print(
        "\nRejection reasons:"
    )

    if len(
        reason_report
    ) > 0:

        print(
            reason_report
            .round(4)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )

    print(
        "\nSource report:"
    )

    print(
        source_report
        .round(3)
        .to_string(
            index=False
        )
    )

    print(
        "\nAccepted diagnostics:"
    )

    print(
        "Average EN words:",
        f"{accepted['en_words'].mean():.2f}"
    )

    print(
        "Average ZH CJK:",
        f"{accepted['zh_cjk_chars'].mean():.2f}"
    )

    print(
        "Average EN Latin ratio:",
        f"{accepted['en_latin_ratio'].mean():.4f}"
    )

    print(
        "Average ZH CJK ratio:",
        f"{accepted['zh_cjk_ratio'].mean():.4f}"
    )

    print(
        "Average ZH CJK / EN word:",
        f"{accepted['zh_cjk_per_en_word'].mean():.4f}"
    )

    print(
        "\nOutput:"
    )

    print(
        accepted_file
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
        "HARD_FILTER_READY_FOR_RISK_ROUTING"
    )


if __name__ == "__main__":

    main()