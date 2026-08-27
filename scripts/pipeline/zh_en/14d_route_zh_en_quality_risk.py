from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Version
# ============================================================

ROUTING_VERSION = "14D_V3"


# ============================================================
# Patterns
# ============================================================

NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?\d+(?:[.,]\d+)*(?![\w])"
)

PERCENT_RE = re.compile(
    r"(?:\d+(?:\.\d+)?)\s*[%％]"
)

URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

REPEATED_PUNCT_RE = re.compile(
    r"([!?！？。，,.])\1{2,}"
)


# ============================================================
# English negation
#
# IMPORTANT:
# Use word-boundary regex rather than:
#
#     if "no" in text
#
# Otherwise:
#     known
#     another
#     northern
#
# can be falsely detected as negation.
# ============================================================

EN_NEGATION_PATTERNS = [

    r"\bnot\b",
    r"\bno\b",
    r"\bnever\b",
    r"\bnone\b",
    r"\bnothing\b",
    r"\bnobody\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bwithout\b",

    r"\bcannot\b",

    r"\bcan't\b",
    r"\bwon't\b",

    r"\bisn't\b",
    r"\baren't\b",
    r"\bwasn't\b",
    r"\bweren't\b",

    r"\bdon't\b",
    r"\bdoesn't\b",
    r"\bdidn't\b",

    r"\bhaven't\b",
    r"\bhasn't\b",
    r"\bhadn't\b",

    r"\bshouldn't\b",
    r"\bwouldn't\b",
    r"\bcouldn't\b",
    r"\bmustn't\b",
]


# ============================================================
# Chinese strong negation
#
# Do NOT use:
#
#     "不" in text
#     "无" in text
#
# because normal lexical words such as:
#
#     不同
#     不知道
#     无耻
#     无偿
#
# can be legitimate translations without representing
# a source-target negation mismatch.
#
# This list intentionally prefers PRECISION over recall.
# Borderline semantics will later be audited by Qwen.
# ============================================================

ZH_STRONG_NEGATION_PATTERNS = [

    r"没有",
    r"没能",
    r"没法",

    r"未能",
    r"未曾",

    r"从未",
    r"从不",
    r"从来不",

    r"不是",
    r"并非",
    r"并不",

    r"不能",
    r"不会",

    r"不再",
    r"不要",
    r"不必",

    r"无需",
    r"无法",

    r"不得",

    r"绝不",
    r"毫不",
]


# ============================================================
# Negative question rewrite
#
# Example:
#
# EN:
#   Isn't that mine?
#
# ZH:
#   那是我的吗？
#
# This is a perfectly natural translation although
# explicit surface negation disappears in Chinese.
# ============================================================

EN_NEGATIVE_QUESTION_RE = re.compile(
    r"\b("
    r"isn't|aren't|wasn't|weren't|"
    r"don't|doesn't|didn't|"
    r"can't|couldn't|"
    r"won't|wouldn't|"
    r"haven't|hasn't|hadn't|"
    r"shouldn't"
    r")\b",
    flags=re.IGNORECASE,
)


# ============================================================
# Currency markers
# ============================================================

EN_CURRENCY = [

    "$",
    "€",
    "£",
    "¥",

    "USD",
    "EUR",
    "GBP",
    "CNY",
    "RMB",

    "dollar",
    "dollars",

    "euro",
    "euros",

    "yuan",
]

ZH_CURRENCY = [

    "$",
    "€",
    "£",
    "¥",

    "美元",
    "欧元",
    "英镑",
    "人民币",

    "元",
    "块",
]


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14D V3 - Quality risk routing "
            "for ZH-EN human parallel corpus."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Number helpers
# ============================================================

def normalize_number(
    value: str,
) -> str:

    value = str(
        value
    )

    value = value.replace(
        ",",
        "",
    )

    return value


def extract_numbers(
    text: str,
) -> list[str]:

    values = NUMBER_RE.findall(
        str(
            text
        )
    )

    return sorted(
        normalize_number(
            value
        )
        for value
        in values
    )


# ============================================================
# Percentage helpers
# ============================================================

def extract_percentages(
    text: str,
) -> list[str]:

    matches = PERCENT_RE.findall(
        str(
            text
        )
    )

    values = []

    for item in matches:

        item = (
            item
            .replace(
                "％",
                "%"
            )
            .replace(
                " ",
                ""
            )
        )

        values.append(
            item
        )

    return sorted(
        values
    )


# ============================================================
# URL / email
# ============================================================

def extract_urls(
    text: str,
) -> list[str]:

    return sorted(
        URL_RE.findall(
            str(
                text
        )
    )


def extract_emails(
    text: str,
) -> list[str]:

    return sorted(
        EMAIL_RE.findall(
            str(
                text
        ).lower()
    )


# ============================================================
# English negation
# ============================================================

def has_english_negation(
    text: str,
) -> bool:

    text = (
        str(
            text
        )
        .lower()
    )

    return any(
        re.search(
            pattern,
            text,
        )
        is not None

        for pattern
        in EN_NEGATION_PATTERNS
    )


# ============================================================
# Chinese negation
# ============================================================

def has_chinese_negation(
    text: str,
) -> bool:

    text = str(
        text
    )

    return any(
        re.search(
            pattern,
            text,
        )
        is not None

        for pattern
        in ZH_STRONG_NEGATION_PATTERNS
    )


# ============================================================
# Negative-question rewrite
# ============================================================

def is_negative_question_rewrite(
    en: str,
    zh: str,
) -> bool:

    en = str(
        en
    ).strip()

    zh = str(
        zh
    ).strip()

    en_negative_question = (
        (
            "?" in en
        )
        and
        (
            EN_NEGATIVE_QUESTION_RE.search(
                en
            )
            is not None
        )
    )

    zh_question = (
        "?" in zh
        or
        "？" in zh
        or
        "吗" in zh
        or
        "么" in zh
        or
        "是不是" in zh
        or
        "难道" in zh
    )

    return bool(
        en_negative_question
        and
        zh_question
    )


# ============================================================
# Currency
# ============================================================

def has_english_currency(
    text: str,
) -> bool:

    lower = (
        str(
            text
        )
        .lower()
    )

    for token in EN_CURRENCY:

        if (
            token.lower()
            in lower
        ):

            return True

    return False


def has_chinese_currency(
    text: str,
) -> bool:

    text = str(
        text
    )

    for token in ZH_CURRENCY:

        if token in text:

            return True

    return False


# ============================================================
# Risk analysis
# ============================================================

def analyze_row(
    row: pd.Series,
) -> dict:

    en = str(
        row["en"]
    )

    zh = str(
        row["zh"]
    )

    flags = []

    # ========================================================
    # Number
    # ========================================================

    en_numbers = (
        extract_numbers(
            en
        )
    )

    zh_numbers = (
        extract_numbers(
            zh
        )
    )

    if (
        en_numbers
        !=
        zh_numbers
    ):

        flags.append(
            "NUMBER_MISMATCH"
        )

    # ========================================================
    # Percentage
    # ========================================================

    en_percent = (
        extract_percentages(
            en
        )
    )

    zh_percent = (
        extract_percentages(
            zh
        )
    )

    if (
        en_percent
        !=
        zh_percent
    ):

        flags.append(
            "PERCENT_MISMATCH"
        )

    # ========================================================
    # Negation
    # ========================================================

    en_neg = (
        has_english_negation(
            en
        )
    )

    zh_neg = (
        has_chinese_negation(
            zh
        )
    )

    negation_question_rewrite = (
        is_negative_question_rewrite(
            en,
            zh,
        )
    )

    if (
        en_neg
        !=
        zh_neg
    ):

        if (
            negation_question_rewrite
        ):

            flags.append(
                "NEGATION_QUESTION_REWRITE"
            )

        else:

            flags.append(
                "NEGATION_MISMATCH"
            )

    # ========================================================
    # URL
    # ========================================================

    en_urls = (
        extract_urls(
            en
        )
    )

    zh_urls = (
        extract_urls(
            zh
        )
    )

    if (
        en_urls
        !=
        zh_urls
    ):

        flags.append(
            "URL_MISMATCH"
        )

    # ========================================================
    # Email
    # ========================================================

    en_emails = (
        extract_emails(
            en
        )
    )

    zh_emails = (
        extract_emails(
            zh
        )
    )

    if (
        en_emails
        !=
        zh_emails
    ):

        flags.append(
            "EMAIL_MISMATCH"
        )

    # ========================================================
    # Currency
    #
    # Presence-only risk signal.
    # Not treated as strong semantic failure by itself.
    # ========================================================

    en_currency = (
        has_english_currency(
            en
        )
    )

    zh_currency = (
        has_chinese_currency(
            zh
        )
    )

    if (
        en_currency
        !=
        zh_currency
    ):

        flags.append(
            "CURRENCY_MISMATCH"
        )

    # ========================================================
    # Repeated punctuation
    # ========================================================

    if (
        REPEATED_PUNCT_RE.search(
            en
        )
        or
        REPEATED_PUNCT_RE.search(
            zh
        )
    ):

        flags.append(
            "REPEATED_PUNCT"
        )

    # ========================================================
    # Very short pair
    #
    # Short translations are NOT bad.
    #
    # Example:
    #
    # Fine! <-> 好吧
    #
    # Only retain as an audit signal.
    # ========================================================

    en_words = int(
        row[
            "en_words"
        ]
    )

    zh_cjk = int(
        row[
            "zh_cjk_chars"
        ]
    )

    if (
        en_words <= 2
        or
        zh_cjk <= 2
    ):

        flags.append(
            "VERY_SHORT_PAIR"
        )

    # ========================================================
    # Soft length ratio
    #
    # 14C hard band was broad.
    # This narrower band is only a risk signal.
    # ========================================================

    length_ratio = (
        zh_cjk
        /
        max(
            en_words,
            1,
        )
    )

    if (
        en_words > 3
        and
        (
            length_ratio < 0.55
            or
            length_ratio > 3.50
        )
    ):

        flags.append(
            "SOFT_LENGTH_RATIO"
        )

    # ========================================================
    # Mixed-script risk
    #
    # Examples such as:
    #
    # The Beatles 由四个音乐家组成。
    #
    # are valid translations, therefore this must NEVER
    # be a hard rejection signal.
    # ========================================================

    en_latin_ratio = float(
        row[
            "en_latin_ratio"
        ]
    )

    zh_cjk_ratio = float(
        row[
            "zh_cjk_ratio"
        ]
    )

    if (
        en_latin_ratio < 0.90
        or
        zh_cjk_ratio < 0.80
    ):

        flags.append(
            "MIXED_SCRIPT_RISK"
        )

    # ========================================================
    # Quality score
    #
    # This is an interpretable routing score.
    #
    # IMPORTANT:
    # It is NOT a probability that the translation is correct.
    # ========================================================

    penalties = {

        # Strong signals
        "NUMBER_MISMATCH":
            30,

        "PERCENT_MISMATCH":
            35,

        "NEGATION_MISMATCH":
            30,

        "URL_MISMATCH":
            40,

        "EMAIL_MISMATCH":
            40,

        # Soft signals
        "NEGATION_QUESTION_REWRITE":
            5,

        "CURRENCY_MISMATCH":
            20,

        "REPEATED_PUNCT":
            10,

        "VERY_SHORT_PAIR":
            5,

        "SOFT_LENGTH_RATIO":
            15,

        "MIXED_SCRIPT_RISK":
            10,
    }

    quality_score = 100

    for flag in flags:

        quality_score -= (
            penalties.get(
                flag,
                0,
            )
        )

    quality_score = max(
        quality_score,
        0,
    )

    # ========================================================
    # Strong risk flags
    #
    # Any one strong flag routes to Qwen.
    #
    # NEGATION_QUESTION_REWRITE is deliberately NOT here.
    # ========================================================

    strong_flags = {

        "NUMBER_MISMATCH",
        "PERCENT_MISMATCH",
        "NEGATION_MISMATCH",
        "URL_MISMATCH",
        "EMAIL_MISMATCH",
    }

    strong_risk = any(
        flag in strong_flags
        for flag in flags
    )

    # ========================================================
    # Soft flags
    # ========================================================

    soft_flags = {

        "NEGATION_QUESTION_REWRITE",

        "CURRENCY_MISMATCH",

        "REPEATED_PUNCT",

        "VERY_SHORT_PAIR",

        "SOFT_LENGTH_RATIO",

        "MIXED_SCRIPT_RISK",
    }

    soft_flag_count = sum(
        flag in soft_flags
        for flag in flags
    )

    # ========================================================
    # Routing
    # ========================================================

    if strong_risk:

        route = (
            "NEEDS_QWEN"
        )

    elif (
        soft_flag_count >= 2
    ):

        route = (
            "NEEDS_QWEN"
        )

    elif (
        quality_score < 80
    ):

        route = (
            "NEEDS_QWEN"
        )

    else:

        route = (
            "AUTO_ACCEPT"
        )

    # ========================================================
    # Output diagnostics
    # ========================================================

    return {

        "risk_flags":
            "|".join(
                flags
            ),

        "risk_flag_count":
            int(
                len(
                    flags
                )
            ),

        "strong_risk":
            bool(
                strong_risk
            ),

        "soft_flag_count":
            int(
                soft_flag_count
            ),

        "quality_score":
            int(
                quality_score
            ),

        "route":
            route,

        # -------------------------
        # Numbers
        # -------------------------

        "en_numbers":
            json.dumps(
                en_numbers,
                ensure_ascii=False,
            ),

        "zh_numbers":
            json.dumps(
                zh_numbers,
                ensure_ascii=False,
            ),

        # -------------------------
        # Percent
        # -------------------------

        "en_percentages":
            json.dumps(
                en_percent,
                ensure_ascii=False,
            ),

        "zh_percentages":
            json.dumps(
                zh_percent,
                ensure_ascii=False,
            ),

        # -------------------------
        # Negation
        # -------------------------

        "en_has_negation":
            bool(
                en_neg
            ),

        "zh_has_negation":
            bool(
                zh_neg
            ),

        "negation_question_rewrite":
            bool(
                negation_question_rewrite
            ),

        # -------------------------
        # URLs / emails
        # -------------------------

        "en_urls":
            json.dumps(
                en_urls,
                ensure_ascii=False,
            ),

        "zh_urls":
            json.dumps(
                zh_urls,
                ensure_ascii=False,
            ),

        "en_emails":
            json.dumps(
                en_emails,
                ensure_ascii=False,
            ),

        "zh_emails":
            json.dumps(
                zh_emails,
                ensure_ascii=False,
            ),

        # -------------------------
        # Other
        # -------------------------

        "en_has_currency":
            bool(
                en_currency
            ),

        "zh_has_currency":
            bool(
                zh_currency
            ),

        "length_ratio":
            float(
                length_ratio
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # Script location:
    #
    # scripts/
    #   pipeline/
    #     zh_en/
    #       14d_route_zh_en_quality_risk.py
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
        "14c_filtered"
        /
        "parallel_hard_filtered_v1.parquet"
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
        "14d_risk_routing"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    routed_file = (
        output_dir
        /
        "parallel_routed_v1.parquet"
    )

    routed_csv = (
        output_dir
        /
        "parallel_routed_v1.csv"
    )

    auto_file = (
        output_dir
        /
        "auto_accept_v1.parquet"
    )

    qwen_file = (
        output_dir
        /
        "needs_qwen_v1.parquet"
    )

    flag_report_file = (
        output_dir
        /
        "risk_flag_report_v1.csv"
    )

    route_report_file = (
        output_dir
        /
        "route_report_v1.csv"
    )

    source_report_file = (
        output_dir
        /
        "routing_by_source_v1.csv"
    )

    report_file = (
        output_dir
        /
        "risk_routing_report_v1.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14D V3 - QUALITY RISK ROUTING"
    )

    print(
        "=" * 110
    )

    print(
        "\nRouting version:"
    )

    print(
        ROUTING_VERSION
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
        routed_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(

            "\nOutput already exists:\n"

            f"{routed_file}\n\n"

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
        "zh_cjk_chars",

        "en_latin_ratio",
        "zh_cjk_ratio",

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
            "Input missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "\nInput rows:",
        len(
            df
        )
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
    # Analyze all pairs
    # ========================================================

    print(
        "\nRunning risk analysis..."
    )

    analyses = []

    total_rows = len(
        df
    )

    for index, (
        _,
        row,
    ) in enumerate(
        df.iterrows(),
        start=1,
    ):

        analyses.append(
            analyze_row(
                row
            )
        )

        if (
            index % 10000 == 0
            or
            index == total_rows
        ):

            print(
                f"{index}/{total_rows}"
            )

    analysis_df = pd.DataFrame(
        analyses
    )

    routed = pd.concat(
        [
            df.reset_index(
                drop=True
            ),

            analysis_df,
        ],
        axis=1,
    )

    # ========================================================
    # Split routes
    # ========================================================

    auto_accept = (
        routed[
            routed[
                "route"
            ]
            ==
            "AUTO_ACCEPT"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    needs_qwen = (
        routed[
            routed[
                "route"
            ]
            ==
            "NEEDS_QWEN"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    if (
        len(
            auto_accept
        )
        +
        len(
            needs_qwen
        )
        !=
        len(
            routed
        )
    ):

        raise RuntimeError(
            "Route partition mismatch."
        )

    if (
        routed[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Normalized pair duplicates found."
        )

    # ========================================================
    # Risk flag counts
    # ========================================================

    flag_counter = Counter()

    for value in routed[
        "risk_flags"
    ]:

        if not value:

            continue

        for flag in (
            str(
                value
            )
            .split(
                "|"
            )
        ):

            if flag:

                flag_counter[
                    flag
                ] += 1

    flag_rows = []

    for (
        flag,
        count,
    ) in (
        flag_counter
        .most_common()
    ):

        flag_rows.append({

            "flag":
                flag,

            "count":
                int(
                    count
                ),

            "percent_of_input":
                float(
                    count
                    /
                    max(
                        len(
                            routed
                        ),
                        1,
                    )
                    *
                    100
                ),
        })

    flag_report = pd.DataFrame(
        flag_rows
    )

    # ========================================================
    # Route report
    # ========================================================

    route_counts = (
        routed[
            "route"
        ]
        .value_counts()
    )

    route_rows = []

    for (
        route,
        count,
    ) in (
        route_counts
        .items()
    ):

        route_rows.append({

            "route":
                route,

            "count":
                int(
                    count
                ),

            "percent":
                float(
                    count
                    /
                    max(
                        len(
                            routed
                        ),
                        1,
                    )
                    *
                    100
                ),
        })

    route_report = pd.DataFrame(
        route_rows
    )

    # ========================================================
    # Source report
    # ========================================================

    source_rows = []

    for (
        source,
        part,
    ) in routed.groupby(
        "source_dataset"
    ):

        auto_count = int(
            (
                part[
                    "route"
                ]
                ==
                "AUTO_ACCEPT"
            )
            .sum()
        )

        qwen_count = int(
            (
                part[
                    "route"
                ]
                ==
                "NEEDS_QWEN"
            )
            .sum()
        )

        source_rows.append({

            "source_dataset":
                str(
                    source
                ),

            "rows":
                int(
                    len(
                        part
                    )
                ),

            "auto_accept":
                auto_count,

            "needs_qwen":
                qwen_count,

            "needs_qwen_percent":
                float(
                    qwen_count
                    /
                    max(
                        len(
                            part
                        ),
                        1,
                    )
                    *
                    100
                ),

            "avg_quality_score":
                float(
                    part[
                        "quality_score"
                    ]
                    .mean()
                ),

            "negation_mismatch":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NEGATION_MISMATCH",
                        regex=False,
                    )
                    .sum()
                ),

            "negation_question_rewrite":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NEGATION_QUESTION_REWRITE",
                        regex=False,
                    )
                    .sum()
                ),

            "number_mismatch":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NUMBER_MISMATCH",
                        regex=False,
                    )
                    .sum()
                ),
        })

    source_report = pd.DataFrame(
        source_rows
    )

    # ========================================================
    # Save
    # ========================================================

    routed.to_parquet(
        routed_file,
        index=False,
    )

    routed.to_csv(
        routed_csv,
        index=False,
        encoding="utf-8-sig",
    )

    auto_accept.to_parquet(
        auto_file,
        index=False,
    )

    needs_qwen.to_parquet(
        qwen_file,
        index=False,
    )

    flag_report.to_csv(
        flag_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    route_report.to_csv(
        route_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    source_report.to_csv(
        source_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "14D",

        "routing_version":
            ROUTING_VERSION,

        "pipeline":
            "zh_en_exp1_v1",

        "input_file":
            str(
                input_file
            ),

        "input_rows":
            int(
                len(
                    routed
                )
            ),

        "routes": {

            str(
                key
            ):
                int(
                    value
                )

            for (
                key,
                value,
            )
            in (
                route_counts
                .items()
            )
        },

        "flags": {

            str(
                key
            ):
                int(
                    value
                )

            for (
                key,
                value,
            )
            in (
                flag_counter
                .items()
            )
        },

        "quality_score": {

            "mean":
                float(
                    routed[
                        "quality_score"
                    ]
                    .mean()
                ),

            "median":
                float(
                    routed[
                        "quality_score"
                    ]
                    .median()
                ),

            "min":
                int(
                    routed[
                        "quality_score"
                    ]
                    .min()
                ),

            "max":
                int(
                    routed[
                        "quality_score"
                    ]
                    .max()
                ),
        },

        "routing_policy": {

            "strong_flags_route_to_qwen": [

                "NUMBER_MISMATCH",

                "PERCENT_MISMATCH",

                "NEGATION_MISMATCH",

                "URL_MISMATCH",

                "EMAIL_MISMATCH",
            ],

            "soft_flags": [

                "NEGATION_QUESTION_REWRITE",

                "CURRENCY_MISMATCH",

                "REPEATED_PUNCT",

                "VERY_SHORT_PAIR",

                "SOFT_LENGTH_RATIO",

                "MIXED_SCRIPT_RISK",
            ],

            "soft_flag_threshold":
                2,

            "quality_score_qwen_threshold":
                80,

            "negation_policy":
                (
                    "High-precision English word-boundary "
                    "patterns plus high-precision Chinese "
                    "strong-negation patterns. Natural "
                    "negative-question rewrites are treated "
                    "as a soft audit signal."
                ),
        },

        "source_report":
            source_rows,

        "final_assertions": {

            "all_rows_routed":
                True,

            "no_duplicate_normalized_pairs":
                True,

            "auto_plus_qwen_equals_input":
                (
                    len(
                        auto_accept
                    )
                    +
                    len(
                        needs_qwen
                    )
                    ==
                    len(
                        routed
                    )
                ),
        },

        "outputs": {

            "routed":
                str(
                    routed_file
                ),

            "routed_csv":
                str(
                    routed_csv
                ),

            "auto_accept":
                str(
                    auto_file
                ),

            "needs_qwen":
                str(
                    qwen_file
                ),

            "flag_report":
                str(
                    flag_report_file
                ),

            "route_report":
                str(
                    route_report_file
                ),

            "source_report":
                str(
                    source_report_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "RISK_ROUTING_V3_COMPLETE",
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
        "STEP 14D V3 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput:",
        len(
            routed
        )
    )

    print(
        "\nRoutes:"
    )

    print(
        route_report
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nRisk flags:"
    )

    if len(
        flag_report
    ) > 0:

        print(
            flag_report
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    else:

        print(
            "None"
        )

    print(
        "\nQuality score:"
    )

    print(
        "Mean:",
        f"{routed['quality_score'].mean():.2f}"
    )

    print(
        "Median:",
        f"{routed['quality_score'].median():.2f}"
    )

    print(
        "Min:",
        int(
            routed[
                "quality_score"
            ]
            .min()
        )
    )

    print(
        "Max:",
        int(
            routed[
                "quality_score"
            ]
            .max()
        )
    )

    print(
        "\nSource report:"
    )

    print(
        source_report
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nNegation diagnostics:"
    )

    neg_mismatch_count = int(
        flag_counter.get(
            "NEGATION_MISMATCH",
            0,
        )
    )

    neg_question_count = int(
        flag_counter.get(
            "NEGATION_QUESTION_REWRITE",
            0,
        )
    )

    print(
        "NEGATION_MISMATCH:",
        neg_mismatch_count
    )

    print(
        "NEGATION_QUESTION_REWRITE:",
        neg_question_count
    )

    print(
        "\nAUTO_ACCEPT:"
    )

    print(
        auto_file
    )

    print(
        "\nNEEDS_QWEN:"
    )

    print(
        qwen_file
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
        "RISK_ROUTING_V3_COMPLETE"
    )


if __name__ == "__main__":

    main()