from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


# ============================================================
# Version
# ============================================================

ROUTING_VERSION = "14D_V4_FINAL_CALIBRATION"


# ============================================================
# Number patterns
#
# IMPORTANT:
#
# Do NOT use \w boundaries here.
#
# In Python Unicode regex:
# Chinese characters are also \w.
#
# Therefore:
#
#   1979年
#
# failed under:
#
#   (?![\w])
#
# V4 only blocks ASCII alphanumeric attachment.
# ============================================================

ARABIC_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?![A-Za-z0-9_])"
)


# Chinese numeric expressions.
#
# Examples:
#
# 一九九七
# 六点半
# 三百
# 两千
# 十二
#
# We DO NOT automatically convert these into Arabic values,
# because contextual forms such as:
#
#   6.30 -> 六点半
#
# are semantic time expressions rather than simple decimals.
#
# These are treated as cross-format numeric evidence.
CHINESE_NUMBER_RE = re.compile(
    r"[零〇○一二两三四五六七八九十百千万亿兆点半]+"
)


# Common English number words.
#
# Used only to recognize cases such as:
#
#   two people -> 2个人
#
# It is NOT used for direct semantic equality.
ENGLISH_NUMBER_WORD_RE = re.compile(
    r"\b(?:"
    r"zero|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|"
    r"first|second|third|fourth|fifth|sixth|seventh|"
    r"eighth|ninth|tenth"
    r")\b",
    flags=re.IGNORECASE,
)


# ============================================================
# Percentage
# ============================================================

PERCENT_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*[%％]"
)


# ============================================================
# URL / Email
# ============================================================

URL_RE = re.compile(
    r"https?://[^\s]+|www\.[^\s]+",
    flags=re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


# ============================================================
# Repeated punctuation
# ============================================================

REPEATED_PUNCT_RE = re.compile(
    r"([!?！？。，,.])\1{2,}"
)


# ============================================================
# English explicit negation
#
# High precision word-boundary patterns.
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
# Chinese broad negation evidence
#
# Used ONLY when English already has explicit negation.
#
# At that point we only ask:
#
# "Does Chinese have some surface evidence of negation?"
#
# Therefore broad characters are acceptable here.
# ============================================================

ZH_NEGATION_EVIDENCE_PATTERNS = [

    r"不",
    r"没",
    r"无",
    r"未",
    r"非",
    r"别",
    r"勿",
    r"莫",
]


# ============================================================
# Chinese strong negation
#
# Used when English has NO explicit negation.
#
# This list is intentionally much stricter.
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
# Currency
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
            "Step 14D V4 Final - "
            "ZH-EN quality risk routing."
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

def canonicalize_arabic_number(
    value: str,
) -> str:

    value = (
        str(value)
        .replace(",", "")
        .strip()
    )

    try:

        number = Decimal(
            value
        )

        if (
            number
            ==
            number.to_integral()
        ):

            return str(
                number.quantize(
                    Decimal("1")
                )
            )

        return format(
            number.normalize(),
            "f",
        )

    except (
        InvalidOperation,
        ValueError,
    ):

        return value


def extract_arabic_numbers(
    text: str,
) -> list[str]:

    matches = (
        ARABIC_NUMBER_RE
        .findall(
            str(text)
        )
    )

    values = [

        canonicalize_arabic_number(
            value
        )

        for value
        in matches
    ]

    return sorted(
        values
    )


def extract_chinese_number_expressions(
    text: str,
) -> list[str]:

    return (
        CHINESE_NUMBER_RE
        .findall(
            str(text)
        )
    )


def extract_english_number_words(
    text: str,
) -> list[str]:

    return [

        match.group(0).lower()

        for match
        in ENGLISH_NUMBER_WORD_RE.finditer(
            str(text)
        )
    ]


# ============================================================
# Number-risk analysis
# ============================================================

def analyze_number_risk(
    en: str,
    zh: str,
) -> dict:

    en_arabic = (
        extract_arabic_numbers(
            en
        )
    )

    zh_arabic = (
        extract_arabic_numbers(
            zh
        )
    )

    zh_chinese_numbers = (
        extract_chinese_number_expressions(
            zh
        )
    )

    en_number_words = (
        extract_english_number_words(
            en
        )
    )

    flag = None

    # --------------------------------------------------------
    # Both sides contain Arabic digits
    #
    # Example:
    #
    # EN: 1979
    # ZH: 1979年
    #
    # V4 correctly extracts both as 1979.
    # --------------------------------------------------------

    if (
        en_arabic
        and
        zh_arabic
    ):

        if (
            en_arabic
            !=
            zh_arabic
        ):

            flag = (
                "NUMBER_MISMATCH"
            )

    # --------------------------------------------------------
    # EN uses Arabic digits,
    # ZH uses Chinese numeric expression.
    #
    # Examples:
    #
    # 1997 -> 一九九七
    #
    # 6.30 -> 六点半
    #
    # This is NOT automatically incorrect.
    # Route to semantic review.
    # --------------------------------------------------------

    elif (
        en_arabic
        and
        not zh_arabic
    ):

        if zh_chinese_numbers:

            flag = (
                "NUMBER_CROSS_FORMAT"
            )

        else:

            flag = (
                "NUMBER_MISSING_ZH"
            )

    # --------------------------------------------------------
    # ZH uses Arabic digits,
    # EN may use number words.
    #
    # Example:
    #
    # two people -> 2个人
    # --------------------------------------------------------

    elif (
        zh_arabic
        and
        not en_arabic
    ):

        if en_number_words:

            flag = (
                "NUMBER_CROSS_FORMAT"
            )

        else:

            flag = (
                "NUMBER_EXTRA_ZH"
            )

    return {

        "number_flag":
            flag,

        "en_arabic_numbers":
            en_arabic,

        "zh_arabic_numbers":
            zh_arabic,

        "zh_chinese_number_expressions":
            zh_chinese_numbers,

        "en_number_words":
            en_number_words,
    }


# ============================================================
# Percentage
# ============================================================

def extract_percentages(
    text: str,
) -> list[str]:

    values = []

    for value in (
        PERCENT_RE.findall(
            str(text)
        )
    ):

        values.append(
            canonicalize_arabic_number(
                value
            )
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
            str(text)
        )
    )


def extract_emails(
    text: str,
) -> list[str]:

    return sorted(
        EMAIL_RE.findall(
            str(text).lower()
        )
    )


# ============================================================
# Negation
# ============================================================

def has_english_negation(
    text: str,
) -> bool:

    text = (
        str(text)
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


def has_chinese_negation_evidence(
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
        in ZH_NEGATION_EVIDENCE_PATTERNS
    )


def has_chinese_strong_negation(
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


def is_negative_question_rewrite(
    en: str,
    zh: str,
) -> bool:

    en = str(en).strip()
    zh = str(zh).strip()

    en_negative_question = (

        "?" in en

        and

        EN_NEGATIVE_QUESTION_RE.search(
            en
        )
        is not None
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

        or
        "能不能" in zh
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
        str(text)
        .lower()
    )

    return any(

        token.lower()
        in lower

        for token
        in EN_CURRENCY
    )


def has_chinese_currency(
    text: str,
) -> bool:

    text = str(
        text
    )

    return any(

        token
        in text

        for token
        in ZH_CURRENCY
    )


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
    # 1. Number
    # ========================================================

    number_result = (
        analyze_number_risk(
            en,
            zh,
        )
    )

    number_flag = (
        number_result[
            "number_flag"
        ]
    )

    if number_flag:

        flags.append(
            number_flag
        )

    # ========================================================
    # 2. Percentage
    # ========================================================

    en_percentages = (
        extract_percentages(
            en
        )
    )

    zh_percentages = (
        extract_percentages(
            zh
        )
    )

    if (
        en_percentages
        !=
        zh_percentages
    ):

        flags.append(
            "PERCENT_MISMATCH"
        )

    # ========================================================
    # 3. Negation
    #
    # NEGATION IS NO LONGER A STRONG ROUTING RULE.
    #
    # Translation can legitimately change surface form:
    #
    # barely
    #   -> 不是很
    #
    # unharmed
    #   -> 没有受到伤害
    #
    # different
    #   -> 不同
    #
    # Therefore this is diagnostic only.
    # ========================================================

    en_neg = (
        has_english_negation(
            en
        )
    )

    zh_neg_evidence = (
        has_chinese_negation_evidence(
            zh
        )
    )

    zh_strong_neg = (
        has_chinese_strong_negation(
            zh
        )
    )

    negative_question_rewrite = (
        is_negative_question_rewrite(
            en,
            zh,
        )
    )

    negation_flag = None

    if en_neg:

        if not zh_neg_evidence:

            if negative_question_rewrite:

                negation_flag = (
                    "NEGATION_QUESTION_REWRITE"
                )

            else:

                negation_flag = (
                    "NEGATION_SURFACE_MISMATCH"
                )

    else:

        if zh_strong_neg:

            negation_flag = (
                "NEGATION_SURFACE_MISMATCH"
            )

    if negation_flag:

        flags.append(
            negation_flag
        )

    # ========================================================
    # 4. URL
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
    # 5. Email
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
    # 6. Currency
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
    # 7. Repeated punctuation
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
    # 8. Very short pair
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
    # 9. Soft length ratio
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
    # 10. Mixed script
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
    # Diagnostic routing score only.
    # NOT correctness probability.
    # ========================================================

    penalties = {

        # ---------------------
        # Strong / review
        # ---------------------

        "NUMBER_MISMATCH":
            30,

        "NUMBER_MISSING_ZH":
            30,

        "NUMBER_EXTRA_ZH":
            30,

        "NUMBER_CROSS_FORMAT":
            15,

        "PERCENT_MISMATCH":
            35,

        "URL_MISMATCH":
            40,

        "EMAIL_MISMATCH":
            40,

        # ---------------------
        # Soft
        # ---------------------

        "NEGATION_SURFACE_MISMATCH":
            5,

        "NEGATION_QUESTION_REWRITE":
            5,

        "CURRENCY_MISMATCH":
            15,

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
    # Strong flags
    # ========================================================

    strong_flags = {

        "NUMBER_MISMATCH",

        "NUMBER_MISSING_ZH",

        "NUMBER_EXTRA_ZH",

        "PERCENT_MISMATCH",

        "URL_MISMATCH",

        "EMAIL_MISMATCH",
    }

    strong_risk = any(

        flag
        in strong_flags

        for flag
        in flags
    )

    # ========================================================
    # Semantic review flags
    #
    # NUMBER_CROSS_FORMAT is not an error,
    # but should be reviewed because deterministic rules
    # cannot safely evaluate:
    #
    # 1997 -> 一九九七
    # 6.30 -> 六点半
    # ========================================================

    semantic_review_flags = {

        "NUMBER_CROSS_FORMAT",
    }

    semantic_review = any(

        flag
        in semantic_review_flags

        for flag
        in flags
    )

    # ========================================================
    # Soft flags
    # ========================================================

    soft_flags = {

        "NEGATION_SURFACE_MISMATCH",

        "NEGATION_QUESTION_REWRITE",

        "CURRENCY_MISMATCH",

        "REPEATED_PUNCT",

        "VERY_SHORT_PAIR",

        "SOFT_LENGTH_RATIO",

        "MIXED_SCRIPT_RISK",
    }

    soft_flag_count = sum(

        flag
        in soft_flags

        for flag
        in flags
    )

    # ========================================================
    # Routing
    # ========================================================

    if strong_risk:

        route = (
            "NEEDS_QWEN"
        )

    elif semantic_review:

        route = (
            "NEEDS_QWEN"
        )

    elif soft_flag_count >= 2:

        route = (
            "NEEDS_QWEN"
        )

    elif quality_score < 80:

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

        "semantic_review":
            bool(
                semantic_review
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

        # ---------------------
        # Number diagnostics
        # ---------------------

        "number_risk_type":
            (
                number_flag
                if number_flag
                else ""
            ),

        "en_arabic_numbers":
            json.dumps(
                number_result[
                    "en_arabic_numbers"
                ],
                ensure_ascii=False,
            ),

        "zh_arabic_numbers":
            json.dumps(
                number_result[
                    "zh_arabic_numbers"
                ],
                ensure_ascii=False,
            ),

        "zh_chinese_number_expressions":
            json.dumps(
                number_result[
                    "zh_chinese_number_expressions"
                ],
                ensure_ascii=False,
            ),

        "en_number_words":
            json.dumps(
                number_result[
                    "en_number_words"
                ],
                ensure_ascii=False,
            ),

        # ---------------------
        # Percentage
        # ---------------------

        "en_percentages":
            json.dumps(
                en_percentages,
                ensure_ascii=False,
            ),

        "zh_percentages":
            json.dumps(
                zh_percentages,
                ensure_ascii=False,
            ),

        # ---------------------
        # Negation
        # ---------------------

        "en_has_explicit_negation":
            bool(
                en_neg
            ),

        "zh_has_negation_evidence":
            bool(
                zh_neg_evidence
            ),

        "zh_has_strong_negation":
            bool(
                zh_strong_neg
            ),

        "negation_question_rewrite":
            bool(
                negative_question_rewrite
            ),

        # ---------------------
        # URLs
        # ---------------------

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

        # ---------------------
        # Emails
        # ---------------------

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

        # ---------------------
        # Other
        # ---------------------

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

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14D V4 FINAL - QUALITY RISK ROUTING"
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
    # Analyze
    # ========================================================

    print(
        "\nRunning risk analysis..."
    )

    analyses = []

    total = len(
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
            index == total
        ):

            print(
                f"{index}/{total}"
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
    # Split
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
    # Flag counts
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
        route_counts.items()
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

            "number_cross_format":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NUMBER_CROSS_FORMAT",
                        regex=False,
                    )
                    .sum()
                ),

            "number_missing_zh":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NUMBER_MISSING_ZH",
                        regex=False,
                    )
                    .sum()
                ),

            "number_extra_zh":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NUMBER_EXTRA_ZH",
                        regex=False,
                    )
                    .sum()
                ),

            "negation_surface_mismatch":
                int(
                    part[
                        "risk_flags"
                    ]
                    .astype(str)
                    .str.contains(
                        "NEGATION_SURFACE_MISMATCH",
                        regex=False,
                    )
                    .sum()
                ),
        })

    source_report = (
        pd.DataFrame(
            source_rows
        )
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

        "input_rows":
            int(
                len(
                    routed
                )
            ),

        "routes": {

            str(key):
                int(value)

            for (
                key,
                value,
            )
            in route_counts.items()
        },

        "flags": {

            str(key):
                int(value)

            for (
                key,
                value,
            )
            in flag_counter.items()
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

            "strong_flags": [

                "NUMBER_MISMATCH",

                "NUMBER_MISSING_ZH",

                "NUMBER_EXTRA_ZH",

                "PERCENT_MISMATCH",

                "URL_MISMATCH",

                "EMAIL_MISMATCH",
            ],

            "semantic_review_flags": [

                "NUMBER_CROSS_FORMAT",
            ],

            "soft_flags": [

                "NEGATION_SURFACE_MISMATCH",

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

            "number_policy":
                (
                    "Arabic digit comparison is exact after "
                    "canonicalization. Chinese-number or "
                    "English-number-word conversions are "
                    "treated as semantic review rather than "
                    "automatic mismatch."
                ),

            "negation_policy":
                (
                    "Surface negation mismatch is diagnostic "
                    "only and does not independently route "
                    "a pair to Qwen."
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
            "RISK_ROUTING_V4_COMPLETE",
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
        "STEP 14D V4 RESULT"
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
        "\nNumber diagnostics:"
    )

    for flag in [

        "NUMBER_MISMATCH",

        "NUMBER_CROSS_FORMAT",

        "NUMBER_MISSING_ZH",

        "NUMBER_EXTRA_ZH",

    ]:

        print(
            f"{flag}:",
            int(
                flag_counter.get(
                    flag,
                    0,
                )
            )
        )

    print(
        "\nNegation diagnostics:"
    )

    print(
        "NEGATION_SURFACE_MISMATCH:",
        int(
            flag_counter.get(
                "NEGATION_SURFACE_MISMATCH",
                0,
            )
        )
    )

    print(
        "NEGATION_QUESTION_REWRITE:",
        int(
            flag_counter.get(
                "NEGATION_QUESTION_REWRITE",
                0,
            )
        )
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
        "RISK_ROUTING_V4_COMPLETE"
    )


if __name__ == "__main__":

    main()