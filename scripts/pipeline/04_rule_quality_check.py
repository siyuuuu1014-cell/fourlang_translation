from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import re

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
    / "03_filtered"
    / "parallel_filtered.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "04_rule_checked"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "parallel_rule_checked.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "parallel_rule_checked.csv"
)

AUTO_ACCEPT_FILE = (
    OUTPUT_DIR
    / "auto_accept.parquet"
)

QWEN_FILE = (
    OUTPUT_DIR
    / "needs_qwen.parquet"
)

HARD_REJECT_FILE = (
    OUTPUT_DIR
    / "hard_reject.parquet"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "rule_quality_report.json"
)


# ============================================================
# 2. Source family
#
# public_5k_v1 / public_5k_v2
# 属于同一个数据家族，不算两个独立来源。
# ============================================================

SOURCE_FAMILY = {
    "public_5k_v1": "public_5k",
    "public_5k_v2": "public_5k",

    "hplt": "hplt",
    "tatoeba": "tatoeba",
    "opus": "opus",

    "exp1": "experiment",
    "exp2": "experiment",

    "unknown": "unknown",
}


def source_to_family(
    source: str,
) -> str:

    source = str(source).strip()

    return SOURCE_FAMILY.get(
        source,
        source,
    )


def calculate_source_families(
    data_sources: str,
):

    sources = [
        item.strip()
        for item in str(data_sources).split("|")
        if item.strip()
    ]

    families = sorted(
        {
            source_to_family(source)
            for source in sources
        }
    )

    return (
        "|".join(families),
        len(families),
    )


# ============================================================
# 3. Number extraction
#
# 当前只严格检查显式数字：
#
# 25
# 12.5
# 100,5
#
# 不尝试：
#
# eight ↔ sakkiz
#
# 那种情况后面交给 Qwen Judge。
# ============================================================

NUMBER_PATTERN = re.compile(
    r"(?<!\w)"
    r"[-+]?"
    r"\d+(?:[.,]\d+)?"
    r"(?!\w)"
)


def normalize_number(
    number: str,
) -> str:

    number = str(number).strip()

    # 统一 decimal separator
    number = number.replace(
        ",",
        ".",
    )

    # 去掉多余 + 号
    if number.startswith("+"):
        number = number[1:]

    return number


def extract_numbers(
    text: str,
):

    return sorted(
        normalize_number(value)
        for value in NUMBER_PATTERN.findall(
            str(text)
        )
    )


# ============================================================
# 4. Time extraction
#
# 08:30
# 8:30
# 18.45
# ============================================================

TIME_PATTERN = re.compile(
    r"\b"
    r"(?:[01]?\d|2[0-3])"
    r"[:.]"
    r"[0-5]\d"
    r"\b"
)


def extract_times(
    text: str,
):

    values = TIME_PATTERN.findall(
        str(text)
    )

    normalized = []

    for value in values:

        value = value.replace(
            ".",
            ":",
        )

        hour, minute = (
            value.split(":")
        )

        # 08:30 和 8:30 视为一致
        value = (
            f"{int(hour)}:"
            f"{minute}"
        )

        normalized.append(
            value
        )

    return sorted(
        normalized
    )


# ============================================================
# 5. Date extraction
#
# 2026-08-25
# 25/08/2026
# 25.08.2026
# ============================================================

DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r")\b"
)


def normalize_date(
    value: str,
) -> str:

    value = str(value)

    value = (
        value
        .replace("/", "-")
        .replace(".", "-")
    )

    return value


def extract_dates(
    text: str,
):

    return sorted(
        normalize_date(value)
        for value in DATE_PATTERN.findall(
            str(text)
        )
    )


# ============================================================
# 6. URL
# ============================================================

URL_PATTERN = re.compile(
    r"https?://[^\s]+",
    re.IGNORECASE,
)


def clean_url(
    value: str,
) -> str:

    return (
        str(value)
        .rstrip(
            ".,;:!?)]}"
        )
    )


def extract_urls(
    text: str,
):

    return sorted(
        clean_url(value)
        for value in URL_PATTERN.findall(
            str(text)
        )
    )


# ============================================================
# 7. Email
# ============================================================

EMAIL_PATTERN = re.compile(
    r"\b"
    r"[A-Za-z0-9._%+-]+"
    r"@"
    r"[A-Za-z0-9.-]+"
    r"\."
    r"[A-Za-z]{2,}"
    r"\b"
)


def extract_emails(
    text: str,
):

    return sorted(
        value.lower()
        for value in EMAIL_PATTERN.findall(
            str(text)
        )
    )


# ============================================================
# 8. English negation
# ============================================================

EN_NEGATION_PATTERN = re.compile(
    r"\b("
    r"not|"
    r"no|"
    r"never|"
    r"none|"
    r"nothing|"
    r"nobody|"
    r"nowhere|"
    r"neither|"
    r"nor|"
    r"without|"
    r"don't|"
    r"doesn't|"
    r"didn't|"
    r"isn't|"
    r"aren't|"
    r"wasn't|"
    r"weren't|"
    r"can't|"
    r"cannot|"
    r"couldn't|"
    r"won't|"
    r"wouldn't|"
    r"shouldn't|"
    r"mustn't|"
    r"haven't|"
    r"hasn't|"
    r"hadn't"
    r")\b",
    re.IGNORECASE,
)


def has_english_negation(
    text: str,
) -> bool:

    return bool(
        EN_NEGATION_PATTERN.search(
            str(text)
        )
    )


# ============================================================
# 9. Uzbek negation
#
# IMPORTANT:
#
# 这里故意使用“保守检测”。
#
# 目的不是完整解决 Uzbek morphology，
# 而是只抓比较明确的否定结构。
#
# 宁愿少抓，也不要像上一版一样大量误报。
# ============================================================

UZ_NEGATION_WORD_PATTERN = re.compile(
    r"\b("
    r"emas|"
    r"yo['’‘ʻ`]?q|"
    r"aslo"
    r")\b",
    re.IGNORECASE,
)


UZ_NEGATIVE_SUFFIXES = (
    # bormaydi
    "maydi",

    # bormayman
    "mayman",

    # bormaymiz
    "maymiz",

    # bormaysiz
    "maysiz",

    # bormaysan
    "maysan",

    # bormadim
    "madim",

    # bormadik
    "madik",

    # bormading
    "mading",

    # bormadingiz
    "madingiz",

    # bormadi
    "madi",

    # bormagan
    "magan",

    # bormaganman
    "maganman",

    # bormagansiz
    "magansiz",

    # bormas
    "mas",

    # bormasdi
    "masdi",

    # bormay
    "may",
)


UZ_TOKEN_PATTERN = re.compile(
    r"[A-Za-z'"
    r"’‘ʻ"
    r"-]+"
)


def normalize_uz_token(
    token: str,
) -> str:

    token = str(token).lower()

    token = (
        token
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
    )

    return token


def has_uzbek_negation(
    text: str,
) -> bool:

    text = str(text)

    # --------------------------------------------------------
    # 明确否定词：
    #
    # emas
    # yo'q
    # aslo
    # --------------------------------------------------------

    if UZ_NEGATION_WORD_PATTERN.search(
        text
    ):
        return True


    # --------------------------------------------------------
    # 保守检查常见否定动词后缀
    # --------------------------------------------------------

    tokens = UZ_TOKEN_PATTERN.findall(
        text
    )


    for token in tokens:

        token = normalize_uz_token(
            token
        )


        for suffix in UZ_NEGATIVE_SUFFIXES:

            # 至少要在 suffix 前面有两个字符，
            # 避免把非常短的普通单词误判。
            if (
                len(token)
                >=
                len(suffix) + 2
                and
                token.endswith(
                    suffix
                )
            ):
                return True


    return False


# ============================================================
# 10. Control characters
# ============================================================

CONTROL_PATTERN = re.compile(
    r"[\x00-\x08"
    r"\x0B"
    r"\x0C"
    r"\x0E-\x1F]"
)


def has_control_chars(
    text: str,
) -> bool:

    return bool(
        CONTROL_PATTERN.search(
            str(text)
        )
    )


# ============================================================
# 11. Repeated punctuation
#
# !!!!!!
# ??????
# ......
# ============================================================

REPEATED_PUNCT_PATTERN = re.compile(
    r"([!?.,;:])\1{4,}"
)


def has_repeated_punctuation(
    text: str,
) -> bool:

    return bool(
        REPEATED_PUNCT_PATTERN.search(
            str(text)
        )
    )


# ============================================================
# 12. Letter ratio
#
# 防止大量：
#
# symbols
# code
# garbage
#
# 注意：
# 当前 Step03 已经排除了 Cyrillic Uzbek。
# ============================================================

LETTER_PATTERN = re.compile(
    r"[A-Za-zÀ-ÿ'"
    r"’‘ʻ]"
)


def letter_ratio(
    text: str,
) -> float:

    text = str(text)

    if not text:
        return 0.0


    letters = len(
        LETTER_PATTERN.findall(
            text
        )
    )


    return (
        letters
        /
        len(text)
    )


# ============================================================
# 13. Source == Target
#
# English 与 Uzbek 完全一样，
# 有可能是：
#
# Google
# Tesla
# OK
#
# 也可能是脏数据。
#
# 因此只作为 REVIEW 风险，
# 不直接删除。
# ============================================================

def comparable_text(
    text: str,
) -> str:

    text = (
        str(text)
        .lower()
    )

    text = re.sub(
        r"[^\w]+",
        "",
        text,
    )

    return text


def same_text(
    source: str,
    target: str,
) -> bool:

    source_clean = comparable_text(
        source
    )

    target_clean = comparable_text(
        target
    )


    # 极短词不判断
    if (
        len(source_clean) < 4
        or
        len(target_clean) < 4
    ):
        return False


    return (
        source_clean
        ==
        target_clean
    )


# ============================================================
# 14. Analyze one row
# ============================================================

def analyze_row(
    row,
):

    source = str(
        row[
            "source_text_normalized"
        ]
    )

    target = str(
        row[
            "target_text_normalized"
        ]
    )


    # ========================================================
    # A. Source provenance
    # ========================================================

    (
        source_families,
        source_family_count,
    ) = calculate_source_families(
        row[
            "data_sources"
        ]
    )


    # ========================================================
    # B. Structured values
    # ========================================================

    source_numbers = (
        extract_numbers(
            source
        )
    )

    target_numbers = (
        extract_numbers(
            target
        )
    )


    source_times = (
        extract_times(
            source
        )
    )

    target_times = (
        extract_times(
            target
        )
    )


    source_dates = (
        extract_dates(
            source
        )
    )

    target_dates = (
        extract_dates(
            target
        )
    )


    source_urls = (
        extract_urls(
            source
        )
    )

    target_urls = (
        extract_urls(
            target
        )
    )


    source_emails = (
        extract_emails(
            source
        )
    )

    target_emails = (
        extract_emails(
            target
        )
    )


    # ========================================================
    # C. Structured consistency
    # ========================================================

    number_consistent = (
        source_numbers
        ==
        target_numbers
    )


    time_consistent = (
        source_times
        ==
        target_times
    )


    date_consistent = (
        source_dates
        ==
        target_dates
    )


    url_consistent = (
        source_urls
        ==
        target_urls
    )


    email_consistent = (
        source_emails
        ==
        target_emails
    )


    # ========================================================
    # D. Negation
    # ========================================================

    source_has_negation = (
        has_english_negation(
            source
        )
    )


    target_has_negation = (
        has_uzbek_negation(
            target
        )
    )


    negation_consistent = (
        source_has_negation
        ==
        target_has_negation
    )


    # ========================================================
    # E. Other quality checks
    # ========================================================

    source_target_same = (
        same_text(
            source,
            target,
        )
    )


    control_chars = (
        has_control_chars(
            source
        )
        or
        has_control_chars(
            target
        )
    )


    repeated_punctuation = (
        has_repeated_punctuation(
            source
        )
        or
        has_repeated_punctuation(
            target
        )
    )


    source_letter_ratio = (
        letter_ratio(
            source
        )
    )


    target_letter_ratio = (
        letter_ratio(
            target
        )
    )


    low_letter_ratio = (
        source_letter_ratio
        <
        0.35
        or
        target_letter_ratio
        <
        0.35
    )


    # ========================================================
    # F. Risk flags
    # ========================================================

    flags = []


    if not number_consistent:

        flags.append(
            "NUMBER_MISMATCH"
        )


    if not time_consistent:

        flags.append(
            "TIME_MISMATCH"
        )


    if not date_consistent:

        flags.append(
            "DATE_MISMATCH"
        )


    if not url_consistent:

        flags.append(
            "URL_MISMATCH"
        )


    if not email_consistent:

        flags.append(
            "EMAIL_MISMATCH"
        )


    if not negation_consistent:

        flags.append(
            "NEGATION_MISMATCH"
        )


    if source_target_same:

        flags.append(
            "SOURCE_TARGET_SAME"
        )


    if control_chars:

        flags.append(
            "CONTROL_CHARACTER"
        )


    if repeated_punctuation:

        flags.append(
            "REPEATED_PUNCT"
        )


    if low_letter_ratio:

        flags.append(
            "LOW_LETTER_RATIO"
        )


    # ========================================================
    # G. Hard reject
    #
    # 当前只对非常明确的数据损坏做 hard reject。
    #
    # 不把数字、时间、否定等直接删除，
    # 因为可能是规则误报。
    # ========================================================

    HARD_REJECT_FLAGS = {
        "CONTROL_CHARACTER",
    }


    hard_reject = any(
        flag in HARD_REJECT_FLAGS
        for flag in flags
    )


    # ========================================================
    # H. Quality score
    #
    # score 只是 Pipeline routing 辅助值，
    # 不是 ground truth。
    # ========================================================

    score = 80.0


    # --------------------------------------------------------
    # 真正独立 source family 才加分。
    #
    # public_5k_v1 + public_5k_v2
    # 仍然只是一个 family。
    # --------------------------------------------------------

    if source_family_count >= 3:

        score += 12.0


    elif source_family_count == 2:

        score += 8.0


    # --------------------------------------------------------
    # 不再根据 occurrence_count 加分。
    # --------------------------------------------------------


    penalty_map = {

        "NUMBER_MISMATCH":
            15.0,

        "TIME_MISMATCH":
            15.0,

        "DATE_MISMATCH":
            15.0,

        "URL_MISMATCH":
            20.0,

        "EMAIL_MISMATCH":
            20.0,

        "NEGATION_MISMATCH":
            15.0,

        "SOURCE_TARGET_SAME":
            10.0,

        "CONTROL_CHARACTER":
            30.0,

        "REPEATED_PUNCT":
            5.0,

        "LOW_LETTER_RATIO":
            8.0,
    }


    for flag in flags:

        score -= penalty_map.get(
            flag,
            0.0,
        )


    score = max(
        0.0,
        min(
            100.0,
            score,
        ),
    )


    # ========================================================
    # I. Routing
    #
    # 重要：
    #
    # 不再因为 source_family_count == 1
    # 就自动送 Qwen。
    # ========================================================

    CRITICAL_FLAGS = {
        "NUMBER_MISMATCH",
        "TIME_MISMATCH",
        "DATE_MISMATCH",
        "URL_MISMATCH",
        "EMAIL_MISMATCH",
        "NEGATION_MISMATCH",
    }


    REVIEW_FLAGS = {
        "SOURCE_TARGET_SAME",
        "LOW_LETTER_RATIO",
        "REPEATED_PUNCT",
    }


    has_critical_risk = any(
        flag in CRITICAL_FLAGS
        for flag in flags
    )


    has_review_risk = any(
        flag in REVIEW_FLAGS
        for flag in flags
    )


    if hard_reject:

        pipeline_route = (
            "HARD_REJECT"
        )


    elif (
        has_critical_risk
        or
        has_review_risk
        or
        score < 75.0
    ):

        pipeline_route = (
            "NEEDS_QWEN"
        )


    else:

        pipeline_route = (
            "AUTO_ACCEPT"
        )


    # ========================================================
    # J. Return
    # ========================================================

    return pd.Series({

        "source_families":
            source_families,

        "source_family_count":
            source_family_count,

        # Structured values

        "source_numbers":
            json.dumps(
                source_numbers,
                ensure_ascii=False,
            ),

        "target_numbers":
            json.dumps(
                target_numbers,
                ensure_ascii=False,
            ),

        "source_times":
            json.dumps(
                source_times,
                ensure_ascii=False,
            ),

        "target_times":
            json.dumps(
                target_times,
                ensure_ascii=False,
            ),

        "source_dates":
            json.dumps(
                source_dates,
                ensure_ascii=False,
            ),

        "target_dates":
            json.dumps(
                target_dates,
                ensure_ascii=False,
            ),

        # Consistency

        "number_consistent":
            number_consistent,

        "time_consistent":
            time_consistent,

        "date_consistent":
            date_consistent,

        "url_consistent":
            url_consistent,

        "email_consistent":
            email_consistent,

        # Negation

        "source_has_negation":
            source_has_negation,

        "target_has_negation":
            target_has_negation,

        "negation_consistent":
            negation_consistent,

        # Other

        "source_target_same":
            source_target_same,

        "control_chars":
            control_chars,

        "repeated_punctuation":
            repeated_punctuation,

        "source_letter_ratio":
            source_letter_ratio,

        "target_letter_ratio":
            target_letter_ratio,

        # Risk

        "risk_flags":
            "|".join(
                flags
            ),

        "risk_count":
            len(flags),

        "quality_score":
            score,

        "pipeline_route":
            pipeline_route,
    })


# ============================================================
# 15. Main
# ============================================================

def main():

    print("=" * 100)
    print("EN-UZ PIPELINE")
    print("STEP 04 - RULE QUALITY CHECK V2")
    print("=" * 100)


    # ========================================================
    # Input validation
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到 Step03 输出：\n"
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
        "\nInput rows:",
        len(df)
    )


    required_columns = [
        "source_text_normalized",
        "target_text_normalized",
        "data_sources",
        "occurrence_count",
        "source_count",
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            f"Step03 数据缺少字段："
            f"{missing_columns}"
        )


    # ========================================================
    # Run rule checks
    # ========================================================

    print(
        "\nRunning rule checks..."
    )


    analysis_df = df.apply(
        analyze_row,
        axis=1,
    )


    result_df = pd.concat(
        [
            df.reset_index(
                drop=True
            ),

            analysis_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    # ========================================================
    # Split routes
    # ========================================================

    auto_accept_df = (
        result_df[
            result_df[
                "pipeline_route"
            ]
            ==
            "AUTO_ACCEPT"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    needs_qwen_df = (
        result_df[
            result_df[
                "pipeline_route"
            ]
            ==
            "NEEDS_QWEN"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    hard_reject_df = (
        result_df[
            result_df[
                "pipeline_route"
            ]
            ==
            "HARD_REJECT"
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # Save full results
    # ========================================================

    print(
        "\nSaving results..."
    )


    result_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )


    result_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    auto_accept_df.to_parquet(
        AUTO_ACCEPT_FILE,
        index=False,
    )


    needs_qwen_df.to_parquet(
        QWEN_FILE,
        index=False,
    )


    hard_reject_df.to_parquet(
        HARD_REJECT_FILE,
        index=False,
    )


    # ========================================================
    # Risk statistics
    # ========================================================

    flag_counter = Counter()


    for value in result_df[
        "risk_flags"
    ]:

        if pd.isna(value):
            continue


        value = str(value).strip()

        if not value:
            continue


        for flag in value.split("|"):

            flag = flag.strip()

            if flag:

                flag_counter[
                    flag
                ] += 1


    # ========================================================
    # Route statistics
    # ========================================================

    route_counts = (
        result_df[
            "pipeline_route"
        ]
        .value_counts()
        .to_dict()
    )


    # ========================================================
    # Family statistics
    # ========================================================

    family_distribution = (
        result_df[
            "source_family_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # ========================================================
    # Risk count distribution
    # ========================================================

    risk_count_distribution = (
        result_df[
            "risk_count"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


    # ========================================================
    # Score distribution
    # ========================================================

    score_stats = (
        result_df[
            "quality_score"
        ]
        .describe()
    )


    # ========================================================
    # Report
    # ========================================================

    report = {

        "pipeline":
            "en_uz",

        "step":
            "04_rule_quality_check_v2",

        "input_rows":
            len(result_df),

        "route_counts": {

            str(key):
                int(value)

            for key, value
            in route_counts.items()
        },

        "route_rates_percent": {

            str(key):
                float(
                    value
                    /
                    len(result_df)
                    *
                    100
                )

            for key, value
            in route_counts.items()
        },

        "risk_flags": {

            str(key):
                int(value)

            for key, value
            in flag_counter.items()
        },

        "risk_count_distribution": {

            str(key):
                int(value)

            for key, value
            in risk_count_distribution.items()
        },

        "source_family_distribution": {

            str(key):
                int(value)

            for key, value
            in family_distribution.items()
        },

        "quality_score": {

            "mean":
                float(
                    score_stats[
                        "mean"
                    ]
                ),

            "std":
                float(
                    score_stats[
                        "std"
                    ]
                ),

            "min":
                float(
                    score_stats[
                        "min"
                    ]
                ),

            "25_percent":
                float(
                    score_stats[
                        "25%"
                    ]
                ),

            "median":
                float(
                    score_stats[
                        "50%"
                    ]
                ),

            "75_percent":
                float(
                    score_stats[
                        "75%"
                    ]
                ),

            "max":
                float(
                    score_stats[
                        "max"
                    ]
                ),
        },

        "auto_accept_rows":
            len(
                auto_accept_df
            ),

        "needs_qwen_rows":
            len(
                needs_qwen_df
            ),

        "hard_reject_rows":
            len(
                hard_reject_df
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
    # Terminal output
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 04 COMPLETE")
    print("=" * 100)


    print(
        "Input:",
        len(
            result_df
        )
    )


    # --------------------------------------------------------
    # Routes
    # --------------------------------------------------------

    print("\nRoutes:")

    print(
        result_df[
            "pipeline_route"
        ]
        .value_counts()
        .to_string()
    )


    print(
        "\nRoute rates:"
    )


    route_rate_df = (
        result_df[
            "pipeline_route"
        ]
        .value_counts()
        .rename_axis(
            "route"
        )
        .reset_index(
            name="count"
        )
    )


    route_rate_df[
        "percent"
    ] = (
        route_rate_df[
            "count"
        ]
        /
        len(result_df)
        *
        100
    )


    print(
        route_rate_df
        .round(2)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Source families
    # --------------------------------------------------------

    print(
        "\nSource family count:"
    )

    print(
        result_df[
            "source_family_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # --------------------------------------------------------
    # Risk flags
    # --------------------------------------------------------

    print(
        "\nRisk flags:"
    )


    if flag_counter:

        for flag, count in (
            flag_counter
            .most_common()
        ):

            percent = (
                count
                /
                len(result_df)
                *
                100
            )


            print(
                f"{flag:<24}: "
                f"{count:<8} "
                f"({percent:.2f}%)"
            )

    else:

        print(
            "No risk flags."
        )


    # --------------------------------------------------------
    # Risk counts
    # --------------------------------------------------------

    print(
        "\nRisk count distribution:"
    )

    print(
        result_df[
            "risk_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    print(
        "\nQuality score:"
    )

    print(
        result_df[
            "quality_score"
        ]
        .describe()
    )


    # --------------------------------------------------------
    # Final route counts
    # --------------------------------------------------------

    print("\nFinal:")

    print(
        "AUTO_ACCEPT:",
        len(
            auto_accept_df
        )
    )

    print(
        "NEEDS_QWEN:",
        len(
            needs_qwen_df
        )
    )

    print(
        "HARD_REJECT:",
        len(
            hard_reject_df
        )
    )


    # --------------------------------------------------------
    # Sample risky rows
    # --------------------------------------------------------

    if len(
        needs_qwen_df
    ) > 0:

        print(
            "\nSample NEEDS_QWEN rows:"
        )


        sample_columns = [
            "source_text_normalized",
            "target_text_normalized",
            "risk_flags",
            "quality_score",
        ]


        print(
            needs_qwen_df[
                sample_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print(
        "\nFiles:"
    )

    print(
        "Full:"
    )

    print(
        OUTPUT_FILE
    )


    print(
        "\nAUTO_ACCEPT:"
    )

    print(
        AUTO_ACCEPT_FILE
    )


    print(
        "\nNEEDS_QWEN:"
    )

    print(
        QWEN_FILE
    )


    print(
        "\nHARD_REJECT:"
    )

    print(
        HARD_REJECT_FILE
    )


    print(
        "\nReport:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":

    main()