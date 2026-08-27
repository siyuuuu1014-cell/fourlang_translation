from __future__ import annotations

import argparse
import json
import re

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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
# Negation lexicons
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
ZH_NEGATIONS = [

    "不",
    "没",
    "没有",
    "无",
    "未",
    "不是",
    "不能",
    "不会",
    "不得",
    "从未",
    "并非",
    "无需",
    "不必",
    "不要",
]


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
            "Step 14D - Risk routing for "
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


def extract_percentages(
    text: str,
) -> list[str]:

    values = []

    for item in PERCENT_RE.findall(
        str(
            text
        )
    ):

        values.append(
            item.replace(
                "％",
                "%"
            )
        )

    return sorted(
        values
    )


def extract_urls(
    text: str,
) -> list[str]:

    return sorted(
        URL_RE.findall(
            str(
                text
        )
    ))

def extract_emails(
    text: str,
) -> list[str]:

    return sorted(
        EMAIL_RE.findall(
            str(
                text
        )
        .lower()
    ))




def has_chinese_negation(
    text: str,
) -> bool:

    text = str(
        text
    )

    for token in ZH_NEGATIONS:

        if token in text:

            return True

    return False


def has_english_currency(
    text: str,
) -> bool:

    lower = str(
        text
    ).lower()

    for token in EN_CURRENCY:

        if token.lower() in lower:

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

    # --------------------------------------------------------
    # Number
    # --------------------------------------------------------

    en_numbers = extract_numbers(
        en
    )

    zh_numbers = extract_numbers(
        zh
    )

    if (
        en_numbers
        !=
        zh_numbers
    ):

        flags.append(
            "NUMBER_MISMATCH"
        )

    # --------------------------------------------------------
    # Percent
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Negation
    # --------------------------------------------------------

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

    if (
        en_neg
        !=
        zh_neg
    ):

        flags.append(
            "NEGATION_MISMATCH"
        )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    en_urls = extract_urls(
        en
    )

    zh_urls = extract_urls(
        zh
    )

    if (
        en_urls
        !=
        zh_urls
    ):

        flags.append(
            "URL_MISMATCH"
        )

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Currency
    #
    # Presence-only check.
    # Currency wording can legitimately change,
    # so this is only a risk signal.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Repeated punctuation
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Very short pair
    #
    # IMPORTANT:
    # This is NOT automatically bad.
    # We track it as an audit signal.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Soft length anomaly
    #
    # Hard filter used 0.2~8.0.
    # Here use a narrower band only as a risk signal.
    # --------------------------------------------------------

    ratio = (
        zh_cjk
        /
        max(
            en_words,
            1
        )
    )

    if (
        en_words > 3
        and
        (
            ratio < 0.55
            or
            ratio > 3.50
        )
    ):

        flags.append(
            "SOFT_LENGTH_RATIO"
        )

    # --------------------------------------------------------
    # Mixed-script risk
    # --------------------------------------------------------

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
    # Interpretability only.
    # It does NOT mean probability of correctness.
    # ========================================================

    penalties = {

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
                0
            )
        )

    quality_score = max(
        quality_score,
        0,
    )

    # ========================================================
    # Routing
    #
    # Strong semantic-risk flags automatically route to Qwen.
    #
    # Soft-only signals do NOT necessarily route.
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

    soft_flag_count = sum(

        flag
        in {
            "CURRENCY_MISMATCH",
            "REPEATED_PUNCT",
            "VERY_SHORT_PAIR",
            "SOFT_LENGTH_RATIO",
            "MIXED_SCRIPT_RISK",
        }

        for flag
        in flags
    )

    if strong_risk:

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

    return {

        "risk_flags":
            "|".join(
                flags
            ),

        "risk_flag_count":
            len(
                flags
            ),

        "quality_score":
            int(
                quality_score
            ),

        "route":
            route,

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

        "en_has_negation":
            bool(
                en_neg
            ),

        "zh_has_negation":
            bool(
                zh_neg
            ),

        "length_ratio":
            float(
                ratio
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
        "STEP 14D - QUALITY RISK ROUTING"
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
        routed_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{routed_file}\n"
            "Use --overwrite to rebuild."
        )

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

    # ========================================================
    # Analyze
    # ========================================================

    analyses = []

    for _, row in df.iterrows():

        analyses.append(
            analyze_row(
                row
            )
        )

    analysis_df = (
        pd.DataFrame(
            analyses
        )
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

    assert (
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
    )

    assert not (
        routed[
            "normalized_pair_hash"
        ]
        .duplicated()
        .any()
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

        for flag in str(
            value
        ).split(
            "|"
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

        source_rows.append({

            "source_dataset":
                source,

            "rows":
                int(
                    len(
                        part
                    )
                ),

            "auto_accept":
                int(
                    (
                        part[
                            "route"
                        ]
                        ==
                        "AUTO_ACCEPT"
                    )
                    .sum()
                ),

            "needs_qwen":
                int(
                    (
                        part[
                            "route"
                        ]
                        ==
                        "NEEDS_QWEN"
                    )
                    .sum()
                ),

            "needs_qwen_percent":
                float(
                    (
                        part[
                            "route"
                        ]
                        ==
                        "NEEDS_QWEN"
                    )
                    .mean()
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

        "pipeline":
            "zh_en_exp1_v1",

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
        },

        "source_report":
            source_rows,

        "outputs": {

            "routed":
                str(
                    routed_file
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
            "RISK_ROUTING_COMPLETE",
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
        "STEP 14D RESULT"
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
        "\nSTATUS:"
    )

    print(
        "RISK_ROUTING_COMPLETE"
    )


if __name__ == "__main__":

    main()