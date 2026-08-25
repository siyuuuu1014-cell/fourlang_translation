from __future__ import annotations

from pathlib import Path
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
    / "01_collected"
    / "parallel_collected.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "02_normalized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "parallel_normalized.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "parallel_normalized.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "normalization_report.json"
)


# ============================================================
# 2. Utilities
# ============================================================

CYRILLIC_PATTERN = re.compile(
    r"[\u0400-\u04FF]"
)

MULTI_SPACE_PATTERN = re.compile(
    r"\s+"
)


def normalize_unicode(
    text: str,
) -> str:

    return unicodedata.normalize(
        "NFKC",
        str(text),
    )


def normalize_spaces(
    text: str,
) -> str:

    text = (
        text
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    text = MULTI_SPACE_PATTERN.sub(
        " ",
        text,
    )

    return text.strip()


# ============================================================
# 3. English normalization
# ============================================================

def normalize_english(
    text: str,
) -> str:

    text = normalize_unicode(
        text
    )

    text = (
        text
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
    )

    text = normalize_spaces(
        text
    )

    return text


# ============================================================
# 4. Uzbek apostrophe normalization
#
# Uzbek Latin 中常见：
# o'
# o‘
# o’
# oʻ
# g'
# g‘
# g’
# gʻ
#
# 统一成 ASCII apostrophe：
# o'
# g'
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


def normalize_uzbek(
    text: str,
) -> str:

    text = normalize_unicode(
        text
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

    text = normalize_spaces(
        text
    )

    return text


# ============================================================
# 5. Cyrillic detection
# ============================================================

def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        CYRILLIC_PATTERN.search(
            str(text)
        )
    )


# ============================================================
# 6. Word count
# ============================================================

WORD_PATTERN = re.compile(
    r"\b[\w'\-]+\b",
    flags=re.UNICODE,
)


def count_words(
    text: str,
) -> int:

    return len(
        WORD_PATTERN.findall(
            str(text)
        )
    )


# ============================================================
# 7. Character count
# ============================================================

def count_chars(
    text: str,
) -> int:

    return len(
        str(text)
    )


# ============================================================
# 8. Main
# ============================================================

def main():

    print("=" * 90)
    print("EN-UZ PIPELINE")
    print("STEP 02 - NORMALIZATION")
    print("=" * 90)


    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"找不到：\n{INPUT_FILE}"
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
        "\nRows:",
        len(df)
    )


    # ========================================================
    # Preserve raw text
    # ========================================================

    df[
        "source_text_raw"
    ] = df[
        "source_text"
    ]

    df[
        "target_text_raw"
    ] = df[
        "target_text"
    ]


    # ========================================================
    # Normalize
    # ========================================================

    print(
        "\nNormalizing English..."
    )

    df[
        "source_text_normalized"
    ] = (
        df[
            "source_text_raw"
        ]
        .astype(str)
        .map(
            normalize_english
        )
    )


    print(
        "Normalizing Uzbek..."
    )

    df[
        "target_text_normalized"
    ] = (
        df[
            "target_text_raw"
        ]
        .astype(str)
        .map(
            normalize_uzbek
        )
    )


    # ========================================================
    # Metadata
    # ========================================================

    df[
        "source_word_count"
    ] = (
        df[
            "source_text_normalized"
        ]
        .map(
            count_words
        )
    )


    df[
        "target_word_count"
    ] = (
        df[
            "target_text_normalized"
        ]
        .map(
            count_words
        )
    )


    df[
        "source_char_count"
    ] = (
        df[
            "source_text_normalized"
        ]
        .map(
            count_chars
        )
    )


    df[
        "target_char_count"
    ] = (
        df[
            "target_text_normalized"
        ]
        .map(
            count_chars
        )
    )


    df[
        "uz_has_cyrillic"
    ] = (
        df[
            "target_text_normalized"
        ]
        .map(
            contains_cyrillic
        )
    )


    # ========================================================
    # Empty after normalization
    # ========================================================

    df[
        "normalized_empty"
    ] = (
        (
            df[
                "source_text_normalized"
            ]
            ==
            ""
        )
        |
        (
            df[
                "target_text_normalized"
            ]
            ==
            ""
        )
    )


    # ========================================================
    # Length ratio
    #
    # 暂时只记录，不在 Step02 删除。
    # ========================================================

    df[
        "length_ratio"
    ] = (
        df[
            "source_char_count"
        ]
        /
        df[
            "target_char_count"
        ]
        .replace(
            0,
            pd.NA,
        )
    )


    # ========================================================
    # Save
    # ========================================================

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )


    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Report
    # ========================================================

    cyrillic_count = int(
        df[
            "uz_has_cyrillic"
        ].sum()
    )


    empty_count = int(
        df[
            "normalized_empty"
        ].sum()
    )


    report = {

        "pipeline":
            "en_uz",

        "step":
            "02_normalization",

        "input_rows":
            len(df),

        "output_rows":
            len(df),

        "normalized_empty":
            empty_count,

        "uz_cyrillic_rows":
            cyrillic_count,

        "uz_cyrillic_rate_percent":
            (
                cyrillic_count
                /
                len(df)
                *
                100
            ),

        "source_word_count": {

            "mean":
                float(
                    df[
                        "source_word_count"
                    ].mean()
                ),

            "max":
                int(
                    df[
                        "source_word_count"
                    ].max()
                ),
        },

        "target_word_count": {

            "mean":
                float(
                    df[
                        "target_word_count"
                    ].mean()
                ),

            "max":
                int(
                    df[
                        "target_word_count"
                    ].max()
                ),
        },
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
    # Final output
    # ========================================================

    print("\n")
    print("=" * 90)
    print("STEP 02 COMPLETE")
    print("=" * 90)


    print(
        "Rows:",
        len(df)
    )

    print(
        "Normalized empty:",
        empty_count
    )

    print(
        "Uzbek Cyrillic:",
        cyrillic_count
    )

    print(
        "Uzbek Cyrillic rate:",
        f"{cyrillic_count / len(df) * 100:.4f}%"
    )


    print(
        "\nEnglish avg words:",
        round(
            df[
                "source_word_count"
            ].mean(),
            2,
        )
    )


    print(
        "Uzbek avg words:",
        round(
            df[
                "target_word_count"
            ].mean(),
            2,
        )
    )


    print(
        "\nFiles:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        OUTPUT_CSV
    )

    print(
        REPORT_FILE
    )


    print(
        "\nSample:"
    )

    print(
        df[
            [
                "data_source",
                "source_text_raw",
                "source_text_normalized",
                "target_text_raw",
                "target_text_normalized",
                "uz_has_cyrillic",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()