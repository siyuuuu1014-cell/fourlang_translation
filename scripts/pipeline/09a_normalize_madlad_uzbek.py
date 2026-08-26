from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import sacrebleu


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

INPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "student"
    / "small100"
    / "heldout_eval_v1"
)

ALL_PREDICTIONS_FILE = (
    INPUT_DIR
    / "all_predictions.parquet"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "script_normalized"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PREDICTIONS_PARQUET = (
    OUTPUT_DIR
    / "all_predictions_script_normalized.parquet"
)

OUTPUT_PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "all_predictions_script_normalized.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "heldout_summary_script_normalized.csv"
)

CHALLENGE_FILE = (
    OUTPUT_DIR
    / "challenge_summary_script_normalized.csv"
)

MADLAD_COMPARE_FILE = (
    OUTPUT_DIR
    / "madlad_raw_vs_latin.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "script_normalization_report.json"
)


# ============================================================
# 2. Cyrillic detection
# ============================================================

CYRILLIC_RE = re.compile(
    r"[\u0400-\u04FF]"
)


def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        CYRILLIC_RE.search(
            str(text)
        )
    )


# ============================================================
# 3. Uzbek Cyrillic -> Latin
# ============================================================
#
# Main Uzbek Cyrillic alphabet:
#
# А Б В Г Д Е Ё Ж З И Й К Л М Н О П Р С Т У Ф Х
# Ц Ч Ш Ъ Ь Э Ю Я Ў Қ Ғ Ҳ
#
# Special:
#
# Ў -> O'
# Ғ -> G'
# Қ -> Q
# Ҳ -> H
# Ч -> Ch
# Ш -> Sh
#
# Е:
#   word start / after vowel / apostrophe -> Ye
#   otherwise -> E
#
# ============================================================

SIMPLE_MAP = {

    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",

    "ё": "yo",
    "ж": "j",
    "з": "z",
    "и": "i",
    "й": "y",

    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",

    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",

    "ф": "f",
    "х": "x",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",

    "ъ": "'",
    "ь": "",

    "э": "e",
    "ю": "yu",
    "я": "ya",

    "ў": "o'",
    "қ": "q",
    "ғ": "g'",
    "ҳ": "h",
}


CYRILLIC_VOWELS = set(
    "аеёиоуўэюяАЕЁИОУЎЭЮЯ"
)


APOSTROPHE_LIKE = {
    "'",
    "’",
    "‘",
    "`",
    "ʻ",
    "ʼ",
    "ʹ",
    "՚",
    "ъ",
    "Ъ",
}


def preserve_case(
    source_char: str,
    replacement: str,
) -> str:

    if not replacement:
        return replacement

    if source_char.isupper():

        if len(replacement) == 1:
            return replacement.upper()

        return (
            replacement[0].upper()
            +
            replacement[1:]
        )

    return replacement


def transliterate_uzbek_cyrillic(
    text: str,
) -> str:

    text = str(text)

    result = []

    for i, char in enumerate(text):

        lower = char.lower()

        # ----------------------------------------------------
        # Е / е requires context
        # ----------------------------------------------------

        if lower == "е":

            previous = (
                text[i - 1]
                if i > 0
                else ""
            )

            at_word_start = (
                i == 0
                or
                not previous.isalpha()
            )

            after_vowel = (
                previous
                in CYRILLIC_VOWELS
            )

            after_apostrophe = (
                previous
                in APOSTROPHE_LIKE
            )

            if (
                at_word_start
                or
                after_vowel
                or
                after_apostrophe
            ):

                replacement = "ye"

            else:

                replacement = "e"

            result.append(
                preserve_case(
                    char,
                    replacement,
                )
            )

            continue

        # ----------------------------------------------------
        # Other Uzbek Cyrillic letters
        # ----------------------------------------------------

        if lower in SIMPLE_MAP:

            replacement = (
                SIMPLE_MAP[
                    lower
                ]
            )

            result.append(
                preserve_case(
                    char,
                    replacement,
                )
            )

            continue

        # ----------------------------------------------------
        # Punctuation / Latin / number etc.
        # ----------------------------------------------------

        result.append(
            char
        )

    return "".join(
        result
    )


# ============================================================
# 4. Common Uzbek normalization
# ============================================================

def normalize_uzbek_latin(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    # --------------------------------------------------------
    # Normalize apostrophe variants
    # --------------------------------------------------------

    for old in [
        "’",
        "‘",
        "`",
        "ʻ",
        "ʼ",
        "ʹ",
        "՚",
        "´",
    ]:

        text = text.replace(
            old,
            "'",
        )

    # --------------------------------------------------------
    # Spaces
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # space before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    # excessive apostrophes
    text = re.sub(
        r"'{2,}",
        "'",
        text,
    )

    return text.strip()


# ============================================================
# 5. Prepare evaluation text
# ============================================================

def prepare_prediction(
    row,
) -> tuple[str, bool]:

    prediction = str(
        row[
            "prediction"
        ]
    )

    direction = str(
        row[
            "direction"
        ]
    )

    converted = False

    # --------------------------------------------------------
    # Only EN -> UZ has Uzbek output
    # --------------------------------------------------------

    if direction == "en_uz":

        if contains_cyrillic(
            prediction
        ):

            prediction = (
                transliterate_uzbek_cyrillic(
                    prediction
                )
            )

            converted = True

        prediction = (
            normalize_uzbek_latin(
                prediction
            )
        )

    return (
        prediction,
        converted,
    )


def prepare_reference(
    row,
) -> str:

    reference = str(
        row[
            "reference"
        ]
    )

    if (
        str(
            row[
                "direction"
            ]
        )
        ==
        "en_uz"
    ):

        reference = (
            normalize_uzbek_latin(
                reference
            )
        )

    return reference


# ============================================================
# 6. Metrics
# ============================================================

def calculate_metrics(
    df: pd.DataFrame,
    prediction_column: str,
    reference_column: str,
) -> dict:

    predictions = (
        df[
            prediction_column
        ]
        .astype(str)
        .tolist()
    )

    references = (
        df[
            reference_column
        ]
        .astype(str)
        .tolist()
    )

    bleu = (
        sacrebleu
        .corpus_bleu(
            predictions,
            [
                references
            ],
        )
        .score
    )

    chrf = (
        sacrebleu
        .CHRF(
            word_order=2
        )
        .corpus_score(
            predictions,
            [
                references
            ],
        )
        .score
    )

    exact = float(
        (
            df[
                prediction_column
            ]
            .astype(str)
            .str.strip()
            ==
            df[
                reference_column
            ]
            .astype(str)
            .str.strip()
        )
        .mean()
        *
        100
    )

    return {
        "samples":
            int(
                len(df)
            ),

        "bleu":
            float(
                bleu
            ),

        "chrf++":
            float(
                chrf
            ),

        "exact_percent":
            exact,

        "avg_latency_seconds":
            float(
                df[
                    "latency_seconds"
                ]
                .mean()
            ),
    }


# ============================================================
# 7. Build summary
# ============================================================

def build_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for model in sorted(
        df[
            "model"
        ]
        .unique()
    ):

        model_df = df[
            df[
                "model"
            ]
            ==
            model
        ]

        for dataset in [
            "benchmark",
            "challenge",
        ]:

            dataset_df = model_df[
                model_df[
                    "dataset"
                ]
                ==
                dataset
            ]

            if len(
                dataset_df
            ) == 0:

                continue

            for direction in [
                "en_uz",
                "uz_en",
            ]:

                part = dataset_df[
                    dataset_df[
                        "direction"
                    ]
                    ==
                    direction
                ]

                if len(part) == 0:
                    continue

                metrics = calculate_metrics(
                    part,
                    prediction_column=
                        "prediction_eval",

                    reference_column=
                        "reference_eval",
                )

                rows.append({

                    "model":
                        model,

                    "dataset":
                        dataset,

                    "category":
                        "ALL",

                    "direction":
                        direction,

                    **metrics,
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 8. Challenge category summary
# ============================================================

def build_challenge_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    challenge_df = df[
        df[
            "dataset"
        ]
        ==
        "challenge"
    ]

    for model in sorted(
        challenge_df[
            "model"
        ]
        .unique()
    ):

        model_df = challenge_df[
            challenge_df[
                "model"
            ]
            ==
            model
        ]

        for category in sorted(
            model_df[
                "category"
            ]
            .dropna()
            .astype(str)
            .unique()
        ):

            category_df = model_df[
                model_df[
                    "category"
                ]
                ==
                category
            ]

            for direction in [
                "en_uz",
                "uz_en",
            ]:

                part = category_df[
                    category_df[
                        "direction"
                    ]
                    ==
                    direction
                ]

                if len(part) == 0:
                    continue

                metrics = calculate_metrics(
                    part,
                    prediction_column=
                        "prediction_eval",

                    reference_column=
                        "reference_eval",
                )

                rows.append({

                    "model":
                        model,

                    "category":
                        category,

                    "direction":
                        direction,

                    **metrics,
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 9. MADLAD raw vs transliterated
# ============================================================

def build_madlad_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    madlad = df[
        (
            df[
                "model"
            ]
            ==
            "madlad400_3b"
        )
        &
        (
            df[
                "direction"
            ]
            ==
            "en_uz"
        )
    ]

    for dataset in [
        "benchmark",
        "challenge",
    ]:

        part = madlad[
            madlad[
                "dataset"
            ]
            ==
            dataset
        ]

        if len(part) == 0:
            continue

        # ----------------------------------------------------
        # Raw
        # ----------------------------------------------------

        raw = calculate_metrics(
            part,
            prediction_column=
                "prediction_raw",

            reference_column=
                "reference",
        )

        # ----------------------------------------------------
        # Latin-normalized
        # ----------------------------------------------------

        normalized = calculate_metrics(
            part,
            prediction_column=
                "prediction_eval",

            reference_column=
                "reference_eval",
        )

        rows.append({

            "dataset":
                dataset,

            "samples":
                len(part),

            "cyrillic_predictions":
                int(
                    part[
                        "was_cyrillic"
                    ]
                    .sum()
                ),

            "cyrillic_rate_percent":
                float(
                    part[
                        "was_cyrillic"
                    ]
                    .mean()
                    *
                    100
                ),

            "raw_bleu":
                raw[
                    "bleu"
                ],

            "normalized_bleu":
                normalized[
                    "bleu"
                ],

            "delta_bleu":
                normalized[
                    "bleu"
                ]
                -
                raw[
                    "bleu"
                ],

            "raw_chrf++":
                raw[
                    "chrf++"
                ],

            "normalized_chrf++":
                normalized[
                    "chrf++"
                ],

            "delta_chrf++":
                normalized[
                    "chrf++"
                ]
                -
                raw[
                    "chrf++"
                ],
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    print("=" * 110)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 09A - MADLAD UZBEK SCRIPT NORMALIZATION")
    print("=" * 110)

    # ========================================================
    # Input
    # ========================================================

    if not ALL_PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "Step09 predictions not found:\n"
            f"{ALL_PREDICTIONS_FILE}"
        )

    print(
        "\nLoading:"
    )

    print(
        ALL_PREDICTIONS_FILE
    )

    df = pd.read_parquet(
        ALL_PREDICTIONS_FILE
    )

    required = [
        "model",
        "dataset",
        "category",
        "direction",
        "source",
        "reference",
        "prediction",
        "latency_seconds",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    print(
        "Rows:",
        len(df)
    )

    print(
        "\nModels:"
    )

    print(
        df[
            "model"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Preserve RAW output
    # ========================================================

    df[
        "prediction_raw"
    ] = (
        df[
            "prediction"
        ]
        .astype(str)
    )

    # ========================================================
    # Normalize
    # ========================================================

    predictions = []

    converted_flags = []

    references = []

    for _, row in df.iterrows():

        (
            prediction,
            converted,
        ) = prepare_prediction(
            row
        )

        reference = (
            prepare_reference(
                row
            )
        )

        predictions.append(
            prediction
        )

        converted_flags.append(
            converted
        )

        references.append(
            reference
        )

    df[
        "prediction_eval"
    ] = predictions

    df[
        "reference_eval"
    ] = references

    df[
        "was_cyrillic"
    ] = converted_flags

    # ========================================================
    # Cyrillic statistics
    # ========================================================

    madlad_en_uz = df[
        (
            df[
                "model"
            ]
            ==
            "madlad400_3b"
        )
        &
        (
            df[
                "direction"
            ]
            ==
            "en_uz"
        )
    ]

    madlad_cyrillic = int(
        madlad_en_uz[
            "was_cyrillic"
        ]
        .sum()
    )

    madlad_total = len(
        madlad_en_uz
    )

    madlad_rate = (
        madlad_cyrillic
        /
        madlad_total
        *
        100
        if madlad_total
        else 0
    )

    print("\n")
    print("=" * 110)
    print("MADLAD SCRIPT STATISTICS")
    print("=" * 110)

    print(
        "MADLAD EN->UZ samples:",
        madlad_total
    )

    print(
        "Cyrillic outputs:",
        madlad_cyrillic
    )

    print(
        "Cyrillic rate:",
        f"{madlad_rate:.2f}%"
    )

    # ========================================================
    # Preview
    # ========================================================

    preview = madlad_en_uz[
        madlad_en_uz[
            "was_cyrillic"
        ]
    ].head(
        10
    )

    print("\n")
    print("=" * 110)
    print("TRANSLITERATION PREVIEW")
    print("=" * 110)

    for _, row in preview.iterrows():

        print()

        print(
            "SOURCE :",
            row[
                "source"
            ]
        )

        print(
            "REF    :",
            row[
                "reference_eval"
            ]
        )

        print(
            "RAW    :",
            row[
                "prediction_raw"
            ]
        )

        print(
            "LATIN  :",
            row[
                "prediction_eval"
            ]
        )

    # ========================================================
    # Summaries
    # ========================================================

    summary = build_summary(
        df
    )

    challenge_summary = (
        build_challenge_summary(
            df
        )
    )

    madlad_compare = (
        build_madlad_comparison(
            df
        )
    )

    # ========================================================
    # Save
    # ========================================================

    df.to_parquet(
        OUTPUT_PREDICTIONS_PARQUET,
        index=False,
    )

    df.to_csv(
        OUTPUT_PREDICTIONS_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    challenge_summary.to_csv(
        CHALLENGE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    madlad_compare.to_csv(
        MADLAD_COMPARE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "09A",

        "purpose":
            (
                "Normalize Uzbek Cyrillic output "
                "to Latin for fair EN-UZ evaluation."
            ),

        "inference_rerun":
            False,

        "input_predictions":
            str(
                ALL_PREDICTIONS_FILE
            ),

        "total_prediction_rows":
            len(df),

        "madlad_en_uz_samples":
            madlad_total,

        "madlad_cyrillic_outputs":
            madlad_cyrillic,

        "madlad_cyrillic_rate_percent":
            madlad_rate,

        "normalization": {

            "unicode":
                "NFKC",

            "apostrophe":
                "ASCII apostrophe",

            "cyrillic_to_latin":
                True,
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
    # Terminal outputs
    # ========================================================

    print("\n")
    print("=" * 120)
    print("MADLAD RAW VS LATIN")
    print("=" * 120)

    print(
        madlad_compare
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 120)
    print("CORRECTED HELD-OUT SUMMARY")
    print("=" * 120)

    print(
        summary[
            [
                "model",
                "dataset",
                "direction",
                "samples",
                "bleu",
                "chrf++",
                "exact_percent",
                "avg_latency_seconds",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 120)
    print("CORRECTED CHALLENGE SUMMARY")
    print("=" * 120)

    print(
        challenge_summary[
            [
                "model",
                "category",
                "direction",
                "samples",
                "bleu",
                "chrf++",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 110)
    print("STEP 09A COMPLETE")
    print("=" * 110)

    print(
        "Corrected summary:"
    )

    print(
        SUMMARY_FILE
    )

    print(
        "\nMADLAD comparison:"
    )

    print(
        MADLAD_COMPARE_FILE
    )

    print(
        "\nPredictions:"
    )

    print(
        OUTPUT_PREDICTIONS_PARQUET
    )


if __name__ == "__main__":
    main()