from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import sacrebleu


def calculate_metrics(
    df: pd.DataFrame,
    direction: str,
) -> dict:

    predictions = (
        df["prediction"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    references = (
        df["reference"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    # ========================================================
    # Target-language-aware BLEU
    # ========================================================

    if direction == "zh_en":

        bleu_tokenizer = "13a"

    elif direction == "en_zh":

        bleu_tokenizer = "zh"

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    bleu = (
        sacrebleu
        .corpus_bleu(
            predictions,
            [references],
            tokenize=bleu_tokenizer,
        )
        .score
    )

    # ========================================================
    # chrF
    # Character only
    # Particularly useful for Chinese
    # ========================================================

    chrf = (
        sacrebleu
        .CHRF(
            word_order=0
        )
        .corpus_score(
            predictions,
            [references],
        )
        .score
    )

    # ========================================================
    # chrF++
    # Keep for consistency with EN-UZ pipeline
    # ========================================================

    chrfpp = (
        sacrebleu
        .CHRF(
            word_order=2
        )
        .corpus_score(
            predictions,
            [references],
        )
        .score
    )

    exact = float(
        (
            df["prediction"]
            .astype(str)
            .str.strip()
            ==
            df["reference"]
            .astype(str)
            .str.strip()
        )
        .mean()
        *
        100
    )

    latency = float(
        df["latency_seconds"]
        .mean()
    )

    return {

        "direction":
            direction,

        "samples":
            int(
                len(df)
            ),

        "bleu":
            float(
                bleu
            ),

        "bleu_tokenizer":
            bleu_tokenizer,

        "chrf":
            float(
                chrf
            ),

        "chrf++":
            float(
                chrfpp
            ),

        "exact_match_percent":
            exact,

        "avg_latency_seconds":
            latency,
    }


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    prediction_file = (
        project_root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "zh_en"
        /
        "baseline_v1"
        /
        "flores_devtest_predictions.parquet"
    )

    output_dir = (
        project_root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "zh_en"
        /
        "baseline_v1"
    )

    output_csv = (
        output_dir
        /
        "baseline_metrics_corrected_v1.csv"
    )

    output_json = (
        output_dir
        /
        "baseline_metrics_corrected_v1.json"
    )

    report_file = (
        output_dir
        /
        "baseline_metric_correction_report_v1.json"
    )

    print("=" * 100)
    print("ZH-EN BASELINE PIPELINE")
    print(
        "STEP 13B-1 - LANGUAGE-AWARE "
        "METRIC RECALCULATION"
    )
    print("=" * 100)

    print(
        "\nPredictions:"
    )

    print(
        prediction_file
    )

    if not prediction_file.exists():

        raise FileNotFoundError(
            prediction_file
        )

    df = pd.read_parquet(
        prediction_file
    )

    required = {
        "direction",
        "prediction",
        "reference",
        "latency_seconds",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "\nRows:",
        len(df)
    )

    print(
        "\nDirections:"
    )

    print(
        df["direction"]
        .value_counts()
        .to_string()
    )

    results = []

    for direction in [
        "zh_en",
        "en_zh",
    ]:

        part = (
            df[
                df["direction"]
                ==
                direction
            ]
            .copy()
        )

        if len(part) == 0:

            raise RuntimeError(
                f"No samples for {direction}"
            )

        result = (
            calculate_metrics(
                part,
                direction,
            )
        )

        results.append(
            result
        )

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    report = {

        "step":
            "13B-1",

        "purpose":
            (
                "Correct ZH-EN baseline evaluation "
                "using target-language-aware BLEU "
                "and character-level chrF."
            ),

        "inference_rerun":
            False,

        "prediction_file":
            str(
                prediction_file
            ),

        "metric_policy": {

            "zh_en": {

                "target_language":
                    "English",

                "bleu_tokenizer":
                    "13a",
            },

            "en_zh": {

                "target_language":
                    "Chinese",

                "bleu_tokenizer":
                    "zh",
            },

            "chrf":
                "CHRF(word_order=0)",

            "chrf++":
                "CHRF(word_order=2)",
        },

        "directions": {

            row[
                "direction"
            ]:
                {
                    key:
                        value

                    for key, value
                    in row.items()

                    if key
                    !=
                    "direction"
                }

            for row
            in results
        },

        "status":
            "CORRECTED_METRICS_READY",
    }

    with open(
        output_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

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

    print("\n")
    print("=" * 110)
    print("CORRECTED BASELINE RESULT")
    print("=" * 110)

    print(
        summary[
            [
                "direction",
                "samples",
                "bleu",
                "bleu_tokenizer",
                "chrf",
                "chrf++",
                "exact_match_percent",
                "avg_latency_seconds",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 100)
    print("METRIC POLICY")
    print("=" * 100)

    print(
        "ZH -> EN:"
    )

    print(
        "  BLEU tokenizer = 13a"
    )

    print()

    print(
        "EN -> ZH:"
    )

    print(
        "  BLEU tokenizer = zh"
    )

    print()

    print(
        "chrF  = character-level"
    )

    print(
        "chrF++ = retained for "
        "cross-experiment reference"
    )

    print("\n")
    print(
        "Inference rerun: False"
    )

    print()

    print(
        "STATUS:"
    )

    print(
        "CORRECTED_METRICS_READY"
    )

    print()

    print(
        "CSV:"
    )

    print(
        output_csv
    )

    print()

    print(
        "JSON:"
    )

    print(
        output_json
    )


if __name__ == "__main__":

    main()