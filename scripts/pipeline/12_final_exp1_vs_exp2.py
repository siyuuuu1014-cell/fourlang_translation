from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch


def load_module(name: str, path: Path):

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_allowed_regression",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def build_summary(
    step09a,
    df,
):

    rows = []

    for dataset in [
        "benchmark",
        "challenge",
    ]:

        for direction in [
            "en_uz",
            "uz_en",
        ]:

            part = df[
                (df["dataset"] == dataset)
                &
                (df["direction"] == direction)
            ]

            metrics = step09a.calculate_metrics(
                part,
                prediction_column="prediction_eval",
                reference_column="reference_eval",
            )

            rows.append({
                "dataset":
                    dataset,

                "direction":
                    direction,

                **metrics,
            })

    return pd.DataFrame(
        rows
    )


def build_challenge_summary(
    step09a,
    df,
):

    rows = []

    challenge = df[
        df["dataset"]
        ==
        "challenge"
    ]

    categories = sorted(
        challenge[
            "category"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    for category in categories:

        for direction in [
            "en_uz",
            "uz_en",
        ]:

            part = challenge[
                (
                    challenge[
                        "category"
                    ]
                    ==
                    category
                )
                &
                (
                    challenge[
                        "direction"
                    ]
                    ==
                    direction
                )
            ]

            if len(part) == 0:
                continue

            metrics = (
                step09a
                .calculate_metrics(
                    part,
                    prediction_column=
                        "prediction_eval",
                    reference_column=
                        "reference_eval",
                )
            )

            rows.append({
                "category":
                    category,

                "direction":
                    direction,

                **metrics,
            })

    return pd.DataFrame(
        rows
    )


def compare(
    exp1,
    exp2,
    keys,
):

    metric_columns = [
        "samples",
        "bleu",
        "chrf++",
        "exact_percent",
        "avg_latency_seconds",
    ]

    a = exp1[
        keys
        +
        metric_columns
    ].copy()

    b = exp2[
        keys
        +
        metric_columns
    ].copy()

    a = a.rename(
        columns={
            "samples":
                "exp1_samples",

            "bleu":
                "exp1_bleu",

            "chrf++":
                "exp1_chrf++",

            "exact_percent":
                "exp1_exact",

            "avg_latency_seconds":
                "exp1_latency",
        }
    )

    b = b.rename(
        columns={
            "samples":
                "exp2_samples",

            "bleu":
                "exp2_bleu",

            "chrf++":
                "exp2_chrf++",

            "exact_percent":
                "exp2_exact",

            "avg_latency_seconds":
                "exp2_latency",
        }
    )

    result = a.merge(
        b,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    result["delta_bleu"] = (
        result["exp2_bleu"]
        -
        result["exp1_bleu"]
    )

    result["delta_chrf++"] = (
        result["exp2_chrf++"]
        -
        result["exp1_chrf++"]
    )

    result["delta_exact"] = (
        result["exp2_exact"]
        -
        result["exp1_exact"]
    )

    result["delta_latency"] = (
        result["exp2_latency"]
        -
        result["exp1_latency"]
    )

    return result


def main():

    args = parse_args()

    root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    pipeline_dir = (
        root
        /
        "scripts"
        /
        "pipeline"
    )

    step09 = load_module(
        "step09",
        pipeline_dir
        /
        "09_eval_heldout_en_uz.py",
    )

    step09a = load_module(
        "step09a",
        pipeline_dir
        /
        "09a_normalize_madlad_uzbek.py",
    )

    benchmark_file = (
        root
        /
        "data"
        /
        "benchmark"
        /
        "en_uz"
        /
        "tatoeba_en_uz_500.csv"
    )

    challenge_file = (
        root
        /
        "data"
        /
        "benchmark"
        /
        "en_uz"
        /
        "challenge_v1.csv"
    )

    exp2_model = (
        root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "exp2_distillation_v1"
        /
        "best_model"
    )

    exp1_prediction_file = (
        root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "heldout_eval_v1"
        /
        "script_normalized"
        /
        "all_predictions_script_normalized.parquet"
    )

    validation_file = (
        root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "exp2_distillation_v1"
        /
        "evaluation"
        /
        "exp1_vs_exp2_validation.json"
    )

    output_dir = (
        root
        /
        "results"
        /
        "student"
        /
        "small100"
        /
        "exp2_distillation_v1"
        /
        "final_heldout_eval"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    exp2_prediction_file = (
        output_dir
        /
        "exp2_predictions.parquet"
    )

    exp2_normalized_file = (
        output_dir
        /
        "exp2_predictions_normalized.parquet"
    )

    comparison_file = (
        output_dir
        /
        "exp1_vs_exp2_heldout.csv"
    )

    category_file = (
        output_dir
        /
        "exp1_vs_exp2_challenge_categories.csv"
    )

    report_file = (
        output_dir
        /
        "final_decision.json"
    )

    print("=" * 110)
    print("EN-UZ STUDENT PIPELINE")
    print(
        "STEP 12 - FINAL EXP1 VS EXP2"
    )
    print("=" * 110)

    for path in [
        benchmark_file,
        challenge_file,
        exp2_model,
        exp1_prediction_file,
        validation_file,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # =====================================================
    # Held-out dataset
    # =====================================================

    benchmark_pairs = (
        step09
        .normalize_pair_dataset(
            step09.read_table(
                benchmark_file
            ),
            "benchmark",
        )
    )

    challenge_pairs = (
        step09
        .normalize_pair_dataset(
            step09.read_table(
                challenge_file
            ),
            "challenge",
        )
    )

    test_df = pd.concat(
        [
            step09.expand_bidirectional(
                benchmark_pairs
            ),

            step09.expand_bidirectional(
                challenge_pairs
            ),
        ],
        ignore_index=True,
    )

    print(
        "\nBenchmark pairs:",
        len(benchmark_pairs)
    )

    print(
        "Challenge pairs:",
        len(challenge_pairs)
    )

    print(
        "Directed samples:",
        len(test_df)
    )

    # =====================================================
    # Exp1 corrected predictions
    # =====================================================

    all_old = pd.read_parquet(
        exp1_prediction_file
    )

    exp1 = (
        all_old[
            all_old["model"]
            ==
            "student_exp1"
        ]
        .copy()
    )

    print(
        "\nExp1 predictions:",
        len(exp1)
    )

    # =====================================================
    # Exp2
    # =====================================================

    if (
        exp2_normalized_file.exists()
        and
        not args.overwrite
    ):

        print(
            "\nUsing existing Exp2 predictions."
        )

        exp2 = pd.read_parquet(
            exp2_normalized_file
        )

    else:

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA unavailable."
            )

        sys.path.insert(
            0,
            str(
                exp2_model
            ),
        )

        from tokenization_small100 import (
            SMALL100Tokenizer
        )

        exp2_raw = (
            step09
            .evaluate_small100(
                model_name=
                    "student_exp2",

                model_path=
                    exp2_model,

                tokenizer_class=
                    SMALL100Tokenizer,

                test_df=
                    test_df,

                batch_size=
                    args.batch_size,

                num_beams=
                    args.num_beams,

                max_source_length=
                    args.max_source_length,

                max_new_tokens=
                    args.max_new_tokens,

                device=
                    torch.device(
                        "cuda"
                    ),
            )
        )

        exp2_raw.to_parquet(
            exp2_prediction_file,
            index=False,
        )

        # ================================================
        # Apply EXACT Step09A normalization
        # ================================================

        exp2 = exp2_raw.copy()

        exp2[
            "prediction_raw"
        ] = exp2[
            "prediction"
        ].astype(str)

        prediction_eval = []
        reference_eval = []
        converted = []

        for _, row in exp2.iterrows():

            pred, was_cyrillic = (
                step09a
                .prepare_prediction(
                    row
                )
            )

            ref = (
                step09a
                .prepare_reference(
                    row
                )
            )

            prediction_eval.append(
                pred
            )

            reference_eval.append(
                ref
            )

            converted.append(
                was_cyrillic
            )

        exp2[
            "prediction_eval"
        ] = prediction_eval

        exp2[
            "reference_eval"
        ] = reference_eval

        exp2[
            "was_cyrillic"
        ] = converted

        exp2.to_parquet(
            exp2_normalized_file,
            index=False,
        )

    # =====================================================
    # Integrity
    # =====================================================

    if len(exp1) != len(test_df):

        raise RuntimeError(
            "Exp1 held-out size mismatch."
        )

    if len(exp2) != len(test_df):

        raise RuntimeError(
            "Exp2 held-out size mismatch."
        )

    exp1_ids = set(
        exp1["sample_id"]
        .astype(str)
    )

    exp2_ids = set(
        exp2["sample_id"]
        .astype(str)
    )

    test_ids = set(
        test_df["sample_id"]
        .astype(str)
    )

    if not (
        exp1_ids
        ==
        exp2_ids
        ==
        test_ids
    ):

        raise RuntimeError(
            "Frozen held-out sample IDs differ."
        )

    # =====================================================
    # Main metrics
    # =====================================================

    exp1_summary = build_summary(
        step09a,
        exp1,
    )

    exp2_summary = build_summary(
        step09a,
        exp2,
    )

    comparison = compare(
        exp1_summary,
        exp2_summary,
        [
            "dataset",
            "direction",
        ],
    )

    # =====================================================
    # Challenge categories
    # =====================================================

    exp1_categories = (
        build_challenge_summary(
            step09a,
            exp1,
        )
    )

    exp2_categories = (
        build_challenge_summary(
            step09a,
            exp2,
        )
    )

    category_comparison = compare(
        exp1_categories,
        exp2_categories,
        [
            "category",
            "direction",
        ],
    )

    # =====================================================
    # Validation result
    # =====================================================

    with open(
        validation_file,
        "r",
        encoding="utf-8",
    ) as f:

        validation = json.load(
            f
        )

    validation_decision = (
        validation[
            "summary"
        ][
            "decision"
        ]
    )

    # =====================================================
    # Final decision
    # =====================================================

    benchmark_en_uz = comparison[
        (
            comparison[
                "dataset"
            ]
            ==
            "benchmark"
        )
        &
        (
            comparison[
                "direction"
            ]
            ==
            "en_uz"
        )
    ].iloc[0]

    mean_delta_bleu = float(
        comparison[
            "delta_bleu"
        ]
        .mean()
    )

    mean_delta_chrf = float(
        comparison[
            "delta_chrf++"
        ]
        .mean()
    )

    min_bleu = float(
        comparison[
            "delta_bleu"
        ]
        .min()
    )

    min_chrf = float(
        comparison[
            "delta_chrf++"
        ]
        .min()
    )

    primary_pass = (
        benchmark_en_uz[
            "delta_bleu"
        ]
        >
        0
        and
        benchmark_en_uz[
            "delta_chrf++"
        ]
        >
        0
    )

    overall_positive = (
        mean_delta_bleu > 0
        and
        mean_delta_chrf > 0
    )

    no_large_regression = (
        min_bleu
        >=
        -args.max_allowed_regression
        and
        min_chrf
        >=
        -args.max_allowed_regression
    )

    if (
        validation_decision
        ==
        "VALIDATION_PASS_BOTH_DIRECTIONS"
        and
        primary_pass
        and
        overall_positive
        and
        no_large_regression
    ):

        decision = (
            "EXP2_ACCEPT"
        )

    elif (
        validation_decision
        ==
        "VALIDATION_PASS_BOTH_DIRECTIONS"
        and
        overall_positive
    ):

        decision = (
            "EXP2_MIXED_REVIEW"
        )

    else:

        decision = (
            "KEEP_EXP1"
        )

    # =====================================================
    # Save
    # =====================================================

    comparison.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig",
    )

    category_comparison.to_csv(
        category_file,
        index=False,
        encoding="utf-8-sig",
    )

    report = {

        "validation_decision":
            validation_decision,

        "benchmark_en_uz_delta_bleu":
            float(
                benchmark_en_uz[
                    "delta_bleu"
                ]
            ),

        "benchmark_en_uz_delta_chrf++":
            float(
                benchmark_en_uz[
                    "delta_chrf++"
                ]
            ),

        "mean_heldout_delta_bleu":
            mean_delta_bleu,

        "mean_heldout_delta_chrf++":
            mean_delta_chrf,

        "minimum_delta_bleu":
            min_bleu,

        "minimum_delta_chrf++":
            min_chrf,

        "max_allowed_regression":
            args.max_allowed_regression,

        "primary_pass":
            bool(
                primary_pass
            ),

        "overall_positive":
            bool(
                overall_positive
            ),

        "no_large_regression":
            bool(
                no_large_regression
            ),

        "decision":
            decision,
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

    # =====================================================
    # Console
    # =====================================================

    print("\n")
    print("=" * 120)
    print(
        "EXP1 VS EXP2 HELD-OUT"
    )
    print("=" * 120)

    print(
        comparison[
            [
                "dataset",
                "direction",
                "exp1_bleu",
                "exp2_bleu",
                "delta_bleu",
                "exp1_chrf++",
                "exp2_chrf++",
                "delta_chrf++",
                "exp1_exact",
                "exp2_exact",
                "delta_exact",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 120)
    print(
        "CHALLENGE CATEGORIES"
    )
    print("=" * 120)

    print(
        category_comparison[
            [
                "category",
                "direction",
                "exp1_bleu",
                "exp2_bleu",
                "delta_bleu",
                "exp1_chrf++",
                "exp2_chrf++",
                "delta_chrf++",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 100)
    print("FINAL DECISION")
    print("=" * 100)

    print(
        "Validation:",
        validation_decision
    )

    print(
        "Tatoeba EN->UZ ΔBLEU:",
        f"{benchmark_en_uz['delta_bleu']:+.4f}"
    )

    print(
        "Tatoeba EN->UZ ΔchrF++:",
        f"{benchmark_en_uz['delta_chrf++']:+.4f}"
    )

    print(
        "Mean ΔBLEU:",
        f"{mean_delta_bleu:+.4f}"
    )

    print(
        "Mean ΔchrF++:",
        f"{mean_delta_chrf:+.4f}"
    )

    print(
        "Minimum ΔBLEU:",
        f"{min_bleu:+.4f}"
    )

    print(
        "Minimum ΔchrF++:",
        f"{min_chrf:+.4f}"
    )

    print(
        "No large regression:",
        no_large_regression
    )

    print(
        "\nDECISION:",
        decision
    )

    print("\n")
    print("=" * 100)
    print("STEP 12 COMPLETE")
    print("=" * 100)

    print(
        "\nComparison:"
    )

    print(
        comparison_file
    )

    print(
        "\nCategory comparison:"
    )

    print(
        category_file
    )

    print(
        "\nFinal report:"
    )

    print(
        report_file
    )


if __name__ == "__main__":

    main()