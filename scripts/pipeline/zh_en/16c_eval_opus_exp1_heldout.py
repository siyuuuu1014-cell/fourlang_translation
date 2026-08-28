from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from sacrebleu.metrics import BLEU, CHRF

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


STEP_VERSION = "16C_V1"


# ============================================================
# Frozen direction configuration
# ============================================================

DIRECTIONS = {

    "zh_en": {

        "display_name":
            "OPUS-MT zh-en Exp1",

        "source_column":
            "zh",

        "target_column":
            "en",

        "bleu_tokenizer":
            "13a",

        "best_model_relative":
            (
                "results/specialists/"
                "zh_en/"
                "opus_mt_zh_en/"
                "exp1_human/"
                "best_model"
            ),

        "training_report_relative":
            (
                "results/specialists/"
                "zh_en/"
                "opus_mt_zh_en/"
                "exp1_human/"
                "training_report.json"
            ),

        "raw_candidate":
            "opus_zh_en",

        "raw_metrics_relative":
            (
                "results/model_bakeoff/"
                "zh_en/"
                "v1/"
                "opus_zh_en/"
                "metrics.json"
            ),
    },

    "en_zh": {

        "display_name":
            "OPUS-MT en-zh Exp1",

        "source_column":
            "en",

        "target_column":
            "zh",

        "bleu_tokenizer":
            "zh",

        "best_model_relative":
            (
                "results/specialists/"
                "en_zh/"
                "opus_mt_en_zh/"
                "exp1_human/"
                "best_model"
            ),

        "training_report_relative":
            (
                "results/specialists/"
                "en_zh/"
                "opus_mt_en_zh/"
                "exp1_human/"
                "training_report.json"
            ),

        "raw_candidate":
            "opus_en_zh",

        "raw_metrics_relative":
            (
                "results/model_bakeoff/"
                "zh_en/"
                "v1/"
                "opus_en_zh/"
                "metrics.json"
            ),
    },
}


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 16C - Evaluate fine-tuned OPUS Exp1 "
            "specialists on frozen FLORES + Tatoeba."
        )
    )

    parser.add_argument(
        "--direction",
        choices=[
            "all",
            "zh_en",
            "en_zh",
        ],
        default="all",
    )

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
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def normalize_exact(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = " ".join(
        text.strip().split()
    )

    return text


def save_json(
    obj: Any,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Load frozen benchmarks
# ============================================================

def load_benchmarks(
    project_root: Path,
):

    benchmark_root = (
        project_root
        / "data"
        / "benchmark"
        / "zh_en"
    )

    files = {

        "flores_devtest":
            benchmark_root
            / "flores_plus_zh_en_devtest_v1.parquet",

        "tatoeba":
            benchmark_root
            / "tatoeba_zh_en_test_v1.parquet",
    }

    benchmarks = {}

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Frozen benchmark missing:\n{path}"
            )

        df = pd.read_parquet(
            path
        )

        required = {
            "en",
            "zh",
        }

        missing = (
            required
            -
            set(df.columns)
        )

        if missing:

            raise RuntimeError(
                f"{name} missing columns: "
                f"{sorted(missing)}"
            )

        df = (
            df[
                [
                    "en",
                    "zh",
                ]
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        benchmarks[
            name
        ] = df

    return benchmarks


# ============================================================
# Raw OPUS metrics from Step 15B
# ============================================================

def load_raw_metrics(
    project_root: Path,
    direction: str,
):

    config = DIRECTIONS[
        direction
    ]

    path = (
        project_root
        /
        config[
            "raw_metrics_relative"
        ]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Raw 15B metrics missing:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        report = json.load(
            f
        )

    rows = []

    for result in report.get(
        "results",
        [],
    ):

        if (
            result.get(
                "direction"
            )
            ==
            direction
        ):

            rows.append(
                result
            )

    if not rows:

        raise RuntimeError(
            f"No raw metrics found for {direction}"
        )

    by_benchmark = {
        row[
            "benchmark"
        ]: row
        for row in rows
    }

    required_benchmarks = {
        "flores_devtest",
        "tatoeba",
    }

    missing = (
        required_benchmarks
        -
        set(
            by_benchmark.keys()
        )
    )

    if missing:

        raise RuntimeError(
            f"Missing raw benchmark metrics: "
            f"{sorted(missing)}"
        )

    return by_benchmark


# ============================================================
# Training metadata
# ============================================================

def load_training_metadata(
    project_root: Path,
    direction: str,
):

    config = DIRECTIONS[
        direction
    ]

    path = (
        project_root
        /
        config[
            "training_report_relative"
        ]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Training report missing:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        report = json.load(
            f
        )

    return {
        "best_epoch":
            report.get(
                "best_epoch"
            ),

        "best_source":
            report.get(
                "best_source"
            ),

        "baseline_validation_loss":
            report.get(
                "baseline_validation_loss"
            ),

        "best_validation_loss":
            report.get(
                "best_validation_loss"
            ),

        "delta_validation_loss":
            report.get(
                "delta_validation_loss"
            ),
    }


# ============================================================
# Load model
# ============================================================

def load_exp1_model(
    project_root: Path,
    direction: str,
):

    config = DIRECTIONS[
        direction
    ]

    model_path = (
        project_root
        /
        config[
            "best_model_relative"
        ]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Exp1 best model missing:\n{model_path}"
        )

    print(
        "\nBest model:"
    )

    print(
        model_path
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
    )

    print(
        "Tokenizer loaded."
    )

    print(
        "Loading Exp1 model..."
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float16,
        )
        .to(
            "cuda"
        )
    )

    model.eval()

    model.config.use_cache = True

    total_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    parameter_memory = sum(
        parameter.numel()
        *
        parameter.element_size()
        for parameter
        in model.parameters()
    )

    model_stats = {

        "total_parameters":
            int(
                total_parameters
            ),

        "loaded_parameter_memory_gib":
            float(
                parameter_memory
                /
                1024 ** 3
            ),
    }

    print(
        "Model loaded."
    )

    print(
        "Parameters:",
        f"{total_parameters:,}"
    )

    return (
        tokenizer,
        model,
        model_stats,
        model_path,
    )


# ============================================================
# Warmup
# ============================================================

def warmup_model(
    tokenizer,
    model,
    text: str,
    max_source_length: int,
):

    encoded = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )

    encoded = {
        key: value.to(
            "cuda"
        )
        for key, value
        in encoded.items()
    }

    with torch.inference_mode():

        _ = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=64,
            use_cache=True,
        )

    torch.cuda.synchronize()


# ============================================================
# Evaluate one frozen benchmark
# ============================================================

def evaluate_benchmark(
    *,
    tokenizer,
    model,
    benchmark_df: pd.DataFrame,
    benchmark_name: str,
    direction: str,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
):

    config = DIRECTIONS[
        direction
    ]

    source_column = (
        config[
            "source_column"
        ]
    )

    target_column = (
        config[
            "target_column"
        ]
    )

    bleu_tokenizer = (
        config[
            "bleu_tokenizer"
        ]
    )

    sources = (
        benchmark_df[
            source_column
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    references = (
        benchmark_df[
            target_column
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        direction,
        "|",
        benchmark_name
    )

    print(
        "=" * 110
    )

    print(
        "Samples:",
        len(sources)
    )

    warmup_model(
        tokenizer=tokenizer,
        model=model,
        text=sources[0],
        max_source_length=(
            max_source_length
        ),
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    predictions = []

    total_generation_seconds = 0.0

    total_end_to_end_seconds = 0.0

    for start in range(
        0,
        len(sources),
        batch_size,
    ):

        batch_sources = (
            sources[
                start:
                start
                +
                batch_size
            ]
        )

        batch_start = (
            time.perf_counter()
        )

        encoded = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        encoded = {
            key: value.to(
                "cuda"
            )
            for key, value
            in encoded.items()
        }

        torch.cuda.synchronize()

        generation_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            generated = (
                model.generate(
                    **encoded,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=(
                        max_new_tokens
                    ),
                    use_cache=True,
                )
            )

        torch.cuda.synchronize()

        generation_seconds = (
            time.perf_counter()
            -
            generation_start
        )

        decoded = (
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        end_to_end_seconds = (
            time.perf_counter()
            -
            batch_start
        )

        total_generation_seconds += (
            generation_seconds
        )

        total_end_to_end_seconds += (
            end_to_end_seconds
        )

        predictions.extend(
            [
                str(text).strip()
                for text in decoded
            ]
        )

        done = min(
            start
            +
            batch_size,
            len(sources),
        )

        if (
            done % 256 == 0
            or
            done == len(sources)
        ):

            print(
                f"{done}/{len(sources)}"
            )

    if (
        len(predictions)
        !=
        len(references)
    ):

        raise RuntimeError(
            "Prediction count mismatch."
        )

    # ========================================================
    # Metrics
    # ========================================================

    bleu_metric = BLEU(
        tokenize=(
            bleu_tokenizer
        )
    )

    chrf_metric = CHRF(
        word_order=0
    )

    chrfpp_metric = CHRF(
        word_order=2
    )

    bleu = (
        bleu_metric
        .corpus_score(
            predictions,
            [
                references
            ],
        )
        .score
    )

    chrf = (
        chrf_metric
        .corpus_score(
            predictions,
            [
                references
            ],
        )
        .score
    )

    chrfpp = (
        chrfpp_metric
        .corpus_score(
            predictions,
            [
                references
            ],
        )
        .score
    )

    exact_count = sum(
        normalize_exact(pred)
        ==
        normalize_exact(ref)
        for pred, ref
        in zip(
            predictions,
            references,
        )
    )

    exact_percent = (
        exact_count
        /
        len(references)
        *
        100
    )

    avg_generation_latency = (
        total_generation_seconds
        /
        len(references)
    )

    avg_end_to_end_latency = (
        total_end_to_end_seconds
        /
        len(references)
    )

    peak_gpu_memory = (
        torch.cuda
        .max_memory_allocated()
        /
        1024 ** 3
    )

    metrics = {

        "benchmark":
            benchmark_name,

        "direction":
            direction,

        "samples":
            int(
                len(references)
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

        "chrfpp":
            float(
                chrfpp
            ),

        "exact_match_percent":
            float(
                exact_percent
            ),

        "avg_generation_latency_seconds":
            float(
                avg_generation_latency
            ),

        "avg_end_to_end_latency_seconds":
            float(
                avg_end_to_end_latency
            ),

        "peak_gpu_memory_gib":
            float(
                peak_gpu_memory
            ),

        "batch_size":
            int(
                batch_size
            ),

        "num_beams":
            int(
                num_beams
            ),
    }

    prediction_df = pd.DataFrame(
        {

            "benchmark":
                benchmark_name,

            "direction":
                direction,

            "sample_index":
                range(
                    len(sources)
                ),

            "source_text":
                sources,

            "reference_text":
                references,

            "prediction_text":
                predictions,
        }
    )

    print(
        "\nBLEU:",
        f"{bleu:.4f}"
    )

    print(
        "chrF:",
        f"{chrf:.4f}"
    )

    print(
        "chrF++:",
        f"{chrfpp:.4f}"
    )

    print(
        "Exact:",
        f"{exact_percent:.3f}%"
    )

    print(
        "Generation latency:",
        f"{avg_generation_latency:.4f}s/sample"
    )

    print(
        "Peak GPU memory:",
        f"{peak_gpu_memory:.3f} GiB"
    )

    return (
        prediction_df,
        metrics,
    )


# ============================================================
# Build raw vs Exp1 comparison
# ============================================================

def build_comparison(
    raw_metrics: dict,
    exp1_metrics: list[dict],
):

    rows = []

    for exp1 in exp1_metrics:

        benchmark = exp1[
            "benchmark"
        ]

        raw = raw_metrics[
            benchmark
        ]

        row = {

            "benchmark":
                benchmark,

            "direction":
                exp1[
                    "direction"
                ],

            # ----------------------------------------------
            # BLEU
            # ----------------------------------------------

            "raw_bleu":
                float(
                    raw[
                        "bleu"
                    ]
                ),

            "exp1_bleu":
                float(
                    exp1[
                        "bleu"
                    ]
                ),

            "delta_bleu":
                float(
                    exp1[
                        "bleu"
                    ]
                    -
                    raw[
                        "bleu"
                    ]
                ),

            # ----------------------------------------------
            # chrF
            # ----------------------------------------------

            "raw_chrf":
                float(
                    raw[
                        "chrf"
                    ]
                ),

            "exp1_chrf":
                float(
                    exp1[
                        "chrf"
                    ]
                ),

            "delta_chrf":
                float(
                    exp1[
                        "chrf"
                    ]
                    -
                    raw[
                        "chrf"
                    ]
                ),

            # ----------------------------------------------
            # chrF++
            # ----------------------------------------------

            "raw_chrfpp":
                float(
                    raw[
                        "chrfpp"
                    ]
                ),

            "exp1_chrfpp":
                float(
                    exp1[
                        "chrfpp"
                    ]
                ),

            "delta_chrfpp":
                float(
                    exp1[
                        "chrfpp"
                    ]
                    -
                    raw[
                        "chrfpp"
                    ]
                ),

            # ----------------------------------------------
            # Exact
            # ----------------------------------------------

            "raw_exact_match_percent":
                float(
                    raw[
                        "exact_match_percent"
                    ]
                ),

            "exp1_exact_match_percent":
                float(
                    exp1[
                        "exact_match_percent"
                    ]
                ),

            "delta_exact_match_percent":
                float(
                    exp1[
                        "exact_match_percent"
                    ]
                    -
                    raw[
                        "exact_match_percent"
                    ]
                ),

            # ----------------------------------------------
            # Efficiency
            # ----------------------------------------------

            "raw_latency_seconds":
                float(
                    raw[
                        "avg_generation_latency_seconds"
                    ]
                ),

            "exp1_latency_seconds":
                float(
                    exp1[
                        "avg_generation_latency_seconds"
                    ]
                ),

            "raw_peak_gpu_memory_gib":
                float(
                    raw[
                        "peak_gpu_memory_gib"
                    ]
                ),

            "exp1_peak_gpu_memory_gib":
                float(
                    exp1[
                        "peak_gpu_memory_gib"
                    ]
                ),
        }

        row[
            "bleu_improved"
        ] = (
            row[
                "delta_bleu"
            ]
            >
            0
        )

        row[
            "chrfpp_improved"
        ] = (
            row[
                "delta_chrfpp"
            ]
            >
            0
        )

        row[
            "both_primary_metrics_improved"
        ] = (
            row[
                "bleu_improved"
            ]
            and
            row[
                "chrfpp_improved"
            ]
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


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

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    output_root = (
        project_root
        / "results"
        / "specialists"
        / "zh_en_exp1_heldout"
        / "v1"
    )

    if (
        output_root.exists()
        and
        args.overwrite
    ):

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN SPECIALIST PIPELINE"
    )

    print(
        "STEP 16C - OPUS EXP1 FROZEN HELD-OUT EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Beam:",
        args.num_beams
    )

    benchmarks = (
        load_benchmarks(
            project_root
        )
    )

    if (
        args.direction
        ==
        "all"
    ):

        directions = [
            "zh_en",
            "en_zh",
        ]

    else:

        directions = [
            args.direction
        ]

    all_predictions = []

    all_metrics = []

    all_comparisons = []

    direction_reports = {}

    # ========================================================
    # Each specialist loaded independently
    # ========================================================

    for direction in directions:

        print("\n")
        print(
            "#" * 110
        )

        print(
            "DIRECTION:",
            direction
        )

        print(
            "#" * 110
        )

        raw_metrics = (
            load_raw_metrics(
                project_root,
                direction,
            )
        )

        training_metadata = (
            load_training_metadata(
                project_root,
                direction,
            )
        )

        (
            tokenizer,
            model,
            model_stats,
            model_path,
        ) = load_exp1_model(
            project_root,
            direction,
        )

        direction_predictions = []

        direction_metrics = []

        for benchmark_name, benchmark_df in (
            benchmarks.items()
        ):

            prediction_df, metrics = (
                evaluate_benchmark(
                    tokenizer=tokenizer,
                    model=model,
                    benchmark_df=(
                        benchmark_df
                    ),
                    benchmark_name=(
                        benchmark_name
                    ),
                    direction=direction,
                    batch_size=(
                        args.batch_size
                    ),
                    num_beams=(
                        args.num_beams
                    ),
                    max_source_length=(
                        args.max_source_length
                    ),
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )
            )

            direction_predictions.append(
                prediction_df
            )

            direction_metrics.append(
                metrics
            )

        comparison = (
            build_comparison(
                raw_metrics,
                direction_metrics,
            )
        )

        # ====================================================
        # Direction summary
        # ====================================================

        mean_delta_bleu = float(
            comparison[
                "delta_bleu"
            ]
            .mean()
        )

        mean_delta_chrfpp = float(
            comparison[
                "delta_chrfpp"
            ]
            .mean()
        )

        strict_primary_pass = bool(
            comparison[
                "both_primary_metrics_improved"
            ]
            .all()
        )

        direction_reports[
            direction
        ] = {

            "display_name":
                DIRECTIONS[
                    direction
                ][
                    "display_name"
                ],

            "best_model":
                str(
                    model_path
                ),

            "training":
                training_metadata,

            "model_stats":
                model_stats,

            "mean_delta_bleu":
                mean_delta_bleu,

            "mean_delta_chrfpp":
                mean_delta_chrfpp,

            "strict_primary_pass":
                strict_primary_pass,
        }

        all_predictions.extend(
            direction_predictions
        )

        all_metrics.extend(
            direction_metrics
        )

        all_comparisons.append(
            comparison
        )

        print("\n")
        print(
            "-" * 110
        )

        print(
            direction,
            "RAW vs EXP1"
        )

        print(
            "-" * 110
        )

        display_columns = [
            "benchmark",
            "raw_bleu",
            "exp1_bleu",
            "delta_bleu",
            "raw_chrfpp",
            "exp1_chrfpp",
            "delta_chrfpp",
            "both_primary_metrics_improved",
        ]

        print(
            comparison[
                display_columns
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

        print(
            "\nMean ΔBLEU:",
            f"{mean_delta_bleu:+.4f}"
        )

        print(
            "Mean ΔchrF++:",
            f"{mean_delta_chrfpp:+.4f}"
        )

        print(
            "Strict primary pass:",
            strict_primary_pass
        )

        del model
        del tokenizer

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # Combine outputs
    # ========================================================

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    comparison_df = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    # ========================================================
    # Overall decision
    # ========================================================

    overall_strict_pass = bool(
        comparison_df[
            "both_primary_metrics_improved"
        ]
        .all()
    )

    all_mean_delta_bleu = float(
        comparison_df[
            "delta_bleu"
        ]
        .mean()
    )

    all_mean_delta_chrfpp = float(
        comparison_df[
            "delta_chrfpp"
        ]
        .mean()
    )

    if overall_strict_pass:

        decision = (
            "EXP1_HELDOUT_PASS_ALL_PRIMARY_METRICS"
        )

    elif (
        all_mean_delta_bleu > 0
        and
        all_mean_delta_chrfpp > 0
    ):

        decision = (
            "EXP1_HELDOUT_OVERALL_IMPROVED_REVIEW_REGRESSIONS"
        )

    else:

        decision = (
            "EXP1_HELDOUT_NOT_CONFIRMED"
        )

    # ========================================================
    # Save
    # ========================================================

    predictions_file = (
        output_root
        / "exp1_heldout_predictions_v1.parquet"
    )

    predictions_csv = (
        output_root
        / "exp1_heldout_predictions_v1.csv"
    )

    metrics_file = (
        output_root
        / "exp1_heldout_metrics_v1.csv"
    )

    comparison_file = (
        output_root
        / "raw_vs_exp1_comparison_v1.csv"
    )

    report_file = (
        output_root
        / "exp1_heldout_report_v1.json"
    )

    predictions_df.to_parquet(
        predictions_file,
        index=False,
    )

    predictions_df.to_csv(
        predictions_csv,
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        metrics_file,
        index=False,
        encoding="utf-8-sig",
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig",
    )

    report = {

        "step":
            "16C",

        "step_version":
            STEP_VERSION,

        "directions":
            directions,

        "generation_policy": {

            "batch_size":
                args.batch_size,

            "num_beams":
                args.num_beams,

            "max_source_length":
                args.max_source_length,

            "max_new_tokens":
                args.max_new_tokens,

            "do_sample":
                False,

            "precision":
                "FP16_INFERENCE",
        },

        "metric_policy": {

            "zh_en_bleu_tokenizer":
                "13a",

            "en_zh_bleu_tokenizer":
                "zh",

            "chrf_word_order":
                0,

            "chrfpp_word_order":
                2,
        },

        "direction_reports":
            direction_reports,

        "overall": {

            "mean_delta_bleu":
                all_mean_delta_bleu,

            "mean_delta_chrfpp":
                all_mean_delta_chrfpp,

            "strict_all_primary_metrics_improved":
                overall_strict_pass,

            "decision":
                decision,
        },

        "outputs": {

            "predictions":
                str(
                    predictions_file
                ),

            "metrics":
                str(
                    metrics_file
                ),

            "comparison":
                str(
                    comparison_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "EXP1_HELDOUT_EVALUATION_COMPLETE",
    }

    save_json(
        report,
        report_file,
    )

    # ========================================================
    # Final console
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 16C RESULT"
    )

    print(
        "=" * 110
    )

    display_columns = [

        "benchmark",

        "direction",

        "raw_bleu",

        "exp1_bleu",

        "delta_bleu",

        "raw_chrfpp",

        "exp1_chrfpp",

        "delta_chrfpp",

        "raw_exact_match_percent",

        "exp1_exact_match_percent",

        "both_primary_metrics_improved",
    ]

    print("\nComparison:")

    print(
        comparison_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nOverall mean ΔBLEU:",
        f"{all_mean_delta_bleu:+.4f}"
    )

    print(
        "Overall mean ΔchrF++:",
        f"{all_mean_delta_chrfpp:+.4f}"
    )

    print(
        "\nAll primary metrics improved:",
        overall_strict_pass
    )

    print(
        "\nDecision:"
    )

    print(
        decision
    )

    print(
        "\nPredictions:"
    )

    print(
        predictions_file
    )

    print(
        "\nComparison:"
    )

    print(
        comparison_file
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
        "EXP1_HELDOUT_EVALUATION_COMPLETE"
    )


if __name__ == "__main__":

    main()