from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch

from transformers import (
    M2M100ForConditionalGeneration,
)


# ============================================================
# Args
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 13D - Original SMaLL-100 "
            "ZH<->EN Tatoeba Baseline"
        )
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
# Pair -> bidirectional
# ============================================================


def expand_bidirectional(
    df: pd.DataFrame,
) -> pd.DataFrame:

    zh_en = pd.DataFrame({

        "sample_id":
            df["pair_id"]
            .astype(str)
            +
            "_zh_en",

        "pair_id":
            df["pair_id"],

        "dataset":
            "tatoeba",

        "split":
            "test",

        "direction":
            "zh_en",

        "source_lang":
            "zh",

        "target_lang":
            "en",

        "source":
            df["zh"],

        "reference":
            df["en"],
    })

    en_zh = pd.DataFrame({

        "sample_id":
            df["pair_id"]
            .astype(str)
            +
            "_en_zh",

        "pair_id":
            df["pair_id"],

        "dataset":
            "tatoeba",

        "split":
            "test",

        "direction":
            "en_zh",

        "source_lang":
            "en",

        "target_lang":
            "zh",

        "source":
            df["en"],

        "reference":
            df["zh"],
    })

    return pd.concat(
        [
            zh_en,
            en_zh,
        ],
        ignore_index=True,
    )


# ============================================================
# Metrics
# ============================================================


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

    # --------------------------------------------------------
    # Target-language-aware BLEU
    # --------------------------------------------------------

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
            [
                references,
            ],
            tokenize=
                bleu_tokenizer,
        )
        .score
    )

    # --------------------------------------------------------
    # Character-level chrF
    # --------------------------------------------------------

    chrf = (
        sacrebleu
        .CHRF(
            word_order=0,
        )
        .corpus_score(
            predictions,
            [
                references,
            ],
        )
        .score
    )

    # --------------------------------------------------------
    # chrF++
    # Keep for cross-experiment reference
    # --------------------------------------------------------

    chrfpp = (
        sacrebleu
        .CHRF(
            word_order=2,
        )
        .corpus_score(
            predictions,
            [
                references,
            ],
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
        df[
            "latency_seconds"
        ]
        .mean()
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


# ============================================================
# Inference
# ============================================================


def evaluate(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
    device,
) -> pd.DataFrame:

    result_rows = []

    directions = [

        (
            "zh_en",
            "en",
        ),

        (
            "en_zh",
            "zh",
        ),
    ]

    for (
        direction,
        target_lang,
    ) in directions:

        part = (
            test_df[
                test_df[
                    "direction"
                ]
                ==
                direction
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        tokenizer.tgt_lang = (
            target_lang
        )

        print("\n")
        print(
            "=" * 90
        )

        print(
            direction.upper()
        )

        print(
            "=" * 90
        )

        print(
            "Target language:",
            target_lang
        )

        print(
            "Samples:",
            len(part)
        )

        for start in range(
            0,
            len(part),
            batch_size,
        ):

            batch = (
                part.iloc[
                    start:
                    start
                    +
                    batch_size
                ]
            )

            sources = (
                batch[
                    "source"
                ]
                .astype(str)
                .tolist()
            )

            inputs = tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=
                    max_source_length,
            )

            inputs = {

                key:
                    value.to(
                        device
                    )

                for key, value
                in inputs.items()
            }

            torch.cuda.synchronize()

            t0 = (
                time.perf_counter()
            )

            with torch.inference_mode():

                outputs = (
                    model.generate(
                        **inputs,

                        num_beams=
                            num_beams,

                        max_new_tokens=
                            max_new_tokens,

                        do_sample=False,

                        use_cache=True,

                        pad_token_id=
                            tokenizer
                            .pad_token_id,
                    )
                )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            predictions = (
                tokenizer
                .batch_decode(
                    outputs,
                    skip_special_tokens=True,
                )
            )

            latency = (
                elapsed
                /
                len(batch)
            )

            for (
                (_, row),
                prediction,
            ) in zip(
                batch.iterrows(),
                predictions,
            ):

                result_rows.append({

                    **row.to_dict(),

                    "model":
                        "small100_original",

                    "prediction":
                        str(
                            prediction
                        ).strip(),

                    "latency_seconds":
                        latency,
                })

            done = min(
                start
                +
                batch_size,
                len(part),
            )

            if (
                done % 250
                <
                batch_size
                or
                done
                ==
                len(part)
            ):

                print(
                    f"{done}/{len(part)}"
                )

    return pd.DataFrame(
        result_rows
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

    model_path = Path(
        "/root/autodl-tmp/models/small100"
    )

    benchmark_file = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
        /
        "tatoeba_zh_en_test_v1.parquet"
    )

    benchmark_report_file = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
        /
        "tatoeba_zh_en_test_v1_report.json"
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
        /
        "tatoeba"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_parquet = (
        output_dir
        /
        "tatoeba_predictions.parquet"
    )

    predictions_csv = (
        output_dir
        /
        "tatoeba_predictions.csv"
    )

    metrics_csv = (
        output_dir
        /
        "tatoeba_metrics.csv"
    )

    metrics_json = (
        output_dir
        /
        "tatoeba_metrics.json"
    )

    report_file = (
        output_dir
        /
        "tatoeba_baseline_report.json"
    )

    print(
        "=" * 100
    )

    print(
        "ZH-EN BASELINE PIPELINE"
    )

    print(
        "STEP 13D - ORIGINAL SMALL-100 "
        "TATOEBA BASELINE"
    )

    print(
        "=" * 100
    )

    print(
        "\nProject root:"
    )

    print(
        project_root
    )

    print(
        "\nModel:"
    )

    print(
        model_path
    )

    print(
        "\nBenchmark:"
    )

    print(
        benchmark_file
    )

    # ========================================================
    # Required files
    # ========================================================

    for path in [

        model_path,
        benchmark_file,
        benchmark_report_file,

    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
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

    # ========================================================
    # Benchmark
    # ========================================================

    benchmark = (
        pd.read_parquet(
            benchmark_file
        )
    )

    required_columns = {

        "pair_id",
        "en",
        "zh",
    }

    missing = (
        required_columns
        -
        set(
            benchmark.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Benchmark missing columns: "
            f"{sorted(missing)}"
        )

    print(
        "\nBenchmark pairs:",
        len(benchmark)
    )

    test_df = (
        expand_bidirectional(
            benchmark
        )
    )

    print(
        "Directed samples:",
        len(test_df)
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer_python = (
        model_path
        /
        "tokenization_small100.py"
    )

    if not tokenizer_python.exists():

        raise FileNotFoundError(
            tokenizer_python
        )

    sys.path.insert(
        0,
        str(
            model_path
        ),
    )

    from tokenization_small100 import (
        SMALL100Tokenizer
    )

    tokenizer = (
        SMALL100Tokenizer
        .from_pretrained(
            str(
                model_path
            ),
            tgt_lang="en",
            local_files_only=True,
        )
    )

    print(
        "\nTokenizer loaded."
    )

    print(
        "zh language id:",
        tokenizer
        .lang_code_to_id
        .get(
            "zh"
        )
    )

    print(
        "en language id:",
        tokenizer
        .lang_code_to_id
        .get(
            "en"
        )
    )

    if (
        tokenizer
        .lang_code_to_id
        .get(
            "zh"
        )
        is None
    ):

        raise RuntimeError(
            "Tokenizer does not support zh."
        )

    if (
        tokenizer
        .lang_code_to_id
        .get(
            "en"
        )
        is None
    ):

        raise RuntimeError(
            "Tokenizer does not support en."
        )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nLoading SMaLL-100..."
    )

    model = (
        M2M100ForConditionalGeneration
        .from_pretrained(
            str(
                model_path
            ),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(
            device
        )
    )

    model.eval()

    model.config.use_cache = True

    print(
        "Model loaded."
    )

    # ========================================================
    # Inference
    # ========================================================

    if (
        predictions_parquet.exists()
        and
        not args.overwrite
    ):

        print(
            "\nUsing existing predictions:"
        )

        print(
            predictions_parquet
        )

        result_df = (
            pd.read_parquet(
                predictions_parquet
            )
        )

    else:

        result_df = evaluate(

            model=
                model,

            tokenizer=
                tokenizer,

            test_df=
                test_df,

            batch_size=
                args.batch_size,

            num_beams=
                args.num_beams,

            max_source_length=
                args
                .max_source_length,

            max_new_tokens=
                args
                .max_new_tokens,

            device=
                device,
        )

        result_df.to_parquet(
            predictions_parquet,
            index=False,
        )

        result_df.to_csv(
            predictions_csv,
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # Release GPU
    # ========================================================

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()

    # ========================================================
    # Integrity
    # ========================================================

    expected_samples = (
        len(benchmark)
        *
        2
    )

    if (
        len(result_df)
        !=
        expected_samples
    ):

        raise RuntimeError(
            "\nInference incomplete.\n"
            f"Expected: {expected_samples}\n"
            f"Found: {len(result_df)}"
        )

    empty_predictions = int(
        result_df[
            "prediction"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "\nEmpty predictions:",
        empty_predictions
    )

    # ========================================================
    # Metrics
    # ========================================================

    results = []

    metrics = {}

    for direction in [

        "zh_en",
        "en_zh",

    ]:

        part = (
            result_df[
                result_df[
                    "direction"
                ]
                ==
                direction
            ]
            .copy()
        )

        metric = (
            calculate_metrics(
                part,
                direction,
            )
        )

        metrics[
            direction
        ] = (
            metric
        )

        results.append({

            "direction":
                direction,

            **metric,
        })

    summary = (
        pd.DataFrame(
            results
        )
    )

    summary.to_csv(
        metrics_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Benchmark metadata
    # ========================================================

    with open(
        benchmark_report_file,
        "r",
        encoding="utf-8",
    ) as f:

        benchmark_report = (
            json.load(
                f
            )
        )

    # ========================================================
    # Report
    # ========================================================

    report = {

        "step":
            "13D",

        "experiment":
            (
                "small100_original_"
                "zh_en_tatoeba_baseline"
            ),

        "model":
            "small100_original",

        "model_path":
            str(
                model_path
            ),

        "benchmark":
            str(
                benchmark_file
            ),

        "benchmark_release":
            benchmark_report
            .get(
                "release"
            ),

        "pairs":
            int(
                len(benchmark)
            ),

        "directed_samples":
            int(
                expected_samples
            ),

        "generation": {

            "batch_size":
                args.batch_size,

            "num_beams":
                args.num_beams,

            "max_source_length":
                args
                .max_source_length,

            "max_new_tokens":
                args
                .max_new_tokens,

            "do_sample":
                False,
        },

        "metric_policy": {

            "zh_en_bleu":
                "13a",

            "en_zh_bleu":
                "zh",

            "chrf":
                "word_order=0",

            "chrf++":
                "word_order=2",
        },

        "directions":
            metrics,

        "status":
            "TATOEBA_BASELINE_COMPLETE",
    }

    with open(
        metrics_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "directions":
                    metrics
            },
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

    # ========================================================
    # Console
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 13D RESULT"
    )

    print(
        "=" * 110
    )

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
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print("\n")
    print(
        "Metric policy:"
    )

    print(
        "ZH -> EN BLEU = 13a"
    )

    print(
        "EN -> ZH BLEU = zh"
    )

    print(
        "chrF = character level"
    )

    print(
        "chrF++ = word_order 2"
    )

    print("\n")
    print(
        "Predictions:"
    )

    print(
        predictions_parquet
    )

    print("\n")
    print(
        "Metrics:"
    )

    print(
        metrics_csv
    )

    print("\n")
    print(
        "Report:"
    )

    print(
        report_file
    )

    print("\n")
    print(
        "STATUS:"
    )

    print(
        "TATOEBA_BASELINE_COMPLETE"
    )


if __name__ == "__main__":

    main()