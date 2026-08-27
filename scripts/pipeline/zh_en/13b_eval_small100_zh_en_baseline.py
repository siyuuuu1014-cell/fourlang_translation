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
            "Step 13B - Original SMaLL-100 "
            "ZH<->EN FLORES+ Baseline"
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
# Expand pair -> 2 directions
# ============================================================


def expand_bidirectional(
    df: pd.DataFrame,
) -> pd.DataFrame:

    zh_en = pd.DataFrame({

        "sample_id":
            df["pair_id"].astype(str)
            +
            "_zh_en",

        "pair_id":
            df["pair_id"],

        "flores_id":
            df["flores_id"],

        "dataset":
            "flores_plus",

        "split":
            df["split"],

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
            df["pair_id"].astype(str)
            +
            "_en_zh",

        "pair_id":
            df["pair_id"],

        "flores_id":
            df["flores_id"],

        "dataset":
            "flores_plus",

        "split":
            df["split"],

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
) -> dict:

    predictions = (
        df["prediction"]
        .astype(str)
        .tolist()
    )

    references = (
        df["reference"]
        .astype(str)
        .tolist()
    )

    bleu = (
        sacrebleu
        .corpus_bleu(
            predictions,
            [
                references,
            ],
        )
        .score
    )

    chrf = (
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

    exact_match = float(
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

        "exact_match_percent":
            exact_match,

        "avg_latency_seconds":
            latency,
    }


# ============================================================
# Evaluate
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

    direction_config = [

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
    ) in direction_config:

        part = (
            test_df[
                test_df["direction"]
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

        print()
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
                batch["source"]
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
                done % 256
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

    # --------------------------------------------------------
    # Because script is now:
    #
    # scripts/pipeline/zh_en/13b_...
    #
    # parents[3] = project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    model_path = (
        Path(
            "/root/autodl-tmp/models/small100"
        )
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
        "flores_plus_zh_en_devtest_v1.parquet"
    )

    manifest_file = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
        /
        "benchmark_manifest_v1.json"
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

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_parquet = (
        output_dir
        /
        "flores_devtest_predictions.parquet"
    )

    predictions_csv = (
        output_dir
        /
        "flores_devtest_predictions.csv"
    )

    metrics_file = (
        output_dir
        /
        "baseline_metrics.json"
    )

    summary_file = (
        output_dir
        /
        "baseline_metrics.csv"
    )

    report_file = (
        output_dir
        /
        "baseline_report.json"
    )

    print(
        "=" * 100
    )

    print(
        "ZH-EN BASELINE PIPELINE"
    )

    print(
        "STEP 13B - ORIGINAL SMALL-100 "
        "FLORES+ BASELINE"
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
    # Integrity
    # ========================================================

    for path in [

        model_path,
        benchmark_file,
        manifest_file,

    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        "GPU:",
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
        "flores_id",
        "zh",
        "en",
        "split",
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

    if len(
        benchmark
    ) != 1012:

        print(
            "\nWARNING:"
        )

        print(
            "Expected FLORES+ devtest "
            "around 1012 pairs, found:",
            len(benchmark)
        )

    test_df = (
        expand_bidirectional(
            benchmark
        )
    )

    print()
    print(
        "Benchmark pairs:",
        len(benchmark)
    )

    print(
        "Directed samples:",
        len(test_df)
    )

    # ========================================================
    # Load tokenizer
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

    # ========================================================
    # Load model
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

    model.config.use_cache = (
        True
    )

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
    # Integrity after inference
    # ========================================================

    if len(
        result_df
    ) != len(
        test_df
    ):

        raise RuntimeError(

            "Inference incomplete.\n"

            f"Expected: {len(test_df)}\n"

            f"Found: {len(result_df)}"
        )

    if (
        result_df[
            "prediction"
        ]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        empty_count = int(
            result_df[
                "prediction"
            ]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(
            "\nWARNING:"
        )

        print(
            "Empty predictions:",
            empty_count
        )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = {}

    summary_rows = []

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

        direction_metrics = (
            calculate_metrics(
                part
            )
        )

        metrics[
            direction
        ] = (
            direction_metrics
        )

        summary_rows.append({

            "direction":
                direction,

            **direction_metrics,
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ========================================================
    # Load benchmark manifest
    # ========================================================

    with open(
        manifest_file,
        "r",
        encoding="utf-8",
    ) as f:

        benchmark_manifest = (
            json.load(
                f
            )
        )

    # ========================================================
    # Report
    # ========================================================

    mean_bleu = float(
        summary_df[
            "bleu"
        ]
        .mean()
    )

    mean_chrf = float(
        summary_df[
            "chrf++"
        ]
        .mean()
    )

    report = {

        "step":
            "13B",

        "experiment":
            (
                "small100_original_"
                "zh_en_flores_plus_baseline"
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

        "benchmark_revision":
            benchmark_manifest.get(
                "dataset_revision"
            ),

        "pairs":
            int(
                len(
                    benchmark
                )
            ),

        "directed_samples":
            int(
                len(
                    test_df
                )
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

        "directions":
            metrics,

        "mean_bleu":
            mean_bleu,

        "mean_chrf++":
            mean_chrf,

        "status":
            "BASELINE_EVALUATION_COMPLETE",
    }

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "model":
                    "small100_original",

                "directions":
                    metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
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
        "=" * 100
    )

    print(
        "STEP 13B RESULT"
    )

    print(
        "=" * 100
    )

    print(
        summary_df[
            [
                "direction",
                "samples",
                "bleu",
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

    print()

    print(
        "Mean BLEU:",
        f"{mean_bleu:.4f}"
    )

    print(
        "Mean chrF++:",
        f"{mean_chrf:.4f}"
    )

    print()

    print(
        "Predictions:"
    )

    print(
        predictions_parquet
    )

    print()

    print(
        "Metrics:"
    )

    print(
        metrics_file
    )

    print()

    print(
        "Report:"
    )

    print(
        report_file
    )

    print()

    print(
        "STATUS:"
    )

    print(
        "BASELINE_EVALUATION_COMPLETE"
    )


if __name__ == "__main__":

    main()