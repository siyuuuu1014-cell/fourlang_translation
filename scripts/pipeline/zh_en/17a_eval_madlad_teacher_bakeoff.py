from __future__ import annotations

import argparse
import gc
import json
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from sacrebleu.metrics import BLEU, CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


STEP_VERSION = "17A_V1"


DIRECTIONS = {

    "zh_en": {
        "source_column": "zh",
        "target_column": "en",
        "target_prefix": "<2en>",
        "bleu_tokenizer": "13a",
    },

    "en_zh": {
        "source_column": "en",
        "target_column": "zh",
        "target_prefix": "<2zh>",
        "bleu_tokenizer": "zh",
    },
}


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 17A - Evaluate MADLAD-400-3B-MT "
            "as ZH-EN teacher candidate."
        )
    )

    parser.add_argument(
        "--model_path",
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
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
    obj,
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
# Resolve MADLAD path
# ============================================================

def resolve_madlad_path(
    explicit_path: str | None,
) -> Path:

    candidates = []

    if explicit_path:

        candidates.append(
            Path(explicit_path)
        )

    candidates.append(
        Path(
            "/root/autodl-tmp/models/"
            "madlad400-3b-mt"
        )
    )

    cache_snapshots = Path(
        "/root/autodl-tmp/huggingface/hub/"
        "models--google--madlad400-3b-mt/"
        "snapshots"
    )

    if cache_snapshots.exists():

        for path in sorted(
            cache_snapshots.iterdir()
        ):

            if path.is_dir():

                candidates.append(
                    path
                )

    for path in candidates:

        if (
            path.exists()
            and
            (
                path
                / "config.json"
            ).exists()
        ):

            return path

    raise FileNotFoundError(
        "\nCannot locate MADLAD model.\n"
        "Checked explicit model path, "
        "/root/autodl-tmp/models/madlad400-3b-mt "
        "and Hugging Face cache snapshots."
    )


# ============================================================
# Frozen benchmarks
# ============================================================

def load_benchmarks(
    project_root: Path,
):

    root = (
        project_root
        / "data"
        / "benchmark"
        / "zh_en"
    )

    files = {

        "flores_devtest":
            root
            / "flores_plus_zh_en_devtest_v1.parquet",

        "tatoeba":
            root
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

        benchmarks[name] = (
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

        print(
            f"{name}: {len(df)} pairs"
        )

    return benchmarks


# ============================================================
# Load OPUS Exp1 reference from Step 16C
# ============================================================

def load_opus_exp1_metrics(
    project_root: Path,
):

    path = (
        project_root
        / "results"
        / "specialists"
        / "zh_en_exp1_heldout"
        / "v1"
        / "exp1_heldout_metrics_v1.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nStep 16C metrics missing:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "benchmark",
        "direction",
        "bleu",
        "chrf",
        "chrfpp",
        "exact_match_percent",
        "avg_generation_latency_seconds",
        "peak_gpu_memory_gib",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "Step 16C metrics missing columns: "
            f"{sorted(missing)}"
        )

    result = {}

    for _, row in df.iterrows():

        key = (
            str(row["benchmark"]),
            str(row["direction"]),
        )

        result[key] = {
            column: (
                row[column]
                if column
                in row.index
                else None
            )
            for column
            in df.columns
        }

    return result


# ============================================================
# Load MADLAD
# ============================================================

def load_madlad(
    model_path: Path,
):

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    print(
        "\nMADLAD path:"
    )

    print(
        model_path
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
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
        "\nLoading MADLAD..."
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
        p.numel()
        for p in model.parameters()
    )

    parameter_memory = sum(
        p.numel()
        *
        p.element_size()
        for p in model.parameters()
    )

    stats = {

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

    print(
        "Parameter memory:",
        f"{stats['loaded_parameter_memory_gib']:.3f} GiB"
    )

    return (
        tokenizer,
        model,
        stats,
    )


# ============================================================
# Build MADLAD prompt
# ============================================================

def build_inputs(
    sources: list[str],
    direction: str,
) -> list[str]:

    prefix = (
        DIRECTIONS[
            direction
        ][
            "target_prefix"
        ]
    )

    return [
        f"{prefix} {text}"
        for text in sources
    ]


# ============================================================
# Warmup
# ============================================================

def warmup(
    tokenizer,
    model,
    text: str,
    direction: str,
    max_source_length: int,
):

    prompt = (
        build_inputs(
            [text],
            direction,
        )
    )

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_source_length,
    )

    encoded = {
        key: value.to("cuda")
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
# Evaluate
# ============================================================

def evaluate_one(
    *,
    tokenizer,
    model,
    df: pd.DataFrame,
    benchmark_name: str,
    direction: str,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
):

    config = (
        DIRECTIONS[
            direction
        ]
    )

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

    sources = (
        df[
            source_column
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    references = (
        df[
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
        "MADLAD",
        "|",
        benchmark_name,
        "|",
        direction,
    )

    print(
        "=" * 110
    )

    print(
        "Samples:",
        len(sources)
    )

    warmup(
        tokenizer=tokenizer,
        model=model,
        text=sources[0],
        direction=direction,
        max_source_length=(
            max_source_length
        ),
    )

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()

    predictions = []

    generation_seconds_total = 0.0

    end_to_end_seconds_total = 0.0

    for start in range(
        0,
        len(sources),
        batch_size,
    ):

        batch_sources = (
            sources[
                start:
                start + batch_size
            ]
        )

        prompts = (
            build_inputs(
                batch_sources,
                direction,
            )
        )

        batch_start = (
            time.perf_counter()
        )

        encoded = tokenizer(
            prompts,
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

        generation_seconds_total += (
            generation_seconds
        )

        end_to_end_seconds_total += (
            end_to_end_seconds
        )

        predictions.extend(
            [
                str(text).strip()
                for text
                in decoded
            ]
        )

        done = min(
            start + batch_size,
            len(sources),
        )

        if (
            done % 128 == 0
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
            config[
                "bleu_tokenizer"
            ]
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
        generation_seconds_total
        /
        len(references)
    )

    avg_end_to_end_latency = (
        end_to_end_seconds_total
        /
        len(references)
    )

    peak_gpu_memory = (
        torch.cuda.max_memory_allocated()
        /
        1024 ** 3
    )

    metrics = {

        "candidate":
            "madlad400_3b_mt",

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
            config[
                "bleu_tokenizer"
            ],

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

            "candidate":
                "madlad400_3b_mt",

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
# Compare MADLAD vs OPUS Exp1
# ============================================================

def build_comparison(
    madlad_metrics: pd.DataFrame,
    opus_metrics: dict,
):

    rows = []

    for _, madlad in (
        madlad_metrics.iterrows()
    ):

        benchmark = str(
            madlad[
                "benchmark"
            ]
        )

        direction = str(
            madlad[
                "direction"
            ]
        )

        key = (
            benchmark,
            direction,
        )

        if key not in opus_metrics:

            raise RuntimeError(
                f"Missing OPUS Exp1 metric for {key}"
            )

        opus = opus_metrics[
            key
        ]

        row = {

            "benchmark":
                benchmark,

            "direction":
                direction,

            # BLEU
            "opus_exp1_bleu":
                float(
                    opus[
                        "bleu"
                    ]
                ),

            "madlad_bleu":
                float(
                    madlad[
                        "bleu"
                    ]
                ),

            "delta_bleu_madlad_minus_opus":
                float(
                    madlad[
                        "bleu"
                    ]
                    -
                    opus[
                        "bleu"
                    ]
                ),

            # chrF++
            "opus_exp1_chrfpp":
                float(
                    opus[
                        "chrfpp"
                    ]
                ),

            "madlad_chrfpp":
                float(
                    madlad[
                        "chrfpp"
                    ]
                ),

            "delta_chrfpp_madlad_minus_opus":
                float(
                    madlad[
                        "chrfpp"
                    ]
                    -
                    opus[
                        "chrfpp"
                    ]
                ),

            # Exact
            "opus_exp1_exact":
                float(
                    opus[
                        "exact_match_percent"
                    ]
                ),

            "madlad_exact":
                float(
                    madlad[
                        "exact_match_percent"
                    ]
                ),

            # Latency
            "opus_exp1_latency":
                float(
                    opus[
                        "avg_generation_latency_seconds"
                    ]
                ),

            "madlad_latency":
                float(
                    madlad[
                        "avg_generation_latency_seconds"
                    ]
                ),

            # GPU
            "opus_exp1_peak_gpu_gib":
                float(
                    opus[
                        "peak_gpu_memory_gib"
                    ]
                ),

            "madlad_peak_gpu_gib":
                float(
                    madlad[
                        "peak_gpu_memory_gib"
                    ]
                ),
        }

        row[
            "madlad_wins_bleu"
        ] = (
            row[
                "delta_bleu_madlad_minus_opus"
            ]
            >
            0
        )

        row[
            "madlad_wins_chrfpp"
        ] = (
            row[
                "delta_chrfpp_madlad_minus_opus"
            ]
            >
            0
        )

        row[
            "madlad_wins_both_primary"
        ] = (
            row[
                "madlad_wins_bleu"
            ]
            and
            row[
                "madlad_wins_chrfpp"
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

    output_root = (
        project_root
        / "results"
        / "teacher_bakeoff"
        / "zh_en"
        / "v1"
        / "madlad400_3b_mt"
    )

    metrics_file = (
        output_root
        / "metrics.json"
    )

    if (
        metrics_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nExisting result found:\n"
            f"{metrics_file}\n\n"
            "Use --overwrite to rerun intentionally."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN TEACHER BAKE-OFF"
    )

    print(
        "STEP 17A - MADLAD-400-3B-MT"
    )

    print(
        "=" * 110
    )

    model_path = (
        resolve_madlad_path(
            args.model_path
        )
    )

    benchmarks = (
        load_benchmarks(
            project_root
        )
    )

    opus_exp1_metrics = (
        load_opus_exp1_metrics(
            project_root
        )
    )

    (
        tokenizer,
        model,
        model_stats,
    ) = load_madlad(
        model_path
    )

    all_predictions = []

    all_metrics = []

    # ========================================================
    # Four frozen evaluations
    # ========================================================

    for benchmark_name, df in (
        benchmarks.items()
    ):

        for direction in [
            "zh_en",
            "en_zh",
        ]:

            (
                prediction_df,
                metrics,
            ) = evaluate_one(
                tokenizer=tokenizer,
                model=model,
                df=df,
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

            all_predictions.append(
                prediction_df
            )

            all_metrics.append(
                metrics
            )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    comparison_df = (
        build_comparison(
            metrics_df,
            opus_exp1_metrics,
        )
    )

    # ========================================================
    # Direction summaries
    # ========================================================

    direction_summary = {}

    for direction in [
        "zh_en",
        "en_zh",
    ]:

        part = (
            comparison_df[
                comparison_df[
                    "direction"
                ]
                ==
                direction
            ]
        )

        mean_delta_bleu = float(
            part[
                "delta_bleu_madlad_minus_opus"
            ]
            .mean()
        )

        mean_delta_chrfpp = float(
            part[
                "delta_chrfpp_madlad_minus_opus"
            ]
            .mean()
        )

        wins_all = bool(
            part[
                "madlad_wins_both_primary"
            ]
            .all()
        )

        direction_summary[
            direction
        ] = {

            "mean_delta_bleu":
                mean_delta_bleu,

            "mean_delta_chrfpp":
                mean_delta_chrfpp,

            "madlad_wins_all_primary":
                wins_all,
        }

    # ========================================================
    # Save
    # ========================================================

    predictions_file = (
        output_root
        / "predictions.parquet"
    )

    comparison_file = (
        output_root
        / "madlad_vs_opus_exp1.csv"
    )

    report_file = (
        output_root
        / "report.json"
    )

    predictions_df.to_parquet(
        predictions_file,
        index=False,
    )

    predictions_df.to_csv(
        output_root
        / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df.to_csv(
        output_root
        / "metrics.csv",
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
            "17A",

        "step_version":
            STEP_VERSION,

        "candidate":
            "google/madlad400-3b-mt",

        "model_path":
            str(
                model_path
            ),

        "license":
            "Apache-2.0",

        "model_stats":
            model_stats,

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
                "FP16",
        },

        "language_control": {

            "zh_en":
                "<2en>",

            "en_zh":
                "<2zh>",
        },

        "metric_policy": {

            "zh_en_bleu_tokenizer":
                "13a",

            "en_zh_bleu_tokenizer":
                "zh",

            "chrf":
                "word_order=0",

            "chrfpp":
                "word_order=2",
        },

        "direction_summary":
            direction_summary,

        "status":
            "MADLAD_TEACHER_BAKEOFF_COMPLETE",

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    save_json(
        report,
        report_file,
    )

    # ========================================================
    # Console summary
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 17A RESULT"
    )

    print(
        "=" * 110
    )

    display_columns = [

        "benchmark",
        "direction",

        "opus_exp1_bleu",
        "madlad_bleu",
        "delta_bleu_madlad_minus_opus",

        "opus_exp1_chrfpp",
        "madlad_chrfpp",
        "delta_chrfpp_madlad_minus_opus",

        "madlad_wins_both_primary",
    ]

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
        "\nDirection summary:"
    )

    for direction, result in (
        direction_summary.items()
    ):

        print(
            "\n",
            direction,
            sep=""
        )

        print(
            "Mean MADLAD - OPUS ΔBLEU:",
            f"{result['mean_delta_bleu']:+.4f}"
        )

        print(
            "Mean MADLAD - OPUS ΔchrF++:",
            f"{result['mean_delta_chrfpp']:+.4f}"
        )

        print(
            "MADLAD wins both primary metrics "
            "on all benchmarks:",
            result[
                "madlad_wins_all_primary"
            ]
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
        "MADLAD_TEACHER_BAKEOFF_COMPLETE"
    )

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":

    main()