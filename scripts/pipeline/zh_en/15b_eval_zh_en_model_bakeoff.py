from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
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
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)


# ============================================================
# Version
# ============================================================

STEP_VERSION = "15B_V1"


# ============================================================
# Candidates
# ============================================================

CANDIDATES = {

    "small100": {
        "display_name": "SMaLL-100",
        "family": "small100",
        "path": "/root/autodl-tmp/models/small100",
        "supported_directions": [
            "zh_en",
            "en_zh",
        ],
        "license": "MIT",
    },

    "m2m100_418m": {
        "display_name": "M2M100-418M",
        "family": "m2m100",
        "path": "/root/autodl-tmp/models/m2m100_418M",
        "supported_directions": [
            "zh_en",
            "en_zh",
        ],
        "license": "MIT",
    },

    "opus_zh_en": {
        "display_name": "OPUS-MT zh-en",
        "family": "opus",
        "path": "/root/autodl-tmp/models/opus-mt-zh-en",
        "supported_directions": [
            "zh_en",
        ],
        "license": "CC-BY-4.0",
    },

    "opus_en_zh": {
        "display_name": "OPUS-MT en-zh",
        "family": "opus",
        "path": "/root/autodl-tmp/models/opus-mt-en-zh",
        "supported_directions": [
            "en_zh",
        ],
        "license": "Apache-2.0",
    },
}


# ============================================================
# Direction config
# ============================================================

DIRECTIONS = {

    "zh_en": {
        "source_lang": "zh",
        "target_lang": "en",
        "source_column": "zh",
        "target_column": "en",
        "bleu_tokenizer": "13a",
    },

    "en_zh": {
        "source_lang": "en",
        "target_lang": "zh",
        "source_column": "en",
        "target_column": "zh",
        "bleu_tokenizer": "zh",
    },
}


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 15B - Raw model bake-off "
            "for frozen ZH-EN benchmarks."
        )
    )

    parser.add_argument(
        "--candidate",
        required=True,
        choices=sorted(
            CANDIDATES.keys()
        ),
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
# Text normalization
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


# ============================================================
# SMALL100 tokenizer loader
# ============================================================

def load_small100_tokenizer(
    model_path: Path,
):

    tokenizer_file = (
        model_path
        /
        "tokenization_small100.py"
    )

    if not tokenizer_file.exists():

        raise FileNotFoundError(
            "\nSMaLL-100 custom tokenizer not found:\n"
            f"{tokenizer_file}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "tokenization_small100",
            tokenizer_file,
        )
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Cannot load SMALL100Tokenizer module."
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    tokenizer = (
        module
        .SMALL100Tokenizer
        .from_pretrained(
            str(model_path)
        )
    )

    return tokenizer


# ============================================================
# Model loading
# ============================================================

def load_candidate(
    candidate_name: str,
):

    config = (
        CANDIDATES[
            candidate_name
        ]
    )

    family = config["family"]

    model_path = Path(
        config["path"]
    )

    if not model_path.exists():

        raise FileNotFoundError(
            "\nCandidate model does not exist:\n"
            f"{model_path}"
        )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required for bake-off."
        )

    print(
        "\nCandidate:",
        config["display_name"]
    )

    print(
        "Family:",
        family
    )

    print(
        "Model path:",
        model_path
    )

    print(
        "License:",
        config["license"]
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    # --------------------------------------------------------
    # SMaLL-100
    # --------------------------------------------------------

    if family == "small100":

        print(
            "\nLoading SMALL100Tokenizer..."
        )

        tokenizer = (
            load_small100_tokenizer(
                model_path
            )
        )

        print(
            "Loading SMaLL-100 model..."
        )

        model = (
            M2M100ForConditionalGeneration
            .from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .to("cuda")
        )

    # --------------------------------------------------------
    # M2M100
    # --------------------------------------------------------

    elif family == "m2m100":

        print(
            "\nLoading M2M100Tokenizer..."
        )

        tokenizer = (
            M2M100Tokenizer
            .from_pretrained(
                str(model_path),
                local_files_only=True,
            )
        )

        print(
            "Loading M2M100-418M..."
        )

        model = (
            M2M100ForConditionalGeneration
            .from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .to("cuda")
        )

    # --------------------------------------------------------
    # OPUS / Marian
    # --------------------------------------------------------

    elif family == "opus":

        print(
            "\nLoading OPUS tokenizer..."
        )

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                str(model_path),
                local_files_only=True,
                use_fast=False,
            )
        )

        print(
            "Loading OPUS model..."
        )

        model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(
                str(model_path),
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            .to("cuda")
        )

    else:

        raise ValueError(
            f"Unknown family: {family}"
        )

    model.eval()

    model.config.use_cache = True

    # --------------------------------------------------------
    # Parameter statistics
    # --------------------------------------------------------

    total_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    parameter_bytes = sum(
        parameter.numel()
        *
        parameter.element_size()
        for parameter
        in model.parameters()
    )

    stats = {
        "total_parameters": int(
            total_params
        ),
        "trainable_parameters": int(
            trainable_params
        ),
        "loaded_parameter_memory_gib": float(
            parameter_bytes
            /
            1024 ** 3
        ),
    }

    print(
        "\nTotal parameters:",
        f"{total_params:,}"
    )

    print(
        "Loaded parameter memory:",
        f"{stats['loaded_parameter_memory_gib']:.3f} GiB"
    )

    return (
        tokenizer,
        model,
        stats,
    )


# ============================================================
# Benchmark loading
# ============================================================

def load_benchmarks(
    project_root: Path,
):

    root = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
    )

    benchmark_files = {

        "flores_devtest": (
            root
            /
            "flores_plus_zh_en_devtest_v1.parquet"
        ),

        "tatoeba": (
            root
            /
            "tatoeba_zh_en_test_v1.parquet"
        ),
    }

    benchmarks = {}

    for name, path in (
        benchmark_files.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                "\nFrozen benchmark missing:\n"
                f"{path}"
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

        print(
            f"{name}: {len(df)} pairs"
        )

    return benchmarks


# ============================================================
# Direction-specific preparation
# ============================================================

def prepare_batch(
    tokenizer,
    family: str,
    direction: str,
    texts: list[str],
    max_source_length: int,
):

    config = (
        DIRECTIONS[
            direction
        ]
    )

    source_lang = (
        config[
            "source_lang"
        ]
    )

    target_lang = (
        config[
            "target_lang"
        ]
    )

    # --------------------------------------------------------
    # SMALL100:
    # target language is placed in input by custom tokenizer.
    # --------------------------------------------------------

    if family == "small100":

        tokenizer.tgt_lang = (
            target_lang
        )

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        generation_extra = {}

    # --------------------------------------------------------
    # M2M100:
    # set source language and force target BOS.
    # --------------------------------------------------------

    elif family == "m2m100":

        tokenizer.src_lang = (
            source_lang
        )

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        generation_extra = {
            "forced_bos_token_id":
                tokenizer.get_lang_id(
                    target_lang
                )
        }

    # --------------------------------------------------------
    # OPUS direct model:
    # no language control token required.
    # --------------------------------------------------------

    elif family == "opus":

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        generation_extra = {}

    else:

        raise ValueError(
            family
        )

    return (
        encoded,
        generation_extra,
    )


# ============================================================
# Warmup
# ============================================================

def warmup(
    tokenizer,
    model,
    family: str,
    direction: str,
    text: str,
    max_source_length: int,
    max_new_tokens: int,
):

    encoded, generation_extra = (
        prepare_batch(
            tokenizer=tokenizer,
            family=family,
            direction=direction,
            texts=[text],
            max_source_length=max_source_length,
        )
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
            max_new_tokens=min(
                max_new_tokens,
                64,
            ),
            use_cache=True,
            **generation_extra,
        )

    torch.cuda.synchronize()


# ============================================================
# Translation
# ============================================================

def translate_direction(
    df: pd.DataFrame,
    tokenizer,
    model,
    candidate_name: str,
    family: str,
    benchmark_name: str,
    direction: str,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
):

    direction_config = (
        DIRECTIONS[
            direction
        ]
    )

    source_column = (
        direction_config[
            "source_column"
        ]
    )

    target_column = (
        direction_config[
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

    print("\n" + "=" * 100)

    print(
        candidate_name,
        "|",
        benchmark_name,
        "|",
        direction,
    )

    print(
        "=" * 100
    )

    print(
        "Samples:",
        len(sources)
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    warmup(
        tokenizer=tokenizer,
        model=model,
        family=family,
        direction=direction,
        text=sources[0],
        max_source_length=max_source_length,
        max_new_tokens=max_new_tokens,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    predictions = []

    batch_generation_times = []

    batch_end_to_end_times = []

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

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

        total_start = (
            time.perf_counter()
        )

        encoded, generation_extra = (
            prepare_batch(
                tokenizer=tokenizer,
                family=family,
                direction=direction,
                texts=batch_sources,
                max_source_length=max_source_length,
            )
        )

        encoded = {
            key: value.to("cuda")
            for key, value
            in encoded.items()
        }

        torch.cuda.synchronize()

        generation_start = (
            time.perf_counter()
        )

        with torch.inference_mode():

            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **generation_extra,
            )

        torch.cuda.synchronize()

        generation_elapsed = (
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

        total_elapsed = (
            time.perf_counter()
            -
            total_start
        )

        predictions.extend(
            [
                str(text).strip()
                for text
                in decoded
            ]
        )

        batch_generation_times.append(
            (
                generation_elapsed,
                len(batch_sources),
            )
        )

        batch_end_to_end_times.append(
            (
                total_elapsed,
                len(batch_sources),
            )
        )

        done = min(
            start + batch_size,
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

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    total_generation_time = sum(
        elapsed
        for elapsed, _
        in batch_generation_times
    )

    total_generation_samples = sum(
        count
        for _, count
        in batch_generation_times
    )

    avg_generation_latency = (
        total_generation_time
        /
        total_generation_samples
    )

    total_end_to_end_time = sum(
        elapsed
        for elapsed, _
        in batch_end_to_end_times
    )

    avg_end_to_end_latency = (
        total_end_to_end_time
        /
        len(sources)
    )

    peak_gpu_memory_gib = (
        torch.cuda.max_memory_allocated()
        /
        1024 ** 3
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    bleu = BLEU(
        tokenize=(
            direction_config[
                "bleu_tokenizer"
            ]
        )
    )

    chrf = CHRF(
        word_order=0
    )

    chrfpp = CHRF(
        word_order=2
    )

    bleu_score = (
        bleu
        .corpus_score(
            predictions,
            [references],
        )
        .score
    )

    chrf_score = (
        chrf
        .corpus_score(
            predictions,
            [references],
        )
        .score
    )

    chrfpp_score = (
        chrfpp
        .corpus_score(
            predictions,
            [references],
        )
        .score
    )

    exact = sum(
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
        exact
        /
        len(references)
        *
        100
    )

    # --------------------------------------------------------
    # Prediction table
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "candidate":
                candidate_name,

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

    metrics = {
        "candidate":
            candidate_name,

        "benchmark":
            benchmark_name,

        "direction":
            direction,

        "samples":
            int(
                len(sources)
            ),

        "bleu":
            float(
                bleu_score
            ),

        "bleu_tokenizer":
            direction_config[
                "bleu_tokenizer"
            ],

        "chrf":
            float(
                chrf_score
            ),

        "chrfpp":
            float(
                chrfpp_score
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
                peak_gpu_memory_gib
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

    print(
        "\nBLEU:",
        f"{bleu_score:.4f}"
    )

    print(
        "chrF:",
        f"{chrf_score:.4f}"
    )

    print(
        "chrF++:",
        f"{chrfpp_score:.4f}"
    )

    print(
        "Exact:",
        f"{exact_percent:.3f}%"
    )

    print(
        "Generation latency/sample:",
        f"{avg_generation_latency:.4f}s"
    )

    print(
        "End-to-end latency/sample:",
        f"{avg_end_to_end_latency:.4f}s"
    )

    print(
        "Peak GPU memory:",
        f"{peak_gpu_memory_gib:.3f} GiB"
    )

    return (
        prediction_df,
        metrics,
    )


# ============================================================
# Build global leaderboard
# ============================================================

def build_leaderboard(
    bakeoff_root: Path,
):

    metric_files = sorted(
        bakeoff_root.glob(
            "*/metrics.json"
        )
    )

    rows = []

    for path in metric_files:

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as f:

                report = json.load(
                    f
                )

        except Exception:

            continue

        for metric in report.get(
            "results",
            [],
        ):

            row = {
                "candidate":
                    report[
                        "candidate"
                    ],

                "display_name":
                    report[
                        "display_name"
                    ],

                "family":
                    report[
                        "family"
                    ],

                "license":
                    report[
                        "license"
                    ],

                "total_parameters":
                    report[
                        "model_stats"
                    ][
                        "total_parameters"
                    ],

                "loaded_parameter_memory_gib":
                    report[
                        "model_stats"
                    ][
                        "loaded_parameter_memory_gib"
                    ],
            }

            row.update(
                metric
            )

            rows.append(
                row
            )

    if not rows:

        return None

    leaderboard = pd.DataFrame(
        rows
    )

    leaderboard = (
        leaderboard
        .sort_values(
            [
                "benchmark",
                "direction",
                "bleu",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    leaderboard.to_csv(
        bakeoff_root
        /
        "leaderboard_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    leaderboard.to_parquet(
        bakeoff_root
        /
        "leaderboard_v1.parquet",
        index=False,
    )

    return leaderboard


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

    candidate_config = (
        CANDIDATES[
            args.candidate
        ]
    )

    family = (
        candidate_config[
            "family"
        ]
    )

    output_root = (
        project_root
        /
        "results"
        /
        "model_bakeoff"
        /
        "zh_en"
        /
        "v1"
    )

    candidate_output = (
        output_root
        /
        args.candidate
    )

    metrics_file = (
        candidate_output
        /
        "metrics.json"
    )

    predictions_file = (
        candidate_output
        /
        "predictions.parquet"
    )

    candidate_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN MODEL BAKE-OFF"
    )

    print(
        "STEP 15B V1 - RAW MODEL EVALUATION"
    )

    print(
        "=" * 110
    )

    print(
        "\nCandidate:",
        args.candidate
    )

    if (
        metrics_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nCandidate result already exists:\n"
            f"{metrics_file}\n\n"
            "Use --overwrite if rerunning intentionally."
        )

    # ========================================================
    # Load frozen benchmarks
    # ========================================================

    print(
        "\nFrozen benchmarks:"
    )

    benchmarks = (
        load_benchmarks(
            project_root
        )
    )

    # ========================================================
    # Load candidate
    # ========================================================

    tokenizer, model, model_stats = (
        load_candidate(
            args.candidate
        )
    )

    # ========================================================
    # Evaluate
    # ========================================================

    all_predictions = []

    all_metrics = []

    for benchmark_name, benchmark_df in (
        benchmarks.items()
    ):

        for direction in (
            candidate_config[
                "supported_directions"
            ]
        ):

            prediction_df, metrics = (
                translate_direction(
                    df=benchmark_df,
                    tokenizer=tokenizer,
                    model=model,
                    candidate_name=args.candidate,
                    family=family,
                    benchmark_name=benchmark_name,
                    direction=direction,
                    batch_size=args.batch_size,
                    num_beams=args.num_beams,
                    max_source_length=args.max_source_length,
                    max_new_tokens=args.max_new_tokens,
                )
            )

            all_predictions.append(
                prediction_df
            )

            all_metrics.append(
                metrics
            )

    # ========================================================
    # Save
    # ========================================================

    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions.to_parquet(
        predictions_file,
        index=False,
    )

    predictions.to_csv(
        candidate_output
        /
        "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "step":
            "15B",

        "step_version":
            STEP_VERSION,

        "candidate":
            args.candidate,

        "display_name":
            candidate_config[
                "display_name"
            ],

        "family":
            family,

        "model_path":
            candidate_config[
                "path"
            ],

        "license":
            candidate_config[
                "license"
            ],

        "supported_directions":
            candidate_config[
                "supported_directions"
            ],

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

            "fp16":
                True,
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

        "results":
            all_metrics,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "RAW_BAKEOFF_COMPLETE",
    }

    with open(
        metrics_file,
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
    # Global leaderboard
    # ========================================================

    leaderboard = (
        build_leaderboard(
            output_root
        )
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 15B RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nCandidate:",
        candidate_config[
            "display_name"
        ]
    )

    print(
        "\nModel statistics:"
    )

    print(
        "Parameters:",
        f"{model_stats['total_parameters']:,}"
    )

    print(
        "Loaded parameter memory:",
        f"{model_stats['loaded_parameter_memory_gib']:.3f} GiB"
    )

    print(
        "\nMetrics:"
    )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    display_columns = [
        "benchmark",
        "direction",
        "samples",
        "bleu",
        "chrf",
        "chrfpp",
        "exact_match_percent",
        "avg_generation_latency_seconds",
        "peak_gpu_memory_gib",
    ]

    print(
        metrics_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nPredictions:"
    )

    print(
        predictions_file
    )

    print(
        "\nMetrics:"
    )

    print(
        metrics_file
    )

    if leaderboard is not None:

        print(
            "\nCurrent leaderboard:"
        )

        leaderboard_display = [
            "display_name",
            "benchmark",
            "direction",
            "bleu",
            "chrfpp",
            "avg_generation_latency_seconds",
            "peak_gpu_memory_gib",
        ]

        print(
            leaderboard[
                leaderboard_display
            ]
            .round(4)
            .to_string(
                index=False
            )
        )

    print(
        "\nSTATUS:"
    )

    print(
        "RAW_BAKEOFF_COMPLETE"
    )

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":

    main()