from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from transformers import M2M100ForConditionalGeneration


# ============================================================
# 1. Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fine-tuned SMaLL-100 "
            "EN-UZ Student Exp1."
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
# 2. JSONL
# ============================================================

def append_jsonl(
    path: Path,
    rows: list[dict],
):

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
            )

            f.write("\n")


def load_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():

        return []

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception:

                print(
                    "[WARNING] Broken checkpoint "
                    "line skipped."
                )

    return rows


# ============================================================
# 3. Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Paths
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    model_path = (
        project_root
        / "results"
        / "student"
        / "small100"
        / "exp1_finetune"
        / "best_model"
    )

    validation_file = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "validation_exp1_bidirectional_v1.parquet"
    )

    baseline_metrics_file = (
        project_root
        / "results"
        / "student"
        / "small100"
        / "exp1_baseline"
        / "validation_metrics.json"
    )

    output_dir = (
        project_root
        / "results"
        / "student"
        / "small100"
        / "exp1_finetune"
        / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        output_dir
        / "eval_checkpoint.jsonl"
    )

    prediction_file = (
        output_dir
        / "validation_predictions.parquet"
    )

    prediction_csv = (
        output_dir
        / "validation_predictions.csv"
    )

    metrics_file = (
        output_dir
        / "validation_metrics.json"
    )

    comparison_file = (
        output_dir
        / "baseline_vs_finetuned.json"
    )

    comparison_csv = (
        output_dir
        / "baseline_vs_finetuned.csv"
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 08C - FINE-TUNED STUDENT EVALUATION")
    print("=" * 100)

    print(
        "Model:",
        model_path
    )

    print(
        "Validation:",
        validation_file
    )

    print(
        "Baseline metrics:",
        baseline_metrics_file
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Beam size:",
        args.num_beams
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
        "\nCUDA:",
        True
    )

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda
    )

    # ========================================================
    # File checks
    # ========================================================

    if not model_path.exists():

        raise FileNotFoundError(
            f"Fine-tuned model not found:\n"
            f"{model_path}"
        )

    if not validation_file.exists():

        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{validation_file}"
        )

    if not baseline_metrics_file.exists():

        raise FileNotFoundError(
            f"Baseline metrics not found:\n"
            f"{baseline_metrics_file}"
        )

    tokenizer_py = (
        model_path
        / "tokenization_small100.py"
    )

    if not tokenizer_py.exists():

        raise FileNotFoundError(
            "SMaLL-100 custom tokenizer "
            "not found:\n"
            f"{tokenizer_py}"
        )

    # ========================================================
    # Import SMALL100Tokenizer
    # ========================================================

    sys.path.insert(
        0,
        str(model_path),
    )

    from tokenization_small100 import (
        SMALL100Tokenizer
    )

    # ========================================================
    # Load validation
    # ========================================================

    print(
        "\nLoading validation dataset..."
    )

    df = pd.read_parquet(
        validation_file
    )

    required_columns = [
        "sample_id",
        "normalized_pair_id",
        "direction",
        "src_lang",
        "tgt_lang",
        "source_text",
        "target_text",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if (
        df[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate sample_id detected."
        )

    print(
        "Validation samples:",
        len(df)
    )

    print(
        "\nDirection distribution:"
    )

    print(
        df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Tokenizer
    # ========================================================

    print(
        "\nLoading SMALL100Tokenizer..."
    )

    tokenizer = (
        SMALL100Tokenizer
        .from_pretrained(
            str(
                model_path
            ),
            tgt_lang="uz",
            local_files_only=True,
        )
    )

    print(
        "Tokenizer:",
        type(
            tokenizer
        ).__name__
    )

    if (
        type(
            tokenizer
        ).__name__
        !=
        "SMALL100Tokenizer"
    ):

        raise RuntimeError(
            "Incorrect tokenizer loaded."
        )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nLoading fine-tuned model..."
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
        "Model:",
        type(
            model
        ).__name__
    )

    print(
        "GPU allocated:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    # ========================================================
    # Resume checkpoint
    # ========================================================

    if args.overwrite:

        if checkpoint_file.exists():

            checkpoint_file.unlink()

        existing_rows = []

    else:

        existing_rows = (
            load_jsonl(
                checkpoint_file
            )
        )

    result_map = {}

    for row in existing_rows:

        result_map[
            str(
                row[
                    "sample_id"
                ]
            )
        ] = row

    completed_ids = set(
        result_map.keys()
    )

    print(
        "\nAlready completed:",
        len(
            completed_ids
        )
    )

    print(
        "Pending:",
        len(df)
        -
        len(
            completed_ids
        )
    )

    # ========================================================
    # Evaluation
    # ========================================================

    total_samples = len(
        df
    )

    processed_this_run = 0

    run_start = (
        time.perf_counter()
    )

    target_configs = [

        (
            "uz",
            "en_uz",
        ),

        (
            "en",
            "uz_en",
        ),
    ]

    for (
        target_lang,
        direction,
    ) in target_configs:

        part = df[
            df[
                "direction"
            ]
            ==
            direction
        ].copy()

        part = part[
            ~part[
                "sample_id"
            ]
            .astype(str)
            .isin(
                completed_ids
            )
        ]

        part = part.reset_index(
            drop=True
        )

        print("\n")
        print("=" * 80)

        print(
            "DIRECTION:",
            direction.upper()
        )

        print(
            "TARGET:",
            target_lang
        )

        print(
            "PENDING:",
            len(part)
        )

        print("=" * 80)

        if len(part) == 0:

            continue

        tokenizer.tgt_lang = (
            target_lang
        )

        for start in range(
            0,
            len(part),
            args.batch_size,
        ):

            end = min(
                start
                +
                args.batch_size,
                len(part),
            )

            batch = (
                part
                .iloc[
                    start:end
                ]
                .copy()
            )

            sources = (
                batch[
                    "source_text"
                ]
                .astype(str)
                .tolist()
            )

            # =================================================
            # Tokenize
            # =================================================

            encoded = tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=
                    args.max_source_length,
            )

            encoded = {
                key:
                    value.to(
                        device
                    )

                for key, value
                in encoded.items()
            }

            # =================================================
            # Generate
            # =================================================

            torch.cuda.synchronize()

            batch_start = (
                time.perf_counter()
            )

            with torch.inference_mode():

                generated = (
                    model.generate(
                        **encoded,

                        max_new_tokens=
                            args.max_new_tokens,

                        num_beams=
                            args.num_beams,

                        do_sample=False,

                        use_cache=True,

                        pad_token_id=
                            tokenizer.pad_token_id,
                    )
                )

            torch.cuda.synchronize()

            batch_seconds = (
                time.perf_counter()
                -
                batch_start
            )

            # =================================================
            # Decode
            # =================================================

            predictions = (
                tokenizer
                .batch_decode(
                    generated,
                    skip_special_tokens=True,
                )
            )

            latency_per_sample = (
                batch_seconds
                /
                len(
                    batch
                )
            )

            batch_rows = []

            for (
                (_, row),
                prediction,
            ) in zip(
                batch.iterrows(),
                predictions,
            ):

                result = {

                    "sample_id":
                        str(
                            row[
                                "sample_id"
                            ]
                        ),

                    "normalized_pair_id":
                        str(
                            row[
                                "normalized_pair_id"
                            ]
                        ),

                    "direction":
                        str(
                            row[
                                "direction"
                            ]
                        ),

                    "src_lang":
                        str(
                            row[
                                "src_lang"
                            ]
                        ),

                    "tgt_lang":
                        str(
                            row[
                                "tgt_lang"
                            ]
                        ),

                    "source":
                        str(
                            row[
                                "source_text"
                            ]
                        ),

                    "reference":
                        str(
                            row[
                                "target_text"
                            ]
                        ),

                    "prediction":
                        str(
                            prediction
                        ),

                    "quality_tier":
                        (
                            str(
                                row[
                                    "quality_tier"
                                ]
                            )
                            if
                            "quality_tier"
                            in row.index
                            else
                            ""
                        ),

                    "data_source":
                        (
                            str(
                                row[
                                    "data_source"
                                ]
                            )
                            if
                            "data_source"
                            in row.index
                            else
                            ""
                        ),

                    "batch_seconds":
                        float(
                            batch_seconds
                        ),

                    "latency_seconds":
                        float(
                            latency_per_sample
                        ),

                    "num_beams":
                        args.num_beams,
                }

                batch_rows.append(
                    result
                )

                result_map[
                    result[
                        "sample_id"
                    ]
                ] = result

            append_jsonl(
                checkpoint_file,
                batch_rows,
            )

            processed_this_run += (
                len(
                    batch_rows
                )
            )

            total_completed = len(
                result_map
            )

            elapsed = (
                time.perf_counter()
                -
                run_start
            )

            speed = (
                processed_this_run
                /
                elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                total_samples
                -
                total_completed
            )

            eta = (
                remaining
                /
                speed
                if speed > 0
                else 0
            )

            if (
                end % 256
                <
                args.batch_size
                or
                end == len(part)
            ):

                print(
                    f"{direction}: "
                    f"{end}/{len(part)}"
                )

                print(
                    f"Overall: "
                    f"{total_completed}/"
                    f"{total_samples}"
                )

                print(
                    "Batch time:",
                    f"{batch_seconds:.3f}s"
                )

                print(
                    "Throughput:",
                    f"{speed:.2f} samples/s"
                )

                print(
                    "ETA:",
                    f"{eta / 60:.1f} min"
                )

    # ========================================================
    # Reload checkpoint
    # ========================================================

    rows = load_jsonl(
        checkpoint_file
    )

    final_map = {}

    for row in rows:

        final_map[
            str(
                row[
                    "sample_id"
                ]
            )
        ] = row

    result_df = pd.DataFrame(
        list(
            final_map.values()
        )
    )

    # ========================================================
    # Integrity
    # ========================================================

    if len(
        result_df
    ) != len(df):

        raise RuntimeError(
            "Evaluation incomplete.\n"
            f"Expected: {len(df)}\n"
            f"Found: {len(result_df)}"
        )

    order_map = {

        str(
            sample_id
        ):
            index

        for index, sample_id
        in enumerate(
            df[
                "sample_id"
            ]
            .astype(str)
        )
    }

    result_df[
        "_order"
    ] = (
        result_df[
            "sample_id"
        ]
        .astype(str)
        .map(
            order_map
        )
    )

    result_df = (
        result_df
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Metrics
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINE-TUNED STUDENT RESULT")
    print("=" * 100)

    chrf_metric = (
        sacrebleu.CHRF(
            word_order=2
        )
    )

    finetuned_metrics = {}

    for direction in [
        "en_uz",
        "uz_en",
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

        predictions = (
            part[
                "prediction"
            ]
            .astype(str)
            .tolist()
        )

        references = (
            part[
                "reference"
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

        chrfpp = (
            chrf_metric
            .corpus_score(
                predictions,
                [
                    references
                ],
            )
            .score
        )

        exact_match = float(
            (
                part[
                    "prediction"
                ]
                .astype(str)
                .str.strip()
                ==
                part[
                    "reference"
                ]
                .astype(str)
                .str.strip()
            )
            .mean()
            *
            100
        )

        avg_latency = float(
            part[
                "latency_seconds"
            ]
            .mean()
        )

        finetuned_metrics[
            direction
        ] = {

            "samples":
                len(part),

            "bleu":
                float(
                    bleu
                ),

            "chrf++":
                float(
                    chrfpp
                ),

            "exact_match_percent":
                exact_match,

            "avg_latency_seconds":
                avg_latency,
        }

        print()
        print(
            direction.upper()
        )

        print(
            "Samples :",
            len(part)
        )

        print(
            "BLEU    :",
            f"{bleu:.4f}"
        )

        print(
            "chrF++  :",
            f"{chrfpp:.4f}"
        )

        print(
            "Exact   :",
            f"{exact_match:.2f}%"
        )

        print(
            "Latency :",
            f"{avg_latency:.4f}s/sample"
        )

    # ========================================================
    # Load baseline
    # ========================================================

    with open(
        baseline_metrics_file,
        "r",
        encoding="utf-8",
    ) as f:

        baseline_data = json.load(
            f
        )

    baseline_metrics = (
        baseline_data[
            "directions"
        ]
    )

    # ========================================================
    # Comparison
    # ========================================================

    print("\n")
    print("=" * 100)
    print("BASELINE VS FINE-TUNED")
    print("=" * 100)

    comparison = {}

    comparison_rows = []

    for direction in [
        "en_uz",
        "uz_en",
    ]:

        base = baseline_metrics[
            direction
        ]

        fine = finetuned_metrics[
            direction
        ]

        delta_bleu = (
            fine[
                "bleu"
            ]
            -
            base[
                "bleu"
            ]
        )

        delta_chrf = (
            fine[
                "chrf++"
            ]
            -
            base[
                "chrf++"
            ]
        )

        delta_exact = (
            fine[
                "exact_match_percent"
            ]
            -
            base[
                "exact_match_percent"
            ]
        )

        delta_latency = (
            fine[
                "avg_latency_seconds"
            ]
            -
            base[
                "avg_latency_seconds"
            ]
        )

        comparison[
            direction
        ] = {

            "baseline":
                base,

            "finetuned":
                fine,

            "delta": {

                "bleu":
                    delta_bleu,

                "chrf++":
                    delta_chrf,

                "exact_match_percent":
                    delta_exact,

                "avg_latency_seconds":
                    delta_latency,
            },
        }

        comparison_rows.append({

            "direction":
                direction,

            "baseline_bleu":
                base[
                    "bleu"
                ],

            "finetuned_bleu":
                fine[
                    "bleu"
                ],

            "delta_bleu":
                delta_bleu,

            "baseline_chrf++":
                base[
                    "chrf++"
                ],

            "finetuned_chrf++":
                fine[
                    "chrf++"
                ],

            "delta_chrf++":
                delta_chrf,

            "baseline_exact":
                base[
                    "exact_match_percent"
                ],

            "finetuned_exact":
                fine[
                    "exact_match_percent"
                ],

            "delta_exact":
                delta_exact,

            "baseline_latency":
                base[
                    "avg_latency_seconds"
                ],

            "finetuned_latency":
                fine[
                    "avg_latency_seconds"
                ],

            "delta_latency":
                delta_latency,
        })

        print()
        print(
            direction.upper()
        )

        print(
            "-" * 70
        )

        print(
            "BLEU:"
        )

        print(
            "  Original :",
            f"{base['bleu']:.4f}"
        )

        print(
            "  Fine-tune:",
            f"{fine['bleu']:.4f}"
        )

        print(
            "  Delta    :",
            f"{delta_bleu:+.4f}"
        )

        print(
            "\nchrF++:"
        )

        print(
            "  Original :",
            f"{base['chrf++']:.4f}"
        )

        print(
            "  Fine-tune:",
            f"{fine['chrf++']:.4f}"
        )

        print(
            "  Delta    :",
            f"{delta_chrf:+.4f}"
        )

        print(
            "\nExact:"
        )

        print(
            "  Original :",
            f"{base['exact_match_percent']:.2f}%"
        )

        print(
            "  Fine-tune:",
            f"{fine['exact_match_percent']:.2f}%"
        )

        print(
            "  Delta    :",
            f"{delta_exact:+.2f}%"
        )

        print(
            "\nLatency:"
        )

        print(
            "  Original :",
            f"{base['avg_latency_seconds']:.4f}s"
        )

        print(
            "  Fine-tune:",
            f"{fine['avg_latency_seconds']:.4f}s"
        )

        print(
            "  Delta    :",
            f"{delta_latency:+.4f}s"
        )

    # ========================================================
    # Overall judgment
    # ========================================================

    mean_bleu_delta = float(
        (
            comparison[
                "en_uz"
            ][
                "delta"
            ][
                "bleu"
            ]
            +
            comparison[
                "uz_en"
            ][
                "delta"
            ][
                "bleu"
            ]
        )
        /
        2
    )

    mean_chrf_delta = float(
        (
            comparison[
                "en_uz"
            ][
                "delta"
            ][
                "chrf++"
            ]
            +
            comparison[
                "uz_en"
            ][
                "delta"
            ][
                "chrf++"
            ]
        )
        /
        2
    )

    if (
        mean_bleu_delta > 0
        and
        mean_chrf_delta > 0
    ):

        decision = (
            "PASS"
        )

    else:

        decision = (
            "REVIEW"
        )

    print("\n")
    print("=" * 100)
    print("EXP1 DECISION")
    print("=" * 100)

    print(
        "Mean ΔBLEU :",
        f"{mean_bleu_delta:+.4f}"
    )

    print(
        "Mean ΔchrF++:",
        f"{mean_chrf_delta:+.4f}"
    )

    print(
        "Decision   :",
        decision
    )

    # ========================================================
    # Save files
    # ========================================================

    result_df.to_parquet(
        prediction_file,
        index=False,
    )

    result_df.to_csv(
        prediction_csv,
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {

        "experiment":
            "small100_en_uz_exp1_finetuned",

        "model_path":
            str(
                model_path
            ),

        "dataset":
            "en_uz_v1_validation_exp1",

        "tokenizer":
            "SMALL100Tokenizer",

        "precision":
            "fp16",

        "generation": {

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
        },

        "directions":
            finetuned_metrics,

        "hardware": {

            "gpu":
                torch.cuda.get_device_name(
                    0
                ),

            "torch":
                torch.__version__,

            "torch_cuda":
                torch.version.cuda,
        },
    }

    with open(
        metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    final_comparison = {

        "experiment":
            "small100_exp1",

        "baseline_model":
            "original small100",

        "finetuned_model":
            str(
                model_path
            ),

        "comparison":
            comparison,

        "summary": {

            "mean_delta_bleu":
                mean_bleu_delta,

            "mean_delta_chrf++":
                mean_chrf_delta,

            "decision":
                decision,
        },
    }

    with open(
        comparison_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            final_comparison,
            f,
            ensure_ascii=False,
            indent=2,
        )

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        comparison_csv,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Complete
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 08C COMPLETE")
    print("=" * 100)

    print(
        "Predictions:"
    )

    print(
        prediction_file
    )

    print(
        "\nMetrics:"
    )

    print(
        metrics_file
    )

    print(
        "\nComparison:"
    )

    print(
        comparison_file
    )

    print(
        "\nComparison CSV:"
    )

    print(
        comparison_csv
    )


if __name__ == "__main__":

    main()