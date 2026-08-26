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
        description="Evaluate original SMaLL-100 on EN-UZ validation set."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/autodl-tmp/models/small100",
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
        help="Ignore existing checkpoint and rerun all samples.",
    )

    return parser.parse_args()


# ============================================================
# 2. JSONL helpers
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
                    json.loads(line)
                )

            except Exception:
                print(
                    "[WARNING] Broken checkpoint line skipped."
                )

    return rows


# ============================================================
# 3. Main
# ============================================================

def main():

    args = parse_args()

    # ========================================================
    # Project paths
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    model_path = Path(
        args.model_path
    )

    input_file = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "validation_exp1_bidirectional_v1.parquet"
    )

    output_dir = (
        project_root
        / "results"
        / "student"
        / "small100"
        / "exp1_baseline"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        output_dir
        / "baseline_checkpoint.jsonl"
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

    # ========================================================
    # Header
    # ========================================================

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 08A - ORIGINAL SMALL-100 BASELINE")
    print("=" * 100)

    print(
        "Model:",
        model_path
    )

    print(
        "Validation:",
        input_file
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
    # Check model
    # ========================================================

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model directory not found:\n"
            f"{model_path}"
        )

    tokenizer_py = (
        model_path
        / "tokenization_small100.py"
    )

    if not tokenizer_py.exists():

        raise FileNotFoundError(
            "SMaLL-100 tokenizer file not found:\n"
            f"{tokenizer_py}"
        )

    # ========================================================
    # Check validation
    # ========================================================

    if not input_file.exists():

        raise FileNotFoundError(
            "Validation dataset not found:\n"
            f"{input_file}"
        )

    # ========================================================
    # Import the REAL SMaLL-100 tokenizer
    # ========================================================

    sys.path.insert(
        0,
        str(model_path)
    )

    from tokenization_small100 import SMALL100Tokenizer

    # ========================================================
    # Data
    # ========================================================

    print(
        "\nLoading validation data..."
    )

    df = pd.read_parquet(
        input_file
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
            f"Missing validation columns: "
            f"{missing}"
        )

    if df[
        "sample_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate sample_id found."
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
    # Expected directions
    # ========================================================

    valid_directions = {
        "en_uz",
        "uz_en",
    }

    found_directions = set(
        df[
            "direction"
        ].astype(str)
    )

    unexpected = (
        found_directions
        -
        valid_directions
    )

    if unexpected:

        raise RuntimeError(
            f"Unexpected directions: "
            f"{unexpected}"
        )

    # ========================================================
    # Load correct tokenizer
    # ========================================================

    print(
        "\nLoading SMALL100Tokenizer..."
    )

    tokenizer = (
        SMALL100Tokenizer
        .from_pretrained(
            str(model_path),
            tgt_lang="uz",
            local_files_only=True,
        )
    )

    print(
        "Tokenizer class:",
        type(tokenizer).__name__
    )

    if (
        type(tokenizer).__name__
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
        "\nLoading SMaLL-100 model..."
    )

    model = (
        M2M100ForConditionalGeneration
        .from_pretrained(
            str(model_path),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(device)
    )

    model.eval()

    model.config.use_cache = True

    print(
        "Model class:",
        type(model).__name__
    )

    print(
        "GPU allocated:",
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    # ========================================================
    # Existing checkpoint
    # ========================================================

    if args.overwrite:

        if checkpoint_file.exists():

            checkpoint_file.unlink()

        existing_rows = []

    else:

        existing_rows = load_jsonl(
            checkpoint_file
        )

    # Deduplicate possible checkpoint rows
    result_map = {}

    for row in existing_rows:

        sample_id = str(
            row[
                "sample_id"
            ]
        )

        result_map[
            sample_id
        ] = row

    completed_ids = set(
        result_map.keys()
    )

    print(
        "\nAlready completed:",
        len(completed_ids)
    )

    print(
        "Pending:",
        len(df)
        -
        len(completed_ids)
    )

    # ========================================================
    # Evaluation
    #
    # Important:
    #
    # Each target language must be processed separately because
    # SMALL100Tokenizer.tgt_lang controls the target prefix.
    # ========================================================

    run_start = time.perf_counter()

    processed_this_run = 0

    total_samples = len(df)

    # Explicit order
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
            f"DIRECTION: "
            f"{direction.upper()}"
        )
        print(
            f"TARGET: {target_lang}"
        )
        print(
            f"PENDING: {len(part)}"
        )
        print("=" * 80)

        if len(part) == 0:
            continue

        # Correct SMaLL-100 language control
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

            source_texts = (
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
                source_texts,
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

            per_sample_latency = (
                batch_seconds
                /
                len(batch)
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
                            per_sample_latency
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

            # =================================================
            # Incremental checkpoint
            # =================================================

            append_jsonl(
                checkpoint_file,
                batch_rows,
            )

            processed_this_run += len(
                batch_rows
            )

            # =================================================
            # Progress
            # =================================================

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
    # Reload checkpoint as source of truth
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

    if len(result_df) != len(df):

        raise RuntimeError(
            "Evaluation incomplete.\n"
            f"Expected: {len(df)}\n"
            f"Found: {len(result_df)}"
        )

    # Original validation ordering
    order_map = {
        str(sample_id):
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
    print("ORIGINAL SMALL-100 BASELINE RESULT")
    print("=" * 100)

    metric_result = {}

    chrf_metric = sacrebleu.CHRF(
        word_order=2
    )

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

        # ----------------------------------------------------
        # BLEU
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # chrF++
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

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
            ].mean()
        )

        metric_result[
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
    # Overall runtime
    # ========================================================

    total_latency_sum = float(
        result_df[
            "latency_seconds"
        ].sum()
    )

    overall_avg_latency = float(
        result_df[
            "latency_seconds"
        ].mean()
    )

    # ========================================================
    # Save metrics
    # ========================================================

    metrics = {

        "experiment":
            "small100_original_baseline",

        "dataset":
            "en_uz_v1_validation_exp1",

        "model":
            "alirezamsh/small100",

        "model_path":
            str(
                model_path
            ),

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

        "total_samples":
            len(
                result_df
            ),

        "directions":
            metric_result,

        "overall": {

            "avg_latency_seconds":
                overall_avg_latency,

            "latency_sum_seconds":
                total_latency_sum,
        },

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

    # ========================================================
    # Save prediction files
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

    # ========================================================
    # Final output
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 08A COMPLETE")
    print("=" * 100)

    print(
        "Total:",
        len(result_df)
    )

    print(
        "\nPredictions:"
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
        "\nOriginal SMaLL-100 baseline completed."
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":
    main()