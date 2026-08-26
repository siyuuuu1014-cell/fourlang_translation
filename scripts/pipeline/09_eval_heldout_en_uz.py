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
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    M2M100ForConditionalGeneration,
)


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmark",
        required=True,
        type=str,
        help="429-pair normal benchmark CSV/Parquet",
    )

    parser.add_argument(
        "--challenge",
        required=True,
        type=str,
        help="300-pair challenge benchmark CSV/Parquet",
    )

    parser.add_argument(
        "--small100_path",
        type=str,
        default="/root/autodl-tmp/models/small100",
    )

    parser.add_argument(
        "--student_path",
        type=str,
        default=(
            "/root/autodl-tmp/fourlang_translation/"
            "results/student/small100/exp1_finetune/best_model"
        ),
    )

    parser.add_argument(
        "--madlad_path",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--small100_batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--madlad_batch_size",
        type=int,
        default=4,
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

    return parser.parse_args()


# ============================================================
# Read input
# ============================================================

def read_table(path: Path) -> pd.DataFrame:

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported file format: {path}"
    )


# ============================================================
# Normalize benchmark schema
# ============================================================

def normalize_pair_dataset(
    df: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:

    df = df.copy()

    # --------------------------------------------------------
    # Case 1
    # en / uz
    # --------------------------------------------------------

    if {
        "en",
        "uz",
    }.issubset(df.columns):

        result = pd.DataFrame({
            "pair_id": (
                df["pair_id"].astype(str)
                if "pair_id" in df.columns
                else [
                    f"{dataset_name}_{i:05d}"
                    for i in range(len(df))
                ]
            ),

            "en": df["en"].astype(str),
            "uz": df["uz"].astype(str),
        })

    # --------------------------------------------------------
    # Case 2
    # normalized pair dataset
    # --------------------------------------------------------

    elif {
        "source_text_normalized",
        "target_text_normalized",
    }.issubset(df.columns):

        result = pd.DataFrame({
            "pair_id": (
                df["normalized_pair_id"].astype(str)
                if "normalized_pair_id" in df.columns
                else [
                    f"{dataset_name}_{i:05d}"
                    for i in range(len(df))
                ]
            ),

            "en":
                df["source_text_normalized"]
                .astype(str),

            "uz":
                df["target_text_normalized"]
                .astype(str),
        })

    # --------------------------------------------------------
    # Case 3
    # source_text / target_text
    # --------------------------------------------------------

    elif {
        "source_text",
        "target_text",
    }.issubset(df.columns):

        result = pd.DataFrame({
            "pair_id": (
                df["normalized_pair_id"].astype(str)
                if "normalized_pair_id" in df.columns
                else [
                    f"{dataset_name}_{i:05d}"
                    for i in range(len(df))
                ]
            ),

            "en":
                df["source_text"]
                .astype(str),

            "uz":
                df["target_text"]
                .astype(str),
        })

    else:

        raise RuntimeError(
            "\nCannot recognize benchmark schema.\n"
            f"Columns:\n{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Preserve challenge category
    # --------------------------------------------------------

    category_candidates = [
        "category",
        "challenge_type",
        "type",
    ]

    category_col = None

    for column in category_candidates:
        if column in df.columns:
            category_col = column
            break

    if category_col:

        result["category"] = (
            df[category_col]
            .fillna("unknown")
            .astype(str)
            .values
        )

    else:

        result["category"] = "ALL"

    result["dataset"] = dataset_name

    # --------------------------------------------------------
    # Remove invalid
    # --------------------------------------------------------

    result = result[
        (result["en"].str.strip() != "")
        &
        (result["uz"].str.strip() != "")
    ].reset_index(drop=True)

    return result


# ============================================================
# Expand pair -> 2 directions
# ============================================================

def expand_bidirectional(
    df: pd.DataFrame,
) -> pd.DataFrame:

    en_uz = pd.DataFrame({
        "sample_id":
            df["pair_id"].astype(str)
            + "_en_uz",

        "pair_id":
            df["pair_id"],

        "dataset":
            df["dataset"],

        "category":
            df["category"],

        "direction":
            "en_uz",

        "src_lang":
            "en",

        "tgt_lang":
            "uz",

        "source":
            df["en"],

        "reference":
            df["uz"],
    })

    uz_en = pd.DataFrame({
        "sample_id":
            df["pair_id"].astype(str)
            + "_uz_en",

        "pair_id":
            df["pair_id"],

        "dataset":
            df["dataset"],

        "category":
            df["category"],

        "direction":
            "uz_en",

        "src_lang":
            "uz",

        "tgt_lang":
            "en",

        "source":
            df["uz"],

        "reference":
            df["en"],
    })

    return pd.concat(
        [
            en_uz,
            uz_en,
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

    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score

    chrf = sacrebleu.CHRF(
        word_order=2
    ).corpus_score(
        predictions,
        [references],
    ).score

    exact = float(
        (
            df["prediction"]
            .astype(str)
            .str.strip()
            ==
            df["reference"]
            .astype(str)
            .str.strip()
        ).mean()
        * 100
    )

    latency = float(
        df["latency_seconds"].mean()
    )

    return {
        "samples": len(df),
        "bleu": float(bleu),
        "chrf++": float(chrf),
        "exact_percent": exact,
        "avg_latency_seconds": latency,
    }


# ============================================================
# Evaluate SMaLL-100
# ============================================================

def evaluate_small100(
    model_name: str,
    model_path: Path,
    tokenizer_class,
    test_df: pd.DataFrame,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
    device,
):

    print("\n")
    print("=" * 100)
    print(f"EVALUATING: {model_name}")
    print("=" * 100)

    tokenizer = tokenizer_class.from_pretrained(
        str(model_path),
        tgt_lang="uz",
        local_files_only=True,
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

    result_rows = []

    direction_config = [
        ("en_uz", "uz"),
        ("uz_en", "en"),
    ]

    for direction, target_lang in direction_config:

        part = (
            test_df[
                test_df["direction"]
                ==
                direction
            ]
            .reset_index(drop=True)
        )

        tokenizer.tgt_lang = target_lang

        print(
            f"\n{direction.upper()} "
            f"samples: {len(part)}"
        )

        for start in range(
            0,
            len(part),
            batch_size,
        ):

            batch = part.iloc[
                start:
                start + batch_size
            ]

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
                max_length=max_source_length,
            )

            inputs = {
                k: v.to(device)
                for k, v
                in inputs.items()
            }

            torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=
                        tokenizer.pad_token_id,
                )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            predictions = (
                tokenizer.batch_decode(
                    output,
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
                        model_name,

                    "prediction":
                        prediction,

                    "latency_seconds":
                        latency,
                })

            done = min(
                start + batch_size,
                len(part),
            )

            if (
                done % 256
                <
                batch_size
                or
                done == len(part)
            ):

                print(
                    f"{done}/{len(part)}"
                )

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(
        result_rows
    )


# ============================================================
# Evaluate MADLAD
# ============================================================

def evaluate_madlad(
    model_path: Path,
    test_df: pd.DataFrame,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
    device,
):

    model_name = "madlad400_3b"

    print("\n")
    print("=" * 100)
    print("EVALUATING: MADLAD-400-3B")
    print("=" * 100)

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            str(model_path),
            dtype=torch.float16,
            local_files_only=True,
        )
        .to(device)
    )

    model.eval()
    model.config.use_cache = True

    result_rows = []

    direction_config = [
        (
            "en_uz",
            "<2uz>",
        ),
        (
            "uz_en",
            "<2en>",
        ),
    ]

    for direction, prefix in direction_config:

        part = (
            test_df[
                test_df["direction"]
                ==
                direction
            ]
            .reset_index(drop=True)
        )

        print(
            f"\n{direction.upper()} "
            f"samples: {len(part)}"
        )

        for start in range(
            0,
            len(part),
            batch_size,
        ):

            batch = part.iloc[
                start:
                start + batch_size
            ]

            sources = [
                f"{prefix} {text}"
                for text
                in (
                    batch["source"]
                    .astype(str)
                    .tolist()
                )
            ]

            inputs = tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_source_length,
            )

            inputs = {
                k: v.to(device)
                for k, v
                in inputs.items()
            }

            torch.cuda.synchronize()

            t0 = time.perf_counter()

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                -
                t0
            )

            predictions = (
                tokenizer.batch_decode(
                    output,
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
                        model_name,

                    "prediction":
                        prediction,

                    "latency_seconds":
                        latency,
                })

            done = min(
                start + batch_size,
                len(part),
            )

            if (
                done % 128
                <
                batch_size
                or
                done == len(part)
            ):

                print(
                    f"{done}/{len(part)}"
                )

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(
        result_rows
    )


# ============================================================
# Build summary
# ============================================================

def build_summary(
    predictions: pd.DataFrame,
):

    rows = []

    # --------------------------------------------------------
    # Overall dataset / direction
    # --------------------------------------------------------

    for model in predictions["model"].unique():

        model_df = predictions[
            predictions["model"]
            ==
            model
        ]

        for dataset in model_df["dataset"].unique():

            dataset_df = model_df[
                model_df["dataset"]
                ==
                dataset
            ]

            for direction in [
                "en_uz",
                "uz_en",
            ]:

                part = dataset_df[
                    dataset_df["direction"]
                    ==
                    direction
                ]

                if len(part) == 0:
                    continue

                metrics = calculate_metrics(
                    part
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

            # ------------------------------------------------
            # Challenge category metrics
            # ------------------------------------------------

            categories = (
                dataset_df[
                    "category"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            for category in categories:

                if category == "ALL":
                    continue

                category_df = dataset_df[
                    dataset_df["category"]
                    ==
                    category
                ]

                for direction in [
                    "en_uz",
                    "uz_en",
                ]:

                    part = category_df[
                        category_df["direction"]
                        ==
                        direction
                    ]

                    if len(part) == 0:
                        continue

                    metrics = (
                        calculate_metrics(
                            part
                        )
                    )

                    rows.append({
                        "model":
                            model,

                        "dataset":
                            dataset,

                        "category":
                            category,

                        "direction":
                            direction,

                        **metrics,
                    })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    output_dir = (
        project_root
        / "results"
        / "student"
        / "small100"
        / "heldout_eval_v1"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark_path = Path(
        args.benchmark
    )

    challenge_path = Path(
        args.challenge
    )

    small100_path = Path(
        args.small100_path
    )

    student_path = Path(
        args.student_path
    )

    madlad_path = Path(
        args.madlad_path
    )

    # ========================================================
    # Check
    # ========================================================

    for path in [
        benchmark_path,
        challenge_path,
        small100_path,
        student_path,
        madlad_path,
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

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 09 - HELD-OUT EVALUATION")
    print("=" * 100)

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "Beam:",
        args.num_beams
    )

    # ========================================================
    # Load benchmarks
    # ========================================================

    benchmark_raw = read_table(
        benchmark_path
    )

    challenge_raw = read_table(
        challenge_path
    )

    benchmark_pairs = (
        normalize_pair_dataset(
            benchmark_raw,
            "benchmark",
        )
    )

    challenge_pairs = (
        normalize_pair_dataset(
            challenge_raw,
            "challenge",
        )
    )

    print(
        "\nBenchmark pairs:",
        len(benchmark_pairs)
    )

    print(
        "Challenge pairs:",
        len(challenge_pairs)
    )

    if "category" in challenge_pairs.columns:

        print(
            "\nChallenge categories:"
        )

        print(
            challenge_pairs[
                "category"
            ]
            .value_counts()
            .to_string()
        )

    test_df = pd.concat(
        [
            expand_bidirectional(
                benchmark_pairs
            ),

            expand_bidirectional(
                challenge_pairs
            ),
        ],
        ignore_index=True,
    )

    print(
        "\nTotal directed samples:",
        len(test_df)
    )

    # ========================================================
    # Correct SMALL100 tokenizer
    # ========================================================

    sys.path.insert(
        0,
        str(small100_path),
    )

    from tokenization_small100 import (
        SMALL100Tokenizer
    )

    # ========================================================
    # Original
    # ========================================================

    original_result = (
        evaluate_small100(
            model_name=
                "small100_original",

            model_path=
                small100_path,

            tokenizer_class=
                SMALL100Tokenizer,

            test_df=
                test_df,

            batch_size=
                args.small100_batch_size,

            num_beams=
                args.num_beams,

            max_source_length=
                args.max_source_length,

            max_new_tokens=
                args.max_new_tokens,

            device=
                device,
        )
    )

    original_result.to_parquet(
        output_dir
        /
        "small100_original_predictions.parquet",
        index=False,
    )

    # ========================================================
    # Fine-tuned Student
    # ========================================================

    student_result = (
        evaluate_small100(
            model_name=
                "student_exp1",

            model_path=
                student_path,

            tokenizer_class=
                SMALL100Tokenizer,

            test_df=
                test_df,

            batch_size=
                args.small100_batch_size,

            num_beams=
                args.num_beams,

            max_source_length=
                args.max_source_length,

            max_new_tokens=
                args.max_new_tokens,

            device=
                device,
        )
    )

    student_result.to_parquet(
        output_dir
        /
        "student_exp1_predictions.parquet",
        index=False,
    )

    # ========================================================
    # MADLAD
    # ========================================================

    teacher_result = (
        evaluate_madlad(
            model_path=
                madlad_path,

            test_df=
                test_df,

            batch_size=
                args.madlad_batch_size,

            num_beams=
                args.num_beams,

            max_source_length=
                args.max_source_length,

            max_new_tokens=
                args.max_new_tokens,

            device=
                device,
        )
    )

    teacher_result.to_parquet(
        output_dir
        /
        "madlad_predictions.parquet",
        index=False,
    )

    # ========================================================
    # Combine
    # ========================================================

    predictions = pd.concat(
        [
            original_result,
            student_result,
            teacher_result,
        ],
        ignore_index=True,
    )

    predictions.to_parquet(
        output_dir
        /
        "all_predictions.parquet",
        index=False,
    )

    predictions.to_csv(
        output_dir
        /
        "all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Metrics
    # ========================================================

    summary = build_summary(
        predictions
    )

    summary.to_csv(
        output_dir
        /
        "heldout_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Terminal
    # ========================================================

    print("\n")
    print("=" * 120)
    print("HELD-OUT SUMMARY")
    print("=" * 120)

    main_summary = summary[
        summary[
            "category"
        ]
        ==
        "ALL"
    ]

    print(
        main_summary[
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
    print("CHALLENGE CATEGORY SUMMARY")
    print("=" * 120)

    challenge_summary = summary[
        (
            summary[
                "dataset"
            ]
            ==
            "challenge"
        )
        &
        (
            summary[
                "category"
            ]
            !=
            "ALL"
        )
    ]

    if len(
        challenge_summary
    ):

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

    else:

        print(
            "No challenge categories found."
        )

    print("\n")
    print("=" * 100)
    print("STEP 09 COMPLETE")
    print("=" * 100)

    print(
        "Results:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()