from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sacrebleu.metrics import CHRF
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


STEP_VERSION = "17B_V1"
SEED = 2026


DIRECTIONS = {

    "zh_en": {
        "source_column": "zh",
        "target_column": "en",
        "madlad_prefix": "<2en>",
        "opus_model_relative": (
            "results/specialists/"
            "zh_en/"
            "opus_mt_zh_en/"
            "exp1_human/"
            "best_model"
        ),
    },

    "en_zh": {
        "source_column": "en",
        "target_column": "zh",
        "madlad_prefix": "<2zh>",
        "opus_model_relative": (
            "results/specialists/"
            "en_zh/"
            "opus_mt_en_zh/"
            "exp1_human/"
            "best_model"
        ),
    },
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 17B - Build OPUS vs MADLAD "
            "teacher disagreement calibration set."
        )
    )

    parser.add_argument(
        "--madlad_model_path",
        default=None,
    )

    # 每个 source_dataset 抽多少 validation pair
    # 600 ALT + 600 Tatoeba
    # 之后双方向 = 最多 2400 directed candidates
    parser.add_argument(
        "--pool_per_source",
        type=int,
        default=600,
    )

    # 最终每个 direction × source 取多少
    # 4 strata × 200 = 800
    parser.add_argument(
        "--selected_per_stratum",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--opus_batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--madlad_batch_size",
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
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Normalization
# ============================================================

def normalize_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def normalize_for_compare(
    text: str,
) -> str:

    text = normalize_text(text)

    text = text.lower()

    return text


# ============================================================
# JSON
# ============================================================

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
# MADLAD path
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

    snapshot_root = Path(
        "/root/autodl-tmp/huggingface/hub/"
        "models--google--madlad400-3b-mt/"
        "snapshots"
    )

    if snapshot_root.exists():

        for path in sorted(
            snapshot_root.iterdir()
        ):

            if path.is_dir():
                candidates.append(path)

    for path in candidates:

        if (
            path.exists()
            and
            (path / "config.json").exists()
        ):
            return path

    raise FileNotFoundError(
        "MADLAD model cannot be located."
    )


# ============================================================
# Load validation
# ============================================================

def load_validation(
    project_root: Path,
) -> pd.DataFrame:

    path = (
        project_root
        / "data"
        / "splits"
        / "zh_en"
        / "v1"
        / "validation_pairs_v1.parquet"
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    df = pd.read_parquet(
        path
    )

    required = {
        "pair_id",
        "en",
        "zh",
        "source_dataset",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            f"Validation missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .copy()
        .reset_index(drop=True)
    )

    print(
        "\nValidation pairs:",
        len(df)
    )

    print(
        "\nSource distribution:"
    )

    print(
        df[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    return df


# ============================================================
# Calibration pool
# ============================================================

def sample_calibration_pool(
    df: pd.DataFrame,
    pool_per_source: int,
    seed: int,
) -> pd.DataFrame:

    parts = []

    available_sources = sorted(
        df[
            "source_dataset"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        "\nAvailable sources:",
        available_sources
    )

    for i, source in enumerate(
        available_sources
    ):

        part = (
            df[
                df[
                    "source_dataset"
                ].astype(str)
                ==
                source
            ]
            .copy()
        )

        n = min(
            pool_per_source,
            len(part),
        )

        sampled = (
            part
            .sample(
                n=n,
                random_state=(
                    seed + i
                ),
                replace=False,
            )
            .copy()
        )

        parts.append(
            sampled
        )

        print(
            f"{source}: "
            f"{len(part)} available "
            f"→ {len(sampled)} sampled"
        )

    pool = pd.concat(
        parts,
        ignore_index=True,
    )

    # deterministic order
    pool = (
        pool
        .sort_values(
            [
                "source_dataset",
                "pair_id",
            ]
        )
        .reset_index(drop=True)
    )

    if pool[
        "pair_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate pair_id in calibration pool."
        )

    return pool


# ============================================================
# Build directed pool
# ============================================================

def build_directed_pool(
    pair_pool: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in pair_pool.iterrows():

        for direction, config in (
            DIRECTIONS.items()
        ):

            source_text = normalize_text(
                row[
                    config[
                        "source_column"
                    ]
                ]
            )

            reference_text = normalize_text(
                row[
                    config[
                        "target_column"
                    ]
                ]
            )

            if (
                not source_text
                or
                not reference_text
            ):
                continue

            rows.append(
                {
                    "sample_id":
                        (
                            f"{row['pair_id']}"
                            f"__{direction}"
                        ),

                    "pair_id":
                        str(
                            row[
                                "pair_id"
                            ]
                        ),

                    "direction":
                        direction,

                    "source_dataset":
                        str(
                            row[
                                "source_dataset"
                            ]
                        ),

                    "source_text":
                        source_text,

                    "reference_text":
                        reference_text,
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result[
        "sample_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate directed sample_id."
        )

    return result


# ============================================================
# Model loading
# ============================================================

def load_opus(
    model_path: Path,
):

    print(
        "\nLoading OPUS:"
    )

    print(
        model_path
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float16,
        )
        .to("cuda")
    )

    model.eval()
    model.config.use_cache = True

    return (
        tokenizer,
        model,
    )


def load_madlad(
    model_path: Path,
):

    print(
        "\nLoading MADLAD:"
    )

    print(
        model_path
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=False,
        )
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float16,
        )
        .to("cuda")
    )

    model.eval()
    model.config.use_cache = True

    return (
        tokenizer,
        model,
    )


# ============================================================
# Generic generation
# ============================================================

def generate_batches(
    *,
    tokenizer,
    model,
    texts: list[str],
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
    prefix: str | None = None,
    label: str,
):

    predictions = []

    total_seconds = 0.0

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:
            start + batch_size
        ]

        if prefix is not None:

            batch = [
                f"{prefix} {text}"
                for text in batch
            ]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        encoded = {
            key:
                value.to(
                    "cuda"
                )
            for key, value
            in encoded.items()
        }

        torch.cuda.synchronize()

        begin = (
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

        total_seconds += (
            time.perf_counter()
            -
            begin
        )

        decoded = (
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        predictions.extend(
            [
                normalize_text(x)
                for x in decoded
            ]
        )

        done = min(
            start + batch_size,
            len(texts),
        )

        if (
            done % 128 == 0
            or
            done == len(texts)
        ):

            print(
                f"{label}: "
                f"{done}/{len(texts)}"
            )

    if (
        len(predictions)
        !=
        len(texts)
    ):

        raise RuntimeError(
            f"{label}: prediction count mismatch."
        )

    avg_seconds = (
        total_seconds
        /
        max(
            len(texts),
            1,
        )
    )

    print(
        f"{label} latency: "
        f"{avg_seconds:.4f}s/sample"
    )

    return predictions


# ============================================================
# OPUS prediction
# ============================================================

def generate_opus_predictions(
    project_root: Path,
    directed_df: pd.DataFrame,
    args,
):

    output = directed_df.copy()

    output[
        "opus_prediction"
    ] = ""

    for direction, config in (
        DIRECTIONS.items()
    ):

        mask = (
            output[
                "direction"
            ]
            ==
            direction
        )

        subset = (
            output.loc[
                mask
            ]
            .copy()
        )

        model_path = (
            project_root
            /
            config[
                "opus_model_relative"
            ]
        )

        if not model_path.exists():

            raise FileNotFoundError(
                model_path
            )

        tokenizer, model = (
            load_opus(
                model_path
            )
        )

        predictions = (
            generate_batches(
                tokenizer=tokenizer,
                model=model,
                texts=(
                    subset[
                        "source_text"
                    ].tolist()
                ),
                batch_size=(
                    args.opus_batch_size
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
                prefix=None,
                label=(
                    f"OPUS {direction}"
                ),
            )
        )

        output.loc[
            mask,
            "opus_prediction"
        ] = predictions

        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    return output


# ============================================================
# MADLAD prediction
# ============================================================

def generate_madlad_predictions(
    directed_df: pd.DataFrame,
    madlad_path: Path,
    args,
):

    output = directed_df.copy()

    output[
        "madlad_prediction"
    ] = ""

    tokenizer, model = (
        load_madlad(
            madlad_path
        )
    )

    for direction, config in (
        DIRECTIONS.items()
    ):

        mask = (
            output[
                "direction"
            ]
            ==
            direction
        )

        subset = (
            output.loc[
                mask
            ]
            .copy()
        )

        predictions = (
            generate_batches(
                tokenizer=tokenizer,
                model=model,
                texts=(
                    subset[
                        "source_text"
                    ].tolist()
                ),
                batch_size=(
                    args.madlad_batch_size
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
                prefix=(
                    config[
                        "madlad_prefix"
                    ]
                ),
                label=(
                    f"MADLAD {direction}"
                ),
            )
        )

        output.loc[
            mask,
            "madlad_prediction"
        ] = predictions

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    return output


# ============================================================
# Disagreement scoring
# ============================================================

def safe_chrfpp(
    hypothesis: str,
    reference: str,
    metric: CHRF,
) -> float:

    hypothesis = normalize_text(
        hypothesis
    )

    reference = normalize_text(
        reference
    )

    if (
        not hypothesis
        or
        not reference
    ):
        return 0.0

    return float(
        metric
        .sentence_score(
            hypothesis,
            [
                reference
            ],
        )
        .score
    )


def sequence_similarity(
    a: str,
    b: str,
) -> float:

    a = normalize_for_compare(a)
    b = normalize_for_compare(b)

    if (
        not a
        and
        not b
    ):
        return 1.0

    return float(
        SequenceMatcher(
            None,
            a,
            b,
        )
        .ratio()
    )


def score_disagreement(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    metric = CHRF(
        word_order=2
    )

    pair_chrfpp = []

    opus_ref = []
    madlad_ref = []

    seq_similarity = []

    exact_same = []

    for _, row in (
        result.iterrows()
    ):

        opus = str(
            row[
                "opus_prediction"
            ]
        )

        madlad = str(
            row[
                "madlad_prediction"
            ]
        )

        reference = str(
            row[
                "reference_text"
            ]
        )

        # Teacher A vs Teacher B
        pair_score = (
            safe_chrfpp(
                opus,
                madlad,
                metric,
            )
        )

        pair_chrfpp.append(
            pair_score
        )

        # diagnostics only
        opus_ref.append(
            safe_chrfpp(
                opus,
                reference,
                metric,
            )
        )

        madlad_ref.append(
            safe_chrfpp(
                madlad,
                reference,
                metric,
            )
        )

        seq_score = (
            sequence_similarity(
                opus,
                madlad,
            )
        )

        seq_similarity.append(
            seq_score
        )

        exact_same.append(
            normalize_for_compare(opus)
            ==
            normalize_for_compare(madlad)
        )

    result[
        "teacher_pair_chrfpp"
    ] = pair_chrfpp

    result[
        "teacher_sequence_similarity"
    ] = seq_similarity

    result[
        "opus_reference_chrfpp"
    ] = opus_ref

    result[
        "madlad_reference_chrfpp"
    ] = madlad_ref

    result[
        "reference_chrfpp_delta_madlad_minus_opus"
    ] = (
        result[
            "madlad_reference_chrfpp"
        ]
        -
        result[
            "opus_reference_chrfpp"
        ]
    )

    result[
        "abs_reference_chrfpp_delta"
    ] = (
        result[
            "reference_chrfpp_delta_madlad_minus_opus"
        ]
        .abs()
    )

    result[
        "teachers_exact_same"
    ] = exact_same

    # --------------------------------------------------------
    # Disagreement score
    #
    # 70% character/word ngram disagreement
    # 30% normalized sequence disagreement
    #
    # IMPORTANT:
    # reference metric is NOT used to decide who wins.
    # It is only retained as a diagnostic field.
    # --------------------------------------------------------

    result[
        "chrfpp_disagreement"
    ] = (
        100.0
        -
        result[
            "teacher_pair_chrfpp"
        ]
    )

    result[
        "sequence_disagreement"
    ] = (
        100.0
        *
        (
            1.0
            -
            result[
                "teacher_sequence_similarity"
            ]
        )
    )

    result[
        "teacher_disagreement_score"
    ] = (
        0.70
        *
        result[
            "chrfpp_disagreement"
        ]
        +
        0.30
        *
        result[
            "sequence_disagreement"
        ]
    )

    return result


# ============================================================
# Select balanced disagreement set
# ============================================================

def select_disagreements(
    scored: pd.DataFrame,
    selected_per_stratum: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    usable = (
        scored[
            (~scored[
                "teachers_exact_same"
            ])
            &
            (
                scored[
                    "opus_prediction"
                ].astype(str)
                .str.len()
                >
                0
            )
            &
            (
                scored[
                    "madlad_prediction"
                ].astype(str)
                .str.len()
                >
                0
            )
        ]
        .copy()
    )

    selected_parts = []

    manifest_rows = []

    grouped = (
        usable.groupby(
            [
                "direction",
                "source_dataset",
            ],
            dropna=False,
        )
    )

    for (
        direction,
        source_dataset
    ), part in grouped:

        part = (
            part
            .sort_values(
                [
                    "teacher_disagreement_score",
                    "abs_reference_chrfpp_delta",
                    "sample_id",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        n = min(
            selected_per_stratum,
            len(part),
        )

        chosen = (
            part
            .head(n)
            .copy()
        )

        chosen[
            "selection_rank_within_stratum"
        ] = range(
            1,
            len(chosen) + 1,
        )

        selected_parts.append(
            chosen
        )

        manifest_rows.append(
            {
                "direction":
                    direction,

                "source_dataset":
                    source_dataset,

                "available_disagreements":
                    int(
                        len(part)
                    ),

                "selected":
                    int(
                        len(chosen)
                    ),

                "mean_disagreement_score":
                    float(
                        chosen[
                            "teacher_disagreement_score"
                        ]
                        .mean()
                    )
                    if len(chosen)
                    else None,

                "median_disagreement_score":
                    float(
                        chosen[
                            "teacher_disagreement_score"
                        ]
                        .median()
                    )
                    if len(chosen)
                    else None,
            }
        )

    if not selected_parts:

        raise RuntimeError(
            "No disagreement samples selected."
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = (
        selected
        .sort_values(
            [
                "direction",
                "source_dataset",
                "selection_rank_within_stratum",
            ]
        )
        .reset_index(drop=True)
    )

    selected[
        "review_id"
    ] = [
        f"zh_en_teacher_{i:05d}"
        for i in range(
            len(selected)
        )
    ]

    # Put review_id first
    columns = (
        [
            "review_id"
        ]
        +
        [
            c
            for c in selected.columns
            if c != "review_id"
        ]
    )

    selected = selected[
        columns
    ]

    manifest = pd.DataFrame(
        manifest_rows
    )

    return (
        selected,
        manifest,
    )


# ============================================================
# Assertions
# ============================================================

def run_assertions(
    selected: pd.DataFrame,
):

    assertions = {}

    assertions[
        "non_empty"
    ] = (
        len(selected)
        >
        0
    )

    assertions[
        "review_id_unique"
    ] = (
        selected[
            "review_id"
        ]
        .is_unique
    )

    assertions[
        "sample_id_unique"
    ] = (
        selected[
            "sample_id"
        ]
        .is_unique
    )

    assertions[
        "no_empty_source"
    ] = bool(
        (
            selected[
                "source_text"
            ]
            .astype(str)
            .str.len()
            >
            0
        )
        .all()
    )

    assertions[
        "no_empty_reference"
    ] = bool(
        (
            selected[
                "reference_text"
            ]
            .astype(str)
            .str.len()
            >
            0
        )
        .all()
    )

    assertions[
        "no_empty_opus"
    ] = bool(
        (
            selected[
                "opus_prediction"
            ]
            .astype(str)
            .str.len()
            >
            0
        )
        .all()
    )

    assertions[
        "no_empty_madlad"
    ] = bool(
        (
            selected[
                "madlad_prediction"
            ]
            .astype(str)
            .str.len()
            >
            0
        )
        .all()
    )

    assertions[
        "teachers_not_identical"
    ] = bool(
        (
            ~selected[
                "teachers_exact_same"
            ]
        )
        .all()
    )

    assertions[
        "valid_directions"
    ] = bool(
        selected[
            "direction"
        ]
        .isin(
            [
                "zh_en",
                "en_zh",
            ]
        )
        .all()
    )

    for key, value in (
        assertions.items()
    ):

        if not value:

            raise RuntimeError(
                f"Assertion failed: {key}"
            )

    return assertions


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    seed_everything(
        args.seed
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    output_root = (
        project_root
        / "data"
        / "teacher_selection"
        / "zh_en"
        / "v1"
        / "17b_disagreement"
    )

    selected_file = (
        output_root
        / "teacher_disagreement_800_v1.parquet"
    )

    if (
        selected_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{selected_file}\n\n"
            "Use --overwrite to rerun."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN TEACHER SELECTION"
    )

    print(
        "STEP 17B - BUILD OPUS VS MADLAD DISAGREEMENT SET"
    )

    print(
        "=" * 110
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This calibration set is built from validation data."
    )

    print(
        "FLORES/Tatoeba held-out predictions are NOT "
        "used for teacher-policy calibration."
    )

    print(
        "These rows must NEVER be inserted into KD training."
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    # ========================================================
    # Validation
    # ========================================================

    validation_df = (
        load_validation(
            project_root
        )
    )

    # ========================================================
    # Pair pool
    # ========================================================

    pair_pool = (
        sample_calibration_pool(
            validation_df,
            pool_per_source=(
                args.pool_per_source
            ),
            seed=args.seed,
        )
    )

    print(
        "\nCalibration pair pool:",
        len(pair_pool)
    )

    # ========================================================
    # Directed pool
    # ========================================================

    directed_df = (
        build_directed_pool(
            pair_pool
        )
    )

    print(
        "Directed candidates:",
        len(directed_df)
    )

    print(
        "\nDirected distribution:"
    )

    print(
        directed_df.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
        .size()
        .to_string()
    )

    # ========================================================
    # OPUS
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "GENERATING OPUS EXP1 PREDICTIONS"
    )

    print(
        "=" * 110
    )

    directed_df = (
        generate_opus_predictions(
            project_root,
            directed_df,
            args,
        )
    )

    # ========================================================
    # MADLAD
    # ========================================================

    madlad_path = (
        resolve_madlad_path(
            args.madlad_model_path
        )
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "GENERATING MADLAD PREDICTIONS"
    )

    print(
        "=" * 110
    )

    directed_df = (
        generate_madlad_predictions(
            directed_df,
            madlad_path,
            args,
        )
    )

    # ========================================================
    # Save complete prediction pool
    # ========================================================

    full_predictions_file = (
        output_root
        / "teacher_calibration_pool_predictions_v1.parquet"
    )

    directed_df.to_parquet(
        full_predictions_file,
        index=False,
    )

    # ========================================================
    # Score disagreement
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "SCORING TEACHER DISAGREEMENT"
    )

    print(
        "=" * 110
    )

    scored = (
        score_disagreement(
            directed_df
        )
    )

    scored_file = (
        output_root
        / "teacher_calibration_pool_scored_v1.parquet"
    )

    scored.to_parquet(
        scored_file,
        index=False,
    )

    # ========================================================
    # Select
    # ========================================================

    selected, manifest = (
        select_disagreements(
            scored,
            selected_per_stratum=(
                args.selected_per_stratum
            ),
        )
    )

    assertions = (
        run_assertions(
            selected
        )
    )

    # ========================================================
    # Save
    # ========================================================

    selected.to_parquet(
        selected_file,
        index=False,
    )

    selected_csv = (
        output_root
        / "teacher_disagreement_800_v1.csv"
    )

    selected.to_csv(
        selected_csv,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_file = (
        output_root
        / "selection_manifest_v1.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
        encoding="utf-8-sig",
    )

    report_file = (
        output_root
        / "teacher_disagreement_report_v1.json"
    )

    report = {

        "step":
            "17B",

        "step_version":
            STEP_VERSION,

        "seed":
            args.seed,

        "source_split":
            "validation",

        "important_policy": {
            "heldout_used_for_teacher_calibration":
                False,

            "allowed_for_kd_training":
                False,

            "purpose":
                (
                    "Teacher pairwise selection calibration only"
                ),
        },

        "pair_pool":
            int(
                len(pair_pool)
            ),

        "directed_candidates":
            int(
                len(directed_df)
            ),

        "selected_rows":
            int(
                len(selected)
            ),

        "pool_per_source":
            int(
                args.pool_per_source
            ),

        "selected_per_stratum":
            int(
                args.selected_per_stratum
            ),

        "generation_policy": {
            "opus_batch_size":
                args.opus_batch_size,

            "madlad_batch_size":
                args.madlad_batch_size,

            "num_beams":
                args.num_beams,

            "max_source_length":
                args.max_source_length,

            "max_new_tokens":
                args.max_new_tokens,

            "do_sample":
                False,
        },

        "disagreement_policy": {
            "teacher_pair_chrfpp_weight":
                0.70,

            "sequence_difference_weight":
                0.30,

            "reference_metric_used_for_winner_selection":
                False,
        },

        "assertions":
            assertions,

        "outputs": {
            "full_predictions":
                str(
                    full_predictions_file
                ),

            "scored_pool":
                str(
                    scored_file
                ),

            "selected_disagreement":
                str(
                    selected_file
                ),

            "manifest":
                str(
                    manifest_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "TEACHER_DISAGREEMENT_SET_READY",
    }

    save_json(
        report,
        report_file,
    )

    # ========================================================
    # Console result
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 17B RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nCalibration pair pool:",
        len(pair_pool)
    )

    print(
        "Directed candidates:",
        len(directed_df)
    )

    print(
        "Selected disagreements:",
        len(selected)
    )

    print(
        "\nSelection manifest:"
    )

    print(
        manifest
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        "\nOverall disagreement:"
    )

    print(
        "Mean:",
        f"{selected['teacher_disagreement_score'].mean():.4f}"
    )

    print(
        "Median:",
        f"{selected['teacher_disagreement_score'].median():.4f}"
    )

    print(
        "Min:",
        f"{selected['teacher_disagreement_score'].min():.4f}"
    )

    print(
        "Max:",
        f"{selected['teacher_disagreement_score'].max():.4f}"
    )

    print(
        "\nReference chrF++ diagnostic:"
    )

    print(
        selected.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )[
            "reference_chrfpp_delta_madlad_minus_opus"
        ]
        .agg(
            [
                "mean",
                "median",
            ]
        )
        .round(4)
        .to_string()
    )

    print(
        "\nAssertions:"
    )

    for key, value in (
        assertions.items()
    ):

        print(
            f"{key}: {value}"
        )

    print(
        "\nSelected disagreement:"
    )

    print(
        selected_file
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
        "TEACHER_DISAGREEMENT_SET_READY"
    )


if __name__ == "__main__":

    main()