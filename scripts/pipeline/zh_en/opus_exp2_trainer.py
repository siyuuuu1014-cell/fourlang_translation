from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


STEP_VERSION = "19_EXP2_KD_V1"
DEFAULT_SEED = 2026


VALID_DIRECTIONS = {
    "zh_en": {
        "source_lang": "zh",
        "target_lang": "en",
        "validation_source_column": "zh",
        "validation_target_column": "en",
        "exp1_best_relative": (
            "results/specialists/"
            "zh_en/opus_mt_zh_en/exp1_human/best_model"
        ),
        "output_relative": (
            "results/specialists/"
            "zh_en/opus_mt_zh_en/exp2_kd_v1"
        ),
    },

    "en_zh": {
        "source_lang": "en",
        "target_lang": "zh",
        "validation_source_column": "en",
        "validation_target_column": "zh",
        "exp1_best_relative": (
            "results/specialists/"
            "en_zh/opus_mt_en_zh/exp1_human/best_model"
        ),
        "output_relative": (
            "results/specialists/"
            "en_zh/opus_mt_en_zh/exp2_kd_v1"
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
# Dataset
# ============================================================

class DirectedTranslationDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
    ):

        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ):

        row = self.df.iloc[index]

        return {
            "training_id":
                str(
                    row.get(
                        "training_id",
                        row.get(
                            "pair_id",
                            index,
                        ),
                    )
                ),

            "source_text":
                str(
                    row[
                        "source_text"
                    ]
                ),

            "target_text":
                str(
                    row[
                        "target_text"
                    ]
                ),

            "training_weight":
                float(
                    row[
                        "training_weight"
                    ]
                ),

            "training_origin":
                str(
                    row.get(
                        "training_origin",
                        "VALIDATION",
                    )
                ),
        }


# ============================================================
# Collator
# ============================================================

class TranslationCollator:

    def __init__(
        self,
        tokenizer,
        max_source_length: int,
        max_target_length: int,
    ):

        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(
        self,
        rows: list[dict[str, Any]],
    ):

        sources = [
            row[
                "source_text"
            ]
            for row in rows
        ]

        targets = [
            row[
                "target_text"
            ]
            for row in rows
        ]

        weights = torch.tensor(
            [
                row[
                    "training_weight"
                ]
                for row in rows
            ],
            dtype=torch.float32,
        )

        source_encoding = self.tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
        )

        target_encoding = self.tokenizer(
            text_target=targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
        )

        labels = (
            target_encoding[
                "input_ids"
            ]
            .clone()
        )

        labels[
            labels
            ==
            self.tokenizer.pad_token_id
        ] = -100

        source_encoding[
            "labels"
        ] = labels

        source_encoding[
            "training_weight"
        ] = weights

        return source_encoding


# ============================================================
# Weighted seq2seq loss
# ============================================================

def compute_per_sample_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
):

    batch_size = labels.size(0)
    sequence_length = labels.size(1)
    vocab_size = logits.size(-1)

    token_losses = (
        F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size,
            ),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        .reshape(
            batch_size,
            sequence_length,
        )
    )

    valid_mask = labels.ne(-100)

    token_counts = (
        valid_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    per_sample_loss = (
        (
            token_losses
            *
            valid_mask
        )
        .sum(dim=1)
        /
        token_counts
    )

    return per_sample_loss


def compute_weighted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    training_weight: torch.Tensor,
):

    per_sample = compute_per_sample_loss(
        logits,
        labels,
    )

    weights = training_weight.to(
        device=per_sample.device,
        dtype=per_sample.dtype,
    )

    denominator = (
        weights
        .sum()
        .clamp_min(
            1e-8
        )
    )

    loss = (
        (
            per_sample
            *
            weights
        )
        .sum()
        /
        denominator
    )

    return (
        loss,
        per_sample,
        weights,
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate_validation_loss(
    model,
    loader,
    device,
    use_amp: bool,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:

        batch.pop(
            "training_weight"
        )

        batch = {
            key:
                value.to(
                    device,
                    non_blocking=True,
                )
            for key, value
            in batch.items()
        }

        labels = batch[
            "labels"
        ]

        with torch.amp.autocast(
            "cuda",
            enabled=use_amp,
        ):

            outputs = model(
                **batch
            )

            per_sample = compute_per_sample_loss(
                outputs.logits,
                labels,
            )

            # IMPORTANT:
            # Frozen validation remains intentionally UNWEIGHTED,
            # matching Step16 Exp1.
            loss_sum = (
                per_sample
                .sum()
                .item()
            )

        total_loss += loss_sum
        total_samples += int(
            labels.size(0)
        )

    if total_samples == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    return (
        total_loss
        /
        total_samples
    )


# ============================================================
# Save helpers
# ============================================================

def save_model(
    model,
    tokenizer,
    output_dir: Path,
):

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        output_dir
    )


def save_latest_checkpoint(
    *,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    baseline_val_loss: float,
    best_val_loss: float,
    best_epoch: int,
    patience_counter: int,
    history: list[dict[str, Any]],
):

    if checkpoint_dir.exists():
        shutil.rmtree(
            checkpoint_dir
        )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        checkpoint_dir
        /
        "model"
    )

    model.save_pretrained(
        model_dir,
        safe_serialization=True,
    )

    tokenizer.save_pretrained(
        model_dir
    )

    state = {
        "epoch":
            int(epoch),

        "global_step":
            int(global_step),

        "baseline_val_loss":
            float(
                baseline_val_loss
            ),

        "best_val_loss":
            float(
                best_val_loss
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "patience_counter":
            int(
                patience_counter
            ),

        "optimizer":
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

        "scaler":
            scaler.state_dict(),

        "history":
            history,
    }

    torch.save(
        state,
        checkpoint_dir
        /
        "training_state.pt",
    )


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
# Data preparation
# ============================================================

def prepare_train_dataframe(
    path: Path,
    direction: str,
):

    df = pd.read_parquet(
        path
    )

    required = {
        "direction",
        "source_text",
        "target_text",
        "training_weight",
        "training_origin",
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Exp2 training file missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df.loc[
            df[
                "direction"
            ]
            .astype(str)
            ==
            direction
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    df[
        "source_text"
    ] = (
        df[
            "source_text"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "target_text"
    ] = (
        df[
            "target_text"
        ]
        .astype(str)
        .str.strip()
    )

    df[
        "training_weight"
    ] = pd.to_numeric(
        df[
            "training_weight"
        ],
        errors="coerce",
    )

    if len(df) == 0:
        raise RuntimeError(
            f"No Exp2 rows for direction={direction}"
        )

    if (
        df[
            "source_text"
        ]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Empty source_text found."
        )

    if (
        df[
            "target_text"
        ]
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Empty target_text found."
        )

    if (
        df[
            "training_weight"
        ]
        .isna()
        .any()
        or
        (
            df[
                "training_weight"
            ]
            <=
            0
        )
        .any()
    ):
        raise RuntimeError(
            "Invalid training_weight found."
        )

    if "training_id" not in df.columns:
        df[
            "training_id"
        ] = [
            f"EXP2_{direction}_{i:07d}"
            for i in range(
                len(df)
            )
        ]

    if not df[
        "training_id"
    ].astype(str).is_unique:
        raise RuntimeError(
            "training_id is not unique "
            f"inside direction={direction}."
        )

    return df


def prepare_validation_dataframe(
    path: Path,
    config: dict[str, Any],
):

    df = pd.read_parquet(
        path
    )

    source_column = (
        config[
            "validation_source_column"
        ]
    )

    target_column = (
        config[
            "validation_target_column"
        ]
    )

    required = {
        source_column,
        target_column,
    }

    missing = (
        required
        -
        set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Validation file missing columns: "
            f"{sorted(missing)}"
        )

    valid = (
        df[
            source_column
        ]
        .notna()
        &
        df[
            target_column
        ]
        .notna()
    )

    df = (
        df.loc[
            valid
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    out = pd.DataFrame(
        {
            "training_id": (
                (
                    "VAL_"
                    +
                    df.index
                    .astype(str)
                )
            ),

            "source_text": (
                df[
                    source_column
                ]
                .astype(str)
                .str.strip()
            ),

            "target_text": (
                df[
                    target_column
                ]
                .astype(str)
                .str.strip()
            ),

            "training_weight":
                1.0,

            "training_origin":
                "FROZEN_VALIDATION",
        }
    )

    if len(out) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    return out


# ============================================================
# CLI
# ============================================================

def build_parser(
    direction: str,
):

    config = VALID_DIRECTIONS[
        direction
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init_model_path",
        default=None,
        help=(
            "Defaults to the direction's Step16 Exp1 best_model."
        ),
    )

    parser.add_argument(
        "--train_file",
        default=(
            "data/distillation/zh_en/v1/"
            "18h_exp2_training/"
            "exp2_train_combined_v1.parquet"
        ),
    )

    parser.add_argument(
        "--validation_file",
        default=(
            "data/splits/zh_en/v1/"
            "validation_pairs_v1.parquet"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


# ============================================================
# Main
# ============================================================

def run_direction(
    direction: str,
):

    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            direction
        )

    args = (
        build_parser(
            direction
        )
        .parse_args()
    )

    config = VALID_DIRECTIONS[
        direction
    ]

    seed_everything(
        args.seed
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    train_file = Path(
        args.train_file
    )

    if not train_file.is_absolute():
        train_file = (
            project_root
            /
            train_file
        )

    validation_file = Path(
        args.validation_file
    )

    if not validation_file.is_absolute():
        validation_file = (
            project_root
            /
            validation_file
        )

    if args.init_model_path:
        init_model_path = Path(
            args.init_model_path
        )

        if not init_model_path.is_absolute():
            init_model_path = (
                project_root
                /
                init_model_path
            )

    else:
        init_model_path = (
            project_root
            /
            config[
                "exp1_best_relative"
            ]
        )

    output_root = (
        project_root
        /
        config[
            "output_relative"
        ]
    )

    best_model_dir = (
        output_root
        /
        "best_model"
    )

    latest_checkpoint_dir = (
        output_root
        /
        "checkpoints"
        /
        "latest"
    )

    history_file = (
        output_root
        /
        "training_history.csv"
    )

    report_file = (
        output_root
        /
        "training_report.json"
    )

    print("=" * 110)
    print("ZH-EN SPECIALIST DISTILLATION PIPELINE")
    print("STEP 19 - EXP2 HUMAN REPLAY + MULTI-TEACHER KD")
    print("=" * 110)

    print(
        "\nDirection:",
        direction
    )

    print(
        "Source:",
        config[
            "source_lang"
        ]
    )

    print(
        "Target:",
        config[
            "target_lang"
        ]
    )

    print(
        "\nExp2 train:"
    )

    print(
        train_file
    )

    print(
        "\nFrozen validation:"
    )

    print(
        validation_file
    )

    print(
        "\nExp1 initialization:"
    )

    print(
        init_model_path
    )

    print(
        "\nExp2 output:"
    )

    print(
        output_root
    )

    # ========================================================
    # Safety
    # ========================================================

    if not train_file.exists():
        raise FileNotFoundError(
            train_file
        )

    if not validation_file.exists():
        raise FileNotFoundError(
            validation_file
        )

    if not init_model_path.exists():
        raise FileNotFoundError(
            init_model_path
        )

    if (
        output_root.exists()
        and
        not args.resume
        and
        not args.overwrite
    ):
        raise RuntimeError(
            "\nOutput already exists:\n"
            f"{output_root}\n\n"
            "Use --resume or --overwrite."
        )

    if (
        args.overwrite
        and
        output_root.exists()
    ):
        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Data
    # ========================================================

    train_df = prepare_train_dataframe(
        train_file,
        direction,
    )

    validation_df = prepare_validation_dataframe(
        validation_file,
        config,
    )

    print(
        "\nTrain directed rows:",
        len(
            train_df
        )
    )

    print(
        "Validation pairs:",
        len(
            validation_df
        )
    )

    print(
        "\nTraining origin distribution:"
    )

    print(
        train_df[
            "training_origin"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTraining weights:"
    )

    print(
        train_df[
            "training_weight"
        ]
        .value_counts()
        .sort_index(
            ascending=False
        )
        .to_string()
    )

    if (
        "selected_teacher"
        in
        train_df.columns
    ):
        kd_only = train_df.loc[
            train_df[
                "training_origin"
            ]
            .astype(str)
            ==
            "TEACHER_KD"
        ]

        if len(kd_only):

            print(
                "\nKD selected Teacher:"
            )

            print(
                kd_only[
                    "selected_teacher"
                ]
                .value_counts()
                .to_string()
            )

            if (
                "target_origin"
                in
                kd_only.columns
            ):

                print(
                    "\nKD target origin:"
                )

                print(
                    kd_only[
                        "target_origin"
                    ]
                    .value_counts()
                    .to_string()
                )

    # ========================================================
    # GPU
    # ========================================================

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    use_amp = True

    # ========================================================
    # Tokenizer
    # ========================================================

    tokenizer_source = (
        latest_checkpoint_dir
        /
        "model"
        if args.resume
        else init_model_path
    )

    print(
        "\nLoading tokenizer from:"
    )

    print(
        tokenizer_source
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            tokenizer_source,
            local_files_only=True,
            use_fast=False,
        )
    )

    print(
        "Tokenizer loaded."
    )

    # ========================================================
    # Dataset / loader
    # ========================================================

    train_dataset = DirectedTranslationDataset(
        train_df
    )

    validation_dataset = DirectedTranslationDataset(
        validation_df
    )

    collator = TranslationCollator(
        tokenizer,
        max_source_length=(
            args.max_source_length
        ),
        max_target_length=(
            args.max_target_length
        ),
    )

    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=True,
        collate_fn=collator,
        num_workers=(
            args.num_workers
        ),
        pin_memory=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            args.batch_size
        ),
        shuffle=False,
        collate_fn=collator,
        num_workers=(
            args.num_workers
        ),
        pin_memory=True,
    )

    # ========================================================
    # Model + resume state
    # ========================================================

    if args.resume:

        state_file = (
            latest_checkpoint_dir
            /
            "training_state.pt"
        )

        checkpoint_model_dir = (
            latest_checkpoint_dir
            /
            "model"
        )

        if not state_file.exists():
            raise FileNotFoundError(
                state_file
            )

        if not checkpoint_model_dir.exists():
            raise FileNotFoundError(
                checkpoint_model_dir
            )

        model_load_path = (
            checkpoint_model_dir
        )

    else:
        model_load_path = (
            init_model_path
        )

    print(
        "\nLoading model from:"
    )

    print(
        model_load_path
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_load_path,
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
        )
        .to(
            device
        )
    )

    model.config.use_cache = False
    model.train()

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Model loaded."
    )

    print(
        "Parameters:",
        f"{total_parameters:,}"
    )

    # ========================================================
    # Optimizer / scheduler
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=(
            args.learning_rate
        ),
        weight_decay=(
            args.weight_decay
        ),
    )

    optimizer_steps_per_epoch = math.ceil(
        len(
            train_loader
        )
        /
        args.gradient_accumulation_steps
    )

    total_training_steps = (
        optimizer_steps_per_epoch
        *
        args.epochs
    )

    warmup_steps = int(
        round(
            total_training_steps
            *
            args.warmup_ratio
        )
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=(
            warmup_steps
        ),
        num_training_steps=(
            total_training_steps
        ),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(
        "\nTraining configuration:"
    )

    print(
        "Epochs:",
        args.epochs
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "Gradient accumulation:",
        args.gradient_accumulation_steps
    )

    print(
        "Effective batch:",
        (
            args.batch_size
            *
            args.gradient_accumulation_steps
        )
    )

    print(
        "Learning rate:",
        args.learning_rate
    )

    print(
        "Optimizer steps/epoch:",
        optimizer_steps_per_epoch
    )

    print(
        "Total steps:",
        total_training_steps
    )

    print(
        "Warmup steps:",
        warmup_steps
    )

    # ========================================================
    # State / baseline / resume
    # ========================================================

    history = []
    start_epoch = 1
    global_step = 0
    best_epoch = 0
    patience_counter = 0

    if args.resume:

        state = torch.load(
            state_file,
            map_location="cpu",
        )

        optimizer.load_state_dict(
            state[
                "optimizer"
            ]
        )

        scheduler.load_state_dict(
            state[
                "scheduler"
            ]
        )

        scaler.load_state_dict(
            state[
                "scaler"
            ]
        )

        start_epoch = (
            int(
                state[
                    "epoch"
                ]
            )
            +
            1
        )

        global_step = int(
            state[
                "global_step"
            ]
        )

        baseline_val_loss = float(
            state[
                "baseline_val_loss"
            ]
        )

        best_val_loss = float(
            state[
                "best_val_loss"
            ]
        )

        best_epoch = int(
            state[
                "best_epoch"
            ]
        )

        patience_counter = int(
            state[
                "patience_counter"
            ]
        )

        history = list(
            state.get(
                "history",
                []
            )
        )

        print(
            "\nResumed state:"
        )

        print(
            "Start epoch:",
            start_epoch
        )

        print(
            "Exp1 baseline validation loss:",
            f"{baseline_val_loss:.6f}"
        )

        print(
            "Current best validation loss:",
            f"{best_val_loss:.6f}"
        )

    else:

        print(
            "\nEvaluating Exp1 best model "
            "on frozen validation..."
        )

        model.config.use_cache = False

        baseline_val_loss = evaluate_validation_loss(
            model,
            validation_loader,
            device,
            use_amp,
        )

        print(
            "Exp1 baseline validation loss:",
            f"{baseline_val_loss:.6f}"
        )

        best_val_loss = baseline_val_loss
        best_epoch = 0

        # Epoch 0 = current Exp1 best. This guarantees Exp2 can never
        # silently replace the final model with a worse KD checkpoint.
        save_model(
            model,
            tokenizer,
            best_model_dir,
        )

        history.append(
            {
                "epoch":
                    0,

                "train_weighted_loss":
                    None,

                "validation_loss":
                    float(
                        baseline_val_loss
                    ),

                "learning_rate":
                    float(
                        args.learning_rate
                    ),

                "best":
                    True,

                "source":
                    "EXP1_BASELINE",
            }
        )

    # ========================================================
    # Training
    # ========================================================

    stopped_early = False

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print(
            "\n"
            +
            "=" * 110
        )

        print(
            f"EPOCH {epoch}/{args.epochs}"
        )

        print(
            "=" * 110
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_weighted_numerator = 0.0
        epoch_weight_sum = 0.0
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            training_weight = batch.pop(
                "training_weight"
            )

            batch = {
                key:
                    value.to(
                        device,
                        non_blocking=True,
                    )
                for key, value
                in batch.items()
            }

            training_weight = training_weight.to(
                device,
                non_blocking=True,
            )

            labels = batch[
                "labels"
            ]

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp,
            ):

                outputs = model(
                    **batch
                )

                (
                    raw_loss,
                    per_sample_loss,
                    sample_weights,
                ) = compute_weighted_loss(
                    outputs.logits,
                    labels,
                    training_weight,
                )

                loss = (
                    raw_loss
                    /
                    args.gradient_accumulation_steps
                )

            scaler.scale(
                loss
            ).backward()

            epoch_weighted_numerator += float(
                (
                    per_sample_loss.detach()
                    *
                    sample_weights.detach()
                )
                .sum()
                .item()
            )

            epoch_weight_sum += float(
                sample_weights
                .detach()
                .sum()
                .item()
            )

            should_step = (
                batch_index
                %
                args.gradient_accumulation_steps
                ==
                0
                or
                batch_index
                ==
                len(
                    train_loader
                )
            )

            if should_step:

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            if (
                batch_index
                %
                200
                ==
                0
                or
                batch_index
                ==
                len(
                    train_loader
                )
            ):

                current_train_loss = (
                    epoch_weighted_numerator
                    /
                    max(
                        epoch_weight_sum,
                        1e-8,
                    )
                )

                current_lr = (
                    optimizer
                    .param_groups[0][
                        "lr"
                    ]
                )

                print(
                    f"{batch_index:>5}/"
                    f"{len(train_loader)}"
                    f" | weighted_loss "
                    f"{current_train_loss:.5f}"
                    f" | lr "
                    f"{current_lr:.8f}"
                )

        train_loss = (
            epoch_weighted_numerator
            /
            max(
                epoch_weight_sum,
                1e-8,
            )
        )

        epoch_seconds = (
            time.perf_counter()
            -
            epoch_start
        )

        # ====================================================
        # Frozen validation
        # ====================================================

        print(
            "\nEvaluating frozen validation..."
        )

        validation_loss = evaluate_validation_loss(
            model,
            validation_loader,
            device,
            use_amp,
        )

        improved = (
            validation_loss
            <
            best_val_loss
        )

        if improved:

            best_val_loss = validation_loss
            best_epoch = epoch
            patience_counter = 0

            save_model(
                model,
                tokenizer,
                best_model_dir,
            )

        else:

            patience_counter += 1

        current_lr = (
            optimizer
            .param_groups[0][
                "lr"
            ]
        )

        history.append(
            {
                "epoch":
                    int(epoch),

                "train_weighted_loss":
                    float(
                        train_loss
                    ),

                "validation_loss":
                    float(
                        validation_loss
                    ),

                "learning_rate":
                    float(
                        current_lr
                    ),

                "epoch_seconds":
                    float(
                        epoch_seconds
                    ),

                "best":
                    bool(
                        improved
                    ),

                "source":
                    (
                        "EXP2_KD"
                        if improved
                        else
                        "EXP2_NOT_BEST"
                    ),
            }
        )

        print(
            "\nEpoch result:"
        )

        print(
            "Train weighted loss:",
            f"{train_loss:.6f}"
        )

        print(
            "Validation loss:",
            f"{validation_loss:.6f}"
        )

        print(
            "Exp1 baseline validation loss:",
            f"{baseline_val_loss:.6f}"
        )

        print(
            "Best validation loss:",
            f"{best_val_loss:.6f}"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Improved over current best:",
            improved
        )

        print(
            "Patience:",
            f"{patience_counter}/"
            f"{args.early_stopping_patience}"
        )

        # Keep ONE full resume checkpoint only.
        save_latest_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_dir=(
                latest_checkpoint_dir
            ),
            epoch=epoch,
            global_step=global_step,
            baseline_val_loss=(
                baseline_val_loss
            ),
            best_val_loss=(
                best_val_loss
            ),
            best_epoch=(
                best_epoch
            ),
            patience_counter=(
                patience_counter
            ),
            history=history,
        )

        pd.DataFrame(
            history
        ).to_csv(
            history_file,
            index=False,
            encoding="utf-8-sig",
        )

        if (
            patience_counter
            >=
            args.early_stopping_patience
        ):

            print(
                "\nEARLY STOPPING"
            )

            stopped_early = True

            break

    # ========================================================
    # Final report
    # ========================================================

    best_source = (
        "EXP1_BASELINE"
        if best_epoch == 0
        else
        "EXP2_KD"
    )

    delta_loss = (
        best_val_loss
        -
        baseline_val_loss
    )

    origin_counts = {
        str(k):
            int(v)
        for k, v
        in train_df[
            "training_origin"
        ]
        .value_counts()
        .to_dict()
        .items()
    }

    kd_rows = int(
        (
            train_df[
                "training_origin"
            ]
            .astype(str)
            ==
            "TEACHER_KD"
        )
        .sum()
    )

    human_rows = int(
        (
            train_df[
                "training_origin"
            ]
            .astype(str)
            ==
            "HUMAN_REPLAY"
        )
        .sum()
    )

    report = {
        "step":
            "19",

        "step_version":
            STEP_VERSION,

        "direction":
            direction,

        "source_lang":
            config[
                "source_lang"
            ],

        "target_lang":
            config[
                "target_lang"
            ],

        "initial_model":
            str(
                init_model_path
            ),

        "initial_model_role":
            "STEP16_EXP1_BEST",

        "train_file":
            str(
                train_file
            ),

        "validation_file":
            str(
                validation_file
            ),

        "train_directed_rows":
            int(
                len(
                    train_df
                )
            ),

        "human_replay_rows":
            human_rows,

        "teacher_kd_rows":
            kd_rows,

        "kd_fraction_percent":
            (
                100.0
                *
                kd_rows
                /
                len(
                    train_df
                )
            ),

        "origin_counts":
            origin_counts,

        "weighted_training":
            True,

        "training_weight_policy": {
            "HUMAN_REPLAY":
                1.0,

            "TEACHER_KD":
                1.0,
        },

        "validation_weighted":
            False,

        "hyperparameters": {
            "epochs":
                args.epochs,

            "batch_size":
                args.batch_size,

            "gradient_accumulation_steps":
                args.gradient_accumulation_steps,

            "effective_batch_size":
                (
                    args.batch_size
                    *
                    args.gradient_accumulation_steps
                ),

            "learning_rate":
                args.learning_rate,

            "weight_decay":
                args.weight_decay,

            "warmup_ratio":
                args.warmup_ratio,

            "max_source_length":
                args.max_source_length,

            "max_target_length":
                args.max_target_length,

            "max_grad_norm":
                args.max_grad_norm,

            "seed":
                args.seed,
        },

        "exp1_baseline_validation_loss":
            float(
                baseline_val_loss
            ),

        "best_validation_loss":
            float(
                best_val_loss
            ),

        "delta_validation_loss":
            float(
                delta_loss
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_source":
            best_source,

        "stopped_early":
            bool(
                stopped_early
            ),

        "best_model":
            str(
                best_model_dir
            ),

        "latest_checkpoint":
            str(
                latest_checkpoint_dir
            ),

        "history":
            history,

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            (
                "EXP2_KD_IMPROVED_VALIDATION"
                if best_epoch > 0
                else
                "EXP1_BASELINE_REMAINS_BEST"
            ),
    }

    save_json(
        report,
        report_file,
    )

    pd.DataFrame(
        history
    ).to_csv(
        history_file,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n")
    print("=" * 110)
    print("STEP 19 EXP2 RESULT")
    print("=" * 110)

    print(
        "\nDirection:",
        direction
    )

    print(
        "\nHuman Replay rows:",
        human_rows
    )

    print(
        "Teacher KD rows:",
        kd_rows
    )

    print(
        "KD fraction:",
        f"{100.0 * kd_rows / len(train_df):.2f}%"
    )

    print(
        "\nExp1 baseline validation loss:",
        f"{baseline_val_loss:.6f}"
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}"
    )

    print(
        "Delta validation loss:",
        f"{delta_loss:+.6f}"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best source:",
        best_source
    )

    print(
        "\nBest model:"
    )

    print(
        best_model_dir
    )

    print(
        "\nTraining report:"
    )

    print(
        report_file
    )

    print(
        "\nSTATUS:"
    )

    if best_epoch > 0:

        print(
            "EXP2_KD_IMPROVED_VALIDATION"
        )

    else:

        print(
            "EXP1_BASELINE_REMAINS_BEST"
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    raise RuntimeError(
        "Do not run opus_exp2_trainer.py directly. "
        "Use 19a_train_opus_zh_en_exp2.py or "
        "19b_train_opus_en_zh_exp2.py."
    )
