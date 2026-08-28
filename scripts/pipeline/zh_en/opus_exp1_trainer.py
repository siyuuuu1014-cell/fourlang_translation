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


STEP_VERSION = "16_EXP1_V1"

DEFAULT_SEED = 2026

VALID_DIRECTIONS = {
    "zh_en": {
        "source_lang": "zh",
        "target_lang": "en",
        "source_column": "zh",
        "target_column": "en",
        "model_path": "/root/autodl-tmp/models/opus-mt-zh-en",
        "output_relative": (
            "results/specialists/"
            "zh_en/opus_mt_zh_en/exp1_human"
        ),
    },

    "en_zh": {
        "source_lang": "en",
        "target_lang": "zh",
        "source_column": "en",
        "target_column": "zh",
        "model_path": "/root/autodl-tmp/models/opus-mt-en-zh",
        "output_relative": (
            "results/specialists/"
            "en_zh/opus_mt_en_zh/exp1_human"
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

class TranslationDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        source_column: str,
        target_column: str,
    ):

        self.df = (
            df
            .reset_index(drop=True)
            .copy()
        )

        self.source_column = source_column
        self.target_column = target_column

    def __len__(self):

        return len(self.df)

    def __getitem__(
        self,
        index: int,
    ):

        row = self.df.iloc[index]

        return {
            "pair_id":
                str(row["pair_id"]),

            "source_text":
                str(
                    row[
                        self.source_column
                    ]
                ),

            "target_text":
                str(
                    row[
                        self.target_column
                    ]
                ),

            "training_weight":
                float(
                    row[
                        "training_weight"
                    ]
                ),

            "quality_tier":
                str(
                    row[
                        "quality_tier"
                    ]
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

        self.max_source_length = (
            max_source_length
        )

        self.max_target_length = (
            max_target_length
        )

    def __call__(
        self,
        rows: list[dict[str, Any]],
    ):

        sources = [
            row["source_text"]
            for row in rows
        ]

        targets = [
            row["target_text"]
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

        source_encoding = (
            self.tokenizer(
                sources,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(
                    self.max_source_length
                ),
            )
        )

        target_encoding = (
            self.tokenizer(
                text_target=targets,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=(
                    self.max_target_length
                ),
            )
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

    batch_size = (
        labels.size(0)
    )

    sequence_length = (
        labels.size(1)
    )

    vocab_size = (
        logits.size(-1)
    )

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

    valid_mask = (
        labels
        .ne(-100)
    )

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

    per_sample = (
        compute_per_sample_loss(
            logits,
            labels,
        )
    )

    weights = (
        training_weight
        .to(
            device=(
                per_sample.device
            ),
            dtype=(
                per_sample.dtype
            ),
        )
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

        weights = batch.pop(
            "training_weight"
        )

        batch = {
            key: value.to(
                device,
                non_blocking=True,
            )
            for key, value
            in batch.items()
        }

        labels = batch[
            "labels"
        ]

        with torch.cuda.amp.autocast(
            enabled=use_amp
        ):

            outputs = model(
                **batch
            )

            per_sample = (
                compute_per_sample_loss(
                    outputs.logits,
                    labels,
                )
            )

            # IMPORTANT:
            # validation is intentionally UNWEIGHTED.
            # The frozen validation set should measure
            # general model fit rather than reproduce
            # training sample weights.
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
# Model saving
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


# ============================================================
# Checkpoint
# ============================================================

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
        / "model"
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

        "best_val_loss":
            float(best_val_loss),

        "best_epoch":
            int(best_epoch),

        "patience_counter":
            int(patience_counter),

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
        / "training_state.pt",
    )


# ============================================================
# JSON helper
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
# Args
# ============================================================

def build_parser(
    direction: str,
):

    config = (
        VALID_DIRECTIONS[
            direction
        ]
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default=(
            config[
                "model_path"
            ]
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
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
        default=3e-5,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
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
# Main training
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

    config = (
        VALID_DIRECTIONS[
            direction
        ]
    )

    seed_everything(
        args.seed
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    train_file = (
        project_root
        / "data"
        / "splits"
        / "zh_en"
        / "v1"
        / "train_pairs_v1.parquet"
    )

    validation_file = (
        project_root
        / "data"
        / "splits"
        / "zh_en"
        / "v1"
        / "validation_pairs_v1.parquet"
    )

    model_path = Path(
        args.model_path
    )

    output_root = (
        project_root
        / config[
            "output_relative"
        ]
    )

    best_model_dir = (
        output_root
        / "best_model"
    )

    latest_checkpoint_dir = (
        output_root
        / "checkpoints"
        / "latest"
    )

    history_file = (
        output_root
        / "training_history.csv"
    )

    report_file = (
        output_root
        / "training_report.json"
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN SPECIALIST PIPELINE"
    )

    print(
        "STEP 16 - OPUS HUMAN EXP1"
    )

    print(
        "=" * 110
    )

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
        "\nTrain:"
    )

    print(
        train_file
    )

    print(
        "\nValidation:"
    )

    print(
        validation_file
    )

    print(
        "\nBase model:"
    )

    print(
        model_path
    )

    print(
        "\nOutput:"
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

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
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
    # Load datasets
    # ========================================================

    train_df = pd.read_parquet(
        train_file
    )

    validation_df = pd.read_parquet(
        validation_file
    )

    required_columns = {
        "pair_id",
        "en",
        "zh",
        "quality_tier",
        "training_weight",
    }

    for name, df in [
        (
            "train",
            train_df,
        ),
        (
            "validation",
            validation_df,
        ),
    ]:

        missing = (
            required_columns
            -
            set(
                df.columns
            )
        )

        if missing:

            raise RuntimeError(
                f"{name} missing columns: "
                f"{sorted(missing)}"
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

    train_df = (
        train_df[
            train_df[
                source_column
            ]
            .notna()
            &
            train_df[
                target_column
            ]
            .notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation_df = (
        validation_df[
            validation_df[
                source_column
            ]
            .notna()
            &
            validation_df[
                target_column
            ]
            .notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "\nTrain pairs:",
        len(train_df)
    )

    print(
        "Validation pairs:",
        len(validation_df)
    )

    print(
        "\nTraining tier distribution:"
    )

    print(
        train_df[
            "quality_tier"
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

    # ========================================================
    # Dataset / loader
    # ========================================================

    train_dataset = (
        TranslationDataset(
            train_df,
            source_column=source_column,
            target_column=target_column,
        )
    )

    validation_dataset = (
        TranslationDataset(
            validation_df,
            source_column=source_column,
            target_column=target_column,
        )
    )

    collator = (
        TranslationCollator(
            tokenizer,
            max_source_length=(
                args.max_source_length
            ),
            max_target_length=(
                args.max_target_length
            ),
        )
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

    validation_loader = (
        DataLoader(
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
    )

    # ========================================================
    # Model
    # ========================================================

    print(
        "\nLoading OPUS model..."
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float16,
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

    optimizer_steps_per_epoch = (
        math.ceil(
            len(train_loader)
            /
            args.gradient_accumulation_steps
        )
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

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_training_steps
            ),
        )
    )

    scaler = (
        torch.cuda.amp.GradScaler(
            enabled=use_amp
        )
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
    # State
    # ========================================================

    history = []

    start_epoch = 1
    global_step = 0

    best_epoch = 0
    patience_counter = 0

    # ========================================================
    # Resume
    # ========================================================

    if args.resume:

        state_file = (
            latest_checkpoint_dir
            / "training_state.pt"
        )

        checkpoint_model_dir = (
            latest_checkpoint_dir
            / "model"
        )

        if not state_file.exists():

            raise FileNotFoundError(
                state_file
            )

        print(
            "\nResuming checkpoint:"
        )

        print(
            latest_checkpoint_dir
        )

        del model

        gc.collect()

        torch.cuda.empty_cache()

        model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(
                checkpoint_model_dir,
                local_files_only=True,
                use_safetensors=True,
                torch_dtype=torch.float16,
            )
            .to(
                device
            )
        )

        model.config.use_cache = False

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

    else:

        # ====================================================
        # Frozen raw-model validation baseline
        # ====================================================

        print(
            "\nEvaluating raw OPUS baseline "
            "on frozen validation..."
        )

        model.config.use_cache = False

        baseline_val_loss = (
            evaluate_validation_loss(
                model,
                validation_loader,
                device,
                use_amp,
            )
        )

        print(
            "Raw baseline validation loss:",
            f"{baseline_val_loss:.6f}"
        )

        best_val_loss = (
            baseline_val_loss
        )

        best_epoch = 0

        # Preserve raw model as epoch-0 best.
        save_model(
            model,
            tokenizer,
            best_model_dir,
        )

        baseline_record = {
            "epoch": 0,
            "train_weighted_loss": None,
            "validation_loss": float(
                baseline_val_loss
            ),
            "learning_rate": float(
                args.learning_rate
            ),
            "best": True,
            "source": "RAW_BASE_MODEL",
        }

        history.append(
            baseline_record
        )

    # ========================================================
    # Training
    # ========================================================

    stopped_early = False

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):

        print("\n" + "=" * 110)

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

        epoch_start = (
            time.perf_counter()
        )

        for batch_index, batch in enumerate(
            train_loader,
            start=1,
        ):

            training_weight = batch.pop(
                "training_weight"
            )

            batch = {
                key: value.to(
                    device,
                    non_blocking=True,
                )
                for key, value
                in batch.items()
            }

            training_weight = (
                training_weight.to(
                    device,
                    non_blocking=True,
                )
            )

            labels = batch[
                "labels"
            ]

            with torch.cuda.amp.autocast(
                enabled=use_amp
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

            # Reporting uses the true weighted objective.
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
                len(train_loader)
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
                batch_index % 200 == 0
                or
                batch_index
                ==
                len(train_loader)
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
        # Validation
        # ====================================================

        print(
            "\nEvaluating frozen validation..."
        )

        validation_loss = (
            evaluate_validation_loss(
                model,
                validation_loader,
                device,
                use_amp,
            )
        )

        improved = (
            validation_loss
            <
            best_val_loss
        )

        if improved:

            best_val_loss = (
                validation_loss
            )

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

        record = {
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
                    "EXP1_FINE_TUNED"
                    if improved
                    else
                    "EXP1_NOT_BEST"
                ),
        }

        history.append(
            record
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
            "Best validation loss:",
            f"{best_val_loss:.6f}"
        )

        print(
            "Best epoch:",
            best_epoch
        )

        print(
            "Improved:",
            improved
        )

        print(
            "Patience:",
            f"{patience_counter}/"
            f"{args.early_stopping_patience}"
        )

        # ====================================================
        # Save ONE resume checkpoint only
        # ====================================================

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

        # ====================================================
        # Early stopping
        # ====================================================

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
        "RAW_BASE_MODEL"
        if best_epoch == 0
        else "EXP1_FINE_TUNED"
    )

    baseline_loss = float(
        history[0][
            "validation_loss"
        ]
    )

    delta_loss = (
        best_val_loss
        -
        baseline_loss
    )

    report = {
        "step":
            "16",

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

        "base_model":
            str(
                model_path
            ),

        "train_file":
            str(
                train_file
            ),

        "validation_file":
            str(
                validation_file
            ),

        "train_pairs":
            int(
                len(train_df)
            ),

        "validation_pairs":
            int(
                len(validation_df)
            ),

        "weighted_training":
            True,

        "training_weight_policy": {
            "GOLD": 1.0,
            "SILVER": 0.8,
            "BRONZE": 0.5,
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

        "baseline_validation_loss":
            baseline_loss,

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
                "EXP1_TRAINING_COMPLETE"
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
    print(
        "=" * 110
    )

    print(
        "STEP 16 EXP1 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nDirection:",
        direction
    )

    print(
        "\nRaw baseline validation loss:",
        f"{baseline_loss:.6f}"
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
            "EXP1_FINE_TUNING_IMPROVED_VALIDATION"
        )

    else:

        print(
            "RAW_MODEL_REMAINS_BEST"
        )

    del model

    gc.collect()

    torch.cuda.empty_cache()