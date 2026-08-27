from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import (
    M2M100ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)


# ============================================================
# 1. Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Initial model. Default: "
            "Exp1 best_model."
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
        default=16,
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
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
        "--max_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# 2. Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Dataset
# ============================================================

class TranslationDataset(Dataset):

    def __init__(self, df):

        self.sources = (
            df["source_text"]
            .astype(str)
            .tolist()
        )

        self.targets = (
            df["target_text"]
            .astype(str)
            .tolist()
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Exp1 human supervision was NOT weighted.
        #
        # Therefore Exp2 keeps HUMAN_REPLAY weight = 1.0
        # to avoid introducing another experimental variable.
        #
        # Teacher KD:
        # HIGH   -> 1.0
        # MEDIUM -> 0.8
        # ----------------------------------------------------

        if "sample_origin" in df.columns:

            origins = (
                df["sample_origin"]
                .fillna("")
                .astype(str)
                .tolist()
            )

        else:

            origins = [
                "HUMAN_REPLAY"
            ] * len(df)

        if "sample_weight" in df.columns:

            stored_weights = (
                pd.to_numeric(
                    df["sample_weight"],
                    errors="coerce",
                )
                .fillna(1.0)
                .astype(float)
                .tolist()
            )

        else:

            stored_weights = [
                1.0
            ] * len(df)

        self.weights = []

        for origin, stored_weight in zip(
            origins,
            stored_weights,
        ):

            if origin == "HUMAN_REPLAY":

                effective_weight = 1.0

            elif origin == "TEACHER_KD":

                effective_weight = float(
                    stored_weight
                )

            else:

                raise RuntimeError(
                    f"Unknown sample_origin: "
                    f"{origin}"
                )

            self.weights.append(
                effective_weight
            )

    def __len__(self):

        return len(
            self.sources
        )

    def __getitem__(
        self,
        index,
    ):

        return {
            "source":
                self.sources[index],

            "target":
                self.targets[index],

            "sample_weight":
                self.weights[index],
        }


# ============================================================
# 4. Direction-specific collator
# ============================================================

class TranslationCollator:

    def __init__(
        self,
        tokenizer,
        target_lang,
        max_length,
    ):

        self.tokenizer = tokenizer
        self.target_lang = target_lang
        self.max_length = max_length

    def __call__(
        self,
        examples,
    ):

        self.tokenizer.tgt_lang = (
            self.target_lang
        )

        sources = [
            x["source"]
            for x in examples
        ]

        targets = [
            x["target"]
            for x in examples
        ]

        weights = torch.tensor(
            [
                x["sample_weight"]
                for x in examples
            ],
            dtype=torch.float32,
        )

        encoded = self.tokenizer(
            sources,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = (
            encoded["labels"]
        )

        labels[
            labels
            ==
            self.tokenizer.pad_token_id
        ] = -100

        encoded["labels"] = (
            labels
        )

        encoded[
            "sample_weight"
        ] = weights

        return encoded


# ============================================================
# 5. Device move
# ============================================================

def move_to_device(
    batch,
    device,
):

    return {
        key: value.to(
            device,
            non_blocking=True,
        )

        for key, value
        in batch.items()
    }


# ============================================================
# 6. Weighted seq2seq loss
# ============================================================

def weighted_seq2seq_loss(
    logits,
    labels,
    sample_weights,
):

    """
    Compute:

        token CE
            ↓
        mean loss per sample
            ↓
        sample-level weighted mean

    Human Replay:
        weight = 1.0

    Teacher HIGH:
        weight = 1.0

    Teacher MEDIUM:
        weight = 0.8
    """

    vocab_size = (
        logits.shape[-1]
    )

    flat_loss = (
        F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size,
            ),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
    )

    token_loss = (
        flat_loss.reshape(
            labels.shape
        )
    )

    valid_mask = (
        labels.ne(-100)
    )

    token_loss = (
        token_loss
        *
        valid_mask
    )

    token_counts = (
        valid_mask
        .sum(dim=1)
        .clamp_min(1)
    )

    per_sample_loss = (
        token_loss.sum(dim=1)
        /
        token_counts
    )

    sample_weights = (
        sample_weights
        .to(
            dtype=per_sample_loss.dtype
        )
    )

    weighted_numerator = (
        per_sample_loss
        *
        sample_weights
    ).sum()

    weight_sum = (
        sample_weights
        .sum()
        .clamp_min(1e-8)
    )

    weighted_loss = (
        weighted_numerator
        /
        weight_sum
    )

    return (
        weighted_loss,
        weighted_numerator.detach(),
        weight_sum.detach(),
    )


# ============================================================
# 7. Copy tokenizer files
# ============================================================

def copy_tokenizer_files(
    source_dir: Path,
    target_dir: Path,
):

    files = [
        "tokenization_small100.py",
        "vocab.json",
        "sentencepiece.bpe.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    for filename in files:

        source = (
            source_dir
            /
            filename
        )

        target = (
            target_dir
            /
            filename
        )

        if source.exists():

            shutil.copy2(
                source,
                target,
            )


# ============================================================
# 8. Save model
# ============================================================

def save_model(
    model,
    tokenizer,
    tokenizer_source_dir,
    output_dir,
):

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

    copy_tokenizer_files(
        tokenizer_source_dir,
        output_dir,
    )


# ============================================================
# 9. Validation
# ============================================================

@torch.inference_mode()
def evaluate_loss(
    model,
    loaders,
    device,
):

    model.eval()

    result = {}

    total_loss_sum = 0.0
    total_samples = 0

    for direction in [
        "en_uz",
        "uz_en",
    ]:

        loader = (
            loaders[
                direction
            ]
        )

        direction_loss_sum = 0.0
        direction_samples = 0

        for batch in loader:

            batch_size = (
                batch[
                    "input_ids"
                ].shape[0]
            )

            # Validation is the original frozen human
            # validation set.
            # sample_weight is irrelevant here.

            batch.pop(
                "sample_weight",
                None,
            )

            batch = move_to_device(
                batch,
                device,
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):

                outputs = model(
                    **batch
                )

                loss = (
                    outputs.loss
                )

            direction_loss_sum += (
                float(
                    loss.item()
                )
                *
                batch_size
            )

            direction_samples += (
                batch_size
            )

        avg_loss = (
            direction_loss_sum
            /
            max(
                direction_samples,
                1,
            )
        )

        result[
            direction
        ] = avg_loss

        total_loss_sum += (
            direction_loss_sum
        )

        total_samples += (
            direction_samples
        )

    overall_loss = (
        total_loss_sum
        /
        max(
            total_samples,
            1,
        )
    )

    result[
        "overall"
    ] = overall_loss

    return result


# ============================================================
# 10. Latest checkpoint
#
# Only ONE resume checkpoint is retained.
#
# This avoids repeating Exp1's large disk usage.
# ============================================================

def save_latest_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    tokenizer_source_dir,
    checkpoint_dir,
    epoch,
    global_step,
    best_val_loss,
    best_epoch,
    no_improve_epochs,
    history,
    baseline_validation,
):

    # Delete old latest checkpoint first.
    # Only one full optimizer checkpoint is retained.

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

    save_model(
        model,
        tokenizer,
        tokenizer_source_dir,
        model_dir,
    )

    state = {
        "epoch":
            epoch,

        "global_step":
            global_step,

        "best_val_loss":
            best_val_loss,

        "best_epoch":
            best_epoch,

        "no_improve_epochs":
            no_improve_epochs,

        "optimizer":
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

        "scaler":
            scaler.state_dict(),

        "history":
            history,

        "baseline_validation":
            baseline_validation,
    }

    torch.save(
        state,
        checkpoint_dir
        /
        "trainer_state.pt",
    )


# ============================================================
# 11. Main
# ============================================================

def main():

    args = parse_args()

    os.environ[
        "TOKENIZERS_PARALLELISM"
    ] = "false"

    set_seed(
        args.seed
    )

    # ========================================================
    # Paths
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    if args.model_path:

        model_path = Path(
            args.model_path
        )

    else:

        model_path = (
            project_root
            / "results"
            / "student"
            / "small100"
            / "exp1_finetune"
            / "best_model"
        )

    train_file = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "11a_exp2_training"
        / "exp2_train_combined_v1.parquet"
    )

    validation_file = (
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
        / "exp2_distillation_v1"
    )

    checkpoint_root = (
        output_dir
        /
        "checkpoints"
    )

    latest_checkpoint_dir = (
        checkpoint_root
        /
        "latest"
    )

    best_model_dir = (
        output_dir
        /
        "best_model"
    )

    history_file = (
        output_dir
        /
        "training_history.csv"
    )

    report_file = (
        output_dir
        /
        "training_report.json"
    )

    # ========================================================
    # Existing output protection
    # ========================================================

    resume_path = (
        Path(
            args.resume_from
        )
        if args.resume_from
        else None
    )

    if (
        output_dir.exists()
        and
        any(
            output_dir.iterdir()
        )
        and
        resume_path is None
    ):

        if args.overwrite:

            print(
                "\nRemoving existing Exp2 output:"
            )

            print(
                output_dir
            )

            shutil.rmtree(
                output_dir
            )

        else:

            raise RuntimeError(
                "\nExp2 output directory already exists:\n"
                f"{output_dir}\n\n"
                "Use --resume_from for resume, or "
                "--overwrite for a fresh run."
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 100)
    print("EN-UZ STUDENT PIPELINE")
    print(
        "STEP 11B - SMALL-100 EXP2 "
        "DISTILLATION FINE-TUNING"
    )
    print("=" * 100)

    print(
        "\nInitial model:"
    )

    print(
        model_path
    )

    print(
        "\nTraining dataset:"
    )

    print(
        train_file
    )

    print(
        "\nValidation dataset:"
    )

    print(
        validation_file
    )

    print(
        "\nEpochs:",
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
        args.batch_size
        *
        args.gradient_accumulation_steps
    )

    print(
        "Learning rate:",
        args.learning_rate
    )

    print(
        "Max length:",
        args.max_length
    )

    print(
        "Early stopping patience:",
        args.early_stopping_patience
    )

    print(
        "\nWeight policy:"
    )

    print(
        "HUMAN_REPLAY : 1.0"
    )

    print(
        "TEACHER HIGH : 1.0"
    )

    print(
        "TEACHER MEDIUM: 0.8"
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
        torch.cuda.get_device_name(0)
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    # ========================================================
    # File checks
    # ========================================================

    for path in [
        model_path,
        train_file,
        validation_file,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # Load data
    # ========================================================

    print(
        "\nLoading datasets..."
    )

    train_df = pd.read_parquet(
        train_file
    )

    val_df = pd.read_parquet(
        validation_file
    )

    required_train_cols = {
        "direction",
        "source_text",
        "target_text",
        "sample_origin",
        "sample_weight",
    }

    missing = (
        required_train_cols
        -
        set(
            train_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Training dataset missing: "
            f"{sorted(missing)}"
        )

    print(
        "Train samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(val_df)
    )

    print(
        "\nTrain origins:"
    )

    print(
        train_df[
            "sample_origin"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTrain directions:"
    )

    print(
        train_df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nOrigin x direction:"
    )

    print(
        pd.crosstab(
            train_df[
                "sample_origin"
            ],
            train_df[
                "direction"
            ],
        )
    )

    print(
        "\nValidation directions:"
    )

    print(
        val_df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Effective weight audit
    # ========================================================

    human_count = int(
        train_df[
            "sample_origin"
        ]
        .eq(
            "HUMAN_REPLAY"
        )
        .sum()
    )

    teacher_count = int(
        train_df[
            "sample_origin"
        ]
        .eq(
            "TEACHER_KD"
        )
        .sum()
    )

    teacher_high_count = int(
        (
            train_df[
                "sample_origin"
            ].eq(
                "TEACHER_KD"
            )
            &
            train_df[
                "teacher_usefulness"
            ].eq(
                "HIGH"
            )
        )
        .sum()
    )

    teacher_medium_count = int(
        (
            train_df[
                "sample_origin"
            ].eq(
                "TEACHER_KD"
            )
            &
            train_df[
                "teacher_usefulness"
            ].eq(
                "MEDIUM"
            )
        )
        .sum()
    )

    print(
        "\nHuman Replay:",
        human_count
    )

    print(
        "Teacher KD:",
        teacher_count
    )

    print(
        "Teacher HIGH:",
        teacher_high_count
    )

    print(
        "Teacher MEDIUM:",
        teacher_medium_count
    )

    # ========================================================
    # Split direction datasets
    # ========================================================

    train_en_uz = (
        train_df[
            train_df[
                "direction"
            ]
            ==
            "en_uz"
        ]
        .reset_index(
            drop=True
        )
    )

    train_uz_en = (
        train_df[
            train_df[
                "direction"
            ]
            ==
            "uz_en"
        ]
        .reset_index(
            drop=True
        )
    )

    val_en_uz = (
        val_df[
            val_df[
                "direction"
            ]
            ==
            "en_uz"
        ]
        .reset_index(
            drop=True
        )
    )

    val_uz_en = (
        val_df[
            val_df[
                "direction"
            ]
            ==
            "uz_en"
        ]
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Correct SMALL100 tokenizer
    # ========================================================

    sys.path.insert(
        0,
        str(
            model_path
        ),
    )

    from tokenization_small100 import (
        SMALL100Tokenizer
    )

    print(
        "\nLoading tokenizer..."
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

    # ========================================================
    # Dataset objects
    # ========================================================

    datasets = {

        "train_en_uz":
            TranslationDataset(
                train_en_uz
            ),

        "train_uz_en":
            TranslationDataset(
                train_uz_en
            ),

        "val_en_uz":
            TranslationDataset(
                val_en_uz.assign(
                    sample_origin=
                        "HUMAN_REPLAY",
                    sample_weight=
                        1.0,
                )
            ),

        "val_uz_en":
            TranslationDataset(
                val_uz_en.assign(
                    sample_origin=
                        "HUMAN_REPLAY",
                    sample_weight=
                        1.0,
                )
            ),
    }

    # ========================================================
    # Collators
    # ========================================================

    en_uz_collator = (
        TranslationCollator(
            tokenizer=
                tokenizer,

            target_lang=
                "uz",

            max_length=
                args.max_length,
        )
    )

    uz_en_collator = (
        TranslationCollator(
            tokenizer=
                tokenizer,

            target_lang=
                "en",

            max_length=
                args.max_length,
        )
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    generator1 = (
        torch.Generator()
    )

    generator1.manual_seed(
        args.seed
    )

    generator2 = (
        torch.Generator()
    )

    generator2.manual_seed(
        args.seed + 1
    )

    train_loaders = {

        "en_uz":
            DataLoader(
                datasets[
                    "train_en_uz"
                ],
                batch_size=
                    args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=
                    en_uz_collator,
                generator=
                    generator1,
            ),

        "uz_en":
            DataLoader(
                datasets[
                    "train_uz_en"
                ],
                batch_size=
                    args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                collate_fn=
                    uz_en_collator,
                generator=
                    generator2,
            ),
    }

    val_loaders = {

        "en_uz":
            DataLoader(
                datasets[
                    "val_en_uz"
                ],
                batch_size=
                    args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=
                    en_uz_collator,
            ),

        "uz_en":
            DataLoader(
                datasets[
                    "val_uz_en"
                ],
                batch_size=
                    args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=
                    uz_en_collator,
            ),
    }

    # ========================================================
    # Model
    # ========================================================

    if resume_path:

        load_model_path = (
            resume_path
            /
            "model"
        )

        print(
            "\nResuming model from:"
        )

        print(
            load_model_path
        )

    else:

        # Key difference from Exp1:
        #
        # Start from EXP1 BEST MODEL.

        load_model_path = (
            model_path
        )

    print(
        "\nLoading model..."
    )

    model = (
        M2M100ForConditionalGeneration
        .from_pretrained(
            str(
                load_model_path
            ),
            local_files_only=True,
        )
        .to(
            device
        )
    )

    model.train()

    model.config.use_cache = False

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
    # Optimizer
    # ========================================================

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=
                args.learning_rate,
            weight_decay=
                args.weight_decay,
        )
    )

    micro_batches_per_epoch = (
        len(
            train_loaders[
                "en_uz"
            ]
        )
        +
        len(
            train_loaders[
                "uz_en"
            ]
        )
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            micro_batches_per_epoch
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
        total_training_steps
        *
        args.warmup_ratio
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_training_steps,
        )
    )

    scaler = (
        torch.cuda.amp.GradScaler()
    )

    print(
        "\nOptimizer steps / epoch:",
        optimizer_steps_per_epoch
    )

    print(
        "Total optimizer steps:",
        total_training_steps
    )

    print(
        "Warmup steps:",
        warmup_steps
    )

    # ========================================================
    # State
    # ========================================================

    start_epoch = 0
    global_step = 0

    best_val_loss = float(
        "inf"
    )

    best_epoch = 0

    no_improve_epochs = 0

    history = []

    baseline_validation = None

    # ========================================================
    # Resume state
    # ========================================================

    if resume_path:

        state_file = (
            resume_path
            /
            "trainer_state.pt"
        )

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
            state.get(
                "best_epoch",
                0,
            )
        )

        no_improve_epochs = int(
            state.get(
                "no_improve_epochs",
                0,
            )
        )

        history = state.get(
            "history",
            [],
        )

        baseline_validation = (
            state.get(
                "baseline_validation"
            )
        )

        print(
            "\nResume epoch:",
            start_epoch + 1
        )

        print(
            "Global step:",
            global_step
        )

        print(
            "Best validation loss:",
            best_val_loss
        )

    # ========================================================
    # Initial validation
    #
    # Critical Exp2 difference:
    #
    # Exp1 best_model itself is the baseline.
    #
    # An Exp2 epoch must BEAT this loss.
    # ========================================================

    if start_epoch == 0:

        print("\n")
        print("=" * 100)
        print(
            "EXP1 BASELINE VALIDATION LOSS"
        )
        print("=" * 100)

        baseline_validation = (
            evaluate_loss(
                model,
                val_loaders,
                device,
            )
        )

        print(
            "EN -> UZ:",
            f"{baseline_validation['en_uz']:.4f}"
        )

        print(
            "UZ -> EN:",
            f"{baseline_validation['uz_en']:.4f}"
        )

        print(
            "Overall :",
            f"{baseline_validation['overall']:.4f}"
        )

        # Exp1 best model is the baseline to beat.

        best_val_loss = float(
            baseline_validation[
                "overall"
            ]
        )

        best_epoch = 0

        # Save baseline into Exp2 best_model.
        #
        # If KD makes the model worse,
        # the best model safely remains Exp1.

        print(
            "\nSaving Exp1 baseline as "
            "initial Exp2 best model..."
        )

        if best_model_dir.exists():

            shutil.rmtree(
                best_model_dir
            )

        save_model(
            model,
            tokenizer,
            model_path,
            best_model_dir,
        )

        with open(
            best_model_dir
            /
            "best_info.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                {
                    "epoch":
                        0,

                    "global_step":
                        0,

                    "val_loss":
                        best_val_loss,

                    "source":
                        "EXP1_BASELINE",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ========================================================
    # Training
    # ========================================================

    train_start = (
        time.perf_counter()
    )

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        print("\n")
        print("=" * 100)

        print(
            f"EPOCH "
            f"{epoch + 1}/"
            f"{args.epochs}"
        )

        print("=" * 100)

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        iterators = {
            direction:
                iter(loader)

            for direction, loader
            in train_loaders.items()
        }

        batch_counts = {
            direction:
                len(loader)

            for direction, loader
            in train_loaders.items()
        }

        max_batches = max(
            batch_counts.values()
        )

        micro_step = 0

        epoch_loss_numerator = 0.0
        epoch_weight_sum = 0.0

        direction_loss_numerator = {
            "en_uz": 0.0,
            "uz_en": 0.0,
        }

        direction_weight_sum = {
            "en_uz": 0.0,
            "uz_en": 0.0,
        }

        direction_samples = {
            "en_uz": 0,
            "uz_en": 0,
        }

        epoch_start = (
            time.perf_counter()
        )

        for index in range(
            max_batches
        ):

            # Same alternating strategy as Exp1.

            for direction in [
                "en_uz",
                "uz_en",
            ]:

                if (
                    index
                    >=
                    batch_counts[
                        direction
                    ]
                ):

                    continue

                batch = next(
                    iterators[
                        direction
                    ]
                )

                current_batch_size = (
                    batch[
                        "input_ids"
                    ].shape[0]
                )

                batch = move_to_device(
                    batch,
                    device,
                )

                sample_weights = (
                    batch.pop(
                        "sample_weight"
                    )
                )

                labels = (
                    batch[
                        "labels"
                    ]
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    outputs = model(
                        **batch
                    )

                    (
                        raw_loss,
                        batch_loss_numerator,
                        batch_weight_sum,
                    ) = weighted_seq2seq_loss(
                        logits=
                            outputs.logits,

                        labels=
                            labels,

                        sample_weights=
                            sample_weights,
                    )

                    loss = (
                        raw_loss
                        /
                        args
                        .gradient_accumulation_steps
                    )

                scaler.scale(
                    loss
                ).backward()

                micro_step += 1

                numerator_value = float(
                    batch_loss_numerator.item()
                )

                weight_value = float(
                    batch_weight_sum.item()
                )

                epoch_loss_numerator += (
                    numerator_value
                )

                epoch_weight_sum += (
                    weight_value
                )

                direction_loss_numerator[
                    direction
                ] += (
                    numerator_value
                )

                direction_weight_sum[
                    direction
                ] += (
                    weight_value
                )

                direction_samples[
                    direction
                ] += (
                    current_batch_size
                )

                # ================================================
                # Optimizer step
                # ================================================

                should_step = (
                    micro_step
                    %
                    args
                    .gradient_accumulation_steps
                    ==
                    0
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

                    # ============================================
                    # Progress
                    # ============================================

                    if (
                        global_step
                        %
                        100
                        ==
                        0
                    ):

                        elapsed = (
                            time.perf_counter()
                            -
                            train_start
                        )

                        avg_loss = (
                            epoch_loss_numerator
                            /
                            max(
                                epoch_weight_sum,
                                1e-8,
                            )
                        )

                        lr = (
                            scheduler
                            .get_last_lr()[0]
                        )

                        print(
                            f"epoch={epoch + 1} "
                            f"step={global_step} "
                            f"weighted_loss="
                            f"{avg_loss:.4f} "
                            f"lr={lr:.2e} "
                            f"gpu="
                            f"{torch.cuda.memory_allocated()/1024**3:.2f}GB "
                            f"time="
                            f"{elapsed/60:.1f}min"
                        )

        # ====================================================
        # Left-over gradient
        # ====================================================

        if (
            micro_step
            %
            args.gradient_accumulation_steps
            !=
            0
        ):

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

        # ====================================================
        # Training loss
        # ====================================================

        train_loss = (
            epoch_loss_numerator
            /
            max(
                epoch_weight_sum,
                1e-8,
            )
        )

        train_en_uz_loss = (
            direction_loss_numerator[
                "en_uz"
            ]
            /
            max(
                direction_weight_sum[
                    "en_uz"
                ],
                1e-8,
            )
        )

        train_uz_en_loss = (
            direction_loss_numerator[
                "uz_en"
            ]
            /
            max(
                direction_weight_sum[
                    "uz_en"
                ],
                1e-8,
            )
        )

        # ====================================================
        # Validation
        # ====================================================

        print(
            "\nRunning validation..."
        )

        val_losses = evaluate_loss(
            model,
            val_loaders,
            device,
        )

        epoch_seconds = (
            time.perf_counter()
            -
            epoch_start
        )

        epoch_record = {

            "epoch":
                epoch + 1,

            "global_step":
                global_step,

            "train_weighted_loss":
                train_loss,

            "train_en_uz_weighted_loss":
                train_en_uz_loss,

            "train_uz_en_weighted_loss":
                train_uz_en_loss,

            "val_loss":
                val_losses[
                    "overall"
                ],

            "val_en_uz_loss":
                val_losses[
                    "en_uz"
                ],

            "val_uz_en_loss":
                val_losses[
                    "uz_en"
                ],

            "learning_rate":
                scheduler
                .get_last_lr()[0],

            "epoch_seconds":
                epoch_seconds,
        }

        history.append(
            epoch_record
        )

        print("\n")

        print(
            "Train weighted :",
            f"{train_loss:.4f}"
        )

        print(
            "Train EN->UZ    :",
            f"{train_en_uz_loss:.4f}"
        )

        print(
            "Train UZ->EN    :",
            f"{train_uz_en_loss:.4f}"
        )

        print(
            "Val loss        :",
            f"{val_losses['overall']:.4f}"
        )

        print(
            "Val EN->UZ      :",
            f"{val_losses['en_uz']:.4f}"
        )

        print(
            "Val UZ->EN      :",
            f"{val_losses['uz_en']:.4f}"
        )

        print(
            "Epoch time      :",
            f"{epoch_seconds/60:.1f} min"
        )

        # ====================================================
        # Best model
        #
        # Must beat Exp1 initial baseline.
        # ====================================================

        improved = (
            val_losses[
                "overall"
            ]
            <
            best_val_loss
        )

        if improved:

            best_val_loss = float(
                val_losses[
                    "overall"
                ]
            )

            best_epoch = (
                epoch + 1
            )

            no_improve_epochs = 0

            print(
                "\nNEW BEST EXP2 MODEL"
            )

            if best_model_dir.exists():

                shutil.rmtree(
                    best_model_dir
                )

            save_model(
                model,
                tokenizer,
                model_path,
                best_model_dir,
            )

            with open(
                best_model_dir
                /
                "best_info.json",
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    {
                        "epoch":
                            best_epoch,

                        "global_step":
                            global_step,

                        "val_loss":
                            best_val_loss,

                        "source":
                            "EXP2_KD",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        else:

            no_improve_epochs += 1

            print(
                "\nNo validation improvement."
            )

            print(
                "Best remains:",
                f"{best_val_loss:.4f}"
            )

        # ====================================================
        # Latest resume checkpoint
        #
        # Only one is retained.
        # ====================================================

        print(
            "\nSaving latest resume checkpoint:"
        )

        print(
            latest_checkpoint_dir
        )

        save_latest_checkpoint(
            model=
                model,

            tokenizer=
                tokenizer,

            optimizer=
                optimizer,

            scheduler=
                scheduler,

            scaler=
                scaler,

            tokenizer_source_dir=
                model_path,

            checkpoint_dir=
                latest_checkpoint_dir,

            epoch=
                epoch,

            global_step=
                global_step,

            best_val_loss=
                best_val_loss,

            best_epoch=
                best_epoch,

            no_improve_epochs=
                no_improve_epochs,

            history=
                history,

            baseline_validation=
                baseline_validation,
        )

        # ====================================================
        # Save history
        # ====================================================

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
            no_improve_epochs
            >=
            args.early_stopping_patience
        ):

            print("\n")
            print("=" * 100)

            print(
                "EARLY STOPPING"
            )

            print(
                "No improvement for",
                no_improve_epochs,
                "epoch(s)."
            )

            print("=" * 100)

            break

    # ========================================================
    # Final report
    # ========================================================

    total_seconds = (
        time.perf_counter()
        -
        train_start
    )

    final_best_source = (
        "EXP1_BASELINE"
        if best_epoch == 0
        else
        "EXP2_KD"
    )

    report = {

        "experiment":
            "small100_en_uz_exp2_distillation_v1",

        "initial_model":
            str(
                model_path
            ),

        "initial_model_stage":
            "EXP1_BEST_MODEL",

        "training_data":
            "HUMAN_REPLAY_PLUS_CLEAN_TEACHER_KD",

        "teacher_used":
            True,

        "teacher_model":
            "google/madlad400-3b-mt",

        "judge_model":
            "Qwen3-8B",

        "distillation_used":
            True,

        "human_samples":
            human_count,

        "teacher_samples":
            teacher_count,

        "teacher_high":
            teacher_high_count,

        "teacher_medium":
            teacher_medium_count,

        "train_samples":
            len(
                train_df
            ),

        "validation_samples":
            len(
                val_df
            ),

        "weight_policy": {

            "human_replay":
                1.0,

            "teacher_high":
                1.0,

            "teacher_medium":
                0.8,

            "important_note":
                (
                    "Human replay is forced to "
                    "weight 1.0 because Exp1 did "
                    "not use sample-level weights."
                ),
        },

        "directions": {

            "en_uz_train":
                len(
                    train_en_uz
                ),

            "uz_en_train":
                len(
                    train_uz_en
                ),

            "en_uz_validation":
                len(
                    val_en_uz
                ),

            "uz_en_validation":
                len(
                    val_uz_en
                ),
        },

        "training_config": {

            "epochs_requested":
                args.epochs,

            "epochs_completed":
                len(
                    history
                ),

            "batch_size":
                args.batch_size,

            "gradient_accumulation_steps":
                args
                .gradient_accumulation_steps,

            "effective_batch_size":
                args.batch_size
                *
                args
                .gradient_accumulation_steps,

            "learning_rate":
                args.learning_rate,

            "weight_decay":
                args.weight_decay,

            "warmup_ratio":
                args.warmup_ratio,

            "max_length":
                args.max_length,

            "max_grad_norm":
                args.max_grad_norm,

            "seed":
                args.seed,

            "amp":
                "FP16",

            "early_stopping_patience":
                args
                .early_stopping_patience,
        },

        "baseline_validation":
            baseline_validation,

        "best_validation_loss":
            best_val_loss,

        "best_epoch":
            best_epoch,

        "best_model_source":
            final_best_source,

        "global_steps":
            global_step,

        "total_training_seconds":
            total_seconds,

        "history":
            history,

        "best_model":
            str(
                best_model_dir
            ),

        "latest_resume_checkpoint":
            str(
                latest_checkpoint_dir
            ),
    }

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
    # Complete
    # ========================================================

    print("\n")
    print("=" * 100)
    print("STEP 11B COMPLETE")
    print("=" * 100)

    print(
        "\nExp1 baseline validation loss:",
        f"{baseline_validation['overall']:.4f}"
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best source:",
        final_best_source
    )

    if best_epoch == 0:

        print(
            "\nWARNING:"
        )

        print(
            "KD training did not beat "
            "the Exp1 baseline validation loss."
        )

        print(
            "The saved best_model therefore "
            "remains the Exp1 baseline."
        )

    else:

        improvement = (
            baseline_validation[
                "overall"
            ]
            -
            best_val_loss
        )

        print(
            "\nValidation loss improvement:",
            f"{improvement:.4f}"
        )

    print(
        "\nBest model:"
    )

    print(
        best_model_dir
    )

    print(
        "\nTraining history:"
    )

    print(
        history_file
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    print(
        "\nLatest resume checkpoint:"
    )

    print(
        latest_checkpoint_dir
    )

    print(
        "\nNext:"
    )

    print(
        "STEP 11C / STEP 12 "
        "Exp1 vs Exp2 evaluation"
    )


if __name__ == "__main__":

    main()