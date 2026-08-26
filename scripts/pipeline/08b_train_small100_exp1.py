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
        default="/root/autodl-tmp/models/small100",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
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
        "--resume_from",
        type=str,
        default=None,
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

    def __len__(self):

        return len(self.sources)

    def __getitem__(self, index):

        return {
            "source":
                self.sources[index],

            "target":
                self.targets[index],
        }


# ============================================================
# 4. Direction-specific collator
#
# 每个 DataLoader 只负责一个 target language。
#
# 这样不会出现 tokenizer.tgt_lang
# 在同一个 batch 中冲突。
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

    def __call__(self, examples):

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

        encoded = self.tokenizer(
            sources,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # padding token 不参与 loss
        labels = encoded["labels"]

        labels[
            labels
            ==
            self.tokenizer.pad_token_id
        ] = -100

        encoded["labels"] = labels

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
# 6. Copy custom tokenizer files
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
# 7. Save model
# ============================================================

def save_model(
    model,
    tokenizer,
    original_model_path,
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
        original_model_path,
        output_dir,
    )


# ============================================================
# 8. Validation
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

        loader = loaders[
            direction
        ]

        direction_loss_sum = 0.0
        direction_samples = 0

        for batch in loader:

            batch_size = (
                batch[
                    "input_ids"
                ].shape[0]
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

                loss = outputs.loss

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
# 9. Save epoch checkpoint
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    original_model_path,
    checkpoint_dir,
    epoch,
    global_step,
    best_val_loss,
    history,
):

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
        original_model_path,
        model_dir,
    )

    state = {
        "epoch":
            epoch,

        "global_step":
            global_step,

        "best_val_loss":
            best_val_loss,

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
        "trainer_state.pt",
    )


# ============================================================
# 10. Main
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

    model_path = Path(
        args.model_path
    )

    train_file = (
        project_root
        / "data"
        / "splits"
        / "en_uz"
        / "v1"
        / "train_exp1_bidirectional_v1.parquet"
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
        / "exp1_finetune"
    )

    checkpoint_root = (
        output_dir
        /
        "checkpoints"
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
    print("STEP 08B - SMALL-100 EXP1 FINE-TUNING")
    print("=" * 100)

    print(
        "Model:",
        model_path
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
        torch.cuda.get_device_name(
            0
        )
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
    # Files
    # ========================================================

    for path in [
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

    print(
        "Train samples:",
        len(train_df)
    )

    print(
        "Validation samples:",
        len(val_df)
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
    # Split direction datasets
    # ========================================================

    train_en_uz = train_df[
        train_df[
            "direction"
        ]
        ==
        "en_uz"
    ].reset_index(
        drop=True
    )

    train_uz_en = train_df[
        train_df[
            "direction"
        ]
        ==
        "uz_en"
    ].reset_index(
        drop=True
    )

    val_en_uz = val_df[
        val_df[
            "direction"
        ]
        ==
        "en_uz"
    ].reset_index(
        drop=True
    )

    val_uz_en = val_df[
        val_df[
            "direction"
        ]
        ==
        "uz_en"
    ].reset_index(
        drop=True
    )

    # ========================================================
    # Correct tokenizer
    # ========================================================

    sys.path.insert(
        0,
        str(model_path),
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
    # Datasets
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
                val_en_uz
            ),

        "val_uz_en":
            TranslationDataset(
                val_uz_en
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

    generator1 = torch.Generator()
    generator1.manual_seed(
        args.seed
    )

    generator2 = torch.Generator()
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
    #
    # FP32 master weights + AMP FP16
    # 比直接把参数全部转 FP16 更稳定。
    # ========================================================

    resume_path = (
        Path(
            args.resume_from
        )
        if args.resume_from
        else None
    )

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=
            args.weight_decay,
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
    # Resume optimizer state
    # ========================================================

    start_epoch = 0
    global_step = 0
    best_val_loss = float(
        "inf"
    )
    history = []

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

        history = state.get(
            "history",
            [],
        )

        print(
            "\nResume epoch:",
            start_epoch + 1
        )

        print(
            "Global step:",
            global_step
        )

    # ========================================================
    # Initial validation
    # ========================================================

    if start_epoch == 0:

        print("\n")
        print("=" * 100)
        print("INITIAL VALIDATION LOSS")
        print("=" * 100)

        initial_val = evaluate_loss(
            model,
            val_loaders,
            device,
        )

        print(
            "EN -> UZ:",
            f"{initial_val['en_uz']:.4f}"
        )

        print(
            "UZ -> EN:",
            f"{initial_val['uz_en']:.4f}"
        )

        print(
            "Overall :",
            f"{initial_val['overall']:.4f}"
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
            f"EPOCH {epoch + 1}/{args.epochs}"
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

        epoch_loss_sum = 0.0
        epoch_samples = 0

        direction_loss = {
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

            # ------------------------------------------------
            # 固定交替：
            #
            # EN→UZ
            # UZ→EN
            #
            # 每个 optimizer update 尽量同时看到两个方向。
            # ------------------------------------------------

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

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    outputs = model(
                        **batch
                    )

                    raw_loss = (
                        outputs.loss
                    )

                    loss = (
                        raw_loss
                        /
                        args.gradient_accumulation_steps
                    )

                scaler.scale(
                    loss
                ).backward()

                micro_step += 1

                epoch_loss_sum += (
                    float(
                        raw_loss.item()
                    )
                    *
                    current_batch_size
                )

                epoch_samples += (
                    current_batch_size
                )

                direction_loss[
                    direction
                ] += (
                    float(
                        raw_loss.item()
                    )
                    *
                    current_batch_size
                )

                direction_samples[
                    direction
                ] += (
                    current_batch_size
                )

                # ============================================
                # Optimizer step
                # ============================================

                should_step = (
                    micro_step
                    %
                    args.gradient_accumulation_steps
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

                    # ========================================
                    # Progress
                    # ========================================

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
                            epoch_loss_sum
                            /
                            max(
                                epoch_samples,
                                1,
                            )
                        )

                        lr = (
                            scheduler
                            .get_last_lr()[0]
                        )

                        print(
                            f"epoch={epoch + 1} "
                            f"step={global_step} "
                            f"loss={avg_loss:.4f} "
                            f"lr={lr:.2e} "
                            f"gpu={torch.cuda.memory_allocated()/1024**3:.2f}GB "
                            f"time={elapsed/60:.1f}min"
                        )

        # ====================================================
        # Left-over accumulated gradient
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
        # Epoch training loss
        # ====================================================

        train_loss = (
            epoch_loss_sum
            /
            max(
                epoch_samples,
                1,
            )
        )

        train_en_uz_loss = (
            direction_loss[
                "en_uz"
            ]
            /
            direction_samples[
                "en_uz"
            ]
        )

        train_uz_en_loss = (
            direction_loss[
                "uz_en"
            ]
            /
            direction_samples[
                "uz_en"
            ]
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

            "train_loss":
                train_loss,

            "train_en_uz_loss":
                train_en_uz_loss,

            "train_uz_en_loss":
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
            "Train loss   :",
            f"{train_loss:.4f}"
        )

        print(
            "Train EN->UZ :",
            f"{train_en_uz_loss:.4f}"
        )

        print(
            "Train UZ->EN :",
            f"{train_uz_en_loss:.4f}"
        )

        print(
            "Val loss     :",
            f"{val_losses['overall']:.4f}"
        )

        print(
            "Val EN->UZ   :",
            f"{val_losses['en_uz']:.4f}"
        )

        print(
            "Val UZ->EN   :",
            f"{val_losses['uz_en']:.4f}"
        )

        print(
            "Epoch time   :",
            f"{epoch_seconds / 60:.1f} min"
        )

        # ====================================================
        # Best model
        # ====================================================

        if (
            val_losses[
                "overall"
            ]
            <
            best_val_loss
        ):

            best_val_loss = (
                val_losses[
                    "overall"
                ]
            )

            print(
                "\nNEW BEST MODEL"
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
                            epoch + 1,

                        "global_step":
                            global_step,

                        "val_loss":
                            best_val_loss,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        # ====================================================
        # Epoch checkpoint
        # ====================================================

        checkpoint_dir = (
            checkpoint_root
            /
            f"epoch_{epoch + 1:02d}"
        )

        print(
            "\nSaving checkpoint:"
        )

        print(
            checkpoint_dir
        )

        save_checkpoint(
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

            original_model_path=
                model_path,

            checkpoint_dir=
                checkpoint_dir,

            epoch=
                epoch,

            global_step=
                global_step,

            best_val_loss=
                best_val_loss,

            history=
                history,
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

    # ========================================================
    # Final report
    # ========================================================

    total_seconds = (
        time.perf_counter()
        -
        train_start
    )

    report = {

        "experiment":
            "small100_en_uz_exp1",

        "base_model":
            "alirezamsh/small100",

        "training_data":
            "GOLD + SILVER",

        "bronze_used":
            False,

        "teacher_used":
            False,

        "distillation_used":
            False,

        "train_samples":
            len(
                train_df
            ),

        "validation_samples":
            len(
                val_df
            ),

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

            "epochs":
                args.epochs,

            "batch_size":
                args.batch_size,

            "gradient_accumulation_steps":
                args.gradient_accumulation_steps,

            "learning_rate":
                args.learning_rate,

            "weight_decay":
                args.weight_decay,

            "warmup_ratio":
                args.warmup_ratio,

            "max_length":
                args.max_length,

            "seed":
                args.seed,

            "amp":
                "FP16",
        },

        "best_validation_loss":
            best_val_loss,

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
    print("STEP 08B COMPLETE")
    print("=" * 100)

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Best model:"
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
        "\nNext: STEP 08C "
        "Fine-tuned Student Evaluation"
    )


if __name__ == "__main__":

    main()