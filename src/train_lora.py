from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from config_utils import load_config, positive_limit, project_path
from data_utils import load_jsonl, tokenize_translation_dataset, validate_languages
from model_utils import load_base_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-memory M2M100 LoRA training")
    parser.add_argument("--config", required=True, help="Path to a TOML config")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume from the latest checkpoint, or from the supplied path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["training"].get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cpu_threads = int(cfg["runtime"].get("cpu_threads", 4))
    torch.set_num_threads(cpu_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    output_dir = project_path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_jsonl(
        project_path(cfg["paths"]["train_file"]),
        limit=positive_limit(cfg["data"].get("max_train_samples")),
        seed=seed,
    )
    valid_dataset = load_jsonl(
        project_path(cfg["paths"]["validation_file"]),
        limit=positive_limit(cfg["data"].get("max_validation_samples")),
        seed=seed,
    )

    supported_languages = cfg["data"].get("supported_languages", ["zh", "en", "ru", "uz"])
    validate_languages(train_dataset, supported_languages)
    validate_languages(valid_dataset, supported_languages)

    model_name = cfg["model"]["base_model"]
    architecture = str(cfg["model"].get("architecture", "m2m100"))
    tokenizer = load_tokenizer(model_name, architecture)
    model = load_base_model(model_name, architecture)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.print_trainable_parameters()

    train_tokens = tokenize_translation_dataset(
        train_dataset,
        tokenizer,
        max_source_length=int(cfg["data"]["max_source_length"]),
        max_target_length=int(cfg["data"]["max_target_length"]),
        architecture=architecture,
    )
    valid_tokens = tokenize_translation_dataset(
        valid_dataset,
        tokenizer,
        max_source_length=int(cfg["data"]["max_source_length"]),
        max_target_length=int(cfg["data"]["max_target_length"]),
        architecture=architecture,
    )

    del train_dataset, valid_dataset

    use_cuda = torch.cuda.is_available()
    requested_bf16 = bool(cfg["runtime"].get("bf16", True))
    use_bf16 = bool(use_cuda and requested_bf16 and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16 and cfg["runtime"].get("fp16_fallback", True))

    train_cfg = cfg["training"]
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        max_steps=int(train_cfg.get("max_steps", -1)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(train_cfg["train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=int(train_cfg.get("eval_steps", 100)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 100)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        dataloader_num_workers=int(cfg["runtime"].get("dataloader_num_workers", 0)),
        dataloader_pin_memory=bool(cfg["runtime"].get("pin_memory", False)),
        predict_with_generate=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
        remove_unused_columns=True,
        save_safetensors=True,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if use_cuda else None,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokens,
        eval_dataset=valid_tokens,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    run_metadata = {
        "config": cfg["_config_path"],
        "base_model": model_name,
        "architecture": architecture,
        "train_samples": len(train_tokens),
        "validation_samples": len(valid_tokens),
        "cuda": use_cuda,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(output_dir / "final_adapter")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
