from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


REQUIRED_COLUMNS = {"src_lang", "tgt_lang", "src_text", "tgt_text"}


def load_jsonl(path: Path, *, limit: int | None = None, seed: int = 42) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    dataset = load_dataset("json", data_files=str(path), split="train")
    missing = REQUIRED_COLUMNS.difference(dataset.column_names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    if limit is not None and limit < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(limit))
    return dataset


def validate_languages(dataset: Dataset, supported_languages: Iterable[str]) -> None:
    supported = set(supported_languages)
    seen_sources = set(dataset.unique("src_lang"))
    seen_targets = set(dataset.unique("tgt_lang"))
    unknown = (seen_sources | seen_targets).difference(supported)
    if unknown:
        raise ValueError(f"Unsupported language codes in dataset: {sorted(unknown)}")


def tokenize_translation_dataset(
    dataset: Dataset,
    tokenizer: Any,
    *,
    max_source_length: int,
    max_target_length: int,
) -> Dataset:
    def preprocess(example: dict[str, Any]) -> dict[str, list[int]]:
        tokenizer.src_lang = example["src_lang"]
        model_inputs = tokenizer(
            example["src_text"],
            max_length=max_source_length,
            truncation=True,
        )

        # M2M100 prefixes the current source-language token. Encoding the target
        # with tgt_lang as src_lang makes every label start with its language ID.
        tokenizer.src_lang = example["tgt_lang"]
        labels = tokenizer(
            example["tgt_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        preprocess,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
