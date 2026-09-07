"""Language-aware, both-side protection across all translation pairs."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from scripts.pipeline_v3.language_normalization import normalize_language_text


def text_key(language: str, text: str) -> str:
    return normalize_language_text(language, text).casefold()


def add_directed_texts(protected: dict, frame: pd.DataFrame) -> None:
    for row in frame.itertuples(index=False):
        protected[row.src_lang].add(text_key(row.src_lang, row.src_text))
        protected[row.tgt_lang].add(text_key(row.tgt_lang, row.tgt_text))


def overlap_mask(frame: pd.DataFrame, protected: dict) -> pd.Series:
    return pd.Series(
        [
            text_key(row.src_lang, row.src_text) in protected.get(row.src_lang, set())
            or text_key(row.tgt_lang, row.tgt_text)
            in protected.get(row.tgt_lang, set())
            for row in frame.itertuples(index=False)
        ],
        index=frame.index,
        dtype=bool,
    )


def protect_splits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    benchmarks: list[pd.DataFrame],
    languages: tuple[str, ...],
    previous_train: pd.DataFrame | None = None,
):
    protected = defaultdict(set)
    for frame in benchmarks:
        for lang in languages:
            if lang not in frame or frame[lang].isna().any():
                raise ValueError(f"Benchmark is missing valid {lang} text.")
            protected[lang].update(text_key(lang, text) for text in frame[lang])
    benchmark_val_overlap = overlap_mask(validation, protected)
    benchmark_train_overlap = overlap_mask(train, protected)
    previous_overlap = pd.Series(False, index=validation.index)
    if previous_train is not None:
        previous_texts = defaultdict(set)
        add_directed_texts(previous_texts, previous_train)
        previous_overlap = overlap_mask(validation, previous_texts)
    clean_validation = validation.loc[
        ~(benchmark_val_overlap | previous_overlap)
    ].copy()
    # Protect ALL original validation texts, even rows removed above.
    add_directed_texts(protected, validation)
    train_overlap = overlap_mask(train, protected)
    clean_train = train.loc[~train_overlap].copy()
    report = {
        "policy": "normalized_language_both_sides_v1",
        "train_rows_removed": int(train_overlap.sum()),
        "train_benchmark_overlap_before": int(benchmark_train_overlap.sum()),
        "validation_benchmark_rows_removed": int(benchmark_val_overlap.sum()),
        "validation_previous_train_rows_removed": int(previous_overlap.sum()),
        "validation_rows_removed": len(validation) - len(clean_validation),
        "protected_overlap_after": int(overlap_mask(clean_train, protected).sum()),
    }
    return clean_train, clean_validation, report
