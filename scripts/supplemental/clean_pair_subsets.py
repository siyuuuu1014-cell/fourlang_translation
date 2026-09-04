from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_v2.data_flow import normalized_key, pair_hash  # noqa: E402
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)


NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
MARKUP_RE = re.compile(r"<[^>]+>|https?://\S+|\{\{.*?\}\}")
REPEATED_PUNCTUATION_RE = re.compile(r"([!?.,])\1{2,}")


@dataclass(frozen=True)
class PairSpec:
    pair: str
    source_lang: str
    target_lang: str
    instructions: dict[str, tuple[str, str]]


PAIR_SPECS = {
    "en-ru.jsonl": PairSpec(
        pair="en_ru",
        source_lang="en",
        target_lang="ru",
        instructions={
            "Переведите следующий текст с английского на русский язык:": (
                "en",
                "ru",
            ),
            "Translate the following text from Russian into English:": (
                "ru",
                "en",
            ),
        },
    ),
    "ru-uz.jsonl": PairSpec(
        pair="uz_ru",
        source_lang="uz",
        target_lang="ru",
        instructions={
            "Quyidagi ruscha matnni o'zbek tiliga (lotin yozuvida) tarjima qiling:": (
                "ru",
                "uz",
            ),
            "Переведите следующий текст с узбекского на русский язык:": (
                "uz",
                "ru",
            ),
        },
    ),
    "ru-zh.jsonl": PairSpec(
        pair="zh_ru",
        source_lang="zh",
        target_lang="ru",
        instructions={
            "请将下面的俄文翻译成简体中文：": ("ru", "zh"),
            "Переведите следующий текст с китайского на русский язык:": (
                "zh",
                "ru",
            ),
        },
    ),
    "uz-zh.jsonl": PairSpec(
        pair="zh_uz",
        source_lang="zh",
        target_lang="uz",
        instructions={
            "请将下面的乌兹别克文翻译成简体中文：": ("uz", "zh"),
            "Quyidagi xitoycha matnni o'zbek tiliga (lotin yozuvida) tarjima qiling:": (
                "zh",
                "uz",
            ),
        },
    ),
}


def _has_expected_script(language: str, text: str) -> bool:
    if language == "zh":
        return bool(re.search(r"[\u3400-\u9fff]", text))
    if language == "uz":
        return bool(re.search(r"[A-Za-z]", text)) and not bool(
            re.search(r"[\u0400-\u052f]", text)
        )
    if language == "ru":
        return bool(re.search(r"[\u0400-\u052f]", text))
    if language == "en":
        return bool(re.search(r"[A-Za-z]", text))
    return any(character.isalpha() for character in text)


def _canonical_pair(
    spec: PairSpec,
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
) -> tuple[str, str]:
    if (source_lang, target_lang) == (spec.source_lang, spec.target_lang):
        return source_text, target_text
    if (source_lang, target_lang) == (spec.target_lang, spec.source_lang):
        return target_text, source_text
    raise ValueError(
        f"Direction {source_lang}-{target_lang} does not belong to {spec.pair}."
    )


def deterministic_rejection_reason(
    source_lang: str,
    target_lang: str,
    source_text: str,
    target_text: str,
    *,
    max_characters: int = 1000,
    max_length_ratio: float = 6.0,
) -> str | None:
    if not source_text or not target_text:
        return "EMPTY"
    if min(len(source_text), len(target_text)) < 2:
        return "TOO_SHORT"
    if max(len(source_text), len(target_text)) > max_characters:
        return "TOO_LONG"
    if MARKUP_RE.search(source_text + " " + target_text):
        return "MARKUP_OR_URL"
    ratio = max(len(source_text), len(target_text)) / max(
        1, min(len(source_text), len(target_text))
    )
    if ratio > max_length_ratio:
        return "CRITICAL_LENGTH_RATIO"
    if normalized_key(source_text) == normalized_key(target_text):
        return "SOURCE_TARGET_SAME"
    if min(
        sum(character.isalpha() for character in source_text),
        sum(character.isalpha() for character in target_text),
    ) < 2:
        return "LOW_LETTER_COUNT"
    if NUMBER_RE.findall(source_text) != NUMBER_RE.findall(target_text):
        return "NUMBER_MISMATCH"
    if not _has_expected_script(source_lang, source_text):
        return "SOURCE_SCRIPT_RISK"
    if not _has_expected_script(target_lang, target_text):
        return "TARGET_SCRIPT_RISK"
    if REPEATED_PUNCTUATION_RE.search(source_text + target_text):
        return "REPEATED_PUNCTUATION"
    return None


def load_protected_sets(
    benchmark_root: Path, spec: PairSpec
) -> tuple[dict[str, set[str]], list[str]]:
    protected = {
        "pairs": set(),
        spec.source_lang: set(),
        spec.target_lang: set(),
    }
    files: list[str] = []
    for path in sorted((benchmark_root / spec.pair).glob("*.parquet")):
        frame = pd.read_parquet(path)
        if not {spec.source_lang, spec.target_lang}.issubset(frame.columns):
            continue
        files.append(str(path))
        for left, right in frame[
            [spec.source_lang, spec.target_lang]
        ].itertuples(index=False, name=None):
            left_text = normalize_language_text(spec.source_lang, str(left))
            right_text = normalize_language_text(spec.target_lang, str(right))
            protected["pairs"].add(pair_hash(left_text, right_text))
            protected[spec.source_lang].add(normalized_key(left_text))
            protected[spec.target_lang].add(normalized_key(right_text))
    return protected, files


def clean_file(
    source_path: Path,
    output_root: Path,
    benchmark_root: Path,
    *,
    rejected_sample_size: int = 100,
) -> dict[str, Any]:
    spec = PAIR_SPECS[source_path.name]
    protected, benchmark_files = load_protected_sets(benchmark_root, spec)
    rejection_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    conversion_counts: Counter[str] = Counter()
    rejected_samples: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    input_rows = 0

    with source_path.open("r", encoding="utf-8") as handle:
        for source_row, line in enumerate(handle, start=1):
            input_rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                reason = "INVALID_JSON"
                rejection_counts[reason] += 1
                if len(rejected_samples) < rejected_sample_size:
                    rejected_samples.append(
                        {"source_row": source_row, "reason": reason, "error": str(exc)}
                    )
                continue

            if not all(key in payload for key in ("instruction", "input", "output")):
                reason = "MISSING_REQUIRED_FIELD"
                rejection_counts[reason] += 1
                continue
            instruction = str(payload["instruction"]).strip()
            direction = spec.instructions.get(instruction)
            if direction is None:
                reason = "UNKNOWN_INSTRUCTION"
                rejection_counts[reason] += 1
                if len(rejected_samples) < rejected_sample_size:
                    rejected_samples.append(
                        {
                            "source_row": source_row,
                            "reason": reason,
                            "instruction": instruction,
                        }
                    )
                continue

            source_lang, target_lang = direction
            direction_counts[f"{source_lang}-{target_lang}"] += 1
            source_raw = str(payload["input"]).strip()
            target_raw = str(payload["output"]).strip()
            try:
                source_text = normalize_language_text(source_lang, source_raw)
                target_text = normalize_language_text(target_lang, target_raw)
            except ValueError as exc:
                reason = "NORMALIZATION_ERROR"
                rejection_counts[reason] += 1
                if len(rejected_samples) < rejected_sample_size:
                    rejected_samples.append(
                        {"source_row": source_row, "reason": reason, "error": str(exc)}
                    )
                continue

            conversion_counts[f"{source_lang}_cells"] += 1
            conversion_counts[f"{target_lang}_cells"] += 1
            conversion_counts[f"{source_lang}_changed"] += source_text != source_raw
            conversion_counts[f"{target_lang}_changed"] += target_text != target_raw
            reason = deterministic_rejection_reason(
                source_lang,
                target_lang,
                source_text,
                target_text,
            )
            canonical_source, canonical_target = _canonical_pair(
                spec,
                source_lang,
                target_lang,
                source_text,
                target_text,
            )
            current_pair_id = pair_hash(canonical_source, canonical_target)
            if reason is None and benchmark_files:
                if (
                    current_pair_id in protected["pairs"]
                    or normalized_key(canonical_source)
                    in protected[spec.source_lang]
                    or normalized_key(canonical_target)
                    in protected[spec.target_lang]
                ):
                    reason = "PROTECTED_BENCHMARK_OVERLAP"
            if reason is not None:
                rejection_counts[reason] += 1
                if len(rejected_samples) < rejected_sample_size:
                    rejected_samples.append(
                        {
                            "source_row": source_row,
                            "reason": reason,
                            "src_lang": source_lang,
                            "tgt_lang": target_lang,
                            "src_text": source_text,
                            "tgt_text": target_text,
                        }
                    )
                continue

            candidates.append(
                {
                    "pair_id": current_pair_id,
                    "pair": spec.pair,
                    "source_lang": spec.source_lang,
                    "target_lang": spec.target_lang,
                    "source_text": canonical_source,
                    "target_text": canonical_target,
                    "observed_direction": f"{source_lang}-{target_lang}",
                    "source_corpus": "translate_languages_pair_subsets",
                    "source_file": source_path.name,
                    "source_row": source_row,
                    "candidate_status": "CLEANED_UNREVIEWED",
                }
            )

    frame = pd.DataFrame(candidates)
    before_exact_dedup = len(frame)
    if frame.empty:
        cleaned = frame
        exact_duplicate_rows = 0
        ambiguous_rows = 0
    else:
        frame = frame.drop_duplicates("pair_id", keep="first").copy()
        exact_duplicate_rows = before_exact_dedup - len(frame)
        source_counts = frame.groupby(
            frame["source_text"].map(normalized_key)
        )["pair_id"].transform("count")
        target_counts = frame.groupby(
            frame["target_text"].map(normalized_key)
        )["pair_id"].transform("count")
        ambiguous = (source_counts != 1) | (target_counts != 1)
        ambiguous_rows = int(ambiguous.sum())
        cleaned = frame[~ambiguous].copy().sort_values("pair_id")
        rejection_counts["EXACT_DUPLICATE"] += exact_duplicate_rows
        rejection_counts["AMBIGUOUS_ONE_TO_MANY"] += ambiguous_rows

    destination = output_root / spec.pair
    destination.mkdir(parents=True, exist_ok=True)
    cleaned_path = destination / "cleaned_candidates.parquet"
    temporary = cleaned_path.with_suffix(".parquet.tmp")
    cleaned.to_parquet(temporary, index=False)
    temporary.replace(cleaned_path)
    rejected_path = destination / "rejected_sample.jsonl"
    rejected_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rejected_samples
        ),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "pair": spec.pair,
        "status": "CLEANED_UNREVIEWED",
        "source_file": str(source_path),
        "input_rows": input_rows,
        "direction_rows": dict(direction_counts),
        "cleaned_rows": len(cleaned),
        "rejections": dict(rejection_counts),
        "normalization": dict(conversion_counts),
        "benchmark_files": benchmark_files,
        "protected_overlap_checked": bool(benchmark_files),
        "requires_protected_recheck": not bool(benchmark_files),
        "existing_training_overlap_checked": False,
        "eligible_for_training": False,
        "notes": [
            "This dataset is isolated from active pipelines.",
            "Qwen review and cross-dataset deduplication are still required.",
            "It must never be used for validation, testing, or model selection.",
        ],
        "outputs": {
            "cleaned_candidates": str(cleaned_path),
            "rejected_sample": str(rejected_path),
        },
    }
    report_path = destination / "cleaning_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean external pair_subsets into isolated, unreviewed candidates."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/supplemental_staging/pair_subsets/v1",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_ROOT / "data/benchmark",
    )
    parser.add_argument("--rejected-sample-size", type=int, default=100)
    args = parser.parse_args()
    if args.rejected_sample_size < 0:
        raise ValueError("--rejected-sample-size cannot be negative.")

    reports = []
    for file_name in PAIR_SPECS:
        source_path = args.source_dir / file_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        report = clean_file(
            source_path,
            args.output_root,
            args.benchmark_root,
            rejected_sample_size=args.rejected_sample_size,
        )
        reports.append(report)
        print(
            f"{file_name}: {report['input_rows']} -> {report['cleaned_rows']} "
            f"cleaned rows",
            flush=True,
        )

    summary_path = args.output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "CLEANED_UNREVIEWED",
                "pairs": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
