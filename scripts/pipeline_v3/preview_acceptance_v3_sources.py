"""Preview 100 held-out corpus sources per language for acceptance v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint  # noqa: E402
from scripts.pipeline_v3.data_safety import text_key  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import (  # noqa: E402
    LANGUAGES,
    _path,
    _read_table,
    normalize_rows,
)

DEFAULT_CONFIG = "configs/multilingual/fourlang.toml"
DEFAULT_OUTPUT = "reports/diagnostics/fourlang/acceptance_v3_source_preview"
ALLOWED_SUFFIXES = {".jsonl", ".json", ".parquet", ".csv"}


def candidate_path_allowed(path: Path) -> bool:
    value = path.as_posix().lower()
    parts = set(path.parts)
    return (
        path.suffix.lower() in ALLOWED_SUFFIXES
        and "test" in path.stem.lower()
        and ("splits" in parts or "clean" in parts)
        and not any(
            marker in value
            for marker in (
                "/benchmark/",
                "/synthetic/",
                "/distillation/",
                "/multilingual/",
            )
        )
    )


def stable_select(rows: list[dict[str, Any]], size: int, seed: int, language: str):
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\n{language}\n{row['src_text']}".encode("utf-8")
        ).hexdigest(),
    )
    return ranked[:size]


def protected_paths(config: dict[str, Any]) -> list[Path]:
    paths = []
    for item in config["pair_data"]:
        for field in ("train", "validation", "kd_train", "exp3_kd_train"):
            if field in item:
                paths.append(_path(item[field]))
    for experiment in ("exp1", "exp2"):
        for name in ("train.jsonl", "validation.jsonl"):
            paths.append(
                PROJECT_ROOT / "data/multilingual/fourlang" / experiment / name
            )
    for field in ("flores_dev", "flores_devtest"):
        paths.append(_path(config["benchmarks"][field]))
    return list(dict.fromkeys(path.resolve() for path in paths))


def load_protected(config: dict[str, Any]):
    values = defaultdict(set)
    hashes = {}
    for path in protected_paths(config):
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = file_sha256(
            path
        )
        raw = _read_table(path)
        benchmark_languages = [language for language in LANGUAGES if language in raw]
        if len(benchmark_languages) == len(LANGUAGES):
            for language in benchmark_languages:
                values[language].update(
                    text_key(language, text)
                    for text in raw[language].fillna("").astype(str)
                    if text.strip()
                )
        else:
            frame = normalize_rows(raw, origin=str(path))
            for row in frame.itertuples(index=False):
                values[row.src_lang].add(text_key(row.src_lang, row.src_text))
                values[row.tgt_lang].add(text_key(row.tgt_lang, row.tgt_text))
    old_sources = PROJECT_ROOT / "acceptance/fourlang_pilot_v1/sources.jsonl"
    if old_sources.is_file():
        hashes[str(old_sources.relative_to(PROJECT_ROOT)).replace("\\", "/")] = (
            file_sha256(old_sources)
        )
        for line in old_sources.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                values[row["src_lang"]].add(
                    text_key(row["src_lang"], row["src_text"])
                )
    return values, hashes


def discover_candidates(protected):
    by_language = defaultdict(dict)
    rejected = Counter()
    files = sorted(
        path
        for path in (PROJECT_ROOT / "data").rglob("*")
        if path.is_file() and candidate_path_allowed(path)
    )
    for path in files:
        try:
            frame = normalize_rows(_read_table(path), origin=str(path))
        except (ValueError, TypeError, KeyError):
            rejected["unsupported_schema"] += 1
            continue
        untrusted = frame["training_source"].astype(str).str.lower().str.contains(
            "teacher|pseudo|synthetic", regex=True
        )
        frame = frame.loc[~untrusted]
        origin = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for row in frame.itertuples(index=False):
            for language, value in (
                (row.src_lang, row.src_text),
                (row.tgt_lang, row.tgt_text),
            ):
                if language not in LANGUAGES:
                    continue
                key = text_key(language, value)
                if key in protected[language]:
                    rejected["protected_exact_overlap"] += 1
                    continue
                if not 8 <= len(value) <= 400:
                    rejected["length"] += 1
                    continue
                if key not in by_language[language]:
                    by_language[language][key] = {
                        "src_lang": language,
                        "src_text": value,
                        "origin": origin,
                        "origin_kind": "AUTHENTIC_HELDOUT_CORPUS_NOT_USER_COLLECTED",
                        "usage": "ACCEPTANCE_ONLY_NOT_FOR_TRAINING",
                        "source_review_status": "PENDING_NATIVE_REVIEW",
                    }
    return by_language, rejected, files


def run(config_path: str | Path, output_path: str | Path, per_language: int):
    if per_language < 1:
        raise ValueError("per-language must be positive.")
    config = load_config(config_path)
    protected, protected_hashes = load_protected(config)
    candidates, rejected, files = discover_candidates(protected)
    seed = int(config["multilingual"]["seed"])
    selected = []
    available = {}
    for language in LANGUAGES:
        pool = list(candidates[language].values())
        available[language] = len(pool)
        selected.extend(stable_select(pool, per_language, seed, language))
    selected_counts = Counter(row["src_lang"] for row in selected)
    selected_origins = Counter(
        (row["src_lang"], row["origin"]) for row in selected
    )
    ready = all(selected_counts[language] == per_language for language in LANGUAGES)
    output = _path(output_path).resolve()
    allowed = (PROJECT_ROOT / DEFAULT_OUTPUT).resolve()
    if output != allowed:
        raise ValueError(f"Output must be {DEFAULT_OUTPUT}.")
    output.mkdir(parents=True, exist_ok=True)
    preview_path = output / "candidate_sources_preview.jsonl"
    preview_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "status": (
            "ACCEPTANCE_V3_SOURCE_PREVIEW_READY_NOT_FROZEN"
            if ready
            else "ACCEPTANCE_V3_SOURCE_PREVIEW_BLOCKED_SHORTAGE"
        ),
        "per_language_target": per_language,
        "planned_directions": len(LANGUAGES) * (len(LANGUAGES) - 1),
        "planned_tasks": per_language * len(LANGUAGES) * (len(LANGUAGES) - 1),
        "available_unique_after_exact_protection": available,
        "selected_by_language": dict(sorted(selected_counts.items())),
        "candidate_files": [
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in files
        ],
        "candidate_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in files
        },
        "protected_file_sha256": protected_hashes,
        "rejected": dict(sorted(rejected.items())),
        "selected_origin_counts": {
            f"{language}:{origin}": count
            for (language, origin), count in sorted(selected_origins.items())
        },
        "exact_protected_overlap_after": sum(
            text_key(row["src_lang"], row["src_text"])
            in protected[row["src_lang"]]
            for row in selected
        ),
        "near_duplicate_check": "NOT_PERFORMED",
        "native_source_review": "PENDING",
        "real_user_data": False,
        "training_started": False,
        "training_data_written": False,
        "acceptance_set_frozen": False,
        "preview_sha256": file_sha256(preview_path),
        "config_fingerprint": fingerprint(
            {
                "seed": seed,
                "per_language": per_language,
                "candidate_files": [str(path) for path in files],
                "protected": protected_hashes,
            }
        ),
        "next_step": (
            "Run semantic near-duplicate screening, then freeze a new acceptance v3."
            if ready
            else "Provide more untouched human-corpus test sources before freezing v3."
        ),
    }
    write_json(output / "summary.json", report)
    print(
        f"{report['status']}: selected={len(selected)} planned_tasks="
        f"{report['planned_tasks']}; no model or training data changed.",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--per-language", type=int, default=100)
    args = parser.parse_args()
    run(args.config, args.output, args.per_language)


if __name__ == "__main__":
    main()
