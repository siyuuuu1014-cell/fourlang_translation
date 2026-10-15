"""Freeze a corpus-based, acceptance-only v3 set after lexical near-duplicate screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint  # noqa: E402
from scripts.pipeline_v3.data_safety import text_key  # noqa: E402
from scripts.pipeline_v3.flores_like_data import NearDuplicateIndex  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import LANGUAGES, _path  # noqa: E402
from scripts.pipeline_v3.preview_acceptance_v3_sources import (  # noqa: E402
    DEFAULT_CONFIG,
    discover_candidates,
    load_protected,
    stable_select,
)

DEFAULT_OUTPUT = "acceptance/fourlang_corpus_v3"
DEFAULT_REPORT = "reports/diagnostics/fourlang/acceptance_v3_freeze"
SHINGLE_SIZE = 5
NEAR_DUPLICATE_JACCARD = 0.80


def select_near_clean(
    candidates: dict[str, dict[str, dict[str, Any]]],
    protected: dict[str, set[str]],
    per_language: int,
    seed: int,
    shingle_size: int = SHINGLE_SIZE,
    threshold: float = NEAR_DUPLICATE_JACCARD,
):
    """Select stable candidates with no lexical near duplicate in protected/selected text."""
    if not 0 < threshold <= 1:
        raise ValueError("near-duplicate threshold must be in (0, 1].")
    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    available_after: dict[str, int] = {}
    for language in LANGUAGES:
        protected_index = NearDuplicateIndex(
            protected[language], shingle_size, threshold
        )
        selected_index = NearDuplicateIndex([], shingle_size, threshold)
        accepted_for_language = []
        clean_available = 0
        for row in stable_select(
            list(candidates[language].values()),
            len(candidates[language]),
            seed,
            language,
        ):
            normalized = text_key(language, row["src_text"])
            if protected_index.matches(normalized):
                rejected[f"{language}:protected_lexical_near_duplicate"] += 1
                continue
            if selected_index.matches(normalized):
                rejected[f"{language}:candidate_lexical_near_duplicate"] += 1
                continue
            clean_available += 1
            if len(accepted_for_language) < per_language:
                selected_index.add(normalized)
                accepted_for_language.append(row)
        available_after[language] = clean_available
        selected.extend(accepted_for_language)
    return selected, available_after, rejected


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_artifacts(selected: list[dict[str, Any]]):
    counters: Counter[str] = Counter()
    sources = []
    for row in sorted(selected, key=lambda item: (item["src_lang"], item["src_text"])):
        language = row["src_lang"]
        counters[language] += 1
        source_id = f"corpus-v3-{language}-{counters[language]:03d}"
        sources.append(
            {
                **row,
                "source_id": source_id,
                "scenario": "heldout_corpus",
                "training_overlap_check": "PASSED_EXACT_AND_LEXICAL_NEAR_DUPLICATE",
            }
        )
    tasks = []
    for source in sources:
        for target in LANGUAGES:
            if target == source["src_lang"]:
                continue
            tasks.append(
                {
                    **source,
                    "task_id": f"{source['source_id']}-{target}",
                    "tgt_lang": target,
                    "reference_text": None,
                    "reference_status": "NOT_PROVIDED",
                    "exp1_output": "",
                    "exp2_output": "",
                    "exp1_grade": "",
                    "exp2_grade": "",
                    "exp1_errors": [],
                    "exp2_errors": [],
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    return sources, tasks


def _publish_locked(directory: Path, files: dict[str, bytes], manifest: dict[str, Any]):
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    expected = {**files, "manifest.json": manifest_bytes}
    if directory.exists():
        existing = [path for path in directory.iterdir() if path.name != ".gitkeep"]
        if set(path.name for path in existing) != set(expected):
            raise ValueError("Frozen output already exists with different files; use a new version.")
        for name, content in expected.items():
            if (directory / name).read_bytes() != content:
                raise ValueError("Frozen output differs; use a new version instead of overwriting.")
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.with_name(directory.name + ".tmp")
    if staging.exists():
        raise ValueError(f"Stale staging directory exists: {staging}")
    staging.mkdir()
    for name, content in expected.items():
        (staging / name).write_bytes(content)
    staging.replace(directory)


def run(
    config_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    per_language: int,
    shingle_size: int,
    threshold: float,
):
    if per_language < 1 or shingle_size < 1:
        raise ValueError("per-language and shingle-size must be positive.")
    config = load_config(config_path)
    protected, protected_hashes = load_protected(config)
    candidates, discovery_rejected, candidate_files = discover_candidates(protected)
    seed = int(config["multilingual"]["seed"])
    selected, available_after, near_rejected = select_near_clean(
        candidates, protected, per_language, seed, shingle_size, threshold
    )
    selected_counts = Counter(row["src_lang"] for row in selected)
    if any(selected_counts[language] != per_language for language in LANGUAGES):
        raise RuntimeError(
            f"Insufficient near-clean sources; nothing frozen: {available_after}"
        )
    sources, tasks = build_artifacts(selected)
    source_bytes = _jsonl_bytes(sources)
    task_bytes = _jsonl_bytes(tasks)
    output = _path(output_path).resolve()
    allowed_output = (PROJECT_ROOT / DEFAULT_OUTPUT).resolve()
    if output != allowed_output:
        raise ValueError(f"Output must be {DEFAULT_OUTPUT}.")
    report_dir = _path(report_path).resolve()
    allowed_report = (PROJECT_ROOT / DEFAULT_REPORT).resolve()
    if report_dir != allowed_report:
        raise ValueError(f"Report must be {DEFAULT_REPORT}.")
    file_hashes = {
        "sources.jsonl": _sha256_bytes(source_bytes),
        "tasks.jsonl": _sha256_bytes(task_bytes),
    }
    manifest = {
        "schema_version": 1,
        "status": "FROZEN_CORPUS_DIAGNOSTIC_PENDING_NATIVE_SOURCE_REVIEW",
        "origin": "AUTHENTIC_HELDOUT_CORPUS_NOT_USER_COLLECTED",
        "sources": len(sources),
        "directions": len(LANGUAGES) * (len(LANGUAGES) - 1),
        "per_direction": per_language,
        "tasks": len(tasks),
        "files": file_hashes,
        "lexical_near_duplicate_check": {
            "method": "normalized character-shingle Jaccard",
            "shingle_size": shingle_size,
            "threshold": threshold,
            "protected_overlap_after": 0,
            "within_selection_overlap_after": 0,
        },
        "source_review": "PENDING_NATIVE_REVIEW",
        "real_user_data": False,
        "training_allowed": False,
        "model_inference_run": False,
        "human_quality_approved": False,
    }
    _publish_locked(
        output,
        {"sources.jsonl": source_bytes, "tasks.jsonl": task_bytes},
        manifest,
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "status": "ACCEPTANCE_V3_FROZEN_FOR_DIAGNOSTIC_INFERENCE",
        "output": str(output.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "sources": len(sources),
        "tasks": len(tasks),
        "selected_by_language": dict(sorted(selected_counts.items())),
        "available_after_lexical_near_duplicate_screening": available_after,
        "lexical_near_duplicate_method": manifest["lexical_near_duplicate_check"],
        "discovery_rejected": dict(sorted(discovery_rejected.items())),
        "near_duplicate_rejected": dict(sorted(near_rejected.items())),
        "candidate_file_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in candidate_files
        },
        "protected_file_sha256": protected_hashes,
        "frozen_file_sha256": file_hashes,
        "source_review": "PENDING_NATIVE_REVIEW",
        "real_user_data": False,
        "training_started": False,
        "training_data_written": False,
        "next_step": "Run Exp1 and Exp2 diagnostic inference on the frozen v3 tasks.",
        "fingerprint": fingerprint(
            {
                "config": config,
                "candidate_files": [str(path) for path in candidate_files],
                "protected": protected_hashes,
                "manifest": manifest,
            }
        ),
    }
    write_json(report_dir / "summary.json", report)
    print(
        f"{report['status']}: sources={len(sources)} tasks={len(tasks)}; "
        "no model or training data changed.",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--per-language", type=int, default=100)
    parser.add_argument("--shingle-size", type=int, default=SHINGLE_SIZE)
    parser.add_argument(
        "--near-duplicate-jaccard", type=float, default=NEAR_DUPLICATE_JACCARD
    )
    args = parser.parse_args()
    run(
        args.config,
        args.output,
        args.report,
        args.per_language,
        args.shingle_size,
        args.near_duplicate_jaccard,
    )


if __name__ == "__main__":
    main()
