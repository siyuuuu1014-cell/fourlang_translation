"""Build an isolated FLORES-like KD source set and a weighted ZH-UZ mix.

This module never trains a model and never writes to the existing v4 dataset.
It profiles FLORES dev, selects independent monolingual sources, prepares the
existing Teacher/Judge pipeline input, and assembles an auditable 40/30/30 mix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import load_config, write_json  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import _read_table, normalize_rows  # noqa: E402
from scripts.pipeline_v3.language_normalization import normalize_language_text  # noqa: E402
from scripts.supplemental.monolingual_v3 import (  # noqa: E402
    _iter_records,
    _source_texts,
    quality_reason,
)

PROJECT_ROOT = PROJECT_ROOT_BOOTSTRAP
LANGUAGES = ("zh", "uz")
DIRECTIONS = ("zh-uz", "uz-zh")


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _read_table(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                default=lambda value: value.item()
                if hasattr(value, "item")
                else str(value),
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_key(language: str, text: Any) -> str:
    return normalize_language_text(language, str(text)).casefold()


def _shingles(text: str, size: int) -> set[str]:
    compact = re.sub(r"\s+", " ", text.casefold()).strip()
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


class NearDuplicateIndex:
    def __init__(self, texts: Iterable[str], size: int, threshold: float):
        self.size = size
        self.threshold = threshold
        self.sets: list[set[str]] = []
        self.lookup: dict[str, list[int]] = defaultdict(list)
        for text in texts:
            self.add(text)

    def add(self, text: str) -> None:
        shingles = _shingles(text, self.size)
        index = len(self.sets)
        self.sets.append(shingles)
        for shingle in shingles:
            self.lookup[shingle].append(index)

    def matches(self, text: str) -> bool:
        shingles = _shingles(text, self.size)
        if not shingles:
            return False
        candidates: Counter[int] = Counter()
        for shingle in shingles:
            candidates.update(self.lookup.get(shingle, ()))
        for index, intersection in candidates.items():
            union = len(shingles) + len(self.sets[index]) - intersection
            if union and intersection / union >= self.threshold:
                return True
        return False


def _benchmark_texts(frame: pd.DataFrame, language: str) -> list[str]:
    if language not in frame.columns:
        raise ValueError(f"Benchmark is missing the {language!r} column.")
    return [_text_key(language, value) for value in frame[language].fillna("")]


def _length_boundaries(texts: list[str]) -> tuple[int, int]:
    lengths = pd.Series([len(text) for text in texts], dtype="int64")
    return max(1, int(lengths.quantile(0.33))), max(2, int(lengths.quantile(0.67)))


def _feature(text: str, boundaries: tuple[int, int]) -> str:
    low, high = boundaries
    length = "short" if len(text) <= low else "medium" if len(text) <= high else "long"
    sentence = "question" if re.search(r"[?？]", text) else "statement"
    digit = "digit" if any(char.isdigit() for char in text) else "no_digit"
    clauses = "multi" if len(re.findall(r"[,，;；:：]", text)) >= 2 else "simple"
    return "|".join((length, sentence, digit, clauses))


def _quotas(keys: list[str], total: int) -> dict[str, int]:
    counts = Counter(keys)
    raw = {key: total * count / len(keys) for key, count in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(raw, key=lambda key: (raw[key] - result[key], key), reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


def _source_group(value: Any) -> str:
    name = str(value).lower()
    for group in ("hplt", "fineweb", "wikipedia", "v2_bronze"):
        if group in name:
            return group
    return "other"


def _stable_rank(seed: int, identity: str) -> str:
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def _extension_id(language: str, text: str) -> str:
    return hashlib.sha256(
        f"zh_uz_flores_like_wikipedia_v1\n{language}\n{_text_key(language, text)}".encode()
    ).hexdigest()


def validate(config: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if config.get("direction", {}).get("pair") != "zh_uz":
        errors.append("direction.pair must be zh_uz")
    mix = config.get("mixture", {})
    shares = [float(mix.get(key, -1)) for key in ("human", "existing_kd", "flores_like_kd")]
    if any(value <= 0 for value in shares) or not math.isclose(sum(shares), 1.0):
        errors.append("mixture human/existing_kd/flores_like_kd must be positive and sum to 1")
    if not math.isclose(float(mix.get("direction_share", 0)), 0.5):
        errors.append("direction_share must be 0.5 for a balanced bidirectional model")
    threshold = float(config.get("selection", {}).get("near_duplicate_jaccard", 0))
    if not 0.0 < threshold <= 1.0:
        errors.append("selection.near_duplicate_jaccard must be in (0, 1]")
    caps = config.get("selection", {}).get("source_caps", {})
    if not caps or any(not 0.0 < float(value) <= 1.0 for value in caps.values()):
        errors.append("selection.source_caps values must be in (0, 1]")
    elif sum(float(value) for value in caps.values()) < 1.0:
        errors.append("selection.source_caps must provide at least 100% total capacity")
    extension = config.get("extension", {})
    extension_sources = extension.get("sources", [])
    if not extension_sources:
        errors.append("extension.sources must contain the ZH and UZ Wikipedia sources")
    elif {str(item.get("language")) for item in extension_sources} != set(LANGUAGES):
        errors.append("extension.sources must contain exactly the zh and uz languages")
    output = _path(config["outputs"]["root"])
    if output.resolve() == _path(config["inputs"]["existing_train"]).parent.resolve():
        errors.append("output root must not overwrite the existing training version")
    report = {"schema_version": 1, "status": "PASS" if not errors else "FAIL", "errors": errors}
    report_path = _path(config["outputs"]["report_root"]) / "validation.json"
    write_json(report_path, report)
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def profile(config: dict[str, Any]) -> dict[str, Any]:
    dev_path = _path(config["inputs"]["flores_dev"])
    dev = _read(dev_path)
    result: dict[str, Any] = {"schema_version": 1, "source": str(dev_path), "rows": len(dev), "languages": {}}
    for language in LANGUAGES:
        texts = _benchmark_texts(dev, language)
        boundaries = _length_boundaries(texts)
        features = [_feature(text, boundaries) for text in texts]
        result["languages"][language] = {
            "length_boundaries_characters": {"short_max": boundaries[0], "medium_max": boundaries[1]},
            "feature_distribution": dict(sorted(Counter(features).items())),
        }
    result["flores_used_as_training_data"] = False
    result["sha256"] = _sha256(dev_path)
    write_json(_path(config["outputs"]["report_root"]) / "flores_profile.json", result)
    return result


def collect_extension(config: dict[str, Any]) -> dict[str, Any]:
    """Collect new, resumable Wikipedia candidates without touching v3 data."""
    validate(config)
    extension = config["extension"]
    filter_config = load_config(extension["filter_config"])
    quality_settings = filter_config["monolingual"]
    extension_paths = [
        _path(value) for value in config["inputs"].get("candidate_extensions", [])
    ]
    if len(extension_paths) != 1:
        raise ValueError("Exactly one inputs.candidate_extensions output is required.")
    output = extension_paths[0]
    state_path = output.with_suffix(".state.json")
    existing_rows = _read(output).to_dict("records") if output.is_file() else []
    if output.is_file() and not state_path.is_file():
        raise RuntimeError(f"Extension data exists without resume state: {state_path}")

    base_pool = _read(_path(config["inputs"]["candidate_pool"]))
    seen = {language: set() for language in LANGUAGES}
    for row in base_pool.itertuples(index=False):
        language = str(row.src_lang)
        if language in seen:
            seen[language].add(_text_key(language, row.src_text))
    excluded = _excluded_source_keys(config)
    for language in LANGUAGES:
        seen[language].update(excluded[language])
    for row in existing_rows:
        seen[str(row["src_lang"])].add(
            _text_key(str(row["src_lang"]), row["src_text"])
        )

    signature_payload = {
        "schema_version": 1,
        "extension": extension,
        "quality_settings": quality_settings,
        "candidate_pool_sha256": _sha256(_path(config["inputs"]["candidate_pool"])),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "signature": signature, "sources": {}}
    )
    if state.get("signature") != signature:
        raise RuntimeError(
            "Wikipedia extension configuration or base pool changed after collection "
            f"started. Preserve {output} and use a new output version intentionally."
        )

    rejection_counts: Counter[str] = Counter()
    checkpoint_rows = int(extension.get("checkpoint_rows", 250))

    def save() -> None:
        _write_parquet(output, existing_rows)
        write_json(state_path, state)

    for source in extension["sources"]:
        source_id = str(source["id"])
        language = str(source["language"])
        target_language = "uz" if language == "zh" else "zh"
        target = int(source["target_rows"])
        source_state = state["sources"].setdefault(
            source_id,
            {"status": "running", "documents_seen": 0, "accepted": 0},
        )
        accepted = sum(
            1 for row in existing_rows if str(row.get("source_corpus")) == source_id
        )
        source_state["accepted"] = accepted
        if accepted >= target:
            source_state["status"] = "completed"
            continue
        documents_seen = int(source_state.get("documents_seen", 0))
        max_documents = int(source.get("max_documents", 100000))
        print(
            f"Collecting {source_id}: resume_documents={documents_seen} "
            f"accepted={accepted}/{target}",
            flush=True,
        )
        try:
            for index, record in enumerate(_iter_records(source)):
                if index < documents_seen:
                    continue
                source_state["documents_seen"] = index + 1
                if index + 1 > max_documents:
                    break
                for raw_text in _source_texts(source, record):
                    reason, text = quality_reason(
                        language, raw_text, quality_settings
                    )
                    key = _text_key(language, text)
                    if reason is None and key in seen[language]:
                        reason = "DUPLICATE_OR_PREVIOUS_SOURCE"
                    if reason is not None:
                        rejection_counts[f"{source_id}:{reason}"] += 1
                        continue
                    existing_rows.append(
                        {
                            "pair_id": _extension_id(language, text),
                            "src_lang": language,
                            "tgt_lang": target_language,
                            "src_text": text,
                            "reference_text": "",
                            "source_corpus": source_id,
                            "source_license": str(source["license"]),
                            "source_record": str(record.get("id", index)),
                        }
                    )
                    seen[language].add(key)
                    accepted += 1
                    source_state["accepted"] = accepted
                    if accepted % checkpoint_rows == 0:
                        save()
                        print(
                            f"Wikipedia checkpoint: {source_id} {accepted}/{target}",
                            flush=True,
                        )
                    if accepted >= target:
                        break
                if accepted >= target:
                    break
        except Exception as error:
            source_state["status"] = "failed"
            source_state["error"] = f"{type(error).__name__}: {error}"
            save()
            raise RuntimeError(
                f"{source_id} collection failed; rerun the same command to resume."
            ) from error
        source_state["status"] = "completed" if accepted >= target else "short"
        source_state.pop("error", None)
        save()

    counts = Counter(str(row["src_lang"]) for row in existing_rows)
    targets = {
        str(source["language"]): int(source["target_rows"])
        for source in extension["sources"]
    }
    shortages = {
        language: target - counts[language]
        for language, target in targets.items()
        if counts[language] < target
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if not shortages else "SHORT",
        "output": str(output),
        "rows": len(existing_rows),
        "by_language": dict(sorted(counts.items())),
        "targets": targets,
        "shortages": shortages,
        "source_state": state["sources"],
        "rejections": dict(sorted(rejection_counts.items())),
        "output_sha256": _sha256(output),
        "base_candidate_pool_modified": False,
    }
    write_json(
        _path(config["outputs"]["report_root"]) / "wikipedia_extension.json",
        report,
    )
    if shortages and bool(extension.get("require_full_targets", True)):
        raise RuntimeError(f"Wikipedia extension targets were not met: {shortages}")
    return report


def _excluded_source_keys(config: dict[str, Any]) -> dict[str, set[str]]:
    excluded = {language: set() for language in LANGUAGES}
    for key in ("previous_selected", "existing_train", "validation"):
        frame = _read(_path(config["inputs"][key]))
        if {"src_lang", "src_text"}.issubset(frame.columns):
            for row in frame.itertuples(index=False):
                language = str(row.src_lang)
                if language in excluded:
                    excluded[language].add(_text_key(language, row.src_text))
                target_language = str(getattr(row, "tgt_lang", ""))
                target_text = getattr(row, "tgt_text", None)
                if target_language in excluded and target_text is not None:
                    excluded[target_language].add(
                        _text_key(target_language, target_text)
                    )
        elif {"source_text", "target_text"}.issubset(frame.columns):
            excluded["zh"].update(
                _text_key("zh", text) for text in frame["source_text"]
            )
            excluded["uz"].update(
                _text_key("uz", text) for text in frame["target_text"]
            )
        else:
            raise ValueError(f"{key} has no recognizable text columns")
    return excluded


def select(config: dict[str, Any], *, stage_only: bool = False) -> dict[str, Any]:
    validate(config)
    profile_report = profile(config)
    settings = config["selection"]
    configured_target = int(
        settings[
            "source_review_candidates_per_direction"
            if stage_only
            else "candidate_sources_per_direction"
        ]
    )
    seed = int(config["direction"]["seed"])
    teacher_root = _path(config["outputs"]["teacher_pipeline_root"])
    pool_path = (
        _path(config["inputs"]["candidate_pool"])
        if stage_only
        else teacher_root / "monolingual_candidates.jsonl"
    )
    pool_paths = [pool_path]
    if stage_only:
        pool_paths.extend(
            _path(value)
            for value in config["inputs"].get("candidate_extensions", [])
        )
    pool = pd.concat([_read(path) for path in pool_paths], ignore_index=True)
    required = {"pair_id", "src_lang", "tgt_lang", "src_text", "source_corpus"}
    if missing := required - set(pool.columns):
        raise ValueError(f"Candidate pool is missing columns: {sorted(missing)}")
    pool = pool[pool["src_lang"].isin(LANGUAGES)].drop_duplicates("pair_id", keep="first")

    if not stage_only and bool(settings.get("require_source_qwen_pass", True)):
        review = _read(teacher_root / "source_judged.parquet")
        review = review.drop_duplicates("pair_id", keep="last")
        passed = review[
            review["judge_parse_ok"].fillna(False).astype(bool)
            & review["judge_label"].fillna("").astype(str).str.upper().eq("PASS")
        ]["pair_id"].astype(str)
        pool = pool[pool["pair_id"].astype(str).isin(set(passed))]

    excluded = _excluded_source_keys(config)
    dev = _read(_path(config["inputs"]["flores_dev"]))
    devtest = _read(_path(config["inputs"]["flores_devtest"]))
    shingle_size = int(settings["shingle_size"])
    threshold = float(settings["near_duplicate_jaccard"])
    caps = {str(key): float(value) for key, value in settings["source_caps"].items()}
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": 2,
        "stage": "source_review_staging" if stage_only else "final_source_selection",
        "candidate_pool": [str(path) for path in pool_paths],
        "by_direction": {},
        "flores_rows_selected": 0,
    }
    shortages: dict[str, int] = {}

    for language in LANGUAGES:
        target_language = "uz" if language == "zh" else "zh"
        boundaries_payload = profile_report["languages"][language]["length_boundaries_characters"]
        boundaries = (int(boundaries_payload["short_max"]), int(boundaries_payload["medium_max"]))
        dev_texts = _benchmark_texts(dev, language)
        protected_texts = dev_texts + _benchmark_texts(devtest, language)
        near = NearDuplicateIndex(protected_texts, shingle_size, threshold)
        rows = []
        rejected = Counter()
        for row in pool[pool["src_lang"] == language].to_dict("records"):
            text = _text_key(language, row["src_text"])
            if text in excluded[language]:
                rejected["previous_or_validation_exact"] += 1
                continue
            if near.matches(text):
                rejected["flores_near_duplicate"] += 1
                continue
            group = _source_group(row["source_corpus"])
            rows.append({**row, "src_text": normalize_language_text(language, row["src_text"]), "source_group": group, "flores_feature": _feature(text, boundaries)})
        rows.sort(key=lambda row: _stable_rank(seed, str(row["pair_id"])))
        target = len(rows) if stage_only and configured_target == 0 else configured_target
        target = min(target, len(rows)) if stage_only else target
        quotas = _quotas([_feature(text, boundaries) for text in dev_texts], target)
        group_limits = {
            group: (target if stage_only else math.floor(target * cap))
            for group, cap in caps.items()
        }
        group_counts: Counter[str] = Counter()
        feature_counts: Counter[str] = Counter()
        chosen_ids: set[str] = set()

        def take(row: dict[str, Any], honor_feature: bool) -> bool:
            group, feature = row["source_group"], row["flores_feature"]
            if group_counts[group] >= group_limits.get(group, group_limits.get("other", 0)):
                return False
            if honor_feature and feature_counts[feature] >= quotas.get(feature, 0):
                return False
            chosen_ids.add(str(row["pair_id"]))
            group_counts[group] += 1
            feature_counts[feature] += 1
            selected.append(row)
            return True

        for row in rows:
            if len(chosen_ids) >= target:
                break
            take(row, True)
        if len(chosen_ids) < target:
            for row in rows:
                if len(chosen_ids) >= target:
                    break
                if str(row["pair_id"]) not in chosen_ids:
                    take(row, False)
        direction = f"{language}-{target_language}"
        report["by_direction"][direction] = {
            "eligible": len(rows), "selected": len(chosen_ids), "target": target,
            "by_source_group": dict(sorted(group_counts.items())),
            "by_feature": dict(sorted(feature_counts.items())), "rejections": dict(sorted(rejected.items())),
        }
        if len(chosen_ids) < target:
            shortages[direction] = target - len(chosen_ids)

    selected.sort(key=lambda row: (str(row["src_lang"]), str(row["pair_id"])))
    output_root = _path(config["outputs"]["root"])
    teacher_rows = [
        {key: row.get(key, "") for key in ("pair_id", "src_lang", "tgt_lang", "src_text", "reference_text", "source_corpus", "source_license", "source_record")}
        for row in selected
    ]
    if stage_only:
        staged_path = teacher_root / "monolingual_candidates.jsonl"
        _write_jsonl(staged_path, teacher_rows)
        report["source_review_input"] = str(staged_path)
    elif not shortages:
        _write_jsonl(output_root / "selected_sources.jsonl", selected)
        _write_jsonl(teacher_root / "kd_candidates.jsonl", teacher_rows)
        report["output"] = str(output_root / "selected_sources.jsonl")
        report["teacher_input"] = str(teacher_root / "kd_candidates.jsonl")
    report["inputs_sha256"] = {
        key: _sha256(_path(config["inputs"][key]))
        for key in ("candidate_pool", "source_review", "previous_selected", "existing_train", "validation", "flores_dev", "flores_devtest")
    }
    report["shortages"] = shortages
    report_name = "source_staging.json" if stage_only else "source_selection.json"
    write_json(_path(config["outputs"]["report_root"]) / report_name, report)
    if shortages:
        detail = ", ".join(
            f"{direction} missing {count}" for direction, count in shortages.items()
        )
        raise RuntimeError(
            "Insufficient Qwen-PASS sources under the configured source caps: "
            f"{detail}. See {report_name}."
        )
    return report


def stage(config: dict[str, Any]) -> dict[str, Any]:
    return select(config, stage_only=True)


def source_calibration_report(config: dict[str, Any]) -> dict[str, Any]:
    """Estimate final source capacity before paying for a complete Qwen audit."""
    validate(config)
    settings = config["selection"]
    target = int(settings["candidate_sources_per_direction"])
    caps = {str(key): float(value) for key, value in settings["source_caps"].items()}
    teacher_root = _path(config["outputs"]["teacher_pipeline_root"])
    candidates = _read(teacher_root / "monolingual_candidates.jsonl").copy()
    calibration_path = teacher_root / "source_judge_calibration.parquet"
    calibration = _read(calibration_path).drop_duplicates("pair_id", keep="last")
    calibration["source_group"] = calibration["source_corpus"].map(_source_group)
    calibration["passed"] = (
        calibration["judge_parse_ok"].fillna(False).astype(bool)
        & calibration["judge_label"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("PASS")
    )
    candidates["source_group"] = candidates["source_corpus"].map(_source_group)
    details: dict[str, Any] = {}
    directions: dict[str, Any] = {}
    for language in LANGUAGES:
        direction = f"{language}-{'uz' if language == 'zh' else 'zh'}"
        expected_capacity = 0.0
        conservative_capacity = 0.0
        for group, available in (
            candidates[candidates["src_lang"] == language]
            .groupby("source_group")
            .size()
            .items()
        ):
            sample = calibration[
                (calibration["src_lang"] == language)
                & (calibration["source_group"] == group)
            ]
            reviewed = len(sample)
            passed = int(sample["passed"].sum())
            rate = passed / reviewed if reviewed else 0.0
            standard_error = (
                math.sqrt(rate * (1.0 - rate) / reviewed) if reviewed else 0.0
            )
            conservative_rate = max(0.0, rate - 1.96 * standard_error)
            cap = target * caps.get(str(group), caps.get("other", 0.0))
            expected = min(float(available) * rate, cap)
            conservative = min(float(available) * conservative_rate, cap)
            expected_capacity += expected
            conservative_capacity += conservative
            details[f"{direction}|{group}"] = {
                "available": int(available),
                "calibration_rows": reviewed,
                "pass_rows": passed,
                "pass_rate": rate,
                "conservative_pass_rate_95": conservative_rate,
                "configured_cap_rows": int(cap),
                "expected_usable_rows": int(expected),
                "conservative_usable_rows": int(conservative),
            }
        decision = (
            "READY"
            if conservative_capacity >= target
            else "LIKELY_READY"
            if expected_capacity >= target
            else "INSUFFICIENT_PROJECTED_CAPACITY"
        )
        directions[direction] = {
            "target": target,
            "expected_usable_rows": int(expected_capacity),
            "conservative_usable_rows": int(conservative_capacity),
            "decision": decision,
        }
    payload = {
        "schema_version": 1,
        "calibration": str(calibration_path),
        "calibration_rows": len(calibration),
        "directions": directions,
        "groups": details,
        "full_source_audit_recommended": all(
            item["decision"] in {"READY", "LIKELY_READY"}
            for item in directions.values()
        ),
    }
    write_json(
        _path(config["outputs"]["report_root"]) / "source_calibration_capacity.json",
        payload,
    )
    return payload


def _teacher_rows(config: dict[str, Any]) -> pd.DataFrame:
    teacher_path = _path(config["outputs"]["teacher_pipeline_root"]) / "teacher_judged.parquet"
    frame = _read(teacher_path)
    frame = frame[
        frame["judge_parse_ok"].fillna(False).astype(bool)
        & frame["judge_label"].fillna("").astype(str).str.upper().eq("PASS")
        & frame["teacher_usefulness"].fillna("").astype(str).str.upper().isin(("HIGH", "MEDIUM"))
    ].copy()
    frame["tgt_text"] = frame["teacher_text"]
    frame["weight"] = frame["teacher_usefulness"].astype(str).str.upper().map(
        {"HIGH": float(config["distillation"]["teacher_high_weight"]), "MEDIUM": float(config["distillation"]["teacher_medium_weight"])}
    )
    frame["training_source"] = "teacher_kd_flores_like_v1"
    return normalize_rows(frame, origin=str(teacher_path))


def _group(value: Any) -> str:
    text = str(value).lower()
    if "flores_like" in text:
        return "flores_like_kd"
    if "teacher" in text or "pseudo" in text or "synthetic" in text:
        return "existing_kd"
    return "human"


def assemble(config: dict[str, Any]) -> dict[str, Any]:
    validate(config)
    existing_path = _path(config["inputs"]["existing_train"])
    validation_path = _path(config["inputs"]["validation"])
    existing = normalize_rows(_read(existing_path), origin=str(existing_path))
    new = _teacher_rows(config)
    new = new.drop_duplicates(["src_lang", "tgt_lang", "src_text", "tgt_text"])
    settings = config["selection"]
    dev = _read(_path(config["inputs"]["flores_dev"]))
    devtest = _read(_path(config["inputs"]["flores_devtest"]))
    protected_exact = {
        language: set(_benchmark_texts(dev, language) + _benchmark_texts(devtest, language))
        for language in LANGUAGES
    }
    protected_near = {
        language: NearDuplicateIndex(
            protected_exact[language],
            int(settings["shingle_size"]),
            float(settings["near_duplicate_jaccard"]),
        )
        for language in LANGUAGES
    }

    for row in existing.itertuples(index=False):
        for language, text in (
            (row.src_lang, row.src_text),
            (row.tgt_lang, row.tgt_text),
        ):
            normalized = _text_key(language, text)
            if (
                normalized in protected_exact[language]
                or protected_near[language].matches(normalized)
            ):
                raise RuntimeError(
                    "Existing v4 data overlaps a protected FLORES split; "
                    "refusing to claim a leak-free derived dataset."
                )
    safe_mask = []
    for row in new.itertuples(index=False):
        source = _text_key(row.src_lang, row.src_text)
        target_text = _text_key(row.tgt_lang, row.tgt_text)
        safe_mask.append(
            source not in protected_exact[row.src_lang]
            and target_text not in protected_exact[row.tgt_lang]
            and not protected_near[row.src_lang].matches(source)
            and not protected_near[row.tgt_lang].matches(target_text)
        )
    new = new.loc[safe_mask].copy()
    old_keys = set(zip(existing["src_lang"], existing["tgt_lang"], existing["src_text"].str.casefold(), strict=True))
    new = new.loc[[
        (row.src_lang, row.tgt_lang, row.src_text.casefold()) not in old_keys
        for row in new.itertuples(index=False)
    ]].copy()
    frame = pd.concat([existing, new], ignore_index=True)
    frame["mix_group"] = frame["training_source"].map(_group)
    frame["direction"] = frame["src_lang"] + "-" + frame["tgt_lang"]
    frame = frame[frame["direction"].isin(DIRECTIONS)].copy()
    minimum = int(config["selection"]["minimum_teacher_rows_per_direction"])
    new_counts = Counter(new["src_lang"] + "-" + new["tgt_lang"])
    shortages = {direction: minimum - new_counts[direction] for direction in DIRECTIONS if new_counts[direction] < minimum}
    if shortages:
        raise RuntimeError(f"Accepted FLORES-like Teacher rows are below minimum: {shortages}")

    mix = config["mixture"]
    target_shares = {key: float(mix[key]) * float(mix["direction_share"]) for key in ("human", "existing_kd", "flores_like_kd")}
    total_scale = float(len(frame))
    before: dict[str, float] = {}
    multipliers: dict[str, float] = {}
    for direction in DIRECTIONS:
        for group, share in target_shares.items():
            mask = frame["direction"].eq(direction) & frame["mix_group"].eq(group)
            mass = float(frame.loc[mask, "weight"].sum())
            key = f"{direction}|{group}"
            if mass <= 0:
                raise RuntimeError(f"Missing required mixture group: {key}")
            multiplier = total_scale * share / mass
            before[key] = mass
            multipliers[key] = multiplier
            frame.loc[mask, "weight"] = frame.loc[mask, "weight"].astype(float) * multiplier
    frame = frame.sample(frac=1, random_state=int(config["direction"]["seed"])).reset_index(drop=True)
    output_root = _path(config["outputs"]["root"])
    train_path = output_root / "train.jsonl"
    validation_output = output_root / "validation.jsonl"
    columns = ["src_lang", "tgt_lang", "src_text", "tgt_text", "weight", "training_source", "origin"]
    _write_jsonl(train_path, frame[columns].to_dict("records"))
    validation = normalize_rows(_read(validation_path), origin=str(validation_path))
    _write_jsonl(validation_output, validation.to_dict("records"))
    after = {
        f"{direction}|{group}": float(frame.loc[frame["direction"].eq(direction) & frame["mix_group"].eq(group), "weight"].sum())
        for direction in DIRECTIONS for group in target_shares
    }
    report = {
        "schema_version": 1, "status": "FLORES_LIKE_V1_BUILT_NOT_TRAINED", "rows": len(frame),
        "new_teacher_rows": dict(sorted(new_counts.items())), "raw_mass": before,
        "weight_multipliers": multipliers, "effective_mass": after,
        "target_global_shares": {f"{direction}|{group}": share for direction in DIRECTIONS for group, share in target_shares.items()},
        "train": {"path": str(train_path), "sha256": _sha256(train_path)},
        "validation": {"path": str(validation_output), "sha256": _sha256(validation_output)},
        "flores_rows_in_training": 0, "existing_v4_modified": False,
    }
    write_json(_path(config["outputs"]["report_root"]) / "assembly.json", report)
    write_json(output_root / "manifest.json", report)
    return report


def status(config: dict[str, Any]) -> dict[str, Any]:
    output_root = _path(config["outputs"]["root"])
    teacher_root = _path(config["outputs"]["teacher_pipeline_root"])
    files = {
        "profile": _path(config["outputs"]["report_root"]) / "flores_profile.json",
        "wikipedia_extension": _path(config["inputs"]["candidate_extensions"][0]),
        "source_review_input": teacher_root / "monolingual_candidates.jsonl",
        "source_review": teacher_root / "source_judged.parquet",
        "selected_sources": output_root / "selected_sources.jsonl",
        "teacher_input": teacher_root / "kd_candidates.jsonl",
        "teacher_generated": teacher_root / "teacher_generated.parquet",
        "teacher_judged": teacher_root / "teacher_judged.parquet",
        "train": output_root / "train.jsonl",
        "manifest": output_root / "manifest.json",
    }
    result = {"schema_version": 1, "files": {key: path.is_file() for key, path in files.items()}}
    if files["manifest"].is_file():
        result["dataset"] = json.loads(files["manifest"].read_text(encoding="utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "profile",
            "extend",
            "stage",
            "source_calibration",
            "select",
            "assemble",
            "status",
        ),
    )
    parser.add_argument("--config", default="configs/directions/zh_uz_flores_like_v1.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    payload = {
        "validate": validate,
        "profile": profile,
        "extend": collect_extension,
        "stage": stage,
        "source_calibration": source_calibration_report,
        "select": select,
        "assemble": assemble,
        "status": status,
    }[args.action](config)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
