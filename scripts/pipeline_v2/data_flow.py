from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

try:
    from .common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        pipeline_namespace,
        project_path,
        write_json,
    )
except ImportError:
    from common import (
        PROJECT_ROOT,
        commercial_candidates,
        load_config,
        pair_info,
        pipeline_namespace,
        project_path,
        write_json,
    )

from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)

QUALITY_WEIGHTS = {"GOLD": 1.0, "SILVER": 0.85, "BRONZE": 0.65}
FIRST_LABELS = {"PASS", "MINOR", "FAIL", "UNCERTAIN"}


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def normalized_key(value: str) -> str:
    return normalize(value).casefold()


def pair_hash(source: str, target: str) -> str:
    return hashlib.sha256(
        f"{normalized_key(source)}\n{normalized_key(target)}".encode()
    ).hexdigest()


def paths(config: dict[str, Any]) -> dict[str, Path]:
    pair, _, _, version = pair_info(config)
    namespace = pipeline_namespace(config)
    pipeline = PROJECT_ROOT / "data" / "pipeline_v2" / namespace
    reports = PROJECT_ROOT / "reports" / "pipeline" / namespace
    return {
        "pipeline": pipeline,
        "reports": reports,
        "candidates": pipeline / "candidates.parquet",
        "routed": pipeline / "rule_routed.parquet",
        "human_review": pipeline / "human_review_input.parquet",
        "human_judged": pipeline / "human_judged.parquet",
        "human_second": pipeline / "human_second_review.parquet",
        "teacher_calibration": pipeline / "teacher_judge_calibration.parquet",
        "teacher_judged": pipeline / "teacher_judged.parquet",
        "approved": PROJECT_ROOT
        / "data"
        / "approved"
        / pair
        / version
        / "approved_pairs.parquet",
        "splits": PROJECT_ROOT / "data" / "splits" / pair / version,
        "distillation": PROJECT_ROOT / "data" / "distillation" / pair / version,
    }


def benchmark_sets(
    config: dict[str, Any], group: str | None = None
) -> dict[str, set[str]]:
    _, source, target, _ = pair_info(config)
    result = {"pairs": set(), "source": set(), "target": set()}
    groups = (group,) if group else ("selection", "final")
    values = [
        value
        for group_name in groups
        for value in config["benchmarks"][group_name].values()
    ]
    for value in values:
        frame = pd.read_parquet(project_path(value))
        left = source if source in frame.columns else "source_text"
        right = target if target in frame.columns else "target_text"
        for source_text, target_text in frame[[left, right]].itertuples(
            index=False, name=None
        ):
            result["pairs"].add(pair_hash(source_text, target_text))
            result["source"].add(normalized_key(source_text))
            result["target"].add(normalized_key(target_text))
    return result


def validate(config: dict[str, Any]) -> None:
    pair, source, target, _ = pair_info(config)
    errors: list[str] = []
    if source == target or pair != f"{source}_{target}":
        errors.append("direction.pair must equal <source_lang>_<target_lang>.")
    scope = str(config.get("pipeline", {}).get("scope", ""))
    data_only = scope in {"pair_data_only", "supplemental_pair_data"}
    roles = ("teacher",) if data_only else ("student", "teacher")
    contract = config.get("text_contract", {})
    if "zh" in {source, target} and (
        contract.get("zh_script") != "simplified"
        or not contract.get("convert_traditional_to_simplified", False)
    ):
        errors.append("ZH data must be converted to Simplified Chinese.")
    if "uz" in {source, target} and (
        contract.get("uz_script") != "latin"
        or not contract.get("transliterate_cyrillic_to_latin", False)
    ):
        errors.append("UZ data must be transliterated to Latin script.")
    if scope == "pair_data_only":
        corpora = config.get("data", {}).get("corpora", [])
        if not corpora:
            errors.append("pair_data_only requires at least one pinned corpus.")
        identities = []
        archives = []
        for corpus in corpora:
            if {
                str(corpus.get("archive_source_lang", "")),
                str(corpus.get("archive_target_lang", "")),
            } != {source, target}:
                errors.append(
                    f"Corpus {corpus.get('name', '?')} does not match {pair}."
                )
            identities.append((corpus.get("name"), corpus.get("version")))
            archives.append(corpus.get("local_archive"))
        if len(identities) != len(set(identities)):
            errors.append("Pinned corpus name/version pairs must be unique.")
        if len(archives) != len(set(archives)):
            errors.append("Pinned corpus archive paths must be unique.")
    for role in roles:
        candidates = commercial_candidates(config, role)
        ids = [item["id"] for item in candidates]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate {role} candidate ids.")
        for item in candidates:
            required_fields = (
                ("license", "family")
                if item.get("family") == "marian_pair"
                else ("license", "revision", "family")
            )
            for required in required_fields:
                if not item.get(required):
                    errors.append(
                        f"{role} candidate {item['id']} is missing {required}."
                    )
            if item.get("family") == "marian_pair" and not all(
                item.get(f"{a}_{b}_revision")
                for a, b in ((source, target), (target, source))
            ):
                errors.append(
                    f"{role} pair candidate {item['id']} is missing a pinned direction revision."
                )
    benchmark_paths = [
        project_path(value)
        for group in ("selection", "final")
        for value in config["benchmarks"][group].values()
    ]
    if len({path.resolve() for path in benchmark_paths}) != len(benchmark_paths):
        errors.append("Selection and final benchmark files must be distinct.")
    report = {
        "schema_version": 3,
        "pair": pair,
        "status": "PASS" if not errors else "FAIL",
        "commercial_use": bool(config["direction"].get("commercial_use", False)),
        "pipeline_scope": scope or "specialist_training",
        "selection_policy": "independent_per_direction",
        "shared_multilingual_training": True,
        "student_candidates": (
            []
            if data_only
            else [
                item["id"] for item in commercial_candidates(config, "student")
            ]
        ),
        "teacher_candidates": [
            item["id"] for item in commercial_candidates(config, "teacher")
        ],
        "benchmark_paths": [
            str(path.relative_to(PROJECT_ROOT)) for path in benchmark_paths
        ],
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths(config)["reports"] / "config_validation.json", report)
    if errors:
        raise SystemExit("Configuration validation failed: " + "; ".join(errors))


def candidates(config: dict[str, Any]) -> None:
    pair, source, target, _ = pair_info(config)
    settings = config["data"]
    rows: list[dict[str, Any]] = []
    with project_path(settings["source_file"]).open(
        "r", encoding="utf-8", errors="replace"
    ) as left:
        with project_path(settings["target_file"]).open(
            "r", encoding="utf-8", errors="replace"
        ) as right:
            for row_number, (source_text, target_text) in enumerate(
                zip(left, right, strict=True), 1
            ):
                source_text, target_text = (
                    normalize_language_text(source, source_text),
                    normalize_language_text(target, target_text),
                )
                rows.append(
                    {
                        "pair_id": pair_hash(source_text, target_text),
                        source: source_text,
                        target: target_text,
                        "source_text": source_text,
                        "target_text": target_text,
                        "source_lang": source,
                        "target_lang": target,
                        "source_corpus": settings["corpus"],
                        "source_row": row_number,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates("pair_id")
    source_counts = frame.groupby(frame["source_text"].map(normalized_key))[
        "pair_id"
    ].transform("count")
    target_counts = frame.groupby(frame["target_text"].map(normalized_key))[
        "pair_id"
    ].transform("count")
    frame = frame[(source_counts == 1) & (target_counts == 1)].reset_index(drop=True)
    output = paths(config)["candidates"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(
        paths(config)["reports"] / "candidates.json",
        {"pair": pair, "status": "CANDIDATE_ONLY", "rows": len(frame)},
    )


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


def rule_assessment(
    source: str,
    target: str,
    settings: dict[str, Any],
    source_lang: str = "en",
    target_lang: str = "ru",
) -> tuple[list[str], float, str]:
    flags: list[str] = []
    hard = False
    source, target = normalize(source), normalize(target)
    if not source or not target:
        flags.append("EMPTY")
        hard = True
    if min(len(source), len(target)) < 2:
        flags.append("TOO_SHORT")
        hard = True
    if max(len(source), len(target)) > int(settings["max_characters"]):
        flags.append("TOO_LONG")
        hard = True
    if re.search(r"<[^>]+>|https?://\S+|\{\{.*?\}\}", source + " " + target):
        flags.append("MARKUP_OR_URL")
        hard = True
    ratio = max(len(source), len(target)) / max(1, min(len(source), len(target)))
    if ratio > float(settings["max_length_ratio"]):
        flags.append("CRITICAL_LENGTH_RATIO")
    if normalized_key(source) == normalized_key(target):
        flags.append("SOURCE_TARGET_SAME")
    if (
        min(
            sum(char.isalpha() for char in source),
            sum(char.isalpha() for char in target),
        )
        < 2
    ):
        flags.append("LOW_LETTER_RATIO")
    if re.search(r"([!?.,])\1{2,}", source + target):
        flags.append("REPEATED_PUNCT")
    if re.findall(r"\d+(?:[.,]\d+)?", source) != re.findall(
            r"\d+(?:[.,]\d+)?", target
    ):
        flags.append("NUMBER_MISMATCH")
        if bool(settings.get("exclude_number_mismatch", False)):
            hard = True
    if not _has_expected_script(source_lang, source):
        flags.append("SOURCE_SCRIPT_RISK")
    if not _has_expected_script(target_lang, target):
        flags.append("TARGET_SCRIPT_RISK")
    penalties = {
        "CRITICAL_LENGTH_RATIO": 40,
        "SOURCE_TARGET_SAME": 35,
        "LOW_LETTER_RATIO": 30,
        "REPEATED_PUNCT": 15,
        "NUMBER_MISMATCH": 25,
        "SOURCE_SCRIPT_RISK": 35,
        "TARGET_SCRIPT_RISK": 35,
    }
    score = max(
        0.0, 100.0 - sum(penalties.get(flag, 100 if hard else 0) for flag in flags)
    )
    route = (
        "HARD_REJECT"
        if hard
        else ("NEEDS_QWEN" if flags or score < 75 else "AUTO_ACCEPT")
    )
    return flags, score, route


def rule_reasons(
    source: str,
    target: str,
    settings: dict[str, Any],
    source_lang: str = "en",
    target_lang: str = "ru",
) -> list[str]:
    return [
        flag.lower()
        for flag in rule_assessment(
            source, target, settings, source_lang, target_lang
        )[0]
    ]


def stratified_auto_accept_audit(
    frame: pd.DataFrame, size: int, seed: int
) -> pd.DataFrame:
    if size <= 0 or frame.empty:
        return frame.head(0)
    working = frame.copy()
    lengths = working[["source_text", "target_text"]].map(len).max(axis=1)
    working["_length_band"] = pd.qcut(
        lengths.rank(method="first"), q=min(4, len(working)), labels=False
    )
    samples = []
    for band, part in working.groupby("_length_band"):
        quota = max(1, round(size * len(part) / len(working)))
        samples.append(
            part.sample(n=min(quota, len(part)), random_state=seed + int(band))
        )
    audit = pd.concat(samples).drop_duplicates("pair_id")
    if len(audit) > size:
        audit = audit.sample(n=size, random_state=seed)
    elif len(audit) < size:
        remaining = working[~working["pair_id"].isin(audit["pair_id"])]
        audit = pd.concat(
            [
                audit,
                remaining.sample(n=size - len(audit), random_state=seed + 10),
            ]
        )
    return audit.drop(columns="_length_band").sort_values("pair_id")


def rules(config: dict[str, Any]) -> None:
    _, source_lang, target_lang, _ = pair_info(config)
    frame = pd.read_parquet(paths(config)["candidates"])
    protected = benchmark_sets(config)
    assessments = []
    for row in frame.itertuples(index=False):
        flags, score, route = rule_assessment(
            row.source_text,
            row.target_text,
            config["data"],
            source_lang,
            target_lang,
        )
        if (
            row.pair_id in protected["pairs"]
            or normalized_key(row.source_text) in protected["source"]
            or normalized_key(row.target_text) in protected["target"]
        ):
            flags.append("PROTECTED_BENCHMARK_OVERLAP")
            route = "HARD_REJECT"
            score = 0.0
        assessments.append((flags, score, route))
    frame["rule_flags"] = [json.dumps(item[0]) for item in assessments]
    frame["quality_score"] = [item[1] for item in assessments]
    frame["pipeline_route"] = [item[2] for item in assessments]
    frame.to_parquet(paths(config)["routed"], index=False)
    force_full_review = bool(config["judge"].get("force_full_human_review", False))
    if force_full_review:
        needs = frame[frame["pipeline_route"] != "HARD_REJECT"].copy()
        audit = frame.head(0).copy()
        needs["qwen_review_type"] = "FULL_REVIEW"
        review = needs.sort_values("pair_id")
    else:
        needs = frame[frame["pipeline_route"] == "NEEDS_QWEN"].copy()
        auto = frame[frame["pipeline_route"] == "AUTO_ACCEPT"].copy()
        audit_size = min(int(config["judge"]["auto_accept_audit_pairs"]), len(auto))
        audit = stratified_auto_accept_audit(
            auto, audit_size, int(config["direction"]["seed"])
        )
        needs["qwen_review_type"] = "NEEDS_QWEN"
        audit["qwen_review_type"] = "AUTO_ACCEPT_AUDIT"
        review = pd.concat([needs, audit], ignore_index=True).sort_values("pair_id")
    review.to_parquet(paths(config)["human_review"], index=False)
    write_json(
        paths(config)["reports"] / "rules.json",
        {
            "input_rows": len(frame),
            "routes": dict(Counter(frame["pipeline_route"])),
            "qwen_full_review_rows": len(needs),
            "auto_accept_audit_rows": len(audit),
            "force_full_human_review": force_full_review,
            "audit_seed": int(config["direction"]["seed"]),
            "protected_overlap_rows": int(
                frame["rule_flags"].str.contains("PROTECTED_BENCHMARK_OVERLAP").sum()
            ),
        },
    )


def _resolve_second(second: str) -> tuple[str, str, float]:
    if second == "FAIL":
        return "REJECT", "REJECT", 0.0
    if second == "UNCERTAIN":
        return "QUARANTINE", "QUARANTINE", 0.0
    if second == "MINOR":
        return "APPROVED", "BRONZE", QUALITY_WEIGHTS["BRONZE"]
    if second == "PASS":
        return "APPROVED", "SILVER", QUALITY_WEIGHTS["SILVER"]
    raise RuntimeError(f"Invalid second label {second!r}.")


def approve(config: dict[str, Any]) -> None:
    routed = pd.read_parquet(paths(config)["routed"])
    first = pd.read_parquet(paths(config)["human_judged"])
    second = pd.read_parquet(paths(config)["human_second"])
    first_map = first.set_index("pair_id").to_dict("index") if len(first) else {}
    second_map = second.set_index("pair_id").to_dict("index") if len(second) else {}
    decisions: list[dict[str, Any]] = []
    for row in routed.to_dict("records"):
        route, pair_id = row["pipeline_route"], row["pair_id"]
        if route == "HARD_REJECT":
            status, tier, weight, source = "REJECT", "REJECT", 0.0, "RULE_HARD_REJECT"
        elif pair_id not in first_map:
            if route != "AUTO_ACCEPT":
                raise RuntimeError(f"Missing Qwen review for {pair_id}")
            status, tier, weight, source = (
                "APPROVED",
                "SILVER",
                QUALITY_WEIGHTS["SILVER"],
                "RULE_LOW_RISK",
            )
        else:
            label = str(first_map[pair_id].get("judge_label", "")).upper()
            if label not in FIRST_LABELS:
                raise RuntimeError(f"Invalid first label for {pair_id}: {label}")
            if label == "PASS":
                status, tier, weight, source = (
                    "APPROVED",
                    "GOLD",
                    QUALITY_WEIGHTS["GOLD"],
                    "QWEN_PASS",
                )
            elif label == "MINOR":
                status, tier, weight, source = (
                    "APPROVED",
                    "BRONZE",
                    QUALITY_WEIGHTS["BRONZE"],
                    "QWEN_MINOR",
                )
            else:
                if pair_id not in second_map:
                    raise RuntimeError(
                        f"Missing independent second review for {pair_id}"
                    )
                second_label = (
                    str(second_map[pair_id].get("judge_label", "UNCERTAIN")).upper()
                    if bool(second_map[pair_id].get("judge_parse_ok", False))
                    else "UNCERTAIN"
                )
                status, tier, weight = _resolve_second(
                    second_label
                )
                source = "QWEN_SECOND_REVIEW"
        row.update(
            final_status=status,
            quality_tier=tier,
            training_weight=weight,
            final_decision_source=source,
        )
        decisions.append(row)
    master = pd.DataFrame(decisions)
    approved = master[master["final_status"] == "APPROVED"].copy()
    output = paths(config)["approved"]
    output.parent.mkdir(parents=True, exist_ok=True)
    approved.to_parquet(output, index=False)
    write_json(
        paths(config)["reports"] / "approved.json",
        {
            "input_rows": len(master),
            "approved_rows": len(approved),
            "rejected_rows": int((master["final_status"] == "REJECT").sum()),
            "quarantined_rows": int((master["final_status"] == "QUARANTINE").sum()),
            "quality_tiers": dict(Counter(approved["quality_tier"])),
        },
    )


def write_directional(
    frame: pd.DataFrame, path: Path, source: str, target: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in frame.itertuples(index=False):
        common = {
            "pair_id": row.pair_id,
            "source_corpus": getattr(row, "source_corpus", "unknown"),
            "quality_tier": row.quality_tier,
            "weight": float(row.training_weight),
        }
        records.extend(
            [
                {
                    **common,
                    "src_lang": source,
                    "tgt_lang": target,
                    "src_text": row.source_text,
                    "tgt_text": row.target_text,
                },
                {
                    **common,
                    "src_lang": target,
                    "tgt_lang": source,
                    "src_text": row.target_text,
                    "tgt_text": row.source_text,
                },
            ]
        )
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )


def select_split_pool(
    frame: pd.DataFrame, settings: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select the explicitly configured, deterministic human-data pool."""
    configured_tiers = [
        str(value).upper() for value in settings.get("training_tiers", [])
    ]
    unknown_tiers = sorted(set(configured_tiers) - set(QUALITY_WEIGHTS))
    if unknown_tiers:
        raise ValueError(f"Unknown data.training_tiers: {unknown_tiers}")
    eligible = (
        frame[frame["quality_tier"].isin(configured_tiers)].copy()
        if configured_tiers
        else frame.copy()
    )
    eligible = eligible.sample(frac=1, random_state=seed).reset_index(drop=True)
    validation_size = int(settings["validation_pairs"])
    test_size = int(settings["test_pairs"])
    configured_train_size = int(settings.get("train_pairs", 0))
    if min(validation_size, test_size, configured_train_size) < 0:
        raise ValueError("Configured split sizes cannot be negative.")
    required = validation_size + test_size + configured_train_size
    if configured_train_size > 0 and len(eligible) < required:
        raise RuntimeError(
            "Insufficient eligible approved pairs for requested splits: "
            f"need {required}, found {len(eligible)}."
        )
    if configured_train_size == 0 and len(eligible) <= validation_size + test_size:
        raise RuntimeError("Insufficient eligible approved pairs for requested splits.")
    selected_size = required if configured_train_size > 0 else len(eligible)
    selected = eligible.iloc[:selected_size].copy()
    return selected, {
        "approved_pairs": len(frame),
        "eligible_pairs": len(eligible),
        "excluded_by_tier": len(frame) - len(eligible),
        "unused_eligible_pairs": len(eligible) - len(selected),
        "training_tiers": configured_tiers or "ALL",
        "configured_train_pairs": configured_train_size or "ALL_REMAINING",
    }


def split(config: dict[str, Any]) -> None:
    _, source, target, _ = pair_info(config)
    frame = pd.read_parquet(paths(config)["approved"])
    protected = benchmark_sets(config)
    leaks = frame[
        frame["pair_id"].isin(protected["pairs"])
        | frame["source_text"].map(normalized_key).isin(protected["source"])
        | frame["target_text"].map(normalized_key).isin(protected["target"])
    ]
    if len(leaks):
        raise RuntimeError(
            f"Protected benchmark leakage detected in {len(leaks)} approved pairs."
        )
    frame, selection_report = select_split_pool(
        frame, config["data"], int(config["direction"]["seed"])
    )
    validation_size, test_size = (
        int(config["data"]["validation_pairs"]),
        int(config["data"]["test_pairs"]),
    )
    train_size = int(config["data"].get("train_pairs", 0))
    train_end = validation_size + test_size + train_size if train_size else None
    parts = {
        "validation": frame.iloc[:validation_size],
        "test": frame.iloc[validation_size : validation_size + test_size],
        "train": frame.iloc[validation_size + test_size : train_end],
    }
    for column in ("pair_id", "source_text", "target_text"):
        sets = [
            set(
                part[column].map(normalized_key)
                if column != "pair_id"
                else part[column]
            )
            for part in parts.values()
        ]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise RuntimeError(f"{column} leakage detected between splits.")
    output = paths(config)["splits"]
    output.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part.to_parquet(output / f"{name}_pairs.parquet", index=False)
        write_directional(part, output / f"{name}.jsonl", source, target)
    write_json(
        paths(config)["reports"] / "split.json",
        {
            **{f"{name}_pairs": len(part) for name, part in parts.items()},
            **selection_report,
            "pair_and_side_disjoint": True,
            "protected_overlap": 0,
        },
    )


def kd_candidates(config: dict[str, Any]) -> None:
    protected = benchmark_sets(config)
    rows = [
        json.loads(line)
        for line in (paths(config)["splits"] / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    limit = int(config["distillation"]["candidate_pairs_per_direction"])
    counts: Counter[str] = Counter()
    seen_sources = set()
    selected = []
    for row in rows:
        direction = f"{row['src_lang']}-{row['tgt_lang']}"
        source_key = normalized_key(row["src_text"])
        side = (
            "source"
            if row["src_lang"] == config["direction"]["source_lang"]
            else "target"
        )
        if (
            source_key in protected[side]
            or (row["src_lang"], source_key) in seen_sources
        ):
            continue
        if limit > 0 and counts[direction] >= limit:
            continue
        seen_sources.add((row["src_lang"], source_key))
        counts[direction] += 1
        selected.append(
            {
                **{
                    key: row[key]
                    for key in ("pair_id", "src_lang", "tgt_lang", "src_text")
                },
                "reference_text": row["tgt_text"],
            }
        )
    output = paths(config)["pipeline"] / "kd_candidates.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    write_json(
        paths(config)["reports"] / "kd_candidates.json",
        {
            "rows": len(selected),
            "directions": dict(counts),
            "protected_source_overlap": 0,
        },
    )


def kd_dataset(config: dict[str, Any]) -> None:
    frame = pd.read_parquet(paths(config)["teacher_judged"])
    protected = benchmark_sets(config)
    accepted = frame[
        (frame["judge_parse_ok"])
        & (frame["judge_label"] == "PASS")
        & frame["teacher_usefulness"].isin(["HIGH", "MEDIUM"])
    ].copy()
    rejected: Counter[str] = Counter()
    teacher_rows = []
    seen = set()
    for row in accepted.itertuples(index=False):
        source_text = normalize_language_text(row.src_lang, row.src_text)
        teacher_text = normalize_language_text(row.tgt_lang, row.teacher_text)
        reason = ""
        if not teacher_text:
            reason = "EMPTY_TEACHER"
        elif normalized_key(source_text) == normalized_key(teacher_text):
            reason = "SOURCE_COPY"
        elif hasattr(row, "reference_text") and normalized_key(
            teacher_text
        ) == normalized_key(row.reference_text):
            reason = "REFERENCE_COPY"
        elif not _has_expected_script(row.tgt_lang, teacher_text):
            reason = "TARGET_SCRIPT"
        side = (
            "source" if row.src_lang == config["direction"]["source_lang"] else "target"
        )
        if normalized_key(source_text) in protected[side]:
            reason = "PROTECTED_SOURCE"
        if row.src_lang == config["direction"]["source_lang"]:
            generated_pair = pair_hash(source_text, teacher_text)
            generated_target_side = "target"
        else:
            generated_pair = pair_hash(teacher_text, source_text)
            generated_target_side = "source"
        if (
            generated_pair in protected["pairs"]
            or normalized_key(teacher_text) in protected[generated_target_side]
        ):
            reason = "PROTECTED_GENERATED_TARGET"
        key = (row.src_lang, normalized_key(source_text), normalized_key(teacher_text))
        if key in seen:
            reason = "DUPLICATE"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(key)
        usefulness = str(row.teacher_usefulness)
        teacher_rows.append(
            {
                "pair_id": row.pair_id,
                "src_lang": row.src_lang,
                "tgt_lang": row.tgt_lang,
                "src_text": source_text,
                "tgt_text": teacher_text,
                "weight": float(
                    config["distillation"][
                        "teacher_high_weight"
                        if usefulness == "HIGH"
                        else "teacher_medium_weight"
                    ]
                ),
                "training_source": "teacher_kd",
                "teacher_usefulness": usefulness,
            }
        )
    human_rows = [
        json.loads(line)
        for line in (paths(config)["splits"] / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    for row in human_rows:
        row["training_source"] = "human_replay"
    output = paths(config)["distillation"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in human_rows + teacher_rows
        ),
        encoding="utf-8",
    )
    (output / "validation.jsonl").write_text(
        (paths(config)["splits"] / "validation.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_json(
        paths(config)["reports"] / "kd_dataset.json",
        {
            "human_rows": len(human_rows),
            "teacher_rows": len(teacher_rows),
            "rejections": dict(rejected),
            "allowed_labels": ["PASS"],
            "allowed_usefulness": ["HIGH", "MEDIUM"],
        },
    )


def teacher_calibration_report(config: dict[str, Any]) -> None:
    frame = pd.read_parquet(paths(config)["teacher_calibration"])
    valid = frame[frame["judge_parse_ok"]]
    if len(valid) != len(frame):
        raise RuntimeError("Teacher calibration contains unparseable Judge results.")
    write_json(
        paths(config)["reports"] / "teacher_judge_policy.json",
        {
            "schema_version": 2,
            "policy": "PASS and usefulness HIGH/MEDIUM; full Teacher audit required",
            "sample_rows": len(frame),
            "labels": dict(Counter(valid["judge_label"])),
            "usefulness": dict(Counter(valid["teacher_usefulness"])),
            "frozen": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=(
            "validate",
            "candidates",
            "rules",
            "approve",
            "split",
            "kd_candidates",
            "kd_dataset",
            "teacher_policy",
        ),
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    {
        "validate": validate,
        "candidates": candidates,
        "rules": rules,
        "approve": approve,
        "split": split,
        "kd_candidates": kd_candidates,
        "kd_dataset": kd_dataset,
        "teacher_policy": teacher_calibration_report,
    }[args.action](config)


if __name__ == "__main__":
    main()
