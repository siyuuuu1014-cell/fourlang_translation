from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_v2.common import (  # noqa: E402
    load_config,
    pair_info,
    pipeline_namespace,
    project_path,
    write_json,
)
from scripts.pipeline_v2.data_flow import (  # noqa: E402
    _has_expected_script,
    benchmark_sets,
    normalized_key,
    pair_hash,
)
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)
from scripts.supplemental.import_pair_v2 import CANTONESE_PATTERN  # noqa: E402


URL_OR_MARKUP = re.compile(r"https?://|www\.|\b\S+@\S+\.\S+\b|<[^>]+>|\{\{.*?\}\}", re.I)
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u052f]")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
REPEATED_PATTERN = re.compile(r"(.)\1{7,}")
CITATION_DEBRIS = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
ZH_BOILERPLATE = re.compile(
    r"本网|本站|免责声明|版权声明|转载.{0,12}(来源|目的)|"
    r"不代表.{0,20}观点|不承担.{0,20}责任"
)
UZBEK_SIGNAL = re.compile(
    r"\b(va|bu|uchun|bilan|ham|emas|edi|ekan|bor|yo'q|o'z|"
    r"bo'l\w*|qil\w*|bo'yicha|orqali|mumkin|kerak|ular|uning|"
    r"ushbu|juda|kabi|yoki|lekin|bir|eng|deb|dan|ga|ning)\b",
    re.I,
)
ZH_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|[\r\n]+")
UZ_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+|[\r\n]+")


def _validate_contract(config: dict[str, Any]) -> None:
    pair, source, target, version = pair_info(config)
    if (pair, source, target, version) != ("zh_uz", "zh", "uz", "v3"):
        raise ValueError("This flow is restricted to the isolated zh_uz v3 pipeline.")
    if pipeline_namespace(config) != "zh_uz_v3":
        raise ValueError("artifacts.pipeline_namespace must be zh_uz_v3.")
    settings = config["monolingual"]
    for language in ("zh", "uz"):
        if int(settings[f"target_{language}_sources"]) < 1:
            raise ValueError(f"target_{language}_sources must be positive.")
        if int(settings[f"collection_target_{language}_sources"]) < int(
            settings[f"target_{language}_sources"]
        ):
            raise ValueError(
                f"collection_target_{language}_sources must be at least the final target."
            )
    ids = [str(item["id"]) for item in config.get("monolingual_sources", [])]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("monolingual_sources must have unique, non-empty ids.")


def _roots(config: dict[str, Any]) -> tuple[Path, Path]:
    namespace = pipeline_namespace(config)
    return (
        PROJECT_ROOT / "data" / "pipeline_v2" / namespace,
        PROJECT_ROOT / "reports" / "pipeline" / namespace,
    )


def _atomic_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _iter_local(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    path = project_path(str(source["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        yield from pd.read_parquet(path).to_dict("records")
        return
    if path.suffix in {".jsonl", ".json"}:
        yield from _read_jsonl(path)
        return
    raise ValueError(f"Unsupported local source format: {path}")


def _iter_hf(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies before streaming HF data.") from exc
    attempts = max(1, int(source.get("connection_attempts", 5)))
    for attempt in range(1, attempts + 1):
        try:
            dataset = load_dataset(
                str(source["dataset_id"]),
                name=str(source["subset"]),
                split=str(source.get("split", "train")),
                streaming=True,
            )
            break
        except (ConnectionError, OSError, TimeoutError) as exc:
            if attempt == attempts:
                raise
            delay = min(5 * (2 ** (attempt - 1)), 40)
            print(
                f"Hugging Face connection retry {attempt}/{attempts} for "
                f"{source['id']} in {delay}s: {type(exc).__name__}",
                flush=True,
            )
            time.sleep(delay)
    yield from dataset


def _hplt_urls(source: dict[str, Any]) -> list[str]:
    request = urllib.request.Request(
        str(source["map_url"]), headers={"User-Agent": "fourlang-translation/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        lines = response.read().decode("utf-8").splitlines()
    bins = tuple(f"/{value}_" for value in source.get("quality_bins", ["10", "9"]))
    urls = [line.strip() for line in lines if line.strip() and bins and any(token in line for token in bins)]
    max_shards = int(source.get("max_shards", 0))
    return urls[:max_shards] if max_shards > 0 else urls


def _iter_hplt(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "Install zstandard before streaming HPLT: python -m pip install "
            "zstandard==0.25.0"
        ) from exc
    urls = _hplt_urls(source)
    if not urls:
        raise RuntimeError(f"No requested HPLT quality-bin shards found for {source['id']}.")
    # HPLT web-register keys vary by language and shard. Arrow's fixed struct
    # casting therefore fails even though the text is valid. Stream the zstd JSONL
    # directly and leave unrelated sparse metadata untouched.
    decompressor = zstandard.ZstdDecompressor()
    for url in urls:
        request = urllib.request.Request(
            url, headers={"User-Agent": "fourlang-translation/0.1"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with decompressor.stream_reader(response) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                    for line in text_stream:
                        if line.strip():
                            yield json.loads(line)


def _iter_records(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    kind = str(source["kind"])
    if kind in {"local_jsonl", "local_parquet", "v2_bronze"}:
        yield from _iter_local(source)
    elif kind == "hf_dataset":
        yield from _iter_hf(source)
    elif kind == "hplt_map":
        yield from _iter_hplt(source)
    else:
        raise ValueError(f"Unknown monolingual source kind: {kind}")


def _source_texts(source: dict[str, Any], record: dict[str, Any]) -> Iterator[str]:
    language = str(source["language"])
    if source["kind"] == "v2_bronze":
        tier = str(record.get("quality_tier", "")).upper()
        if tier != str(source.get("quality_tier", "BRONZE")).upper():
            return
    field = str(source["text_field"])
    value = record.get(field)
    if value is None:
        return
    text = str(value)
    if not bool(source.get("split_documents", True)):
        yield text
        return
    pattern = ZH_SENTENCE_BOUNDARY if language == "zh" else UZ_SENTENCE_BOUNDARY
    yield from (part for part in pattern.split(text) if part.strip())


def quality_reason(language: str, text: str, settings: dict[str, Any]) -> tuple[str | None, str]:
    try:
        normalized = normalize_language_text(language, text)
    except (TypeError, ValueError):
        return "NORMALIZATION_ERROR", str(text).strip()
    if not normalized or "\ufffd" in normalized:
        return "EMPTY_OR_REPLACEMENT", normalized
    length = len(normalized)
    if length < int(settings[f"min_{language}_characters"]):
        return "TOO_SHORT", normalized
    if length > int(settings[f"max_{language}_characters"]):
        return "TOO_LONG", normalized
    if URL_OR_MARKUP.search(normalized):
        return "URL_EMAIL_OR_MARKUP", normalized
    if REPEATED_PATTERN.search(normalized):
        return "REPEATED_CHARACTER", normalized
    if CITATION_DEBRIS.search(normalized):
        return "CITATION_DEBRIS", normalized
    if sum(character.isdigit() for character in normalized) / max(length, 1) > float(
        settings.get("max_digit_ratio", 0.25)
    ):
        return "TOO_MANY_DIGITS", normalized
    if language == "zh":
        if not CJK_PATTERN.search(normalized):
            return "ZH_SCRIPT_RISK", normalized
        if CANTONESE_PATTERN.search(normalized):
            return "NON_MANDARIN_CHINESE", normalized
        if ZH_BOILERPLATE.search(normalized):
            return "ZH_BOILERPLATE", normalized
        if normalized.count(" ") >= 2 and not re.search(r"[，,。！？!?；;：:]", normalized):
            return "ZH_KEYWORD_FRAGMENT", normalized
        letters = [character for character in normalized if character.isalpha()]
        cjk_ratio = sum(bool(CJK_PATTERN.fullmatch(character)) for character in letters) / max(
            len(letters), 1
        )
        if cjk_ratio < float(settings.get("min_zh_cjk_ratio", 0.6)):
            return "ZH_SCRIPT_RATIO", normalized
    else:
        if CYRILLIC_PATTERN.search(normalized) or CJK_PATTERN.search(normalized):
            return "UZ_SCRIPT_RISK", normalized
        letters = [character for character in normalized if character.isalpha()]
        latin = sum("a" <= character.casefold() <= "z" for character in letters)
        if latin / max(len(letters), 1) < float(settings.get("min_uz_latin_ratio", 0.85)):
            return "UZ_SCRIPT_RATIO", normalized
        if not UZBEK_SIGNAL.search(normalized):
            return "UZ_LANGUAGE_SIGNAL", normalized
    return None, normalized


def _existing_sources(config: dict[str, Any]) -> dict[str, set[str]]:
    result = {"zh": set(), "uz": set()}
    for value in config["monolingual"].get("deduplicate_against", []):
        path = project_path(str(value))
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            for record in frame.to_dict("records"):
                if "source_text" in record and "target_text" in record:
                    result["zh"].add(normalized_key(record["source_text"]))
                    result["uz"].add(normalized_key(record["target_text"]))
                elif record.get("src_lang") in result:
                    result[str(record["src_lang"])].add(normalized_key(record["src_text"]))
        else:
            for record in _read_jsonl(path):
                language = str(record.get("src_lang", ""))
                if language in result:
                    result[language].add(normalized_key(record["src_text"]))
    protected = benchmark_sets(config)
    result["zh"].update(protected["source"])
    result["uz"].update(protected["target"])
    return result


def _candidate_id(language: str, text: str) -> str:
    return hashlib.sha256(f"zh_uz_v3\n{language}\n{normalized_key(text)}".encode()).hexdigest()


def validate(config: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(config)
    _, report_root = _roots(config)
    report = {
        "schema_version": 1,
        "pair": "zh_uz",
        "version": "v3",
        "namespace": pipeline_namespace(config),
        "target_sources": {
            "zh": int(config["monolingual"]["target_zh_sources"]),
            "uz": int(config["monolingual"]["target_uz_sources"]),
        },
        "collection_targets": {
            "zh": int(config["monolingual"]["collection_target_zh_sources"]),
            "uz": int(config["monolingual"]["collection_target_uz_sources"]),
        },
        "sources": [str(item["id"]) for item in config["monolingual_sources"]],
        "isolated_from_v2": True,
    }
    write_json(report_root / "config_validation.json", report)
    return report


def collect(
    config: dict[str, Any], *, allow_config_extension: bool = False
) -> dict[str, Any]:
    _validate_contract(config)
    settings = config["monolingual"]
    pipeline_root, report_root = _roots(config)
    checkpoint_path = pipeline_root / "monolingual_collected.parquet"
    state_path = pipeline_root / "monolingual_collection_state.json"
    checkpoint_every = int(settings.get("checkpoint_rows", 250))
    existing = _existing_sources(config)
    collected_raw = (
        pd.read_parquet(checkpoint_path).to_dict("records")
        if checkpoint_path.is_file()
        else []
    )
    collected: list[dict[str, Any]] = []
    stale_rejections: Counter[str] = Counter()
    for row in collected_raw:
        reason, normalized = quality_reason(
            str(row["src_lang"]), str(row["src_text"]), settings
        )
        if reason:
            stale_rejections[f"checkpoint:{reason}"] += 1
            continue
        row["src_text"] = normalized
        collected.append(row)
    seen = {language: set(existing[language]) for language in ("zh", "uz")}
    for row in collected:
        seen[str(row["src_lang"])].add(normalized_key(row["src_text"]))
    state: dict[str, Any] = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "sources": {}}
    )
    signature = hashlib.sha256(
        json.dumps(
            {
                "monolingual": settings,
                "sources": config["monolingual_sources"],
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    previous_signature = state.get("config_signature")
    if previous_signature and previous_signature != signature and not allow_config_extension:
        raise RuntimeError(
            "The v3 collection config changed after checkpoints were created. "
            f"Move data/pipeline_v2/{pipeline_namespace(config)}/monolingual_* "
            "before intentionally starting a different collection."
        )
    state["config_signature"] = signature
    accepted_by_source = Counter(str(row["source_corpus"]) for row in collected)
    for source_id, source_state in state["sources"].items():
        source_state["accepted"] = accepted_by_source[source_id]
    counts = Counter(str(row["src_lang"]) for row in collected)
    rejection_counts: Counter[str] = Counter(stale_rejections)

    def save_checkpoint() -> None:
        _atomic_parquet(collected, checkpoint_path)
        write_json(state_path, state)

    for source in config["monolingual_sources"]:
        if not bool(source.get("enabled", True)):
            continue
        source_id = str(source["id"])
        language = str(source["language"])
        target = int(settings[f"collection_target_{language}_sources"])
        source_limit = int(source["max_rows"])
        source_state = state["sources"].setdefault(
            source_id,
            {"status": "running", "documents_seen": 0, "accepted": 0},
        )
        if counts[language] >= target:
            continue
        documents_seen = int(source_state.get("documents_seen", 0))
        accepted_here = int(source_state.get("accepted", 0))
        if source_state.get("status") == "completed" and accepted_here >= source_limit:
            continue
        source_state["status"] = "running"
        max_documents = int(source.get("max_documents", 100_000))
        min_language_score = float(source.get("min_language_score", 0.0))
        print(
            f"Collecting {source_id}: resume_documents={documents_seen} "
            f"resume_accepted={accepted_here} target_{language}={target}",
            flush=True,
        )
        try:
            for index, record in enumerate(_iter_records(source)):
                if index < documents_seen:
                    continue
                source_state["documents_seen"] = index + 1
                if index + 1 > max_documents:
                    break
                if min_language_score and float(record.get("language_score", 1.0)) < min_language_score:
                    rejection_counts[f"{source_id}:LOW_LANGUAGE_SCORE"] += 1
                    continue
                for raw_text in _source_texts(source, record):
                    reason, text = quality_reason(language, raw_text, settings)
                    key = normalized_key(text)
                    if reason is None and key in seen[language]:
                        reason = "DUPLICATE_OR_PROTECTED_SOURCE"
                    if reason is not None:
                        rejection_counts[f"{source_id}:{reason}"] += 1
                        continue
                    candidate_id = _candidate_id(language, text)
                    collected.append(
                        {
                            "pair_id": candidate_id,
                            "src_lang": language,
                            "tgt_lang": "uz" if language == "zh" else "zh",
                            "src_text": text,
                            "reference_text": "",
                            "source_corpus": source_id,
                            "source_license": str(source.get("license", "unknown")),
                            "source_record": str(record.get("id", index)),
                        }
                    )
                    seen[language].add(key)
                    counts[language] += 1
                    accepted_here += 1
                    source_state["accepted"] = accepted_here
                    if len(collected) % checkpoint_every == 0:
                        save_checkpoint()
                        print(
                            f"Collection checkpoint: zh={counts['zh']}/{settings['collection_target_zh_sources']} "
                            f"uz={counts['uz']}/{settings['collection_target_uz_sources']} "
                            f"source={source_id}",
                            flush=True,
                        )
                    if accepted_here >= source_limit or counts[language] >= target:
                        break
                if accepted_here >= source_limit or counts[language] >= target:
                    break
        except Exception as exc:
            source_state["status"] = "failed"
            source_state["error"] = f"{type(exc).__name__}: {exc}"
            save_checkpoint()
            raise RuntimeError(
                f"Source {source_id!r} failed after {source_state['documents_seen']} documents; "
                "the checkpoint is safe and the same command can be rerun."
            ) from exc
        source_state["status"] = "completed"
        source_state.pop("error", None)
        save_checkpoint()
        print(
            f"Source complete: {source_id} accepted={accepted_here} "
            f"documents={source_state['documents_seen']}",
            flush=True,
        )

    targets = {
        "zh": int(settings["collection_target_zh_sources"]),
        "uz": int(settings["collection_target_uz_sources"]),
    }
    shortages = {language: targets[language] - counts[language] for language in targets if counts[language] < targets[language]}
    selected: list[dict[str, Any]] = []
    for language in ("zh", "uz"):
        rows = [row for row in collected if row["src_lang"] == language]
        rows.sort(key=lambda row: hashlib.sha256(f"{config['direction']['seed']}:{row['pair_id']}".encode()).hexdigest())
        selected.extend(rows[: targets[language]])
    selected.sort(key=lambda row: (row["src_lang"], row["pair_id"]))
    _write_jsonl(selected, pipeline_root / "monolingual_candidates.jsonl")
    # Invalidate the earlier pre-audit candidate list. The selection stage is the
    # only stage allowed to materialize Teacher-ready sources.
    _write_jsonl([], pipeline_root / "kd_candidates.jsonl")
    sample_size = int(settings.get("audit_sample_rows_per_language", 100))
    audit_sample = []
    for language in ("zh", "uz"):
        audit_sample.extend([row for row in selected if row["src_lang"] == language][:sample_size])
    _write_jsonl(audit_sample, pipeline_root / "source_audit_sample.jsonl")
    by_source = Counter((row["src_lang"], row["source_corpus"]) for row in selected)
    selected_counts = Counter(str(row["src_lang"]) for row in selected)
    report = {
        "schema_version": 1,
        "candidate_rows": len(selected),
        "directions": {
            "zh-uz": selected_counts["zh"],
            "uz-zh": selected_counts["uz"],
        },
        "targets": targets,
        "shortages": shortages,
        "selected_by_source": {
            f"{language}:{source_id}": count
            for (language, source_id), count in sorted(by_source.items())
        },
        "rejections": dict(rejection_counts),
        "checkpoint": str(checkpoint_path),
        "audit_sample": str(pipeline_root / "source_audit_sample.jsonl"),
        "eligible_for_source_review": not shortages,
    }
    write_json(report_root / "monolingual_collection.json", report)
    if shortages and bool(settings.get("require_full_targets", True)):
        raise RuntimeError(f"Monolingual source targets were not met: {shortages}")
    return report


def select_sources(config: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(config)
    settings = config["monolingual"]
    pipeline_root, report_root = _roots(config)
    frame = pd.read_parquet(pipeline_root / "source_judged.parquet")
    accepted = frame[
        frame["judge_parse_ok"].fillna(False)
        & (frame["judge_label"].fillna("UNCERTAIN").str.upper() == "PASS")
    ].copy()
    targets = {
        "zh": int(settings["target_zh_sources"]),
        "uz": int(settings["target_uz_sources"]),
    }
    selected: list[dict[str, Any]] = []
    available: Counter[str] = Counter()
    for language in ("zh", "uz"):
        rows = accepted[accepted["src_lang"] == language].to_dict("records")
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{config['direction']['seed']}:{row['pair_id']}".encode()
            ).hexdigest()
        )
        available[language] = len(rows)
        selected.extend(
            {
                key: row[key]
                for key in (
                    "pair_id",
                    "src_lang",
                    "tgt_lang",
                    "src_text",
                    "reference_text",
                    "source_corpus",
                    "source_license",
                    "source_record",
                )
            }
            for row in rows[: targets[language]]
        )
    shortages = {
        language: targets[language] - available[language]
        for language in targets
        if available[language] < targets[language]
    }
    selected.sort(key=lambda row: (row["src_lang"], row["pair_id"]))
    labels = Counter(
        (
            str(row.src_lang),
            str(row.judge_label) if bool(row.judge_parse_ok) else "UNPARSEABLE",
        )
        for row in frame.itertuples(index=False)
    )
    report = {
        "schema_version": 1,
        "reviewed_rows": len(frame),
        "available_pass": dict(available),
        "selected": Counter(str(row["src_lang"]) for row in selected),
        "targets": targets,
        "shortages": shortages,
        "labels": {
            f"{language}:{label}": count
            for (language, label), count in sorted(labels.items())
        },
        "eligible_for_teacher_generation": not shortages,
    }
    write_json(report_root / "source_selection.json", report)
    if shortages:
        raise RuntimeError(f"Qwen-approved monolingual sources are below target: {shortages}")
    _write_jsonl(selected, pipeline_root / "kd_candidates.jsonl")
    return report


def finalize(config: dict[str, Any]) -> dict[str, Any]:
    _validate_contract(config)
    pipeline_root, report_root = _roots(config)
    judged_path = pipeline_root / "teacher_judged.parquet"
    frame = pd.read_parquet(judged_path)
    accepted = frame[
        frame["judge_parse_ok"].fillna(False)
        & (frame["judge_label"] == "PASS")
        & frame["teacher_usefulness"].isin(["HIGH", "MEDIUM"])
    ].copy()
    protected = benchmark_sets(config)
    base_train_path = project_path(config["monolingual"]["base_train"])
    base_validation_path = project_path(config["monolingual"]["base_validation"])
    base_rows = _read_jsonl(base_train_path)
    seen = {
        (
            str(row["src_lang"]),
            normalized_key(row["src_text"]),
            normalized_key(row["tgt_text"]),
        )
        for row in base_rows
    }
    rejected: Counter[str] = Counter()
    teacher_rows: list[dict[str, Any]] = []
    for row in accepted.to_dict("records"):
        source_lang = str(row["src_lang"])
        target_lang = str(row["tgt_lang"])
        source_text = normalize_language_text(source_lang, row["src_text"])
        teacher_text = normalize_language_text(target_lang, row["teacher_text"])
        reason = ""
        if not teacher_text:
            reason = "EMPTY_TEACHER"
        elif normalized_key(source_text) == normalized_key(teacher_text):
            reason = "SOURCE_COPY"
        elif not _has_expected_script(target_lang, teacher_text):
            reason = "TARGET_SCRIPT"
        source_side = "source" if source_lang == "zh" else "target"
        target_side = "target" if target_lang == "uz" else "source"
        if normalized_key(source_text) in protected[source_side]:
            reason = "PROTECTED_SOURCE"
        canonical_pair = (
            pair_hash(source_text, teacher_text)
            if source_lang == "zh"
            else pair_hash(teacher_text, source_text)
        )
        if canonical_pair in protected["pairs"] or normalized_key(teacher_text) in protected[target_side]:
            reason = "PROTECTED_GENERATED_TARGET"
        key = (source_lang, normalized_key(source_text), normalized_key(teacher_text))
        if key in seen:
            reason = "DUPLICATE"
        if reason:
            rejected[reason] += 1
            continue
        seen.add(key)
        usefulness = str(row["teacher_usefulness"])
        teacher_rows.append(
            {
                "pair_id": str(row["pair_id"]),
                "src_lang": source_lang,
                "tgt_lang": target_lang,
                "src_text": source_text,
                "tgt_text": teacher_text,
                "weight": float(
                    config["distillation"][
                        "teacher_high_weight" if usefulness == "HIGH" else "teacher_medium_weight"
                    ]
                ),
                "training_source": "teacher_kd_v3",
                "teacher_usefulness": usefulness,
                "teacher_id": str(row.get("teacher_id", "nllb200_3_3b")),
                "source_corpus": str(row.get("source_corpus", "unknown")),
            }
        )
    counts = Counter(f"{row['src_lang']}-{row['tgt_lang']}" for row in teacher_rows)
    minimum = int(config["monolingual"].get("minimum_accepted_per_direction", 0))
    shortages = {
        direction: minimum - counts[direction]
        for direction in ("zh-uz", "uz-zh")
        if counts[direction] < minimum
    }
    output_root = PROJECT_ROOT / "data" / "distillation" / "zh_uz" / "v3"
    report = {
        "schema_version": 1,
        "base_v2_rows": len(base_rows),
        "teacher_rows": len(teacher_rows),
        "teacher_rows_by_direction": dict(counts),
        "minimum_accepted_per_direction": minimum,
        "shortages": shortages,
        "rejections": dict(rejected),
        "allowed_labels": ["PASS"],
        "allowed_usefulness": ["HIGH", "MEDIUM"],
        "ready_for_training": not shortages,
    }
    write_json(report_root / "kd_dataset.json", report)
    if shortages and bool(config["monolingual"].get("require_minimum_accepted", True)):
        raise RuntimeError(f"Accepted v3 Teacher data is below the quality floor: {shortages}")
    _write_jsonl(base_rows + teacher_rows, output_root / "train.jsonl")
    _write_jsonl(_read_jsonl(base_validation_path), output_root / "validation.jsonl")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated ZH-UZ v3 monolingual augmentation.")
    parser.add_argument(
        "action", choices=("validate", "collect", "select_sources", "finalize")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "collect":
        result = collect(config, allow_config_extension=args.extend)
    else:
        result = {
            "validate": validate,
            "select_sources": select_sources,
            "finalize": finalize,
        }[args.action](config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
