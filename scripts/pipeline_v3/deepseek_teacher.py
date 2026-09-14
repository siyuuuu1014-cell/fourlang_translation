"""Checkpointed DeepSeek API Teacher generation for the ZH-UZ pilot."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import load_config, read_json, write_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint  # noqa: E402
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)

PROJECT_ROOT = PROJECT_ROOT_BOOTSTRAP
DIRECTIONS = ("zh-uz", "uz-zh")
ARABIC_NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
HAN_RE = re.compile(r"[\u3400-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
REPEATED_TOKEN_RE = re.compile(r"(?i)(?:^|\s)([^\s]+)(?:\s+\1){3,}(?:\s|$)")
SCALED_RANGE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*[-–—~至到]\s*(\d+(?:[.,]\d+)?)\s*"
    r"-?\s*(千万|百万|十万|万|亿|ming|million|milliard)",
    re.IGNORECASE,
)
CHINESE_NUMBER_RE = re.compile(r"[零〇一二两三四五六七八九十百千万亿]+")
CHINESE_SCALE_AFTER_ARABIC_RE = re.compile(r"\s*(?:多\s*)?(千万|百万|十万|万|亿)")
LATIN_SCALE_AFTER_ARABIC_RE = re.compile(
    r"\s*(ming|million|milliard)(?:dan|ga|ni|ning|lik|lar)?\b",
    re.IGNORECASE,
)
UZ_MONTHS = {
    "yanvar": 1,
    "fevral": 2,
    "mart": 3,
    "aprel": 4,
    "may": 5,
    "iyun": 6,
    "iyul": 7,
    "avgust": 8,
    "sentabr": 9,
    "sentyabr": 9,
    "oktabr": 10,
    "oktyabr": 10,
    "noyabr": 11,
    "dekabr": 12,
}
UZ_NUMBER_PHRASES = {
    "o'n ikkinchi": 12,
    "o'n birinchi": 11,
    "birinchi": 1,
    "ikkinchi": 2,
    "uchinchi": 3,
    "to'rtinchi": 4,
    "beshinchi": 5,
    "oltinchi": 6,
    "yettinchi": 7,
    "sakkizinchi": 8,
    "to'qqizinchi": 9,
    "o'ninchi": 10,
    "yigirma": 20,
    "qirq": 40,
    "to'qqiz": 9,
    "sakkiz": 8,
    "yetti": 7,
    "olti": 6,
    "besh": 5,
    "to'rt": 4,
    "uch": 3,
    "ikki": 2,
    "bir": 1,
}
SOURCE_SPAM_PATTERNS = (
    (
        "GAMBLING_OR_BETTING",
        re.compile(
            r"(?i)(?:kazino|qimor|mostbet|bukmeker|pul\s+tikish|"
            r"tikish\s+(?:bozori|sayti|o'yin)|slot\s+(?:video\s+)?o'yin|"
            r"赌场|博彩|赌博|下注|投注|老虎机|真人娱乐场|幸运\s*8|六合彩)"
        ),
    ),
    (
        "SEO_OR_PROMO_PREFIX",
        re.compile(r"(?i)^\s*(?:top\s*\d+|澳洲幸运\s*\d+公式)"),
    ),
)
SCALE_VALUES = {
    "万": Decimal(10_000),
    "十万": Decimal(100_000),
    "百万": Decimal(1_000_000),
    "千万": Decimal(10_000_000),
    "亿": Decimal(100_000_000),
    "ming": Decimal(1_000),
    "million": Decimal(1_000_000),
    "milliard": Decimal(1_000_000_000),
}

SYSTEM_PROMPT = """You are a professional Chinese-Uzbek translator creating clean
knowledge-distillation data. Translate every item faithfully and completely. Preserve the
speaker, action, object, condition, negation, numbers, dates, units, names, and logical
relations. Use natural Simplified Chinese for target zh and natural Latin-script Uzbek for
target uz. Do not explain, summarize, add facts, omit facts, or copy the source. Return one
JSON object only in this exact shape:
{"items":[{"id":"input id","translation":"translation only"}]}
Return every input id exactly once and no additional ids."""


class FatalDeepSeekError(RuntimeError):
    """A request/configuration error that should stop queued paid requests."""


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def row_key(row: dict[str, Any]) -> str:
    return str(row["pair_id"])


def direction(row: dict[str, Any]) -> str:
    return f"{row['src_lang']}-{row['tgt_lang']}"


def _decimal_key(value: Decimal) -> str:
    normalized = value.normalize()
    return format(normalized, "f")


def _plain_number(token: str) -> Decimal | None:
    token = token.strip()
    if not token:
        return None
    if token.count(":") or token.count(".") + token.count(",") > 1:
        return None
    if "," in token:
        left, right = token.split(",", maxsplit=1)
        token = left + right if len(right) == 3 else left + "." + right
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def _token_values(token: str) -> list[Decimal]:
    if token.count(":") or token.count(".") + token.count(",") > 1:
        values = []
        for part in re.split(r"[.,:]", token):
            if part:
                values.append(Decimal(part))
        return values
    value = _plain_number(token)
    return [value] if value is not None else []


def _chinese_integer(token: str) -> int | None:
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1_000, "万": 10_000, "亿": 100_000_000}
    if all(character in digits for character in token):
        return int("".join(str(digits[character]) for character in token))
    total = section = number = 0
    for character in token:
        if character in digits:
            number = digits[character]
            continue
        unit = units.get(character)
        if unit is None:
            return None
        if unit < 10_000:
            section += (number or 1) * unit
        else:
            section = (section + number) * unit
            total += section
            section = 0
        number = 0
    return total + section + number


def _scaled_number_matches(text: str) -> tuple[list[Decimal], list[tuple[int, int]]]:
    values: list[Decimal] = []
    occupied: list[tuple[int, int]] = []
    for match in SCALED_RANGE_RE.finditer(text):
        first = _plain_number(match.group(1))
        second = _plain_number(match.group(2))
        scale = SCALE_VALUES[match.group(3).casefold()]
        if first is not None and second is not None:
            values.extend((first * scale, second * scale))
            occupied.append(match.span())
    return values, occupied


def _overlaps(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in occupied)


def numeric_values(
    text: str, language: str, *, include_language_words: bool = True
) -> Counter[str]:
    values, occupied = _scaled_number_matches(text)
    for match in ARABIC_NUMBER_RE.finditer(text):
        if _overlaps(match.span(), occupied):
            continue
        token_values = _token_values(match.group())
        suffix = text[match.end() : match.end() + 20]
        scale_match = (
            CHINESE_SCALE_AFTER_ARABIC_RE.match(suffix)
            if language == "zh"
            else LATIN_SCALE_AFTER_ARABIC_RE.match(suffix)
        )
        if scale_match and len(token_values) == 1:
            scale = SCALE_VALUES[scale_match.group(1).casefold()]
            token_values = [token_values[0] * scale]
            occupied.append((match.start(), match.end() + scale_match.end()))
        values.extend(token_values)
    if language == "zh" and include_language_words:
        for match in CHINESE_NUMBER_RE.finditer(text):
            if _overlaps(match.span(), occupied):
                continue
            parsed = _chinese_integer(match.group())
            if parsed is not None:
                values.append(Decimal(parsed))
        if "元旦" in text:
            values.append(Decimal(1))
    elif language == "uz" and include_language_words:
        lowered = text.casefold()
        for month, value in UZ_MONTHS.items():
            values.extend(
                Decimal(value)
                for _ in re.finditer(
                    rf"\b{re.escape(month)}(?:ning|dan|ida|iga|da|de|ga|ni)?\b",
                    lowered,
                )
            )
        for phrase, value in UZ_NUMBER_PHRASES.items():
            pattern = rf"(?<![\w']){re.escape(phrase)}(?:ta|tasi)?(?![\w'])"
            values.extend(Decimal(value) for _ in re.finditer(pattern, lowered))
        values.extend(
            Decimal(20) for _ in re.finditer(r"\bXX\s+asr", text, re.IGNORECASE)
        )
    return Counter(_decimal_key(value) for value in values)


def numbers_preserved(
    source: str, target: str, source_lang: str, target_lang: str
) -> bool:
    # Preserve the original check's high-precision scope: explicit Arabic
    # numbers in the source are mandatory. Language words are parsed only in
    # the translation so equivalent forms such as 6 -> "oltita" are accepted.
    required = numeric_values(source, source_lang, include_language_words=False)
    available = numeric_values(target, target_lang)
    if re.search(r"(?i)\b(?:km|cm|mm|m)2\b", source) and "平方" in target:
        required["2"] -= 1
    if re.search(r"(?i)\b(?:km|cm|mm|m)3\b", source) and "立方" in target:
        required["3"] -= 1
    return all(available[value] >= count for value, count in required.items())


def source_filter_reasons(text: str) -> list[str]:
    normalized = " ".join(str(text).split())
    return [
        reason for reason, pattern in SOURCE_SPAM_PATTERNS if pattern.search(normalized)
    ]


def select_for_mode(
    rows: list[dict[str, Any]], config: dict[str, Any], full: bool
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejections: Counter[str] = Counter()
    eligible = rows
    if full and config.get("source_filter", {}).get("enabled", True):
        eligible = []
        for row in rows:
            reasons = source_filter_reasons(str(row.get("src_text", "")))
            if reasons:
                rejections.update(reasons)
            else:
                eligible.append(row)
    if full:
        configured_limit = config.get("full", {}).get("rows_per_direction")
        limit = int(configured_limit) if configured_limit is not None else None
    else:
        limit = int(config["pilot"]["rows_per_direction"])
    return (
        select_rows(eligible, int(config["pipeline"]["seed"]), limit),
        rejections,
    )


def select_rows(
    rows: list[dict[str, Any]], seed: int, limit_per_direction: int | None
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {item: [] for item in DIRECTIONS}
    seen: set[str] = set()
    for row in rows:
        key = row_key(row)
        current = direction(row)
        if key in seen:
            raise RuntimeError(f"Duplicate pair_id in Teacher input: {key}")
        if current not in grouped:
            raise RuntimeError(f"Unexpected Teacher direction: {current}")
        if not str(row.get("src_text", "")).strip():
            raise RuntimeError(f"Empty source text: {key}")
        seen.add(key)
        grouped[current].append(row)
    selected = []
    for current in DIRECTIONS:
        candidates = sorted(
            grouped[current],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row_key(row)}".encode()
            ).hexdigest(),
        )
        if limit_per_direction is not None:
            if len(candidates) < limit_per_direction:
                raise RuntimeError(
                    f"{current} has only {len(candidates)}/{limit_per_direction} rows"
                )
            candidates = candidates[:limit_per_direction]
        selected.extend(candidates)
    return selected


def parse_translation_response(content: str, expected_ids: list[str]) -> dict[str, str]:
    payload = json.loads(content)
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("DeepSeek JSON response has no items array")
    translations: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("DeepSeek response item is not an object")
        key = str(item.get("id", ""))
        text = str(item.get("translation", "")).strip()
        if key in translations or not key or not text:
            raise ValueError(
                "DeepSeek response contains duplicate/empty id or translation"
            )
        translations[key] = text
    if set(translations) != set(expected_ids):
        raise ValueError("DeepSeek response ids do not exactly match the request")
    return translations


def hard_check(
    row: dict[str, Any], translation: str, config: dict[str, Any]
) -> list[str]:
    failures = []
    source = str(row["src_text"]).strip()
    target = str(row["tgt_lang"])
    quality = config["quality"]
    try:
        normalized = normalize_language_text(target, translation)
    except ValueError:
        normalized = translation.strip()
        failures.append("TARGET_TEXT_NORMALIZATION_FAILED")
    if not normalized:
        failures.append("EMPTY_TRANSLATION")
    if quality.get("reject_source_copy") and normalized.casefold() == source.casefold():
        failures.append("SOURCE_COPIED")
    if quality.get("require_arabic_numbers"):
        if not numbers_preserved(
            source,
            normalized,
            str(row["src_lang"]),
            target,
        ):
            failures.append("ARABIC_NUMBER_MISMATCH")
    if quality.get("reject_excessive_repetition") and REPEATED_TOKEN_RE.search(
        normalized
    ):
        failures.append("EXCESSIVE_REPETITION")
    if quality.get("require_target_script_signal"):
        if target == "zh" and not HAN_RE.search(normalized):
            failures.append("ZH_SCRIPT_SIGNAL_MISSING")
        if target == "uz" and (
            CYRILLIC_RE.search(translation) or not LATIN_RE.search(normalized)
        ):
            failures.append("UZ_LATIN_SIGNAL_MISSING")
    return sorted(set(failures))


def normalized_teacher_text(target: str, translation: str) -> str:
    try:
        return normalize_language_text(target, translation)
    except ValueError:
        return translation.strip()


def request_batch(
    rows: list[dict[str, Any]], config: dict[str, Any], api_key: str
) -> dict[str, Any]:
    api = config["api"]
    current_direction = direction(rows[0])
    if any(direction(row) != current_direction for row in rows):
        raise ValueError("One API request may contain only one translation direction")
    items = [
        {
            "id": row_key(row),
            "source_language": row["src_lang"],
            "target_language": row["tgt_lang"],
            "text": row["src_text"],
        }
        for row in rows
    ]
    body: dict[str, Any] = {
        "model": api["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Translate this JSON input:\n"
                + json.dumps({"items": items}, ensure_ascii=False),
            },
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": int(api["max_tokens"]),
        "temperature": float(api["temperature"]),
        "thinking": {"type": "disabled" if not api.get("thinking") else "enabled"},
    }
    last_error: Exception | None = None
    for attempt in range(int(api["max_retries"])):
        try:
            response = httpx.post(
                str(api["base_url"]).rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=float(api["timeout_seconds"]),
            )
            if 400 <= response.status_code < 500 and response.status_code not in {
                408,
                429,
            }:
                raise FatalDeepSeekError(
                    f"DeepSeek rejected the request with HTTP {response.status_code}"
                )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            translations = parse_translation_response(
                content, [row_key(row) for row in rows]
            )
            return {
                "response_id": str(payload.get("id", "")),
                "system_fingerprint": str(payload.get("system_fingerprint", "")),
                "model": str(payload.get("model", api["model"])),
                "usage": payload.get("usage", {}),
                "translations": translations,
            }
        except FatalDeepSeekError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            last_error = error
            if attempt + 1 < int(api["max_retries"]):
                time.sleep(min(2**attempt, 8) + random.random())
    raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")


def output_root(config: dict[str, Any], full: bool) -> Path:
    return project_path(config["output"]["root"]) / ("full" if full else "pilot")


def manifest_payload(
    config: dict[str, Any],
    input_path: Path,
    selected: list[dict[str, Any]],
    full: bool,
    source_filter_rejections: Counter[str] | None = None,
) -> dict[str, Any]:
    generation_contract = {
        "api": config["api"],
        "quality": config["quality"],
        "system_prompt": SYSTEM_PROMPT,
    }
    if full:
        generation_contract["source_filter"] = config.get("source_filter", {})
        generation_contract["full_selection"] = config.get("full", {})
    return {
        "schema_version": 1,
        "mode": "full" if full else "pilot",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "selected_rows": len(selected),
        "by_direction": {
            item: sum(direction(row) == item for row in selected) for item in DIRECTIONS
        },
        "generation_contract": generation_contract,
        "generation_fingerprint": fingerprint(generation_contract),
        "source_filter_rejections": dict(source_filter_rejections or {}),
        "status": "running",
    }


def load_state(
    path: Path, selected: list[dict[str, Any]], signature: str
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    allowed = {row_key(row) for row in selected}
    completed: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = row_key(row)
        if key not in allowed:
            raise RuntimeError(f"Checkpoint contains an unexpected row: {key}")
        if row.get("generation_fingerprint") != signature:
            raise RuntimeError("Checkpoint generation settings have changed")
        if key in completed:
            raise RuntimeError(f"Checkpoint contains a duplicate row: {key}")
        completed[key] = row
    return completed


def plan(config: dict[str, Any], full: bool) -> dict[str, Any]:
    input_path = project_path(config["input"]["candidates"])
    rows = read_jsonl(input_path)
    selected, source_rejections = select_for_mode(rows, config, full)
    manifest = manifest_payload(config, input_path, selected, full, source_rejections)
    root = output_root(config, full)
    completed = load_state(
        root / "generations.jsonl",
        selected,
        manifest["generation_fingerprint"],
    )
    pending = len(selected) - len(completed)
    return {
        **manifest,
        "status": (
            "completed" if not pending else "running" if completed else "not_started"
        ),
        "completed_rows": len(completed),
        "pending_rows": pending,
        "estimated_pending_requests": math.ceil(
            pending / int(config["api"]["batch_size"])
        ),
        "output": str(root),
    }


def with_runtime_overrides(
    config: dict[str, Any],
    *,
    batch_size_override: int | None = None,
    concurrency_override: int | None = None,
) -> dict[str, Any]:
    if batch_size_override is not None and batch_size_override < 1:
        raise ValueError("batch_size_override must be at least 1")
    if concurrency_override is not None and concurrency_override < 1:
        raise ValueError("concurrency_override must be at least 1")
    runtime_config = {**config, "api": dict(config["api"])}
    if batch_size_override is not None:
        runtime_config["api"]["batch_size"] = batch_size_override
    if concurrency_override is not None:
        runtime_config["api"]["concurrency"] = concurrency_override
    return runtime_config


def generate(
    config: dict[str, Any],
    full: bool,
    *,
    batch_size_override: int | None = None,
    concurrency_override: int | None = None,
) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    runtime_config = with_runtime_overrides(
        config,
        batch_size_override=batch_size_override,
        concurrency_override=concurrency_override,
    )

    input_path = project_path(config["input"]["candidates"])
    rows = read_jsonl(input_path)
    selected, source_rejections = select_for_mode(rows, config, full)
    root = output_root(config, full)
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_payload(config, input_path, selected, full, source_rejections)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if (
            existing.get("generation_fingerprint") != manifest["generation_fingerprint"]
            or existing.get("input_sha256") != manifest["input_sha256"]
        ):
            raise RuntimeError(
                "Existing run has different inputs or generation settings"
            )
    else:
        write_json(manifest_path, manifest)
    checkpoint_path = root / "generations.jsonl"
    completed = load_state(
        checkpoint_path, selected, manifest["generation_fingerprint"]
    )
    batch_size = int(runtime_config["api"]["batch_size"])
    pending = [row for row in selected if row_key(row) not in completed]
    batches = []
    for current in DIRECTIONS:
        direction_rows = [row for row in pending if direction(row) == current]
        batches.extend(
            direction_rows[start : start + batch_size]
            for start in range(0, len(direction_rows), batch_size)
        )
    if pending:
        print(
            f"pending={len(pending)} batches={len(batches)} "
            f"concurrency={runtime_config['api']['concurrency']}",
            flush=True,
        )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(runtime_config["api"]["concurrency"])
    )
    recoverable_errors: list[str] = []
    try:
        futures = {
            executor.submit(request_batch, batch, runtime_config, api_key): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                response = future.result()
            except FatalDeepSeekError:
                for pending_future in futures:
                    pending_future.cancel()
                raise
            except RuntimeError as error:
                recoverable_errors.append(
                    f"{direction(batch[0])}:{row_key(batch[0])}: {error}"
                )
                continue
            generated = []
            for index, row in enumerate(batch):
                key = row_key(row)
                teacher_text = response["translations"][key]
                failures = hard_check(row, teacher_text, config)
                generated.append(
                    {
                        **row,
                        "teacher_text": normalized_teacher_text(
                            str(row["tgt_lang"]), teacher_text
                        ),
                        "teacher_id": str(config["api"]["model"]),
                        "api_response_id": response["response_id"],
                        "api_system_fingerprint": response["system_fingerprint"],
                        "hard_check_pass": not failures,
                        "hard_check_failures_json": json.dumps(failures),
                        "api_usage_json": (
                            json.dumps(response["usage"], sort_keys=True)
                            if index == 0
                            else ""
                        ),
                        "generation_fingerprint": manifest["generation_fingerprint"],
                    }
                )
            append_jsonl(checkpoint_path, generated)
            completed.update({row_key(row): row for row in generated})
            print(f"completed={len(completed)}/{len(selected)}", flush=True)
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    if recoverable_errors:
        raise RuntimeError(
            f"{len(recoverable_errors)} DeepSeek batches failed; successful batches were "
            "checkpointed. Rerun to resume. First error: " + recoverable_errors[0]
        )
    ordered = [completed[row_key(row)] for row in selected]
    frame = pd.DataFrame(ordered)
    parquet_path = root / "teacher_generated.parquet"
    temporary = parquet_path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(parquet_path)
    failures = frame[~frame["hard_check_pass"].astype(bool)]
    usage_totals: dict[str, int] = {}
    for value in frame["api_usage_json"]:
        if not value:
            continue
        for key, amount in json.loads(value).items():
            if isinstance(amount, int):
                usage_totals[key] = usage_totals.get(key, 0) + amount
    report = {
        **manifest,
        "status": "completed",
        "completed_rows": len(frame),
        "hard_check_pass_rows": int(frame["hard_check_pass"].sum()),
        "hard_check_reject_rows": len(failures),
        "hard_check_reasons": json.loads(
            failures["hard_check_failures_json"].value_counts().to_json()
        ),
        "api_usage_totals": usage_totals,
        "teacher_generated": str(parquet_path),
        "checkpoint": str(checkpoint_path),
    }
    write_json(root / "report.json", report)
    write_json(manifest_path, {**manifest, "status": "completed"})
    return report


def _reason_counts(values: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        counts.update(json.loads(str(value)))
    return dict(sorted(counts.items()))


def reaudit(config: dict[str, Any], full: bool) -> dict[str, Any]:
    root = output_root(config, full)
    generated_path = root / "teacher_generated.parquet"
    if not generated_path.is_file():
        raise FileNotFoundError(generated_path)
    frame = pd.read_parquet(generated_path)
    required = {"pair_id", "src_lang", "tgt_lang", "src_text", "teacher_text"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Teacher output is missing columns: {sorted(missing)}")

    audited_rows = []
    for raw in frame.to_dict(orient="records"):
        translation_failures = hard_check(raw, str(raw["teacher_text"]), config)
        source_failures = (
            source_filter_reasons(str(raw["src_text"]))
            if config.get("source_filter", {}).get("enabled", True)
            else []
        )
        audited_rows.append(
            {
                **raw,
                "original_hard_check_pass": bool(raw.get("hard_check_pass", False)),
                "revised_translation_pass": not translation_failures,
                "revised_translation_failures_json": json.dumps(translation_failures),
                "source_filter_pass": not source_failures,
                "source_filter_failures_json": json.dumps(source_failures),
                "combined_usable": not translation_failures and not source_failures,
            }
        )
    audited = pd.DataFrame(audited_rows)
    parquet_path = root / "reaudit.parquet"
    parquet_temporary = parquet_path.with_suffix(".parquet.tmp")
    audited.to_parquet(parquet_temporary, index=False)
    parquet_temporary.replace(parquet_path)
    jsonl_path = root / "reaudit.jsonl"
    jsonl_temporary = jsonl_path.with_suffix(".jsonl.tmp")
    audited.to_json(
        jsonl_temporary,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    jsonl_temporary.replace(jsonl_path)

    directions: dict[str, dict[str, Any]] = {}
    for current in DIRECTIONS:
        subset = audited[
            (audited["src_lang"].astype(str) + "-" + audited["tgt_lang"].astype(str))
            == current
        ]
        directions[current] = {
            "rows": len(subset),
            "original_hard_pass_rows": int(subset["original_hard_check_pass"].sum()),
            "revised_translation_pass_rows": int(
                subset["revised_translation_pass"].sum()
            ),
            "source_filter_pass_rows": int(subset["source_filter_pass"].sum()),
            "combined_usable_rows": int(subset["combined_usable"].sum()),
            "combined_usable_rate": (
                float(subset["combined_usable"].mean()) if len(subset) else 0.0
            ),
        }
    report = {
        "schema_version": 1,
        "status": "REAUDIT_READY_SEMANTIC_REVIEW_REQUIRED",
        "mode": "full" if full else "pilot",
        "input": str(generated_path),
        "rows": len(audited),
        "directions": directions,
        "revised_translation_failure_reasons": _reason_counts(
            audited["revised_translation_failures_json"]
        ),
        "source_filter_failure_reasons": _reason_counts(
            audited["source_filter_failures_json"]
        ),
        "reaudit_parquet": str(parquet_path),
        "reaudit_jsonl": str(jsonl_path),
        "original_teacher_output_modified": False,
        "api_called": False,
        "manual_review_note": (
            "Rule checks cannot detect every entity or meaning error; preserve the "
            "completed 100-row semantic audit as selection evidence."
        ),
    }
    write_json(root / "reaudit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "generate", "reaudit"))
    parser.add_argument(
        "--config", default="configs/directions/zh_uz_deepseek_teacher_v1.toml"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly generate every selected source instead of the 400-row pilot.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=(
            "Runtime request batch-size override. It preserves the existing "
            "generation checkpoint."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help=(
            "Runtime request concurrency override. It preserves the existing "
            "generation checkpoint."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "generate":
        result = generate(
            config,
            args.full,
            batch_size_override=args.batch_size,
            concurrency_override=args.concurrency,
        )
    elif args.action == "reaudit":
        result = reaudit(config, args.full)
    else:
        result = plan(config, args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
