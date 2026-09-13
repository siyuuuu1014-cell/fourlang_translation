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


def parse_translation_response(
    content: str, expected_ids: list[str]
) -> dict[str, str]:
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
            raise ValueError("DeepSeek response contains duplicate/empty id or translation")
        translations[key] = text
    if set(translations) != set(expected_ids):
        raise ValueError("DeepSeek response ids do not exactly match the request")
    return translations


def hard_check(row: dict[str, Any], translation: str, config: dict[str, Any]) -> list[str]:
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
        source_numbers = ARABIC_NUMBER_RE.findall(source)
        target_numbers = ARABIC_NUMBER_RE.findall(normalized)
        if sorted(source_numbers) != sorted(target_numbers):
            failures.append("ARABIC_NUMBER_MISMATCH")
    if quality.get("reject_excessive_repetition") and REPEATED_TOKEN_RE.search(normalized):
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
    config: dict[str, Any], input_path: Path, selected: list[dict[str, Any]], full: bool
) -> dict[str, Any]:
    generation_contract = {
        "api": config["api"],
        "quality": config["quality"],
        "system_prompt": SYSTEM_PROMPT,
    }
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
        "status": "running",
    }


def load_state(path: Path, selected: list[dict[str, Any]], signature: str) -> dict[str, dict[str, Any]]:
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
    limit = None if full else int(config["pilot"]["rows_per_direction"])
    selected = select_rows(rows, int(config["pipeline"]["seed"]), limit)
    manifest = manifest_payload(config, input_path, selected, full)
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


def generate(config: dict[str, Any], full: bool) -> dict[str, Any]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    input_path = project_path(config["input"]["candidates"])
    rows = read_jsonl(input_path)
    limit = None if full else int(config["pilot"]["rows_per_direction"])
    selected = select_rows(rows, int(config["pipeline"]["seed"]), limit)
    root = output_root(config, full)
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest_payload(config, input_path, selected, full)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("generation_fingerprint") != manifest["generation_fingerprint"] or existing.get("input_sha256") != manifest["input_sha256"]:
            raise RuntimeError("Existing run has different inputs or generation settings")
    else:
        write_json(manifest_path, manifest)
    checkpoint_path = root / "generations.jsonl"
    completed = load_state(
        checkpoint_path, selected, manifest["generation_fingerprint"]
    )
    batch_size = int(config["api"]["batch_size"])
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
            f"concurrency={config['api']['concurrency']}",
            flush=True,
        )
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(config["api"]["concurrency"])
    )
    recoverable_errors: list[str] = []
    try:
        futures = {
            executor.submit(request_batch, batch, config, api_key): batch
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "status", "generate"))
    parser.add_argument(
        "--config", default="configs/directions/zh_uz_deepseek_teacher_v1.toml"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Explicitly generate every selected source instead of the 400-row pilot.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    result = generate(config, args.full) if args.action == "generate" else plan(config, args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
