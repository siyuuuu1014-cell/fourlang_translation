from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .common import PROJECT_ROOT, load_config, pair_info
except ImportError:
    from common import PROJECT_ROOT, load_config, pair_info

LABELS = {"PASS", "MINOR", "FAIL", "UNCERTAIN"}
USEFULNESS = {"HIGH", "MEDIUM", "LOW", "REJECT"}
JUDGE_SCHEMA_VERSION = 2


def parse_result(text: str, *, teacher: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "judge_parse_ok": False,
        "judge_label": "UNCERTAIN",
        "judge_reason": "unparseable",
        "semantic_consistent": False,
        "omission": False,
        "addition": False,
        "mistranslation": False,
        "number_error": False,
        "entity_error": False,
        "negation_error": False,
    }
    if teacher:
        result["teacher_usefulness"] = "REJECT"
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return result
    try:
        payload = json.loads(match.group(0))
        label = str(payload["label"]).upper()
        if label not in LABELS:
            return result
        if teacher and "teacher_usefulness" not in payload:
            return result
        usefulness = str(payload.get("teacher_usefulness", "REJECT")).upper()
        if teacher and usefulness not in USEFULNESS:
            return result
        result.update(
            {
                "judge_parse_ok": True,
                "judge_label": label,
                "judge_reason": str(payload.get("reason", ""))[:500],
                "semantic_consistent": bool(
                    payload.get("semantic_consistent", label in {"PASS", "MINOR"})
                ),
                "omission": bool(payload.get("omission", False)),
                "addition": bool(payload.get("addition", False)),
                "mistranslation": bool(payload.get("mistranslation", False)),
                "number_error": bool(payload.get("number_error", False)),
                "entity_error": bool(payload.get("entity_error", False)),
                "negation_error": bool(payload.get("negation_error", False)),
            }
        )
        if teacher:
            result["teacher_usefulness"] = (
                "REJECT" if label in {"FAIL", "UNCERTAIN"} else usefulness
            )
    except (TypeError, KeyError, json.JSONDecodeError):
        return result
    return result


def prompt(mode: str, src_lang: str, tgt_lang: str, source: str, target: str) -> str:
    second = mode == "human_second"
    teacher = mode == "teacher"
    origin = "a translation Teacher" if teacher else "a human parallel corpus"
    independence = (
        "This is an independent second review. Ignore any possible earlier verdict. "
        if second
        else ""
    )
    extra = ', "teacher_usefulness": "HIGH|MEDIUM|LOW|REJECT"' if teacher else ""
    return f"""You are a strict bilingual translation quality auditor. {independence}
The candidate comes from {origin}. Compare meaning, omissions, additions, fluency, language,
names, numbers, time expressions and negation. Do not rewrite the translation.
PASS means fully usable. MINOR means usable with a small non-substantive issue.
FAIL means a substantive error. UNCERTAIN means it cannot be judged reliably.
Source language: {src_lang}
Target language: {tgt_lang}
Source: {source}
Candidate translation: {target}
Return one JSON object only:
{{"label":"PASS|MINOR|FAIL|UNCERTAIN","semantic_consistent":true,"omission":false,
"addition":false,"mistranslation":false,"number_error":false,"entity_error":false,
"negation_error":false{extra},"reason":"short reason"}}"""


def io_paths(config: dict[str, Any], mode: str, calibration: bool) -> tuple[Path, Path]:
    pair, _, _, _ = pair_info(config)
    base = PROJECT_ROOT / "data" / "pipeline_v2" / pair
    if mode == "human":
        return base / "human_review_input.parquet", base / "human_judged.parquet"
    if mode == "human_second":
        return base / "human_judged.parquet", base / "human_second_review.parquet"
    return base / "teacher_generated.parquet", base / (
        "teacher_judge_calibration.parquet" if calibration else "teacher_judged.parquet"
    )


def judge_id(row: dict[str, Any]) -> str:
    return f"{row['pair_id']}:{row.get('src_lang', row.get('source_lang'))}:{row.get('tgt_lang', row.get('target_lang'))}"


def second_review_mask(frame: pd.DataFrame) -> pd.Series:
    return (~frame["judge_parse_ok"].fillna(False).astype(bool)) | frame[
        "judge_label"
    ].fillna("UNCERTAIN").astype(str).str.upper().isin(["FAIL", "UNCERTAIN"])


def save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("judge_id", keep="last").to_parquet(
        path, index=False
    )


def render_prompt(tokenizer: Any, mode: str, record: dict[str, Any]) -> str:
    source = str(record.get("src_text", record.get("source_text", "")))
    target = str(record.get("teacher_text", record.get("target_text", "")))
    src_lang = str(record.get("src_lang", record.get("source_lang", "")))
    tgt_lang = str(record.get("tgt_lang", record.get("target_lang", "")))
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt(mode, src_lang, tgt_lang, source, target),
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    rendered_prompts: list[str],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven Qwen translation Judge."
    )
    parser.add_argument("mode", choices=("human", "human_second", "teacher"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_path, output_path = io_paths(config, args.mode, args.calibration)
    frame = pd.read_parquet(source_path)
    if args.mode == "human_second":
        frame = frame[second_review_mask(frame)].copy()
    if args.calibration:
        count = min(int(config["judge"]["teacher_calibration_pairs"]), len(frame))
        frame = frame.sample(
            n=count, random_state=int(config["direction"]["seed"])
        ).sort_values("pair_id")
    frame["judge_id"] = [judge_id(row) for row in frame.to_dict("records")]
    if output_path.exists() and not args.overwrite:
        existing = pd.read_parquet(output_path)
        required_columns = {
            "judge_id",
            "judge_parse_ok",
            "judge_label",
            "judge_schema_version",
        }
        if args.mode == "teacher":
            required_columns.add("teacher_usefulness")
        if not required_columns.issubset(existing.columns):
            existing = existing.head(0)
        completed = (
            set(
                existing.loc[
                    existing["judge_parse_ok"]
                    & (existing["judge_schema_version"] == JUDGE_SCHEMA_VERSION),
                    "judge_id",
                ].astype(str)
            )
            if len(existing)
            else set()
        )
        pending = frame[~frame["judge_id"].isin(completed)]
        result = existing.to_dict("records")
    else:
        pending = frame
        result = []
    if pending.empty:
        if result:
            save(result, output_path)
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(output_path, index=False)
        return
    model_path = str(config["judge"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    pending_records = pending.to_dict("records")
    initial_batch_size = int(config["judge"]["batch_size"])
    if initial_batch_size < 1:
        raise ValueError("judge.batch_size must be at least 1.")
    current_batch_size = initial_batch_size
    max_input_tokens = int(config["judge"].get("max_input_tokens", 1536))
    max_new_tokens = int(config["judge"]["max_new_tokens"])
    processed = 0
    last_saved = 0
    print(
        f"pending={len(pending_records)} initial_batch_size={initial_batch_size} "
        f"max_input_tokens={max_input_tokens}"
    )
    while processed < len(pending_records):
        batch = pending_records[processed : processed + current_batch_size]
        rendered = [render_prompt(tokenizer, args.mode, record) for record in batch]
        try:
            answers = generate_batch(
                model,
                tokenizer,
                rendered,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            print(f"CUDA OOM: reducing batch_size to {current_batch_size}")
            continue
        for record, answer in zip(batch, answers, strict=True):
            record.update(parse_result(answer, teacher=args.mode == "teacher"))
            record["judge_schema_version"] = JUDGE_SCHEMA_VERSION
            record["judge_raw"] = answer[:2000]
            result.append(record)
        processed += len(batch)
        if processed - last_saved >= 100 or processed == len(pending_records):
            save(result, output_path)
            last_saved = processed
            print(
                f"judged {processed}/{len(pending_records)} "
                f"batch_size={current_batch_size}"
            )
    save(result, output_path)


if __name__ == "__main__":
    main()
