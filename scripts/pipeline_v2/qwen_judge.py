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
    from .common import PROJECT_ROOT, load_config, pipeline_namespace
except ImportError:
    from common import PROJECT_ROOT, load_config, pipeline_namespace

LABELS = {"PASS", "MINOR", "FAIL", "UNCERTAIN"}
USEFULNESS = {"HIGH", "MEDIUM", "LOW", "REJECT"}
JUDGE_SCHEMA_VERSION = 2


def parse_result(
    text: str, *, teacher: bool = False, allow_minor: bool = True
) -> dict[str, Any]:
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
        forced_uncertain = label == "MINOR" and not allow_minor
        if forced_uncertain:
            label = "UNCERTAIN"
        if teacher and "teacher_usefulness" not in payload:
            return result
        usefulness = str(payload.get("teacher_usefulness", "REJECT")).upper()
        if teacher and usefulness not in USEFULNESS:
            return result
        result.update(
            {
                "judge_parse_ok": True,
                "judge_label": label,
                "judge_reason": (
                    "strict second review returned disallowed MINOR verdict"
                    if forced_uncertain
                    else str(payload.get("reason", ""))[:500]
                ),
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


def prompt(
    mode: str,
    src_lang: str,
    tgt_lang: str,
    source: str,
    target: str,
    *,
    source_policy: str = "strict_v1",
) -> str:
    if mode == "source":
        language_name = "Simplified Mandarin Chinese" if src_lang == "zh" else "Latin-script Uzbek"
        if source_policy == "natural_sentence_entities_allowed_v2":
            return f"""You are a careful native-language corpus curator.
Judge whether the text is a complete, coherent, natural sentence in {language_name}.
Proper names, place names, organization and product names, established loanwords, Latin
acronyms, Arabic numerals, dates, measurements, citations used inside a real sentence,
and ordinary Unicode punctuation are allowed. For Chinese, do not reject a sentence only
because it contains a legitimate Latin-script name or acronym. For Uzbek, do not call text
Cyrillic unless it actually contains Cyrillic letters; Latin-script foreign names and
historical names are allowed. Do not reject a natural encyclopedic definition merely for
containing parenthetical information or foreign entities.
Reject fragments, headings, keyword/entity lists, navigation, advertisements, gambling/SEO
copy, disclaimers, citation debris without a sentence, corrupted text, substantial
code-switching, wrong-language text, and clearly unnatural machine-translated wording.
Judge the sentence as written; do not correct or rewrite it. PASS means fully usable as a
translation source. FAIL requires a concrete substantive defect. UNCERTAIN means native-level
quality genuinely cannot be established. Do not use foreign names, numbers, or punctuation
alone as the reason for FAIL.
Language: {src_lang}
Text: {source}
Return one JSON object only:
{{"label":"PASS|FAIL|UNCERTAIN","reason":"short, concrete reason"}}"""
        if source_policy != "strict_v1":
            raise ValueError(f"Unsupported source Judge policy: {source_policy!r}")
        return f"""You are a strict native-language corpus curator.
Judge whether the text is a complete, coherent, natural sentence in {language_name}.
Reject text in another language, mixed-language text, keyword or entity lists, navigation or
SEO fragments, disclaimers and boilerplate, citation debris, corrupted text, and unnatural
machine-translated wording. Do not correct or rewrite the text.
PASS means fully suitable as a clean monolingual translation source.
FAIL means unsuitable. UNCERTAIN means native-level quality cannot be established reliably.
Language: {src_lang}
Text: {source}
Return one JSON object only:
{{"label":"PASS|FAIL|UNCERTAIN","reason":"short reason"}}"""
    second = mode in {"human_second", "teacher_second"}
    teacher = mode in {"teacher", "teacher_second"}
    origin = "a translation Teacher" if teacher else "a human parallel corpus"
    independence = (
        "This is an independent second review. Ignore any possible earlier verdict. "
        if second
        else ""
    )
    extra = ', "teacher_usefulness": "HIGH|MEDIUM|LOW|REJECT"' if teacher else ""
    language_contract = []
    if "zh" in {src_lang, tgt_lang}:
        language_contract.append(
            "Chinese must be standard Simplified Mandarin, not Traditional Chinese or Cantonese."
        )
    if "uz" in {src_lang, tgt_lang}:
        language_contract.append(
            "Uzbek must use the Latin script, not Cyrillic."
        )
    contract_text = " ".join(language_contract)
    verdict_contract = (
        "This is a strict adjudication of a previously borderline Teacher translation. "
        "Return PASS only when it is accurate, complete, sufficiently natural, and safe for "
        "knowledge-distillation training. Alternative wording alone is not an error. Return "
        "FAIL for any concrete semantic, omission, addition, terminology, entity, number, "
        "negation, truncation, or target-language defect. Return UNCERTAIN only when reliable "
        "judgment is impossible. Do not return MINOR."
        if mode == "teacher_second"
        else "PASS means fully usable. MINOR means usable with a small non-substantive issue. "
        "FAIL means a substantive error. UNCERTAIN means it cannot be judged reliably."
    )
    return f"""You are a strict bilingual translation quality auditor. {independence}
The candidate comes from {origin}. Compare meaning, omissions, additions, fluency, language,
names, numbers, time expressions and negation. Do not rewrite the translation.
{contract_text}
For Teacher candidates, also reject when the source itself is in the wrong language,
is a keyword/list fragment, boilerplate, corrupted, or not a coherent natural sentence.
{verdict_contract}
Source language: {src_lang}
Target language: {tgt_lang}
Source: {source}
Candidate translation: {target}
Return one JSON object only:
{{"label":"PASS|MINOR|FAIL|UNCERTAIN","semantic_consistent":true,"omission":false,
"addition":false,"mistranslation":false,"number_error":false,"entity_error":false,
"negation_error":false{extra},"reason":"short reason"}}"""


def io_paths(config: dict[str, Any], mode: str, calibration: bool) -> tuple[Path, Path]:
    base = PROJECT_ROOT / "data" / "pipeline_v2" / pipeline_namespace(config)
    if mode == "source":
        return base / "monolingual_candidates.jsonl", base / (
            "source_judge_calibration.parquet"
            if calibration
            else "source_judged.parquet"
        )
    if mode == "human":
        return base / "human_review_input.parquet", base / "human_judged.parquet"
    if mode == "human_second":
        return base / "human_judged.parquet", base / "human_second_review.parquet"
    if mode == "teacher_second":
        return base / (
            "teacher_judge_calibration.parquet"
            if calibration
            else "teacher_judged.parquet"
        ), base / (
            "teacher_minor_second_review_calibration.parquet"
            if calibration
            else "teacher_minor_second_review.parquet"
        )
    return base / "teacher_generated.parquet", base / (
        "teacher_judge_calibration.parquet" if calibration else "teacher_judged.parquet"
    )


def judge_id(row: dict[str, Any]) -> str:
    return f"{row['pair_id']}:{row.get('src_lang', row.get('source_lang'))}:{row.get('tgt_lang', row.get('target_lang'))}"


def second_review_mask(frame: pd.DataFrame) -> pd.Series:
    return (~frame["judge_parse_ok"].fillna(False).astype(bool)) | frame[
        "judge_label"
    ].fillna("UNCERTAIN").astype(str).str.upper().isin(["FAIL", "UNCERTAIN"])


def teacher_minor_review_mask(frame: pd.DataFrame) -> pd.Series:
    """Only send semantically clean first-pass MINOR rows to strict adjudication."""
    mask = (
        frame["judge_parse_ok"].fillna(False).astype(bool)
        & frame["judge_label"].fillna("").astype(str).str.upper().eq("MINOR")
        & frame["semantic_consistent"].fillna(False).astype(bool)
    )
    for field in (
        "omission",
        "addition",
        "mistranslation",
        "number_error",
        "entity_error",
        "negation_error",
    ):
        mask &= ~frame[field].fillna(False).astype(bool)
    return mask


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _same_input(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return str(left) == str(right)


def reusable_judgments(
    frame: pd.DataFrame, paths: list[Path], mode: str
) -> pd.DataFrame:
    """Load only judgments whose complete audited input is unchanged."""
    expected = {
        str(row["judge_id"]): row for row in frame.to_dict("records")
    }
    matched: list[dict[str, Any]] = []
    teacher = mode == "teacher"
    fields = ["pair_id", "src_lang", "tgt_lang", "src_text"]
    if teacher:
        fields.extend(["teacher_text", "teacher_id"])
    for path in paths:
        if not path.is_file():
            continue
        previous = pd.read_parquet(path)
        if "judge_id" not in previous.columns:
            continue
        for row in previous.to_dict("records"):
            current = expected.get(str(row.get("judge_id", "")))
            if current is None:
                continue
            if all(_same_input(row.get(field), current.get(field)) for field in fields):
                matched.append(row)
    if not matched:
        return frame.head(0).copy()
    return pd.DataFrame(matched).drop_duplicates("judge_id", keep="last")


def stratified_source_sample(
    frame: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    if count >= len(frame):
        return frame.sort_values("pair_id")
    groups = list(
        frame.groupby(["src_lang", "source_corpus"], sort=True, dropna=False).groups.items()
    )
    per_group = max(1, count // len(groups))
    selected: list[Any] = []
    for offset, (_, indices) in enumerate(groups):
        group_indices = list(indices)
        take = min(per_group, len(group_indices))
        selected.extend(
            frame.loc[group_indices].sample(n=take, random_state=seed + offset).index
        )
    remaining = count - len(selected)
    if remaining > 0:
        pool = frame.drop(index=selected)
        selected.extend(pool.sample(n=remaining, random_state=seed + 10_000).index)
    return frame.loc[selected].sort_values("pair_id")


def save(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop_duplicates("judge_id", keep="last").to_parquet(
        path, index=False
    )


def render_prompt(
    tokenizer: Any,
    mode: str,
    record: dict[str, Any],
    *,
    source_policy: str = "strict_v1",
) -> str:
    source = str(record.get("src_text", record.get("source_text", "")))
    target = str(record.get("teacher_text", record.get("target_text", "")))
    src_lang = str(record.get("src_lang", record.get("source_lang", "")))
    tgt_lang = str(record.get("tgt_lang", record.get("target_lang", "")))
    return tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt(
                    mode,
                    src_lang,
                    tgt_lang,
                    source,
                    target,
                    source_policy=source_policy,
                ),
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
    parser.add_argument(
        "mode",
        choices=("source", "human", "human_second", "teacher", "teacher_second"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    source_policy = str(config.get("judge", {}).get("source_policy", "strict_v1"))
    source_path, output_path = io_paths(config, args.mode, args.calibration)
    if source_path.suffix == ".jsonl":
        frame = pd.DataFrame(
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        frame = pd.read_parquet(source_path)
    if args.mode == "human_second":
        frame = frame[second_review_mask(frame)].copy()
    if args.mode == "teacher_second":
        frame = frame[teacher_minor_review_mask(frame)].copy()
    if args.calibration:
        count_key = (
            "source_calibration_pairs"
            if args.mode == "source"
            else "teacher_calibration_pairs"
        )
        count = min(int(config["judge"][count_key]), len(frame))
        if args.mode == "source":
            frame = stratified_source_sample(
                frame, count, int(config["direction"]["seed"])
            )
        else:
            frame = frame.sample(
                n=count, random_state=int(config["direction"]["seed"])
            ).sort_values("pair_id")
    frame["judge_id"] = [judge_id(row) for row in frame.to_dict("records")]
    frame["judge_policy"] = (
        source_policy if args.mode == "source" else "translation_v1"
    )
    existing_parts: list[pd.DataFrame] = []
    if output_path.exists() and not args.overwrite:
        existing_parts.append(pd.read_parquet(output_path))
    if not args.overwrite and args.mode in {"source", "teacher"}:
        reuse_key = "source_judged" if args.mode == "source" else "teacher_judged"
        configured = config.get("reuse", {}).get(reuse_key, [])
        if isinstance(configured, (str, Path)):
            configured = [configured]
        reuse_paths = [_project_path(value) for value in configured]
        if args.mode == "teacher" and not args.calibration:
            calibration_path = output_path.with_name("teacher_judge_calibration.parquet")
            reuse_paths.append(calibration_path)
        reused = reusable_judgments(frame, reuse_paths, args.mode)
        if len(reused):
            print(f"Reusing {len(reused)} compatible {args.mode} judgments.")
            existing_parts.insert(0, reused)
    if existing_parts:
        existing = pd.concat(existing_parts, ignore_index=True).drop_duplicates(
            "judge_id", keep="last"
        )
        required_columns = {
            "judge_id",
            "judge_parse_ok",
            "judge_label",
            "judge_schema_version",
            "judge_policy",
        }
        if args.mode in {"teacher", "teacher_second"}:
            required_columns.add("teacher_usefulness")
        if not required_columns.issubset(existing.columns):
            existing = existing.head(0)
        else:
            # Input pools can be refined between resumable review runs. Never
            # retain a verdict for a row that is no longer in the current pool.
            existing = existing[
                existing["judge_id"].isin(set(frame["judge_id"].astype(str)))
                & existing["judge_policy"].eq(frame["judge_policy"].iloc[0])
            ].copy()
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
    initial_batch_size = int(config["judge"]["batch_size"])
    if initial_batch_size < 1:
        raise ValueError("judge.batch_size must be at least 1.")
    max_input_tokens = int(config["judge"].get("max_input_tokens", 1536))
    max_new_tokens = int(config["judge"]["max_new_tokens"])
    model_device = str(config["judge"].get("model_device", "auto")).lower()
    if model_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("judge.model_device=cuda but CUDA is unavailable")
        device_map: str | dict[str, int] = {"": 0}
    elif model_device == "auto":
        device_map = "auto"
    else:
        raise ValueError("judge.model_device must be 'auto' or 'cuda'")
    print(
        f"pending={len(pending)} initial_batch_size={initial_batch_size} "
        f"max_input_tokens={max_input_tokens} model_device={model_device}",
        flush=True,
    )
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
        device_map=device_map,
        low_cpu_mem_usage=True,
    ).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    pending_records = pending.to_dict("records")
    current_batch_size = initial_batch_size
    processed = 0
    last_saved = 0
    while processed < len(pending_records):
        batch = pending_records[processed : processed + current_batch_size]
        rendered = [
            render_prompt(
                tokenizer,
                args.mode,
                record,
                source_policy=source_policy,
            )
            for record in batch
        ]
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
            if args.mode == "teacher_second":
                record["first_judge_label"] = record.get("judge_label")
                record["first_judge_reason"] = record.get("judge_reason")
            record.update(
                parse_result(
                    answer,
                    teacher=args.mode in {"teacher", "teacher_second"},
                    allow_minor=args.mode != "teacher_second",
                )
            )
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
