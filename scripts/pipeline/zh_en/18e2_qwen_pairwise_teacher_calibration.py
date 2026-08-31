from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_VERSION = "18E2_V1"

DEFAULT_QWEN_PATH = "/root/autodl-tmp/models/Qwen3-8B"

VALID_WINNERS = {
    "A",
    "B",
    "TIE",
    "BOTH_BAD",
}

FINAL_WINNERS = {
    "OPUS",
    "MADLAD",
    "TIE",
    "BOTH_BAD",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18E2 - Qwen3-8B dual-order pairwise calibration "
            "for OPUS vs MADLAD Teacher routing on 800 train-only rows."
        )
    )

    p.add_argument("--project_root", default=None)

    p.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/"
            "qwen_teacher_routing_calibration_800_v1.parquet"
        ),
    )

    p.add_argument(
        "--model_path",
        default=DEFAULT_QWEN_PATH,
    )

    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/"
            "18e2_pairwise"
        ),
    )

    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    p.add_argument(
        "--max_input_tokens",
        type=int,
        default=1536,
    )

    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=384,
    )

    p.add_argument(
        "--max_retries",
        type=int,
        default=2,
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)

    if not p.is_absolute():
        p = root / p

    return p.resolve()


def append_jsonl(rows: list[dict], path: Path):
    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return pd.DataFrame(
        rows
    )


def save_json(obj: dict, path: Path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def cleanup_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_name():
    if not torch.cuda.is_available():
        return "CPU"

    return torch.cuda.get_device_name(0)


def load_qwen(model_path: Path):
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )

    print("Tokenizer loaded.")

    print("Loading Qwen3-8B...")

    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        **kwargs,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    # Remove inherited sampling-only values so deterministic generation
    # does not emit temperature/top_p/top_k warnings.
    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass

    print("Model loaded.")

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Parameters:", f"{params:,}")

    return tokenizer, model, params


def clean_text(value) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()


def bool_field(obj, key):
    value = obj.get(key)

    if isinstance(value, bool):
        return value

    return None


def build_pairwise_prompt(
    row: pd.Series,
    candidate_a_model: str,
    candidate_a_text: str,
    candidate_b_model: str,
    candidate_b_text: str,
) -> str:
    source_lang = str(
        row["source_lang"]
    )

    target_lang = str(
        row["target_lang"]
    )

    source = clean_text(
        row["source_text"]
    )

    human = clean_text(
        row["human_reference"]
    )

    return f"""You are evaluating two machine-translation candidates.

Your task is NOT to imitate the human reference wording.
Judge semantic faithfulness to the SOURCE first.

The human reference is only auxiliary evidence.
It may be noisy, incomplete, stylistically different, or one of several valid translations.
Never reject a candidate merely because it differs lexically from the reference.

SOURCE LANGUAGE: {source_lang}
TARGET LANGUAGE: {target_lang}

SOURCE:
{source}

HUMAN REFERENCE (auxiliary only):
{human}

CANDIDATE A ({candidate_a_model}):
{candidate_a_text}

CANDIDATE B ({candidate_b_model}):
{candidate_b_text}

Evaluate each candidate independently against SOURCE.

A translation has a MAJOR ERROR if it materially changes factual meaning, action/state,
participant roles, entity identity, quantity, negation/event status, time/event order,
location/direction, omits important information, or adds unsupported information.

A translation may still be acceptable with small stylistic, fluency, wording,
or non-critical nuance differences.

Then choose:
- A: Candidate A is meaningfully better.
- B: Candidate B is meaningfully better.
- TIE: both are acceptable and neither has a meaningful semantic advantage.
- BOTH_BAD: both contain substantive semantic problems and should not be used as KD targets.

LOGICAL CONSISTENCY RULES:
1. If A has major_error=true and B has major_error=false, winner MUST NOT be A.
2. If B has major_error=true and A has major_error=false, winner MUST NOT be B.
3. BOTH_BAD normally requires both candidates to be unacceptable or substantively wrong.
4. acceptable=true and major_error=true is logically inconsistent. If major_error=true,
   acceptable should normally be false.
5. Winner must agree with your own candidate assessments.

Return exactly ONE JSON object.
No markdown.
No text outside JSON.
No chain-of-thought.

Schema:
{{
  "winner": "A|B|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "A": {{
    "acceptable": true,
    "major_error": false,
    "minor_error": false,
    "core_meaning_preserved": true,
    "main_action_state_match": true,
    "participant_roles_match": true,
    "entity_match": true,
    "quantity_match": true,
    "negation_event_status_match": true,
    "time_event_order_match": true,
    "location_direction_match": true,
    "important_information_preserved": true,
    "unsupported_information_added": false
  }},
  "B": {{
    "acceptable": true,
    "major_error": false,
    "minor_error": false,
    "core_meaning_preserved": true,
    "main_action_state_match": true,
    "participant_roles_match": true,
    "entity_match": true,
    "quantity_match": true,
    "negation_event_status_match": true,
    "time_event_order_match": true,
    "location_direction_match": true,
    "important_information_preserved": true,
    "unsupported_information_added": false
  }},
  "reason": "brief evidence-based reason"
}}"""


def build_adjudication_prompt(
    row: pd.Series,
) -> str:
    source = clean_text(
        row["source_text"]
    )

    human = clean_text(
        row["human_reference"]
    )

    opus = clean_text(
        row["opus_prediction"]
    )

    madlad = clean_text(
        row["madlad_prediction"]
    )

    return f"""You are the final adjudicator for two machine-translation candidates.

Judge semantic faithfulness to SOURCE first.
The human reference is auxiliary only and may itself be noisy.

SOURCE LANGUAGE: {row["source_lang"]}
TARGET LANGUAGE: {row["target_lang"]}

SOURCE:
{source}

HUMAN REFERENCE (auxiliary only):
{human}

OPUS:
{opus}

MADLAD:
{madlad}

Choose exactly one final result:
- OPUS: OPUS is meaningfully better.
- MADLAD: MADLAD is meaningfully better.
- TIE: both are acceptable and neither has a meaningful semantic advantage.
- BOTH_BAD: both contain substantive semantic errors and should not be used for KD.

A major semantic error includes material changes to core meaning, action/state,
participant roles, entity identity, quantities, negation/event status, time/event order,
location/direction, important omissions, or unsupported additions.

Do not prefer a candidate merely because it is closer in wording to the human reference.
Do not penalize valid paraphrases.

Return exactly ONE JSON object.
No markdown.
No text outside JSON.
No chain-of-thought.

Schema:
{{
  "winner": "OPUS|MADLAD|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "opus_acceptable": true,
  "opus_major_error": false,
  "madlad_acceptable": true,
  "madlad_major_error": false,
  "reason": "brief evidence-based reason"
}}"""


def extract_json_object(text: str):
    text = str(text).strip()

    if not text:
        return None

    try:
        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    # Prefer the last JSON-looking object if model emitted surrounding text.
    candidates = re.findall(
        r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}",
        text,
        flags=re.DOTALL,
    )

    for candidate in reversed(
        candidates
    ):
        try:
            obj = json.loads(
                candidate
            )

            if isinstance(obj, dict):
                return obj

        except Exception:
            continue

    first = text.find("{")
    last = text.rfind("}")

    if (
        first >= 0
        and
        last > first
    ):
        try:
            obj = json.loads(
                text[
                    first:last + 1
                ]
            )

            if isinstance(obj, dict):
                return obj

        except Exception:
            pass

    return None


def validate_candidate_block(
    block,
):
    if not isinstance(
        block,
        dict,
    ):
        return False

    required = [
        "acceptable",
        "major_error",
        "minor_error",
        "core_meaning_preserved",
        "main_action_state_match",
        "participant_roles_match",
        "entity_match",
        "quantity_match",
        "negation_event_status_match",
        "time_event_order_match",
        "location_direction_match",
        "important_information_preserved",
        "unsupported_information_added",
    ]

    for key in required:
        if not isinstance(
            block.get(key),
            bool,
        ):
            return False

    return True


def pairwise_logically_consistent(
    obj,
):
    if not isinstance(
        obj,
        dict,
    ):
        return False, [
            "not_object",
        ]

    errors = []

    winner = obj.get(
        "winner"
    )

    if winner not in VALID_WINNERS:
        errors.append(
            "invalid_winner"
        )

    a = obj.get("A")
    b = obj.get("B")

    if not validate_candidate_block(a):
        errors.append(
            "invalid_A_block"
        )

    if not validate_candidate_block(b):
        errors.append(
            "invalid_B_block"
        )

    if errors:
        return False, errors

    if (
        a["acceptable"]
        and
        a["major_error"]
    ):
        errors.append(
            "A_acceptable_major_conflict"
        )

    if (
        b["acceptable"]
        and
        b["major_error"]
    ):
        errors.append(
            "B_acceptable_major_conflict"
        )

    if (
        a["major_error"]
        and
        not b["major_error"]
        and
        winner == "A"
    ):
        errors.append(
            "winner_A_has_unique_major_error"
        )

    if (
        b["major_error"]
        and
        not a["major_error"]
        and
        winner == "B"
    ):
        errors.append(
            "winner_B_has_unique_major_error"
        )

    if (
        winner == "BOTH_BAD"
        and
        a["acceptable"]
        and
        b["acceptable"]
        and
        not a["major_error"]
        and
        not b["major_error"]
    ):
        errors.append(
            "both_bad_but_both_acceptable"
        )

    return (
        len(errors) == 0,
        errors,
    )


def adjudication_logically_consistent(
    obj,
):
    if not isinstance(
        obj,
        dict,
    ):
        return False, [
            "not_object",
        ]

    errors = []

    winner = obj.get(
        "winner"
    )

    if winner not in FINAL_WINNERS:
        errors.append(
            "invalid_winner"
        )

    bool_keys = [
        "opus_acceptable",
        "opus_major_error",
        "madlad_acceptable",
        "madlad_major_error",
    ]

    for key in bool_keys:
        if not isinstance(
            obj.get(key),
            bool,
        ):
            errors.append(
                f"invalid_{key}"
            )

    if errors:
        return False, errors

    if (
        obj[
            "opus_acceptable"
        ]
        and
        obj[
            "opus_major_error"
        ]
    ):
        errors.append(
            "opus_acceptable_major_conflict"
        )

    if (
        obj[
            "madlad_acceptable"
        ]
        and
        obj[
            "madlad_major_error"
        ]
    ):
        errors.append(
            "madlad_acceptable_major_conflict"
        )

    if (
        obj[
            "opus_major_error"
        ]
        and
        not obj[
            "madlad_major_error"
        ]
        and
        winner == "OPUS"
    ):
        errors.append(
            "winner_opus_has_unique_major_error"
        )

    if (
        obj[
            "madlad_major_error"
        ]
        and
        not obj[
            "opus_major_error"
        ]
        and
        winner == "MADLAD"
    ):
        errors.append(
            "winner_madlad_has_unique_major_error"
        )

    return (
        len(errors) == 0,
        errors,
    )


def map_ab_to_model(
    winner_ab: str,
    candidate_a_model: str,
    candidate_b_model: str,
) -> str:
    if winner_ab == "A":
        return candidate_a_model

    if winner_ab == "B":
        return candidate_b_model

    if winner_ab in {
        "TIE",
        "BOTH_BAD",
    }:
        return winner_ab

    return "PARSE_ERROR"


def render_chat_prompt(
    tokenizer,
    prompt: str,
):
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate_batch(
    *,
    prompts: list[str],
    tokenizer,
    model,
    max_input_tokens: int,
    max_new_tokens: int,
):
    rendered = [
        render_chat_prompt(
            tokenizer,
            prompt,
        )
        for prompt in prompts
    ]

    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt",
    )

    input_lengths = (
        encoded[
            "attention_mask"
        ]
        .sum(dim=1)
        .tolist()
    )

    if torch.cuda.is_available():
        encoded = {
            k: v.cuda(
                non_blocking=True
            )
            for k, v
            in encoded.items()
        }

        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        t0
    )

    outputs = []

    prompt_width = encoded[
        "input_ids"
    ].shape[1]

    for sequence in generated:
        generated_only = sequence[
            prompt_width:
        ]

        text = tokenizer.decode(
            generated_only,
            skip_special_tokens=True,
        ).strip()

        outputs.append(
            text
        )

    return (
        outputs,
        input_lengths,
        elapsed,
    )


def judge_pairwise_rows(
    *,
    rows: pd.DataFrame,
    pass_name: str,
    tokenizer,
    model,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    max_retries: int,
    checkpoint_path: Path,
):
    existing = load_jsonl(
        checkpoint_path
    )

    completed = (
        set(
            existing[
                "qwen_review_id"
            ]
            .astype(str)
            .tolist()
        )
        if len(existing)
        else set()
    )

    pending = rows.loc[
        ~rows[
            "qwen_review_id"
        ]
        .astype(str)
        .isin(
            completed
        )
    ].copy()

    print(
        f"\n{pass_name}:"
    )

    print(
        "Existing checkpoint rows:",
        len(completed),
    )

    print(
        "Pending rows:",
        len(pending),
    )

    new_records = []

    total = len(
        pending
    )

    for start in range(
        0,
        total,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            total,
        )

        batch = (
            pending.iloc[
                start:stop
            ]
            .copy()
        )

        prompts = []

        metadata = []

        for _, row in batch.iterrows():
            review_id = str(
                row[
                    "qwen_review_id"
                ]
            )

            # Pass1 uses deterministic mixed A/B placement;
            # Pass2 reverses exactly.
            base_swap = (
                int(
                    re.sub(
                        r"\D",
                        "",
                        review_id,
                    )
                    or
                    "0"
                )
                %
                2
                ==
                1
            )

            if pass_name == "PASS2":
                swap = not base_swap
            else:
                swap = base_swap

            if not swap:
                a_model = "OPUS"
                a_text = str(
                    row[
                        "opus_prediction"
                    ]
                )
                b_model = "MADLAD"
                b_text = str(
                    row[
                        "madlad_prediction"
                    ]
                )
            else:
                a_model = "MADLAD"
                a_text = str(
                    row[
                        "madlad_prediction"
                    ]
                )
                b_model = "OPUS"
                b_text = str(
                    row[
                        "opus_prediction"
                    ]
                )

            prompts.append(
                build_pairwise_prompt(
                    row=row,
                    candidate_a_model=a_model,
                    candidate_a_text=a_text,
                    candidate_b_model=b_model,
                    candidate_b_text=b_text,
                )
            )

            metadata.append(
                {
                    "review_id": review_id,
                    "a_model": a_model,
                    "b_model": b_model,
                    "swap": bool(
                        swap
                    ),
                    "row": row,
                }
            )

        # First attempt for whole batch.
        outputs, input_lengths, elapsed = generate_batch(
            prompts=prompts,
            tokenizer=tokenizer,
            model=model,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
        )

        batch_records = []

        retry_items = []

        for idx, (
            raw,
            meta,
            input_len,
        ) in enumerate(
            zip(
                outputs,
                metadata,
                input_lengths,
            )
        ):
            obj = extract_json_object(
                raw
            )

            valid, logic_errors = (
                pairwise_logically_consistent(
                    obj
                )
            )

            if valid:
                winner_ab = str(
                    obj[
                        "winner"
                    ]
                )

                mapped = map_ab_to_model(
                    winner_ab,
                    meta[
                        "a_model"
                    ],
                    meta[
                        "b_model"
                    ],
                )

                record = {
                    "qwen_review_id": meta[
                        "review_id"
                    ],
                    "pass_name": pass_name,
                    "candidate_a_model": meta[
                        "a_model"
                    ],
                    "candidate_b_model": meta[
                        "b_model"
                    ],
                    "candidate_order_swapped": meta[
                        "swap"
                    ],
                    "winner_ab": winner_ab,
                    "winner_model": mapped,
                    "confidence": str(
                        obj.get(
                            "confidence",
                            "",
                        )
                    ),
                    "A": obj[
                        "A"
                    ],
                    "B": obj[
                        "B"
                    ],
                    "reason": str(
                        obj.get(
                            "reason",
                            "",
                        )
                    ),
                    "parse_success": True,
                    "logical_consistency": True,
                    "logic_errors": [],
                    "input_tokens": int(
                        input_len
                    ),
                    "raw_qwen_output": raw,
                    "retry_count": 0,
                }

                batch_records.append(
                    record
                )

            else:
                retry_items.append(
                    (
                        idx,
                        raw,
                        obj,
                        logic_errors,
                        meta,
                        input_len,
                    )
                )

        # Retry problematic rows individually.
        for (
            _idx,
            raw_initial,
            obj_initial,
            logic_initial,
            meta,
            input_len,
        ) in retry_items:
            row = meta[
                "row"
            ]

            prompt = build_pairwise_prompt(
                row=row,
                candidate_a_model=meta[
                    "a_model"
                ],
                candidate_a_text=(
                    str(
                        row[
                            "opus_prediction"
                        ]
                    )
                    if meta[
                        "a_model"
                    ]
                    ==
                    "OPUS"
                    else str(
                        row[
                            "madlad_prediction"
                        ]
                    )
                ),
                candidate_b_model=meta[
                    "b_model"
                ],
                candidate_b_text=(
                    str(
                        row[
                            "opus_prediction"
                        ]
                    )
                    if meta[
                        "b_model"
                    ]
                    ==
                    "OPUS"
                    else str(
                        row[
                            "madlad_prediction"
                        ]
                    )
                ),
            )

            final_raw = (
                raw_initial
            )

            final_obj = (
                obj_initial
            )

            final_logic = (
                logic_initial
            )

            retry_count = 0

            valid = False

            for retry in range(
                1,
                max_retries + 1,
            ):
                retry_count = retry

                retry_prompt = (
                    prompt
                    +
                    "\n\nIMPORTANT RETRY: "
                    "Your previous response was invalid or logically inconsistent. "
                    "Re-check your candidate assessments and ensure winner is consistent "
                    "with major_error/acceptable fields. Return only valid JSON."
                )

                out, lens, _elapsed = generate_batch(
                    prompts=[
                        retry_prompt
                    ],
                    tokenizer=tokenizer,
                    model=model,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )

                final_raw = out[
                    0
                ]

                final_obj = extract_json_object(
                    final_raw
                )

                valid, final_logic = (
                    pairwise_logically_consistent(
                        final_obj
                    )
                )

                input_len = lens[
                    0
                ]

                if valid:
                    break

            if valid:
                winner_ab = str(
                    final_obj[
                        "winner"
                    ]
                )

                mapped = map_ab_to_model(
                    winner_ab,
                    meta[
                        "a_model"
                    ],
                    meta[
                        "b_model"
                    ],
                )

                record = {
                    "qwen_review_id": meta[
                        "review_id"
                    ],
                    "pass_name": pass_name,
                    "candidate_a_model": meta[
                        "a_model"
                    ],
                    "candidate_b_model": meta[
                        "b_model"
                    ],
                    "candidate_order_swapped": meta[
                        "swap"
                    ],
                    "winner_ab": winner_ab,
                    "winner_model": mapped,
                    "confidence": str(
                        final_obj.get(
                            "confidence",
                            "",
                        )
                    ),
                    "A": final_obj[
                        "A"
                    ],
                    "B": final_obj[
                        "B"
                    ],
                    "reason": str(
                        final_obj.get(
                            "reason",
                            "",
                        )
                    ),
                    "parse_success": True,
                    "logical_consistency": True,
                    "logic_errors": [],
                    "input_tokens": int(
                        input_len
                    ),
                    "raw_qwen_output": final_raw,
                    "retry_count": int(
                        retry_count
                    ),
                }

            else:
                record = {
                    "qwen_review_id": meta[
                        "review_id"
                    ],
                    "pass_name": pass_name,
                    "candidate_a_model": meta[
                        "a_model"
                    ],
                    "candidate_b_model": meta[
                        "b_model"
                    ],
                    "candidate_order_swapped": meta[
                        "swap"
                    ],
                    "winner_ab": "PARSE_ERROR",
                    "winner_model": "PARSE_ERROR",
                    "confidence": "",
                    "A": {},
                    "B": {},
                    "reason": "",
                    "parse_success": False,
                    "logical_consistency": False,
                    "logic_errors": (
                        final_logic
                        if isinstance(
                            final_logic,
                            list,
                        )
                        else [
                            str(
                                final_logic
                            )
                        ]
                    ),
                    "input_tokens": int(
                        input_len
                    ),
                    "raw_qwen_output": final_raw,
                    "retry_count": int(
                        retry_count
                    ),
                }

            batch_records.append(
                record
            )

        append_jsonl(
            batch_records,
            checkpoint_path,
        )

        new_records.extend(
            batch_records
        )

        counts = Counter(
            r[
                "winner_model"
            ]
            for r in batch_records
        )

        parse_n = sum(
            1
            for r in batch_records
            if r[
                "parse_success"
            ]
        )

        print(
            f"{stop}/{total}"
            f" | parse {parse_n}/{len(batch_records)}"
            f" | {dict(counts)}"
        )

    new_df = pd.DataFrame(
        new_records
    )

    if len(existing):
        out = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )
    else:
        out = new_df

    if (
        out[
            "qwen_review_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate review IDs in {pass_name}."
        )

    return out


def judge_adjudications(
    *,
    rows: pd.DataFrame,
    tokenizer,
    model,
    max_input_tokens: int,
    max_new_tokens: int,
    max_retries: int,
    checkpoint_path: Path,
):
    existing = load_jsonl(
        checkpoint_path
    )

    completed = (
        set(
            existing[
                "qwen_review_id"
            ]
            .astype(str)
            .tolist()
        )
        if len(existing)
        else set()
    )

    pending = rows.loc[
        ~rows[
            "qwen_review_id"
        ]
        .astype(str)
        .isin(
            completed
        )
    ].copy()

    print(
        "\nADJUDICATION:"
    )

    print(
        "Existing checkpoint rows:",
        len(completed),
    )

    print(
        "Pending rows:",
        len(pending),
    )

    records = []

    total = len(
        pending
    )

    for i, (_, row) in enumerate(
        pending.iterrows(),
        start=1,
    ):
        prompt = build_adjudication_prompt(
            row
        )

        final_raw = ""

        final_obj = None

        logic_errors = []

        input_tokens = 0

        retry_count = 0

        valid = False

        for attempt in range(
            0,
            max_retries + 1,
        ):
            retry_count = attempt

            current_prompt = prompt

            if attempt > 0:
                current_prompt += (
                    "\n\nIMPORTANT RETRY: your previous answer was invalid or "
                    "logically inconsistent. Re-evaluate both named candidates and "
                    "return only one logically consistent JSON object."
                )

            outs, lens, _elapsed = generate_batch(
                prompts=[
                    current_prompt
                ],
                tokenizer=tokenizer,
                model=model,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
            )

            final_raw = outs[
                0
            ]

            input_tokens = int(
                lens[
                    0
                ]
            )

            final_obj = extract_json_object(
                final_raw
            )

            valid, logic_errors = (
                adjudication_logically_consistent(
                    final_obj
                )
            )

            if valid:
                break

        if valid:
            record = {
                "qwen_review_id": str(
                    row[
                        "qwen_review_id"
                    ]
                ),
                "winner_model": str(
                    final_obj[
                        "winner"
                    ]
                ),
                "confidence": str(
                    final_obj.get(
                        "confidence",
                        "",
                    )
                ),
                "opus_acceptable": bool(
                    final_obj[
                        "opus_acceptable"
                    ]
                ),
                "opus_major_error": bool(
                    final_obj[
                        "opus_major_error"
                    ]
                ),
                "madlad_acceptable": bool(
                    final_obj[
                        "madlad_acceptable"
                    ]
                ),
                "madlad_major_error": bool(
                    final_obj[
                        "madlad_major_error"
                    ]
                ),
                "reason": str(
                    final_obj.get(
                        "reason",
                        "",
                    )
                ),
                "parse_success": True,
                "logical_consistency": True,
                "logic_errors": [],
                "input_tokens": input_tokens,
                "raw_qwen_output": final_raw,
                "retry_count": retry_count,
            }

        else:
            record = {
                "qwen_review_id": str(
                    row[
                        "qwen_review_id"
                    ]
                ),
                "winner_model": "PARSE_ERROR",
                "confidence": "",
                "opus_acceptable": False,
                "opus_major_error": False,
                "madlad_acceptable": False,
                "madlad_major_error": False,
                "reason": "",
                "parse_success": False,
                "logical_consistency": False,
                "logic_errors": logic_errors,
                "input_tokens": input_tokens,
                "raw_qwen_output": final_raw,
                "retry_count": retry_count,
            }

        append_jsonl(
            [
                record
            ],
            checkpoint_path,
        )

        records.append(
            record
        )

        if (
            i % 25 == 0
            or
            i == total
        ):
            print(
                f"{i}/{total}"
                f" | winner={record['winner_model']}"
                f" | parse={record['parse_success']}"
            )

    new_df = pd.DataFrame(
        records
    )

    if len(existing):
        out = pd.concat(
            [
                existing,
                new_df,
            ],
            ignore_index=True,
        )

    else:
        out = new_df

    return out


def candidate_assessment_by_model(
    result_row: pd.Series,
    model_name: str,
):
    a_model = str(
        result_row[
            "candidate_a_model"
        ]
    )

    b_model = str(
        result_row[
            "candidate_b_model"
        ]
    )

    if model_name == a_model:
        return result_row[
            "A"
        ]

    if model_name == b_model:
        return result_row[
            "B"
        ]

    return {}


def main():
    args = parse_args()

    root = (
        Path(
            args.project_root
        ).resolve()
        if args.project_root
        else infer_project_root()
    )

    input_path = resolve_path(
        root,
        args.input,
    )

    model_path = resolve_path(
        root,
        args.model_path,
    )

    outdir = resolve_path(
        root,
        args.output_dir,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            input_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pass1_checkpoint = (
        outdir
        /
        "checkpoint_pass1_v1.jsonl"
    )

    pass2_checkpoint = (
        outdir
        /
        "checkpoint_pass2_v1.jsonl"
    )

    adjudication_checkpoint = (
        outdir
        /
        "checkpoint_adjudication_v1.jsonl"
    )

    final_output = (
        outdir
        /
        "qwen_teacher_routing_calibration_results_v1.parquet"
    )

    summary_output = (
        outdir
        /
        "qwen_teacher_routing_calibration_summary_v1.csv"
    )

    report_output = (
        outdir
        /
        "qwen_teacher_routing_calibration_report_v1.json"
    )

    if args.overwrite:
        for p in [
            pass1_checkpoint,
            pass2_checkpoint,
            adjudication_checkpoint,
            final_output,
            summary_output,
            report_output,
        ]:
            if p.exists():
                p.unlink()

    else:
        if final_output.exists():
            raise RuntimeError(
                f"Final output already exists:\n{final_output}\n"
                "Use --overwrite to rebuild."
            )

    print(
        "=" * 110
    )

    print(
        "ZH-EN DISTILLATION PIPELINE"
    )

    print(
        "STEP 18E2 - QWEN3-8B DUAL-ORDER TEACHER ROUTING CALIBRATION"
    )

    print(
        "=" * 110
    )

    print(
        "\nInput:"
    )

    print(
        input_path
    )

    print(
        "\nModel:"
    )

    print(
        model_path
    )

    print(
        "\nGPU:"
    )

    print(
        gpu_name()
    )

    df = pd.read_parquet(
        input_path
    ).copy()

    required = {
        "qwen_review_id",
        "kd_candidate_id",
        "direction",
        "source_dataset",
        "calibration_band",
        "source_lang",
        "target_lang",
        "source_text",
        "human_reference",
        "opus_prediction",
        "madlad_prediction",
        "teacher_disagreement_score",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if (
        df[
            "qwen_review_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate qwen_review_id."
        )

    print(
        "\nRows:",
        len(df),
    )

    tokenizer, model, params = load_qwen(
        model_path
    )

    pass1 = judge_pairwise_rows(
        rows=df,
        pass_name="PASS1",
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        checkpoint_path=pass1_checkpoint,
    )

    pass2 = judge_pairwise_rows(
        rows=df,
        pass_name="PASS2",
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        checkpoint_path=pass2_checkpoint,
    )

    p1 = pass1.rename(
        columns={
            c: f"pass1_{c}"
            for c in pass1.columns
            if c
            !=
            "qwen_review_id"
        }
    )

    p2 = pass2.rename(
        columns={
            c: f"pass2_{c}"
            for c in pass2.columns
            if c
            !=
            "qwen_review_id"
        }
    )

    merged = (
        df.merge(
            p1,
            on="qwen_review_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            p2,
            on="qwen_review_id",
            how="left",
            validate="one_to_one",
        )
    )

    merged[
        "dual_order_consistent"
    ] = (
        merged[
            "pass1_winner_model"
        ]
        ==
        merged[
            "pass2_winner_model"
        ]
    ) & (
        merged[
            "pass1_parse_success"
        ]
        ==
        True
    ) & (
        merged[
            "pass2_parse_success"
        ]
        ==
        True
    )

    adjudication_rows = merged.loc[
        ~merged[
            "dual_order_consistent"
        ]
    ].copy()

    adjudication = judge_adjudications(
        rows=adjudication_rows,
        tokenizer=tokenizer,
        model=model,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        max_retries=args.max_retries,
        checkpoint_path=adjudication_checkpoint,
    )

    if len(
        adjudication
    ):
        adjudication_merge = adjudication.rename(
            columns={
                c: f"adjudication_{c}"
                for c in adjudication.columns
                if c
                !=
                "qwen_review_id"
            }
        )

        merged = merged.merge(
            adjudication_merge,
            on="qwen_review_id",
            how="left",
            validate="one_to_one",
        )

    else:
        for c in [
            "adjudication_winner_model",
            "adjudication_confidence",
            "adjudication_opus_acceptable",
            "adjudication_opus_major_error",
            "adjudication_madlad_acceptable",
            "adjudication_madlad_major_error",
            "adjudication_reason",
            "adjudication_parse_success",
            "adjudication_logical_consistency",
        ]:
            merged[
                c
            ] = None

    merged[
        "final_winner"
    ] = merged[
        "pass1_winner_model"
    ]

    merged[
        "final_resolution"
    ] = "DUAL_ORDER_CONSENSUS"

    inconsistent = (
        ~merged[
            "dual_order_consistent"
        ]
    )

    merged.loc[
        inconsistent,
        "final_winner",
    ] = merged.loc[
        inconsistent,
        "adjudication_winner_model",
    ]

    merged.loc[
        inconsistent,
        "final_resolution",
    ] = "INDEPENDENT_ADJUDICATION"

    # Extract candidate assessments by model for both passes.
    pass1_opus_acceptable = []
    pass1_opus_major = []
    pass1_mad_acceptable = []
    pass1_mad_major = []

    pass2_opus_acceptable = []
    pass2_opus_major = []
    pass2_mad_acceptable = []
    pass2_mad_major = []

    for _, row in merged.iterrows():
        p1row = pd.Series(
            {
                "candidate_a_model": row[
                    "pass1_candidate_a_model"
                ],
                "candidate_b_model": row[
                    "pass1_candidate_b_model"
                ],
                "A": row[
                    "pass1_A"
                ],
                "B": row[
                    "pass1_B"
                ],
            }
        )

        p2row = pd.Series(
            {
                "candidate_a_model": row[
                    "pass2_candidate_a_model"
                ],
                "candidate_b_model": row[
                    "pass2_candidate_b_model"
                ],
                "A": row[
                    "pass2_A"
                ],
                "B": row[
                    "pass2_B"
                ],
            }
        )

        p1_opus = candidate_assessment_by_model(
            p1row,
            "OPUS",
        )

        p1_mad = candidate_assessment_by_model(
            p1row,
            "MADLAD",
        )

        p2_opus = candidate_assessment_by_model(
            p2row,
            "OPUS",
        )

        p2_mad = candidate_assessment_by_model(
            p2row,
            "MADLAD",
        )

        pass1_opus_acceptable.append(
            bool(
                p1_opus.get(
                    "acceptable",
                    False,
                )
            )
        )

        pass1_opus_major.append(
            bool(
                p1_opus.get(
                    "major_error",
                    False,
                )
            )
        )

        pass1_mad_acceptable.append(
            bool(
                p1_mad.get(
                    "acceptable",
                    False,
                )
            )
        )

        pass1_mad_major.append(
            bool(
                p1_mad.get(
                    "major_error",
                    False,
                )
            )
        )

        pass2_opus_acceptable.append(
            bool(
                p2_opus.get(
                    "acceptable",
                    False,
                )
            )
        )

        pass2_opus_major.append(
            bool(
                p2_opus.get(
                    "major_error",
                    False,
                )
            )
        )

        pass2_mad_acceptable.append(
            bool(
                p2_mad.get(
                    "acceptable",
                    False,
                )
            )
        )

        pass2_mad_major.append(
            bool(
                p2_mad.get(
                    "major_error",
                    False,
                )
            )
        )

    merged[
        "pass1_opus_acceptable"
    ] = pass1_opus_acceptable

    merged[
        "pass1_opus_major_error"
    ] = pass1_opus_major

    merged[
        "pass1_madlad_acceptable"
    ] = pass1_mad_acceptable

    merged[
        "pass1_madlad_major_error"
    ] = pass1_mad_major

    merged[
        "pass2_opus_acceptable"
    ] = pass2_opus_acceptable

    merged[
        "pass2_opus_major_error"
    ] = pass2_opus_major

    merged[
        "pass2_madlad_acceptable"
    ] = pass2_mad_acceptable

    merged[
        "pass2_madlad_major_error"
    ] = pass2_mad_major

    # Final quality diagnostics:
    # if adjudicated, use adjudication fields;
    # if consensus, require both passes to agree on acceptable/major state
    merged[
        "final_opus_acceptable"
    ] = (
        merged[
            "pass1_opus_acceptable"
        ]
        &
        merged[
            "pass2_opus_acceptable"
        ]
    )

    merged[
        "final_opus_major_error"
    ] = (
        merged[
            "pass1_opus_major_error"
        ]
        |
        merged[
            "pass2_opus_major_error"
        ]
    )

    merged[
        "final_madlad_acceptable"
    ] = (
        merged[
            "pass1_madlad_acceptable"
        ]
        &
        merged[
            "pass2_madlad_acceptable"
        ]
    )

    merged[
        "final_madlad_major_error"
    ] = (
        merged[
            "pass1_madlad_major_error"
        ]
        |
        merged[
            "pass2_madlad_major_error"
        ]
    )

    if len(
        adjudication
    ):
        merged.loc[
            inconsistent,
            "final_opus_acceptable",
        ] = merged.loc[
            inconsistent,
            "adjudication_opus_acceptable",
        ]

        merged.loc[
            inconsistent,
            "final_opus_major_error",
        ] = merged.loc[
            inconsistent,
            "adjudication_opus_major_error",
        ]

        merged.loc[
            inconsistent,
            "final_madlad_acceptable",
        ] = merged.loc[
            inconsistent,
            "adjudication_madlad_acceptable",
        ]

        merged.loc[
            inconsistent,
            "final_madlad_major_error",
        ] = merged.loc[
            inconsistent,
            "adjudication_madlad_major_error",
        ]

    merged[
        "final_resolved"
    ] = merged[
        "final_winner"
    ].isin(
        FINAL_WINNERS
    )

    unresolved = int(
        (
            ~merged[
                "final_resolved"
            ]
        ).sum()
    )

    assertions = {
        "rows_preserved": (
            len(
                merged
            )
            ==
            len(
                df
            )
        ),
        "review_id_unique": (
            merged[
                "qwen_review_id"
            ]
            .is_unique
        ),
        "pass1_complete": (
            merged[
                "pass1_winner_model"
            ]
            .notna()
            .all()
        ),
        "pass2_complete": (
            merged[
                "pass2_winner_model"
            ]
            .notna()
            .all()
        ),
        "final_resolution_complete": (
            unresolved
            ==
            0
        ),
        "final_winner_valid": (
            merged[
                "final_winner"
            ]
            .isin(
                FINAL_WINNERS
            )
            .all()
        ),
        "no_prompt_truncation": (
            (
                merged[
                    "pass1_input_tokens"
                ]
                <
                args.max_input_tokens
            ).all()
            and
            (
                merged[
                    "pass2_input_tokens"
                ]
                <
                args.max_input_tokens
            ).all()
        ),
    }

    failed = [
        k
        for k, v in assertions.items()
        if not bool(v)
    ]

    if failed:
        raise RuntimeError(
            "STEP18E2 assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    merged.to_parquet(
        final_output,
        index=False,
    )

    summary = (
        merged.groupby(
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "final_winner",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "qwen_review_id",
                "size",
            ),
            mean_disagreement_score=(
                "teacher_disagreement_score",
                "mean",
            ),
        )
        .reset_index()
    )

    group_totals = (
        summary.groupby(
            [
                "direction",
                "source_dataset",
                "calibration_band",
            ]
        )[
            "rows"
        ]
        .transform(
            "sum"
        )
    )

    summary[
        "winner_percent_within_band"
    ] = (
        100.0
        *
        summary[
            "rows"
        ]
        /
        group_totals
    )

    summary.to_csv(
        summary_output,
        index=False,
        encoding="utf-8-sig",
    )

    final_counts = (
        merged[
            "final_winner"
        ]
        .value_counts()
        .to_dict()
    )

    report = {
        "step": "18E2",
        "step_version": STEP_VERSION,
        "input": str(
            input_path
        ),
        "model": {
            "path": str(
                model_path
            ),
            "parameters": int(
                params
            ),
        },
        "generation": {
            "batch_size": int(
                args.batch_size
            ),
            "max_input_tokens": int(
                args.max_input_tokens
            ),
            "max_new_tokens": int(
                args.max_new_tokens
            ),
            "do_sample": False,
            "max_retries": int(
                args.max_retries
            ),
        },
        "counts": {
            "rows": int(
                len(
                    merged
                )
            ),
            "dual_order_consistent_rows": int(
                merged[
                    "dual_order_consistent"
                ]
                .sum()
            ),
            "adjudication_used_rows": int(
                (
                    ~merged[
                        "dual_order_consistent"
                    ]
                ).sum()
            ),
            "unresolved_rows": int(
                unresolved
            ),
            "final_winner": {
                str(
                    k
                ): int(
                    v
                )
                for k, v
                in final_counts.items()
            },
        },
        "assertions": {
            k: bool(v)
            for k, v
            in assertions.items()
        },
        "outputs": {
            "results": str(
                final_output
            ),
            "summary": str(
                summary_output
            ),
            "pass1_checkpoint": str(
                pass1_checkpoint
            ),
            "pass2_checkpoint": str(
                pass2_checkpoint
            ),
            "adjudication_checkpoint": str(
                adjudication_checkpoint
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18E3_GATE_POLICY_CALIBRATION"
        ),
    }

    save_json(
        report,
        report_output,
    )

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "STEP 18E2 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nRows:",
        len(
            merged
        ),
    )

    print(
        "\nPass 1 parse:",
        int(
            merged[
                "pass1_parse_success"
            ].sum()
        ),
        "/",
        len(
            merged
        ),
    )

    print(
        "Pass 2 parse:",
        int(
            merged[
                "pass2_parse_success"
            ].sum()
        ),
        "/",
        len(
            merged
        ),
    )

    print(
        "\nDual-order consistency:",
        int(
            merged[
                "dual_order_consistent"
            ].sum()
        ),
        "/",
        len(
            merged
        ),
        f"({100.0 * merged['dual_order_consistent'].mean():.2f}%)",
    )

    print(
        "\nAdjudication used:",
        int(
            (
                ~merged[
                    "dual_order_consistent"
                ]
            ).sum()
        ),
    )

    print(
        "\nFinal resolution:",
        int(
            merged[
                "final_resolved"
            ].sum()
        ),
        "/",
        len(
            merged
        ),
    )

    print(
        "Unresolved:",
        unresolved,
    )

    print(
        "\nFinal winners:"
    )

    print(
        merged[
            "final_winner"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDirection × source × band × final winner:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nAssertions:"
    )

    for k, v in assertions.items():
        print(
            f"{k}: {bool(v)}"
        )

    print(
        "\nResults:"
    )

    print(
        final_output
    )

    print(
        "\nSummary:"
    )

    print(
        summary_output
    )

    print(
        "\nReport:"
    )

    print(
        report_output
    )

    print(
        "\nSTATUS:"
    )

    print(
        "READY_FOR_STEP_18E3_GATE_POLICY_CALIBRATION"
    )

    del model
    del tokenizer

    cleanup_cuda()


if __name__ == "__main__":
    main()
