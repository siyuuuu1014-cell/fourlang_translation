from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_VERSION = "17C_V2"
PROMPT_VERSION = "ZH_EN_TEACHER_PAIRWISE_V2"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"

VALID_WINNERS = {"A", "B", "TIE", "BOTH_BAD"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}

BOOL_FIELDS = [
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

CRITICAL_TRUE_FIELDS = [
    "core_meaning_preserved",
    "main_action_state_match",
    "participant_roles_match",
    "negation_event_status_match",
    "important_information_preserved",
]


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 17C V2 - Qwen3-8B pairwise teacher judge "
            "for OPUS Exp1 vs MADLAD."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["calibration", "full"],
        default="calibration",
    )

    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--calibration_size",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=3072,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# I/O
# ============================================================

def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

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


def append_jsonl(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )
            except Exception:
                # Ignore a partially-written last line after an interrupted run.
                continue

    return rows


def safe_unlink(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


# ============================================================
# Input
# ============================================================

def load_disagreement_set(
    project_root: Path,
) -> pd.DataFrame:
    path = (
        project_root
        / "data"
        / "teacher_selection"
        / "zh_en"
        / "v1"
        / "17b_disagreement"
        / "teacher_disagreement_800_v1.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"17B disagreement set missing:\n{path}"
        )

    df = pd.read_parquet(path)

    required = {
        "review_id",
        "sample_id",
        "pair_id",
        "direction",
        "source_dataset",
        "source_text",
        "reference_text",
        "opus_prediction",
        "madlad_prediction",
        "teacher_disagreement_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "17B input missing columns: "
            f"{sorted(missing)}"
        )

    if df["review_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate review_id found."
        )

    print(
        "\n17B input rows:",
        len(df),
    )

    print(
        "\nInput distribution:"
    )

    print(
        df.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
        .size()
        .to_string()
    )

    return (
        df.copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Calibration selection
# ============================================================

def select_calibration_rows(
    df: pd.DataFrame,
    calibration_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    IMPORTANT:
    This intentionally mirrors the V1 calibration sampling policy
    so V1 and V2 evaluate the same 200 rows when seed/size match.
    """
    calibration_size = min(
        calibration_size,
        len(df),
    )

    strata = (
        df[
            [
                "direction",
                "source_dataset",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "direction",
                "source_dataset",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if len(strata) == 0:
        raise RuntimeError(
            "No calibration strata."
        )

    base = (
        calibration_size
        //
        len(strata)
    )

    remainder = (
        calibration_size
        %
        len(strata)
    )

    parts = []

    for i, row in strata.iterrows():
        direction = str(
            row["direction"]
        )

        source_dataset = str(
            row["source_dataset"]
        )

        part = (
            df[
                (
                    df["direction"]
                    ==
                    direction
                )
                &
                (
                    df["source_dataset"]
                    .astype(str)
                    ==
                    source_dataset
                )
            ]
            .copy()
        )

        requested = (
            base
            +
            (
                1
                if i < remainder
                else 0
            )
        )

        requested = min(
            requested,
            len(part),
        )

        selected = (
            part.sample(
                n=requested,
                random_state=(
                    seed + i
                ),
                replace=False,
            )
            .copy()
        )

        parts.append(selected)

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    out = (
        out.sort_values(
            [
                "direction",
                "source_dataset",
                "review_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "\nCalibration selected:",
        len(out),
    )

    print(
        "\nCalibration distribution:"
    )

    print(
        out.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
        .size()
        .to_string()
    )

    return out


# ============================================================
# Candidate randomization
# ============================================================

def deterministic_swap(
    review_id: str,
    seed: int,
) -> bool:
    digest = hashlib.sha256(
        f"{seed}|{review_id}".encode(
            "utf-8"
        )
    ).hexdigest()

    return bool(
        int(
            digest[:8],
            16,
        )
        %
        2
    )


def assign_candidates(
    row: pd.Series,
    seed: int,
) -> dict:
    swap = deterministic_swap(
        str(
            row["review_id"]
        ),
        seed,
    )

    opus = str(
        row["opus_prediction"]
    )

    madlad = str(
        row["madlad_prediction"]
    )

    if not swap:
        return {
            "candidate_a_model": "OPUS",
            "candidate_a_text": opus,
            "candidate_b_model": "MADLAD",
            "candidate_b_text": madlad,
            "candidate_order_swapped": False,
        }

    return {
        "candidate_a_model": "MADLAD",
        "candidate_a_text": madlad,
        "candidate_b_model": "OPUS",
        "candidate_b_text": opus,
        "candidate_order_swapped": True,
    }


# ============================================================
# Prompt V2
# ============================================================

SYSTEM_PROMPT = """
You are a strict bilingual translation evaluator for Chinese and English.

Compare two candidate translations of the SAME source sentence.

SOURCE is authoritative.
REFERENCE is auxiliary only and may itself be imperfect.

Judge semantic fidelity to SOURCE, not lexical similarity to REFERENCE.
Valid paraphrases, equivalent numeric formats, natural Chinese wording,
and plausible transliterations must not be penalized merely for differing
from the reference.

Evaluate candidate A and candidate B independently first.
Only after evaluating both candidates, choose the final winner.

Your final winner MUST be logically consistent with your own major_error fields:
- if A has major_error=true and B has major_error=false, winner MUST be B
- if B has major_error=true and A has major_error=false, winner MUST be A
- if both have major_error=true, winner MUST be BOTH_BAD
- if neither has a major error and they are essentially equal, winner may be TIE

Return exactly ONE valid JSON object.
No markdown.
No chain-of-thought.
No text outside JSON.
""".strip()


RUBRIC = """
MAJOR ERROR:
A substantive error that changes the factual proposition or essential meaning,
including core meaning, main action/state, participant roles, important entity,
negation/event status, major time/order relation, major location/direction
relation, important omission, or unsupported factual addition.

MINOR ERROR:
A limited issue that does not materially change the factual proposition,
such as slight awkwardness or a small non-critical nuance loss.

Formatting alone is not a semantic error. Examples:
- 160 million and 1.6亿 may be equivalent
- 9:00 p.m. and 晚上9点 may be equivalent
- plausible name transliterations may both be acceptable

Use this exact JSON schema:

{
  "winner": "A|B|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "A": {
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
  },
  "B": {
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
  },
  "reason": "brief evidence-based explanation"
}
""".strip()


def build_user_prompt(
    row: pd.Series,
    assignment: dict,
) -> str:
    return f"""
PAIRWISE TRANSLATION JUDGMENT

DIRECTION:
{row["direction"]}

SOURCE:
{row["source_text"]}

REFERENCE — auxiliary only:
{row["reference_text"]}

CANDIDATE A:
{assignment["candidate_a_text"]}

CANDIDATE B:
{assignment["candidate_b_text"]}

{RUBRIC}
""".strip()


# ============================================================
# Model
# ============================================================

def load_qwen(
    model_path: Path,
):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
    )

    tokenizer.padding_side = (
        "left"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "Tokenizer loaded."
    )

    print(
        "\nLoading Qwen3-8B..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16,
        )
        .to("cuda")
    )

    model.eval()

    # Deterministic judging and suppress irrelevant generation warnings.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print(
        "Model loaded."
    )

    print(
        "Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    return (
        tokenizer,
        model,
    )


def apply_chat_template(
    tokenizer,
    user_prompt: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
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


# ============================================================
# Token safety
# ============================================================

def count_tokens(
    tokenizer,
    text: str,
) -> int:
    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
    )


def truncate_text_middle(
    tokenizer,
    text: str,
    max_tokens: int,
) -> str:
    ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    if len(ids) <= max_tokens:
        return text

    max_tokens = max(
        32,
        max_tokens,
    )

    first_n = (
        max_tokens
        //
        2
    )

    last_n = (
        max_tokens
        -
        first_n
    )

    return tokenizer.decode(
        ids[:first_n]
        +
        ids[-last_n:],
        skip_special_tokens=True,
    )


def build_safe_chat_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    max_input_tokens: int,
) -> tuple[
    str,
    dict,
]:
    user_prompt = build_user_prompt(
        row,
        assignment,
    )

    chat_prompt = apply_chat_template(
        tokenizer,
        user_prompt,
    )

    original_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    metadata = {
        "original_prompt_tokens": int(
            original_tokens
        ),
        "final_prompt_tokens": int(
            original_tokens
        ),
        "prompt_truncated": False,
    }

    if original_tokens <= max_input_tokens:
        return (
            chat_prompt,
            metadata,
        )

    # Keep all semantic fields. Compress each field independently
    # rather than allowing tokenizer truncation to remove the SOURCE.
    field_budget = 400

    compact_prompt = f"""
PAIRWISE TRANSLATION JUDGMENT

DIRECTION:
{row["direction"]}

SOURCE:
{truncate_text_middle(tokenizer, str(row["source_text"]), field_budget)}

REFERENCE — auxiliary only:
{truncate_text_middle(tokenizer, str(row["reference_text"]), field_budget)}

CANDIDATE A:
{truncate_text_middle(tokenizer, assignment["candidate_a_text"], field_budget)}

CANDIDATE B:
{truncate_text_middle(tokenizer, assignment["candidate_b_text"], field_budget)}

{RUBRIC}
""".strip()

    chat_prompt = apply_chat_template(
        tokenizer,
        compact_prompt,
    )

    final_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    if final_tokens > max_input_tokens:
        raise RuntimeError(
            "\nPrompt still exceeds token limit "
            "after safe field compression.\n"
            f"review_id={row['review_id']}\n"
            f"tokens={final_tokens}\n"
            f"limit={max_input_tokens}"
        )

    metadata[
        "final_prompt_tokens"
    ] = int(
        final_tokens
    )

    metadata[
        "prompt_truncated"
    ] = True

    return (
        chat_prompt,
        metadata,
    )


# ============================================================
# JSON parsing + logical validation
# ============================================================

def extract_json_object(
    text: str,
) -> str | None:
    text = str(
        text
    ).strip()

    text = re.sub(
        r"^```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```$",
        "",
        text,
    )

    start = text.find(
        "{"
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(
        start,
        len(text),
    ):
        ch = text[i]

        if escape:
            escape = False
            continue

        if (
            ch == "\\"
            and
            in_string
        ):
            escape = True
            continue

        if ch == '"':
            in_string = (
                not in_string
            )
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1

        elif ch == "}":
            depth -= 1

            if depth == 0:
                return text[
                    start:
                    i + 1
                ]

    return None


def validate_candidate_block(
    value: Any,
) -> tuple[
    bool,
    str | None,
]:
    if not isinstance(
        value,
        dict,
    ):
        return (
            False,
            "BLOCK_NOT_OBJECT",
        )

    for field in BOOL_FIELDS:
        if field not in value:
            return (
                False,
                f"MISSING_FIELD:{field}",
            )

        if not isinstance(
            value[field],
            bool,
        ):
            return (
                False,
                f"NON_BOOL_FIELD:{field}",
            )

    # major_error and minor_error are mutually exclusive.
    if (
        value["major_error"]
        and
        value["minor_error"]
    ):
        return (
            False,
            "MAJOR_AND_MINOR_BOTH_TRUE",
        )

    # Critical semantic failures cannot be called non-major.
    critical_failed = [
        field
        for field in CRITICAL_TRUE_FIELDS
        if value[field] is False
    ]

    if (
        critical_failed
        and
        not value["major_error"]
    ):
        return (
            False,
            (
                "CRITICAL_DIMENSION_FAILED_BUT_MAJOR_ERROR_FALSE:"
                +
                "|".join(
                    critical_failed
                )
            ),
        )

    return (
        True,
        None,
    )


def validate_judgment_logic(
    obj: dict,
) -> tuple[
    bool,
    str | None,
]:
    winner = obj[
        "winner"
    ]

    a = obj[
        "A"
    ]

    b = obj[
        "B"
    ]

    if (
        a["major_error"]
        and
        not b["major_error"]
        and
        winner != "B"
    ):
        return (
            False,
            "A_MAJOR_B_CLEAN_WINNER_MUST_BE_B",
        )

    if (
        b["major_error"]
        and
        not a["major_error"]
        and
        winner != "A"
    ):
        return (
            False,
            "B_MAJOR_A_CLEAN_WINNER_MUST_BE_A",
        )

    if (
        a["major_error"]
        and
        b["major_error"]
        and
        winner != "BOTH_BAD"
    ):
        return (
            False,
            "BOTH_MAJOR_WINNER_MUST_BE_BOTH_BAD",
        )

    if (
        not a["major_error"]
        and
        not b["major_error"]
        and
        winner == "BOTH_BAD"
    ):
        return (
            False,
            "BOTH_CLEAN_CANNOT_BE_BOTH_BAD",
        )

    return (
        True,
        None,
    )


def parse_qwen_result(
    raw_text: str,
) -> tuple[
    dict | None,
    str | None,
]:
    json_text = extract_json_object(
        raw_text
    )

    if json_text is None:
        return (
            None,
            "NO_JSON_OBJECT",
        )

    try:
        obj = json.loads(
            json_text
        )

    except Exception as exc:
        return (
            None,
            (
                "JSON_DECODE_ERROR:"
                f"{type(exc).__name__}"
            ),
        )

    if not isinstance(
        obj,
        dict,
    ):
        return (
            None,
            "ROOT_NOT_OBJECT",
        )

    winner = str(
        obj.get(
            "winner",
            "",
        )
    ).strip().upper()

    confidence = str(
        obj.get(
            "confidence",
            "",
        )
    ).strip().upper()

    if winner not in VALID_WINNERS:
        return (
            None,
            f"INVALID_WINNER:{winner}",
        )

    if confidence not in VALID_CONFIDENCE:
        return (
            None,
            (
                "INVALID_CONFIDENCE:"
                f"{confidence}"
            ),
        )

    a_ok, a_error = validate_candidate_block(
        obj.get("A")
    )

    if not a_ok:
        return (
            None,
            (
                "INVALID_A_BLOCK:"
                f"{a_error}"
            ),
        )

    b_ok, b_error = validate_candidate_block(
        obj.get("B")
    )

    if not b_ok:
        return (
            None,
            (
                "INVALID_B_BLOCK:"
                f"{b_error}"
            ),
        )

    reason = str(
        obj.get(
            "reason",
            "",
        )
    ).strip()

    if not reason:
        return (
            None,
            "EMPTY_REASON",
        )

    obj["winner"] = (
        winner
    )

    obj["confidence"] = (
        confidence
    )

    obj["reason"] = (
        reason
    )

    logic_ok, logic_error = (
        validate_judgment_logic(
            obj
        )
    )

    if not logic_ok:
        return (
            None,
            (
                "LOGICAL_INCONSISTENCY:"
                f"{logic_error}"
            ),
        )

    return (
        obj,
        None,
    )


# ============================================================
# Map A/B to model names
# ============================================================

def map_result_to_models(
    parsed: dict,
    assignment: dict,
) -> dict:
    model_a = assignment[
        "candidate_a_model"
    ]

    model_b = assignment[
        "candidate_b_model"
    ]

    winner_ab = parsed[
        "winner"
    ]

    if winner_ab == "A":
        winner_model = model_a

    elif winner_ab == "B":
        winner_model = model_b

    elif winner_ab == "TIE":
        winner_model = "TIE"

    else:
        winner_model = "BOTH_BAD"

    blocks = {
        model_a: parsed["A"],
        model_b: parsed["B"],
    }

    opus = blocks[
        "OPUS"
    ]

    madlad = blocks[
        "MADLAD"
    ]

    result = {
        "judge_winner_ab": winner_ab,
        "judge_winner": winner_model,
        "judge_confidence": parsed["confidence"],
        "judge_reason": parsed["reason"],
    }

    for field in BOOL_FIELDS:
        result[
            f"opus_{field}"
        ] = bool(
            opus[field]
        )

        result[
            f"madlad_{field}"
        ] = bool(
            madlad[field]
        )

    # Derived from major_error. Qwen does NOT emit this field anymore.
    result[
        "opus_acceptable"
    ] = (
        not result[
            "opus_major_error"
        ]
    )

    result[
        "madlad_acceptable"
    ] = (
        not result[
            "madlad_major_error"
        ]
    )

    return result


# ============================================================
# Generation
# ============================================================

def generate_batch(
    *,
    tokenizer,
    model,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )

    encoded = {
        key:
            value.to(
                "cuda"
            )
        for key, value
        in encoded.items()
    }

    input_length = (
        encoded[
            "input_ids"
        ]
        .shape[1]
    )

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[
        :,
        input_length:
    ]

    decoded = (
        tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
    )

    return [
        str(text).strip()
        for text in decoded
    ]


# ============================================================
# Retry
# ============================================================

def build_retry_chat_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    validation_error: str,
    raw_output: str,
) -> str:
    retry_system = """
You are correcting a logically inconsistent or unparsable translation judgment.

SOURCE is authoritative.
Evaluate both candidates independently again.
Then make the final winner consistent with your major_error fields.

Mandatory consistency:
- A major, B clean => winner B
- B major, A clean => winner A
- both major => BOTH_BAD
- neither major and essentially equal => TIE is allowed

Return exactly one valid JSON object and nothing else.
""".strip()

    retry_user = f"""
The previous judgment failed automatic validation.

VALIDATION ERROR:
{validation_error}

DIRECTION:
{row["direction"]}

SOURCE:
{row["source_text"]}

REFERENCE — auxiliary only:
{row["reference_text"]}

CANDIDATE A:
{assignment["candidate_a_text"]}

CANDIDATE B:
{assignment["candidate_b_text"]}

PREVIOUS INVALID OUTPUT:
{str(raw_output)[:1800]}

{RUBRIC}
""".strip()

    messages = [
        {
            "role": "system",
            "content": retry_system,
        },
        {
            "role": "user",
            "content": retry_user,
        },
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


# ============================================================
# Judge
# ============================================================

def judge_rows(
    *,
    df: pd.DataFrame,
    tokenizer,
    model,
    args,
    checkpoint_file: Path,
) -> pd.DataFrame:
    existing_rows = load_jsonl(
        checkpoint_file
    )

    existing = {}

    for row in existing_rows:
        review_id = str(
            row.get(
                "review_id",
                "",
            )
        )

        if review_id:
            existing[
                review_id
            ] = row

    completed_ids = set(
        existing.keys()
    )

    print(
        "\nExisting checkpoint rows:",
        len(completed_ids),
    )

    pending_df = (
        df[
            ~df[
                "review_id"
            ]
            .astype(str)
            .isin(
                completed_ids
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print(
        "Pending rows:",
        len(pending_df),
    )

    processed_this_run = 0

    for start in range(
        0,
        len(pending_df),
        args.batch_size,
    ):
        batch_df = (
            pending_df.iloc[
                start:
                start + args.batch_size
            ]
            .copy()
        )

        prompts = []
        metadata_rows = []

        for _, row in batch_df.iterrows():
            assignment = assign_candidates(
                row,
                args.seed,
            )

            prompt, prompt_meta = (
                build_safe_chat_prompt(
                    tokenizer,
                    row,
                    assignment,
                    args.max_input_tokens,
                )
            )

            prompts.append(
                prompt
            )

            metadata_rows.append(
                (
                    row,
                    assignment,
                    prompt_meta,
                )
            )

        raw_outputs = generate_batch(
            tokenizer=tokenizer,
            model=model,
            prompts=prompts,
            max_input_tokens=(
                args.max_input_tokens
            ),
            max_new_tokens=(
                args.max_new_tokens
            ),
        )

        batch_counter = Counter()
        parse_ok_count = 0

        for (
            row,
            assignment,
            prompt_meta,
        ), raw_output in zip(
            metadata_rows,
            raw_outputs,
        ):
            parsed, parse_error = (
                parse_qwen_result(
                    raw_output
                )
            )

            retry_count = 0
            validation_error_history = []

            if parse_error:
                validation_error_history.append(
                    parse_error
                )

            while (
                parsed is None
                and
                retry_count
                <
                args.max_retries
            ):
                retry_count += 1

                retry_prompt = (
                    build_retry_chat_prompt(
                        tokenizer,
                        row,
                        assignment,
                        parse_error or "UNKNOWN",
                        raw_output,
                    )
                )

                retry_tokens = count_tokens(
                    tokenizer,
                    retry_prompt,
                )

                if (
                    retry_tokens
                    >
                    args.max_input_tokens
                ):
                    # Retry again without embedding a long invalid response.
                    retry_prompt = (
                        build_retry_chat_prompt(
                            tokenizer,
                            row,
                            assignment,
                            parse_error or "UNKNOWN",
                            "",
                        )
                    )

                raw_output = generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    prompts=[
                        retry_prompt
                    ],
                    max_input_tokens=(
                        args.max_input_tokens
                    ),
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                )[0]

                parsed, parse_error = (
                    parse_qwen_result(
                        raw_output
                    )
                )

                if parse_error:
                    validation_error_history.append(
                        parse_error
                    )

            result = {
                "review_id": str(
                    row["review_id"]
                ),
                "sample_id": str(
                    row["sample_id"]
                ),
                "pair_id": str(
                    row["pair_id"]
                ),
                "direction": str(
                    row["direction"]
                ),
                "source_dataset": str(
                    row["source_dataset"]
                ),
                "source_text": str(
                    row["source_text"]
                ),
                "reference_text": str(
                    row["reference_text"]
                ),
                "opus_prediction": str(
                    row["opus_prediction"]
                ),
                "madlad_prediction": str(
                    row["madlad_prediction"]
                ),
                "teacher_disagreement_score": float(
                    row[
                        "teacher_disagreement_score"
                    ]
                ),
                "candidate_a_model": assignment[
                    "candidate_a_model"
                ],
                "candidate_b_model": assignment[
                    "candidate_b_model"
                ],
                "candidate_order_swapped": bool(
                    assignment[
                        "candidate_order_swapped"
                    ]
                ),
                "prompt_version": PROMPT_VERSION,
                "original_prompt_tokens": int(
                    prompt_meta[
                        "original_prompt_tokens"
                    ]
                ),
                "final_prompt_tokens": int(
                    prompt_meta[
                        "final_prompt_tokens"
                    ]
                ),
                "prompt_truncated": bool(
                    prompt_meta[
                        "prompt_truncated"
                    ]
                ),
                "parse_success": bool(
                    parsed is not None
                ),
                "parse_error": (
                    ""
                    if parsed is not None
                    else str(
                        parse_error
                    )
                ),
                "validation_error_history": (
                    " || ".join(
                        validation_error_history
                    )
                ),
                "retry_count": int(
                    retry_count
                ),
                "raw_qwen_output": str(
                    raw_output
                ),
            }

            if parsed is not None:
                result.update(
                    map_result_to_models(
                        parsed,
                        assignment,
                    )
                )

                result[
                    "logical_consistency_pass"
                ] = True

                parse_ok_count += 1

                batch_counter[
                    result[
                        "judge_winner"
                    ]
                ] += 1

            else:
                result[
                    "judge_winner"
                ] = "PARSE_ERROR"

                result[
                    "logical_consistency_pass"
                ] = False

                batch_counter[
                    "PARSE_ERROR"
                ] += 1

            append_jsonl(
                result,
                checkpoint_file,
            )

            existing[
                result["review_id"]
            ] = result

            processed_this_run += 1

        total_done = (
            len(completed_ids)
            +
            processed_this_run
        )

        print(
            f"{total_done}/{len(df)}"
            f" | parse {parse_ok_count}/{len(batch_df)}"
            f" | {dict(batch_counter)}"
        )

    final_rows = []

    for _, row in df.iterrows():
        review_id = str(
            row["review_id"]
        )

        if review_id not in existing:
            raise RuntimeError(
                f"Missing result for {review_id}"
            )

        final_rows.append(
            existing[
                review_id
            ]
        )

    return pd.DataFrame(
        final_rows
    )


# ============================================================
# Reports
# ============================================================

def build_summary(
    result_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict,
]:
    valid = (
        result_df[
            result_df[
                "parse_success"
            ]
            ==
            True
        ]
        .copy()
    )

    rows = []

    for (
        direction,
        source_dataset,
    ), part in valid.groupby(
        [
            "direction",
            "source_dataset",
        ]
    ):
        total = len(part)

        winner_counts = (
            part[
                "judge_winner"
            ]
            .value_counts()
            .to_dict()
        )

        row = {
            "direction": direction,
            "source_dataset": source_dataset,
            "rows": int(total),
        }

        for winner in [
            "OPUS",
            "MADLAD",
            "TIE",
            "BOTH_BAD",
        ]:
            count = int(
                winner_counts.get(
                    winner,
                    0,
                )
            )

            row[
                f"{winner.lower()}_count"
            ] = count

            row[
                f"{winner.lower()}_percent"
            ] = (
                float(
                    count
                    /
                    total
                    *
                    100
                )
                if total
                else 0.0
            )

        row[
            "opus_acceptable_percent"
        ] = float(
            part[
                "opus_acceptable"
            ]
            .mean()
            *
            100
        )

        row[
            "madlad_acceptable_percent"
        ] = float(
            part[
                "madlad_acceptable"
            ]
            .mean()
            *
            100
        )

        row[
            "opus_major_error_percent"
        ] = float(
            part[
                "opus_major_error"
            ]
            .mean()
            *
            100
        )

        row[
            "madlad_major_error_percent"
        ] = float(
            part[
                "madlad_major_error"
            ]
            .mean()
            *
            100
        )

        row[
            "retried_rows"
        ] = int(
            (
                part[
                    "retry_count"
                ]
                >
                0
            )
            .sum()
        )

        row[
            "mean_retry_count"
        ] = float(
            part[
                "retry_count"
            ]
            .mean()
        )

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    overall_counts = (
        valid[
            "judge_winner"
        ]
        .value_counts()
        .to_dict()
    )

    overall = {
        "rows": int(
            len(result_df)
        ),
        "parse_success": int(
            result_df[
                "parse_success"
            ]
            .sum()
        ),
        "parse_success_percent": float(
            result_df[
                "parse_success"
            ]
            .mean()
            *
            100
        ),
        "prompt_truncated_rows": int(
            result_df[
                "prompt_truncated"
            ]
            .sum()
        ),
        "logical_consistency_pass_rows": int(
            result_df[
                "logical_consistency_pass"
            ]
            .fillna(
                False
            )
            .sum()
        ),
        "retried_rows": int(
            (
                result_df[
                    "retry_count"
                ]
                >
                0
            )
            .sum()
        ),
        "retry_events": int(
            result_df[
                "retry_count"
            ]
            .sum()
        ),
        "winner_counts": {
            winner: int(
                overall_counts.get(
                    winner,
                    0,
                )
            )
            for winner in [
                "OPUS",
                "MADLAD",
                "TIE",
                "BOTH_BAD",
            ]
        },
    }

    if len(valid):
        overall[
            "opus_acceptable_percent"
        ] = float(
            valid[
                "opus_acceptable"
            ]
            .mean()
            *
            100
        )

        overall[
            "madlad_acceptable_percent"
        ] = float(
            valid[
                "madlad_acceptable"
            ]
            .mean()
            *
            100
        )

        overall[
            "opus_major_error_percent"
        ] = float(
            valid[
                "opus_major_error"
            ]
            .mean()
            *
            100
        )

        overall[
            "madlad_major_error_percent"
        ] = float(
            valid[
                "madlad_major_error"
            ]
            .mean()
            *
            100
        )

    return (
        summary_df,
        overall,
    )


def build_position_bias_report(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Model-specific position diagnostic:
    compare a model's win rate when it appears in A vs when it appears in B.
    TIE/BOTH_BAD stay in the denominator.
    """
    valid = (
        result_df[
            result_df[
                "parse_success"
            ]
            ==
            True
        ]
        .copy()
    )

    rows = []

    for direction, direction_part in valid.groupby(
        "direction"
    ):
        for model_name in [
            "OPUS",
            "MADLAD",
        ]:
            for position in [
                "A",
                "B",
            ]:
                position_column = (
                    "candidate_a_model"
                    if position == "A"
                    else "candidate_b_model"
                )

                part = (
                    direction_part[
                        direction_part[
                            position_column
                        ]
                        ==
                        model_name
                    ]
                    .copy()
                )

                if part.empty:
                    continue

                model_wins = int(
                    (
                        part[
                            "judge_winner"
                        ]
                        ==
                        model_name
                    )
                    .sum()
                )

                opponent_name = (
                    "MADLAD"
                    if model_name == "OPUS"
                    else "OPUS"
                )

                opponent_wins = int(
                    (
                        part[
                            "judge_winner"
                        ]
                        ==
                        opponent_name
                    )
                    .sum()
                )

                ties = int(
                    (
                        part[
                            "judge_winner"
                        ]
                        ==
                        "TIE"
                    )
                    .sum()
                )

                both_bad = int(
                    (
                        part[
                            "judge_winner"
                        ]
                        ==
                        "BOTH_BAD"
                    )
                    .sum()
                )

                rows.append(
                    {
                        "direction": direction,
                        "model": model_name,
                        "position": position,
                        "rows": int(
                            len(part)
                        ),
                        "model_win_percent": float(
                            model_wins
                            /
                            len(part)
                            *
                            100
                        ),
                        "opponent_win_percent": float(
                            opponent_wins
                            /
                            len(part)
                            *
                            100
                        ),
                        "tie_percent": float(
                            ties
                            /
                            len(part)
                            *
                            100
                        ),
                        "both_bad_percent": float(
                            both_bad
                            /
                            len(part)
                            *
                            100
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_dimension_report(
    result_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = (
        result_df[
            result_df[
                "parse_success"
            ]
            ==
            True
        ]
        .copy()
    )

    dimensions = [
        (
            "core_meaning",
            "core_meaning_preserved",
            True,
        ),
        (
            "main_action_state",
            "main_action_state_match",
            True,
        ),
        (
            "participant_roles",
            "participant_roles_match",
            True,
        ),
        (
            "entity",
            "entity_match",
            True,
        ),
        (
            "quantity",
            "quantity_match",
            True,
        ),
        (
            "negation_event_status",
            "negation_event_status_match",
            True,
        ),
        (
            "time_event_order",
            "time_event_order_match",
            True,
        ),
        (
            "location_direction",
            "location_direction_match",
            True,
        ),
        (
            "important_information",
            "important_information_preserved",
            True,
        ),
        (
            "unsupported_addition",
            "unsupported_information_added",
            False,
        ),
    ]

    rows = []

    for (
        direction,
        source_dataset,
    ), part in valid.groupby(
        [
            "direction",
            "source_dataset",
        ]
    ):
        for (
            display_name,
            field,
            expected_true,
        ) in dimensions:
            for model_name in [
                "opus",
                "madlad",
            ]:
                column = (
                    f"{model_name}_{field}"
                )

                values = (
                    part[
                        column
                    ]
                    .astype(
                        bool
                    )
                )

                if expected_true:
                    error_percent = float(
                        (
                            ~values
                        )
                        .mean()
                        *
                        100
                    )

                else:
                    error_percent = float(
                        values
                        .mean()
                        *
                        100
                    )

                rows.append(
                    {
                        "direction": direction,
                        "source_dataset": source_dataset,
                        "model": model_name.upper(),
                        "dimension": display_name,
                        "error_percent": error_percent,
                        "rows": int(
                            len(part)
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def save_audit_samples(
    result_df: pd.DataFrame,
    path: Path,
) -> None:
    valid = (
        result_df[
            result_df[
                "parse_success"
            ]
            ==
            True
        ]
        .copy()
    )

    lines = []

    for winner in [
        "OPUS",
        "MADLAD",
        "TIE",
        "BOTH_BAD",
    ]:
        part = (
            valid[
                valid[
                    "judge_winner"
                ]
                ==
                winner
            ]
            .copy()
        )

        if part.empty:
            continue

        part = (
            part.sort_values(
                "teacher_disagreement_score",
                ascending=False,
            )
            .head(8)
        )

        lines.extend(
            [
                "",
                "=" * 120,
                f"WINNER: {winner}",
                "=" * 120,
            ]
        )

        for _, row in part.iterrows():
            lines.extend(
                [
                    "",
                    "-" * 120,
                    f"REVIEW: {row['review_id']}",
                    f"DIRECTION: {row['direction']}",
                    (
                        "SOURCE DATASET: "
                        f"{row['source_dataset']}"
                    ),
                    (
                        f"A={row['candidate_a_model']} "
                        f"| B={row['candidate_b_model']}"
                    ),
                    (
                        f"winner_ab={row['judge_winner_ab']} "
                        f"| winner={row['judge_winner']}"
                    ),
                    (
                        "retry_count="
                        f"{row['retry_count']}"
                    ),
                    "",
                    "SOURCE:",
                    str(
                        row[
                            "source_text"
                        ]
                    ),
                    "",
                    "REFERENCE:",
                    str(
                        row[
                            "reference_text"
                        ]
                    ),
                    "",
                    "OPUS:",
                    str(
                        row[
                            "opus_prediction"
                        ]
                    ),
                    "",
                    "MADLAD:",
                    str(
                        row[
                            "madlad_prediction"
                        ]
                    ),
                    "",
                    "QWEN REASON:",
                    str(
                        row[
                            "judge_reason"
                        ]
                    ),
                ]
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    seed_everything(
        args.seed
    )

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    model_path = Path(
        args.model_path
    )

    input_df = (
        load_disagreement_set(
            project_root
        )
    )

    if args.mode == "calibration":
        work_df = (
            select_calibration_rows(
                input_df,
                calibration_size=(
                    args.calibration_size
                ),
                seed=args.seed,
            )
        )

        mode_dir = (
            "calibration"
        )

        output_name = (
            f"qwen_pairwise_calibration_"
            f"{len(work_df)}_v2"
        )

    else:
        work_df = (
            input_df
            .copy()
            .reset_index(
                drop=True
            )
        )

        mode_dir = (
            "full"
        )

        output_name = (
            "qwen_pairwise_800_v2"
        )

    output_root = (
        project_root
        / "data"
        / "teacher_selection"
        / "zh_en"
        / "v1"
        / "17c_qwen_pairwise"
        / mode_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        output_root
        /
        f"{output_name}_checkpoint.jsonl"
    )

    parquet_file = (
        output_root
        /
        f"{output_name}.parquet"
    )

    csv_file = (
        output_root
        /
        f"{output_name}.csv"
    )

    summary_file = (
        output_root
        /
        f"{output_name}_summary.csv"
    )

    dimension_file = (
        output_root
        /
        f"{output_name}_dimension_errors.csv"
    )

    position_file = (
        output_root
        /
        f"{output_name}_position_bias.csv"
    )

    report_file = (
        output_root
        /
        f"{output_name}_report.json"
    )

    audit_file = (
        output_root
        /
        f"{output_name}_audit.txt"
    )

    v2_files = [
        checkpoint_file,
        parquet_file,
        csv_file,
        summary_file,
        dimension_file,
        position_file,
        report_file,
        audit_file,
    ]

    # IMPORTANT:
    # Only remove V2 files. Never remove the V1 calibration artifacts.
    if args.overwrite:
        for path in v2_files:
            safe_unlink(
                path
            )

    if (
        parquet_file.exists()
        and
        not args.overwrite
        and
        not args.resume
    ):
        raise RuntimeError(
            "\nExisting V2 output found:\n"
            f"{parquet_file}\n\n"
            "Use --overwrite or --resume."
        )

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "ZH-EN TEACHER SELECTION"
    )

    print(
        "STEP 17C V2 - QWEN3-8B PAIRWISE TEACHER JUDGE"
    )

    print(
        "=" * 110
    )

    print(
        "\nMode:",
        args.mode,
    )

    print(
        "Rows:",
        len(work_df),
    )

    print(
        "Prompt version:",
        PROMPT_VERSION,
    )

    print(
        "Max input tokens:",
        args.max_input_tokens,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Max retries:",
        args.max_retries,
    )

    print(
        "\nV2 policy:"
    )

    print(
        "- SOURCE authoritative"
    )

    print(
        "- REFERENCE auxiliary only"
    )

    print(
        "- deterministic A/B randomization"
    )

    print(
        "- acceptable is derived in code, not emitted by Qwen"
    )

    print(
        "- winner/major-error conflicts trigger retry"
    )

    print(
        "- critical semantic contradictions trigger retry"
    )

    print(
        "- V1 calibration artifacts are preserved"
    )

    print(
        "- validation-derived rows are NEVER used for KD training"
    )

    tokenizer, model = load_qwen(
        model_path
    )

    result_df = judge_rows(
        df=work_df,
        tokenizer=tokenizer,
        model=model,
        args=args,
        checkpoint_file=checkpoint_file,
    )

    result_df.to_parquet(
        parquet_file,
        index=False,
    )

    result_df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df, overall = (
        build_summary(
            result_df
        )
    )

    dimension_df = (
        build_dimension_report(
            result_df
        )
    )

    position_df = (
        build_position_bias_report(
            result_df
        )
    )

    summary_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    dimension_df.to_csv(
        dimension_file,
        index=False,
        encoding="utf-8-sig",
    )

    position_df.to_csv(
        position_file,
        index=False,
        encoding="utf-8-sig",
    )

    save_audit_samples(
        result_df,
        audit_file,
    )

    report = {
        "step": "17C",
        "step_version": STEP_VERSION,
        "prompt_version": PROMPT_VERSION,
        "mode": args.mode,
        "input_rows": int(
            len(work_df)
        ),
        "qwen_model": str(
            model_path
        ),
        "policy": {
            "source_authoritative": True,
            "reference_auxiliary_only": True,
            "candidate_order_randomized": True,
            "candidate_order_seed": args.seed,
            "acceptable_emitted_by_qwen": False,
            "acceptable_derived_as_not_major_error": True,
            "logical_consistency_validation": True,
            "critical_dimension_consistency_validation": True,
            "allowed_for_kd_training": False,
            "purpose": (
                "teacher-policy calibration only"
            ),
        },
        "generation": {
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "max_retries": args.max_retries,
        },
        "overall": overall,
        "outputs": {
            "results": str(
                parquet_file
            ),
            "summary": str(
                summary_file
            ),
            "dimension_errors": str(
                dimension_file
            ),
            "position_bias": str(
                position_file
            ),
            "audit": str(
                audit_file
            ),
            "checkpoint": str(
                checkpoint_file
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "status": (
            "V2_CALIBRATION_COMPLETE_REVIEW_REQUIRED"
            if args.mode == "calibration"
            else
            "V2_PAIRWISE_FULL_COMPLETE"
        ),
    }

    save_json(
        report,
        report_file,
    )

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "STEP 17C V2 RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nMode:",
        args.mode,
    )

    print(
        "Rows:",
        len(result_df),
    )

    print(
        "\nParse:"
    )

    print(
        f"{overall['parse_success']}/"
        f"{overall['rows']} "
        f"({overall['parse_success_percent']:.2f}%)"
    )

    print(
        "\nPrompt truncated rows:",
        overall[
            "prompt_truncated_rows"
        ],
    )

    print(
        "Logical consistency pass rows:",
        overall[
            "logical_consistency_pass_rows"
        ],
    )

    print(
        "Retried rows:",
        overall[
            "retried_rows"
        ],
    )

    print(
        "Retry events:",
        overall[
            "retry_events"
        ],
    )

    print(
        "\nOverall winners:"
    )

    for winner, count in (
        overall[
            "winner_counts"
        ]
        .items()
    ):
        print(
            f"{winner}: {count}"
        )

    print(
        "\nStratified winner distribution:"
    )

    if not summary_df.empty:
        display_columns = [
            "direction",
            "source_dataset",
            "rows",
            "opus_count",
            "opus_percent",
            "madlad_count",
            "madlad_percent",
            "tie_count",
            "tie_percent",
            "both_bad_count",
            "both_bad_percent",
            "opus_acceptable_percent",
            "madlad_acceptable_percent",
            "opus_major_error_percent",
            "madlad_major_error_percent",
            "retried_rows",
        ]

        print(
            summary_df[
                display_columns
            ]
            .round(3)
            .to_string(
                index=False
            )
        )

    print(
        "\nPosition-bias diagnostic:"
    )

    if not position_df.empty:
        print(
            position_df
            .round(3)
            .to_string(
                index=False
            )
        )

    print(
        "\nOverall candidate quality:"
    )

    for key in [
        "opus_acceptable_percent",
        "madlad_acceptable_percent",
        "opus_major_error_percent",
        "madlad_major_error_percent",
    ]:
        if key in overall:
            print(
                f"{key}: "
                f"{overall[key]:.2f}%"
            )

    print(
        "\nResults:"
    )

    print(
        parquet_file
    )

    print(
        "\nAudit:"
    )

    print(
        audit_file
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    print(
        "\nSTATUS:"
    )

    print(
        report[
            "status"
        ]
    )

    del model
    del tokenizer

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
