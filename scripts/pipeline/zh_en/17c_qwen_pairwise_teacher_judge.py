from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


STEP_VERSION = "17C_V1"
PROMPT_VERSION = "ZH_EN_TEACHER_PAIRWISE_V1"

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"

VALID_WINNERS = {
    "A",
    "B",
    "TIE",
    "BOTH_BAD",
}

VALID_CONFIDENCE = {
    "HIGH",
    "MEDIUM",
    "LOW",
}


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):

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
            "Step 17C - Qwen3-8B pairwise teacher judge "
            "for OPUS Exp1 vs MADLAD."
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "calibration",
            "full",
        ],
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
        default=1,
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
# JSON helpers
# ============================================================

def save_json(
    obj: Any,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(
    obj: dict,
    path: Path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            +
            "\n"
        )


def load_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():

        return []

    rows = []

    with open(
        path,
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
                pass

    return rows


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

    df = pd.read_parquet(
        path
    )

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

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:

        raise RuntimeError(
            "17B input missing columns: "
            f"{sorted(missing)}"
        )

    if df[
        "review_id"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate review_id found."
        )

    print(
        "\n17B input rows:",
        len(df)
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
        df
        .copy()
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

    if calibration_size > len(df):

        calibration_size = len(df)

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
        .reset_index(drop=True)
    )

    num_strata = len(
        strata
    )

    if num_strata == 0:

        raise RuntimeError(
            "No calibration strata."
        )

    base = (
        calibration_size
        //
        num_strata
    )

    remainder = (
        calibration_size
        %
        num_strata
    )

    parts = []

    for i, row in (
        strata.iterrows()
    ):

        direction = str(
            row[
                "direction"
            ]
        )

        source_dataset = str(
            row[
                "source_dataset"
            ]
        )

        part = (
            df[
                (
                    df[
                        "direction"
                    ]
                    ==
                    direction
                )
                &
                (
                    df[
                        "source_dataset"
                    ].astype(str)
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

        # ----------------------------------------------------
        # Calibration should include both high and medium
        # disagreement instead of only the top-most cases.
        # Input 17B is already a high-disagreement pool.
        # ----------------------------------------------------

        selected = (
            part
            .sample(
                n=requested,
                random_state=(
                    seed + i
                ),
                replace=False,
            )
            .copy()
        )

        parts.append(
            selected
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                "direction",
                "source_dataset",
                "review_id",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        "\nCalibration selected:",
        len(result)
    )

    print(
        "\nCalibration distribution:"
    )

    print(
        result.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
        .size()
        .to_string()
    )

    return result


# ============================================================
# Candidate randomization
# ============================================================

def deterministic_swap(
    review_id: str,
    seed: int,
) -> bool:

    text = (
        f"{seed}|{review_id}"
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()

    value = int(
        digest[:8],
        16,
    )

    return bool(
        value % 2
    )


def assign_candidates(
    row: pd.Series,
    seed: int,
) -> dict:

    swap = (
        deterministic_swap(
            str(
                row[
                    "review_id"
                ]
            ),
            seed,
        )
    )

    opus = str(
        row[
            "opus_prediction"
        ]
    )

    madlad = str(
        row[
            "madlad_prediction"
        ]
    )

    if not swap:

        return {
            "candidate_a_model":
                "OPUS",

            "candidate_a_text":
                opus,

            "candidate_b_model":
                "MADLAD",

            "candidate_b_text":
                madlad,

            "candidate_order_swapped":
                False,
        }

    return {
        "candidate_a_model":
            "MADLAD",

        "candidate_a_text":
            madlad,

        "candidate_b_model":
            "OPUS",

        "candidate_b_text":
            opus,

        "candidate_order_swapped":
            True,
    }


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = """
You are a strict bilingual translation evaluator for Chinese and English.

Your task is to compare two candidate translations of the SAME source sentence.

The SOURCE is authoritative.
The REFERENCE is only an auxiliary example and may itself be imperfect.

Judge semantic fidelity to SOURCE, not lexical similarity to REFERENCE.

Do not reward a candidate merely because its wording resembles the reference.
Do not penalize a candidate for valid paraphrasing, equivalent number formats,
natural Chinese expressions, or plausible transliteration variants.

Return only one valid JSON object.
No markdown.
No chain-of-thought.
No text outside JSON.
""".strip()


RUBRIC = """
Evaluation rules:

A major error includes any substantive change to:
- core meaning
- main action or state
- participant or semantic roles
- entity identity
- number, date, money or quantity
- negation or event status
- time or event order
- location or direction
- important information through omission
- unsupported factual addition

A minor error is a limited issue that does not materially change the factual
proposition, such as slight awkwardness or a small non-critical nuance loss.

Fluency alone is not a major error unless it makes the meaning incorrect or
seriously difficult to understand.

Winner policy:
- If A has a major error and B does not, choose B.
- If B has a major error and A does not, choose A.
- If both preserve the meaning but one is clearly more accurate, choose it.
- If both are acceptable and essentially equal, choose TIE.
- If both contain serious semantic errors, choose BOTH_BAD.

Use this exact JSON schema:

{
  "winner": "A|B|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "A": {
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
  },
  "B": {
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
  },
  "reason": "brief evidence-based explanation"
}
""".strip()


def build_user_prompt(
    row: pd.Series,
    assignment: dict,
) -> str:

    direction = str(
        row[
            "direction"
        ]
    )

    source_text = str(
        row[
            "source_text"
        ]
    )

    reference_text = str(
        row[
            "reference_text"
        ]
    )

    candidate_a = (
        assignment[
            "candidate_a_text"
        ]
    )

    candidate_b = (
        assignment[
            "candidate_b_text"
        ]
    )

    prompt = f"""
PAIRWISE TRANSLATION JUDGMENT

DIRECTION:
{direction}

SOURCE:
{source_text}

REFERENCE — auxiliary only:
{reference_text}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

{RUBRIC}
""".strip()

    return prompt


# ============================================================
# Tokenizer / model
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
        torch.cuda.get_device_name(0)
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
        .to(
            "cuda"
        )
    )

    model.eval()

    print(
        "Model loaded."
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        f"{total_parameters:,}"
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# Chat formatting
# ============================================================

def apply_chat_template(
    tokenizer,
    user_prompt: str,
) -> str:

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },
        {
            "role":
                "user",

            "content":
                user_prompt,
        },
    ]

    # Qwen3 versions differ in whether
    # enable_thinking is accepted.
    try:

        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    except TypeError:

        return (
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


# ============================================================
# Prompt token safeguards
# ============================================================

def count_tokens(
    tokenizer,
    text: str,
) -> int:

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
        )[
            "input_ids"
        ]
    )


def truncate_text_middle(
    tokenizer,
    text: str,
    max_tokens: int,
) -> str:

    ids = tokenizer(
        text,
        add_special_tokens=False,
    )[
        "input_ids"
    ]

    if len(ids) <= max_tokens:

        return text

    if max_tokens < 32:

        max_tokens = 32

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

    kept = (
        ids[:first_n]
        +
        ids[-last_n:]
    )

    return (
        tokenizer.decode(
            kept,
            skip_special_tokens=True,
        )
    )


def build_safe_chat_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    max_input_tokens: int,
) -> tuple[str, dict]:

    # Usually these validation sentences are short and
    # this branch will never truncate.
    user_prompt = (
        build_user_prompt(
            row,
            assignment,
        )
    )

    chat_prompt = (
        apply_chat_template(
            tokenizer,
            user_prompt,
        )
    )

    original_tokens = (
        count_tokens(
            tokenizer,
            chat_prompt,
        )
    )

    metadata = {
        "original_prompt_tokens":
            int(
                original_tokens
            ),

        "final_prompt_tokens":
            int(
                original_tokens
            ),

        "prompt_truncated":
            False,
    }

    if (
        original_tokens
        <=
        max_input_tokens
    ):

        return (
            chat_prompt,
            metadata,
        )

    # --------------------------------------------------------
    # Safety fallback:
    # keep all four semantic fields.
    #
    # Each gets a bounded token budget.
    # Rules remain unchanged.
    # --------------------------------------------------------

    direction = str(
        row[
            "direction"
        ]
    )

    source = str(
        row[
            "source_text"
        ]
    )

    reference = str(
        row[
            "reference_text"
        ]
    )

    candidate_a = str(
        assignment[
            "candidate_a_text"
        ]
    )

    candidate_b = str(
        assignment[
            "candidate_b_text"
        ]
    )

    # 4 fields × ~400 = 1600.
    # Leaves substantial room for system/rubric/schema.
    field_budget = 400

    source = truncate_text_middle(
        tokenizer,
        source,
        field_budget,
    )

    reference = truncate_text_middle(
        tokenizer,
        reference,
        field_budget,
    )

    candidate_a = truncate_text_middle(
        tokenizer,
        candidate_a,
        field_budget,
    )

    candidate_b = truncate_text_middle(
        tokenizer,
        candidate_b,
        field_budget,
    )

    compact_prompt = f"""
PAIRWISE TRANSLATION JUDGMENT

DIRECTION:
{direction}

SOURCE:
{source}

REFERENCE — auxiliary only:
{reference}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

{RUBRIC}
""".strip()

    chat_prompt = (
        apply_chat_template(
            tokenizer,
            compact_prompt,
        )
    )

    final_tokens = (
        count_tokens(
            tokenizer,
            chat_prompt,
        )
    )

    if (
        final_tokens
        >
        max_input_tokens
    ):

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
# JSON parsing
# ============================================================

def extract_json_object(
    text: str,
) -> str | None:

    text = str(
        text
    ).strip()

    # Remove markdown fences if model ignored instruction.
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


BOOL_FIELDS = [
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


def validate_candidate_block(
    value: Any,
) -> bool:

    if not isinstance(
        value,
        dict,
    ):

        return False

    for field in BOOL_FIELDS:

        if field not in value:

            return False

        if not isinstance(
            value[
                field
            ],
            bool,
        ):

            return False

    return True


def parse_qwen_result(
    raw_text: str,
) -> tuple[
    dict | None,
    str | None,
]:

    json_text = (
        extract_json_object(
            raw_text
        )
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
            (
                "INVALID_WINNER:"
                f"{winner}"
            ),
        )

    if (
        confidence
        not in
        VALID_CONFIDENCE
    ):

        return (
            None,
            (
                "INVALID_CONFIDENCE:"
                f"{confidence}"
            ),
        )

    if not validate_candidate_block(
        obj.get(
            "A"
        )
    ):

        return (
            None,
            "INVALID_A_BLOCK",
        )

    if not validate_candidate_block(
        obj.get(
            "B"
        )
    ):

        return (
            None,
            "INVALID_B_BLOCK",
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

    obj[
        "winner"
    ] = winner

    obj[
        "confidence"
    ] = confidence

    obj[
        "reason"
    ] = reason

    return (
        obj,
        None,
    )


# ============================================================
# Map A/B back to models
# ============================================================

def map_result_to_models(
    parsed: dict,
    assignment: dict,
) -> dict:

    model_a = (
        assignment[
            "candidate_a_model"
        ]
    )

    model_b = (
        assignment[
            "candidate_b_model"
        ]
    )

    winner_ab = (
        parsed[
            "winner"
        ]
    )

    if winner_ab == "A":

        winner_model = (
            model_a
        )

    elif winner_ab == "B":

        winner_model = (
            model_b
        )

    elif winner_ab == "TIE":

        winner_model = (
            "TIE"
        )

    else:

        winner_model = (
            "BOTH_BAD"
        )

    blocks = {
        model_a:
            parsed[
                "A"
            ],

        model_b:
            parsed[
                "B"
            ],
    }

    opus = blocks[
        "OPUS"
    ]

    madlad = blocks[
        "MADLAD"
    ]

    result = {

        "judge_winner_ab":
            winner_ab,

        "judge_winner":
            winner_model,

        "judge_confidence":
            parsed[
                "confidence"
            ],

        "judge_reason":
            parsed[
                "reason"
            ],
    }

    for field in BOOL_FIELDS:

        result[
            f"opus_{field}"
        ] = bool(
            opus[
                field
            ]
        )

        result[
            f"madlad_{field}"
        ] = bool(
            madlad[
                field
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

        generated = (
            model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=(
                    max_new_tokens
                ),
                use_cache=True,
                pad_token_id=(
                    tokenizer.pad_token_id
                ),
                eos_token_id=(
                    tokenizer.eos_token_id
                ),
            )
        )

    new_tokens = (
        generated[
            :,
            input_length:
        ]
    )

    decoded = (
        tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )
    )

    return [
        str(x).strip()
        for x in decoded
    ]


# ============================================================
# Retry prompt
# ============================================================

def build_retry_prompt(
    tokenizer,
    original_prompt: str,
    raw_output: str,
) -> str:

    messages = [
        {
            "role":
                "system",

            "content":
                (
                    "Return exactly one valid JSON object. "
                    "Do not include markdown or commentary."
                ),
        },
        {
            "role":
                "user",

            "content":
                f"""
Your previous answer could not be parsed.

Re-evaluate the ORIGINAL translation comparison below and return ONLY
one valid JSON object using the requested schema.

ORIGINAL TASK:
{original_prompt}

INVALID PREVIOUS OUTPUT:
{raw_output[:2000]}
""".strip(),
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
# Judge rows
# ============================================================

def judge_rows(
    *,
    df: pd.DataFrame,
    tokenizer,
    model,
    args,
    checkpoint_file: Path,
) -> pd.DataFrame:

    existing_rows = (
        load_jsonl(
            checkpoint_file
        )
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
        len(
            completed_ids
        )
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
        len(
            pending_df
        )
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
                start
                +
                args.batch_size
            ]
            .copy()
        )

        prompts = []
        metadata_rows = []

        for _, row in (
            batch_df.iterrows()
        ):

            assignment = (
                assign_candidates(
                    row,
                    args.seed,
                )
            )

            (
                prompt,
                prompt_meta,
            ) = (
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
                    prompt,
                    prompt_meta,
                )
            )

        raw_outputs = (
            generate_batch(
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
        )

        batch_counter = Counter()

        parse_ok_count = 0

        for (
            row,
            assignment,
            prompt,
            prompt_meta,
        ), raw_output in zip(
            metadata_rows,
            raw_outputs,
        ):

            (
                parsed,
                parse_error,
            ) = (
                parse_qwen_result(
                    raw_output
                )
            )

            retry_count = 0

            # ------------------------------------------------
            # Retry only parsing failures.
            # ------------------------------------------------

            while (
                parsed is None
                and
                retry_count
                <
                args.max_retries
            ):

                retry_count += 1

                retry_prompt = (
                    build_retry_prompt(
                        tokenizer,
                        prompt,
                        raw_output,
                    )
                )

                retry_output = (
                    generate_batch(
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
                )

                raw_output = (
                    retry_output
                )

                (
                    parsed,
                    parse_error,
                ) = (
                    parse_qwen_result(
                        raw_output
                    )
                )

            result = {
                "review_id":
                    str(
                        row[
                            "review_id"
                        ]
                    ),

                "sample_id":
                    str(
                        row[
                            "sample_id"
                        ]
                    ),

                "pair_id":
                    str(
                        row[
                            "pair_id"
                        ]
                    ),

                "direction":
                    str(
                        row[
                            "direction"
                        ]
                    ),

                "source_dataset":
                    str(
                        row[
                            "source_dataset"
                        ]
                    ),

                "source_text":
                    str(
                        row[
                            "source_text"
                        ]
                    ),

                "reference_text":
                    str(
                        row[
                            "reference_text"
                        ]
                    ),

                "opus_prediction":
                    str(
                        row[
                            "opus_prediction"
                        ]
                    ),

                "madlad_prediction":
                    str(
                        row[
                            "madlad_prediction"
                        ]
                    ),

                "teacher_disagreement_score":
                    float(
                        row[
                            "teacher_disagreement_score"
                        ]
                    ),

                "candidate_a_model":
                    assignment[
                        "candidate_a_model"
                    ],

                "candidate_b_model":
                    assignment[
                        "candidate_b_model"
                    ],

                "candidate_order_swapped":
                    bool(
                        assignment[
                            "candidate_order_swapped"
                        ]
                    ),

                "prompt_version":
                    PROMPT_VERSION,

                "original_prompt_tokens":
                    int(
                        prompt_meta[
                            "original_prompt_tokens"
                        ]
                    ),

                "final_prompt_tokens":
                    int(
                        prompt_meta[
                            "final_prompt_tokens"
                        ]
                    ),

                "prompt_truncated":
                    bool(
                        prompt_meta[
                            "prompt_truncated"
                        ]
                    ),

                "parse_success":
                    bool(
                        parsed
                        is not None
                    ),

                "parse_error":
                    (
                        ""
                        if parsed
                        is not None
                        else str(
                            parse_error
                        )
                    ),

                "retry_count":
                    int(
                        retry_count
                    ),

                "raw_qwen_output":
                    str(
                        raw_output
                    ),
            }

            if parsed is not None:

                mapped = (
                    map_result_to_models(
                        parsed,
                        assignment,
                    )
                )

                result.update(
                    mapped
                )

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

                batch_counter[
                    "PARSE_ERROR"
                ] += 1

            append_jsonl(
                result,
                checkpoint_file,
            )

            existing[
                result[
                    "review_id"
                ]
            ] = result

            processed_this_run += 1

        total_done = (
            len(
                completed_ids
            )
            +
            processed_this_run
        )

        print(
            f"{total_done}/{len(df)}"
            f" | parse "
            f"{parse_ok_count}/"
            f"{len(batch_df)}"
            f" | "
            f"{dict(batch_counter)}"
        )

    # ========================================================
    # Build ordered final dataframe
    # ========================================================

    final_rows = []

    for _, row in df.iterrows():

        review_id = str(
            row[
                "review_id"
            ]
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
# Summary
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
        source_dataset
    ), part in (
        valid.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
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
            "direction":
                direction,

            "source_dataset":
                source_dataset,

            "rows":
                int(
                    total
                ),
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

        # Candidate health
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
        "rows":
            int(
                len(
                    result_df
                )
            ),

        "parse_success":
            int(
                result_df[
                    "parse_success"
                ]
                .sum()
            ),

        "parse_success_percent":
            float(
                result_df[
                    "parse_success"
                ]
                .mean()
                *
                100
            ),

        "prompt_truncated_rows":
            int(
                result_df[
                    "prompt_truncated"
                ]
                .sum()
            ),

        "winner_counts":
            {
                winner:
                    int(
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


# ============================================================
# Dimension error report
# ============================================================

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
        source_dataset
    ), part in (
        valid.groupby(
            [
                "direction",
                "source_dataset",
            ]
        )
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
                    .astype(bool)
                )

                if expected_true:

                    error_rate = float(
                        (
                            ~values
                        )
                        .mean()
                        *
                        100
                    )

                else:

                    # unsupported_information_added:
                    # True itself is the error.
                    error_rate = float(
                        values
                        .mean()
                        *
                        100
                    )

                rows.append(
                    {
                        "direction":
                            direction,

                        "source_dataset":
                            source_dataset,

                        "model":
                            model_name.upper(),

                        "dimension":
                            display_name,

                        "error_percent":
                            error_rate,

                        "rows":
                            int(
                                len(part)
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Audit text
# ============================================================

def save_audit_samples(
    result_df: pd.DataFrame,
    path: Path,
):

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

    samples = []

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

        # Show strongest-disagreement examples.
        part = (
            part
            .sort_values(
                "teacher_disagreement_score",
                ascending=False,
            )
            .head(8)
        )

        samples.append(
            "\n"
            +
            "=" * 120
        )

        samples.append(
            f"WINNER: {winner}"
        )

        samples.append(
            "=" * 120
        )

        for _, row in (
            part.iterrows()
        ):

            samples.append(
                "\n"
                +
                "-" * 120
            )

            samples.append(
                f"REVIEW: {row['review_id']}"
            )

            samples.append(
                f"DIRECTION: {row['direction']}"
            )

            samples.append(
                f"SOURCE: {row['source_dataset']}"
            )

            samples.append(
                "\nSOURCE TEXT:"
            )

            samples.append(
                str(
                    row[
                        "source_text"
                    ]
                )
            )

            samples.append(
                "\nREFERENCE:"
            )

            samples.append(
                str(
                    row[
                        "reference_text"
                    ]
                )
            )

            samples.append(
                "\nOPUS:"
            )

            samples.append(
                str(
                    row[
                        "opus_prediction"
                    ]
                )
            )

            samples.append(
                "\nMADLAD:"
            )

            samples.append(
                str(
                    row[
                        "madlad_prediction"
                    ]
                )
            )

            samples.append(
                "\nQWEN:"
            )

            samples.append(
                f"winner={row['judge_winner']}"
            )

            samples.append(
                f"confidence={row['judge_confidence']}"
            )

            samples.append(
                str(
                    row[
                        "judge_reason"
                    ]
                )
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                samples
            )
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

    # ========================================================
    # Mode
    # ========================================================

    if (
        args.mode
        ==
        "calibration"
    ):

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
            f"{len(work_df)}_v1"
        )

    else:

        work_df = (
            input_df
            .copy()
            .reset_index(drop=True)
        )

        mode_dir = (
            "full"
        )

        output_name = (
            "qwen_pairwise_800_v1"
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

    # ========================================================
    # Output policy
    # ========================================================

    if (
        args.overwrite
        and
        output_root.exists()
    ):

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        parquet_file.exists()
        and
        not args.overwrite
        and
        not args.resume
    ):

        raise RuntimeError(
            "\nExisting output found:\n"
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
        "STEP 17C - QWEN3-8B PAIRWISE TEACHER JUDGE"
    )

    print(
        "=" * 110
    )

    print(
        "\nMode:",
        args.mode
    )

    print(
        "Rows:",
        len(
            work_df
        )
    )

    print(
        "Prompt version:",
        PROMPT_VERSION
    )

    print(
        "Max input tokens:",
        args.max_input_tokens
    )

    print(
        "Batch size:",
        args.batch_size
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "SOURCE is authoritative."
    )

    print(
        "REFERENCE is auxiliary only."
    )

    print(
        "Candidate A/B order is deterministically randomized."
    )

    print(
        "These validation-derived rows are NOT allowed "
        "into KD training."
    )

    # ========================================================
    # Model
    # ========================================================

    tokenizer, model = (
        load_qwen(
            model_path
        )
    )

    # ========================================================
    # Judge
    # ========================================================

    result_df = (
        judge_rows(
            df=work_df,
            tokenizer=tokenizer,
            model=model,
            args=args,
            checkpoint_file=(
                checkpoint_file
            ),
        )
    )

    # ========================================================
    # Save raw results
    # ========================================================

    result_df.to_parquet(
        parquet_file,
        index=False,
    )

    result_df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Reports
    # ========================================================

    (
        summary_df,
        overall,
    ) = (
        build_summary(
            result_df
        )
    )

    dimension_df = (
        build_dimension_report(
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

    save_audit_samples(
        result_df,
        audit_file,
    )

    report = {

        "step":
            "17C",

        "step_version":
            STEP_VERSION,

        "prompt_version":
            PROMPT_VERSION,

        "mode":
            args.mode,

        "input_rows":
            int(
                len(
                    work_df
                )
            ),

        "qwen_model":
            str(
                model_path
            ),

        "policy": {

            "source_authoritative":
                True,

            "reference_auxiliary_only":
                True,

            "candidate_order_randomized":
                True,

            "candidate_order_seed":
                args.seed,

            "allowed_for_kd_training":
                False,

            "purpose":
                "teacher-policy calibration",
        },

        "generation": {

            "batch_size":
                args.batch_size,

            "max_input_tokens":
                args.max_input_tokens,

            "max_new_tokens":
                args.max_new_tokens,

            "do_sample":
                False,

            "max_retries":
                args.max_retries,
        },

        "overall":
            overall,

        "outputs": {

            "results":
                str(
                    parquet_file
                ),

            "summary":
                str(
                    summary_file
                ),

            "dimension_errors":
                str(
                    dimension_file
                ),

            "audit":
                str(
                    audit_file
                ),

            "checkpoint":
                str(
                    checkpoint_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            (
                "CALIBRATION_COMPLETE_REVIEW_REQUIRED"
                if
                args.mode
                ==
                "calibration"
                else
                "PAIRWISE_FULL_COMPLETE"
            ),
    }

    save_json(
        report,
        report_file,
    )

    # ========================================================
    # Console
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 17C RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nMode:",
        args.mode
    )

    print(
        "Rows:",
        len(
            result_df
        )
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
        ]
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
    ]

    if not summary_df.empty:

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
        "\nAudit samples:"
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