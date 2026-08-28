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


STEP_VERSION = "17C_V3"
PROMPT_VERSION = "ZH_EN_TEACHER_PAIRWISE_V3"
ADJUDICATION_VERSION = "ZH_EN_TEACHER_ADJUDICATION_V1"

DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/Qwen3-8B"

VALID_PAIR_WINNERS = {"A", "B", "TIE", "BOTH_BAD"}
VALID_CONFIDENCE = {"HIGH", "MEDIUM", "LOW"}
VALID_QUALITY = {"GOOD", "MINOR", "MAJOR"}
VALID_PRIMARY_ERRORS = {
    "NONE",
    "MEANING",
    "OMISSION",
    "ADDITION",
    "ENTITY",
    "NUMBER",
    "NEGATION",
    "TIME",
    "LOCATION",
    "ROLE",
    "FLUENCY",
    "OTHER",
}

QUALITY_SEVERITY = {
    "GOOD": 0,
    "MINOR": 1,
    "MAJOR": 2,
}


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
            "Step 17C V3 - dual-order Qwen3-8B pairwise teacher judge "
            "for OPUS Exp1 vs MADLAD with independent adjudication."
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
        default=384,
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
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
# Generic I/O
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

    rows = []

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
                # Ignore a partially-written line after interruption.
                continue

    return rows


def safe_unlink(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


# ============================================================
# Load 17B disagreement data
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
        "\n17B distribution:"
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
# Same calibration selection as V1 / V2
# ============================================================

def select_calibration_rows(
    df: pd.DataFrame,
    calibration_size: int,
    seed: int,
) -> pd.DataFrame:
    """
    Mirrors V1/V2 sampling exactly:
    - same four direction x source strata
    - same random_state = seed + stratum index
    - same sorting

    Therefore seed=2026,size=200 should reproduce the same 200 rows.
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

        n = min(
            base
            +
            (
                1
                if i < remainder
                else 0
            ),
            len(part),
        )

        selected = (
            part.sample(
                n=n,
                replace=False,
                random_state=(
                    seed + i
                ),
            )
            .copy()
        )

        parts.append(
            selected
        )

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


def verify_same_calibration_as_v2(
    project_root: Path,
    work_df: pd.DataFrame,
) -> None:
    """
    If V2 calibration exists, verify V3 uses exactly the same review_ids.
    This is diagnostic only but prevents accidental apples-to-oranges comparison.
    """
    v2_path = (
        project_root
        / "data"
        / "teacher_selection"
        / "zh_en"
        / "v1"
        / "17c_qwen_pairwise"
        / "calibration"
        / f"qwen_pairwise_calibration_{len(work_df)}_v2.parquet"
    )

    if not v2_path.exists():
        print(
            "\nV2 calibration not found; "
            "same-sample verification skipped."
        )
        return

    v2 = pd.read_parquet(v2_path)

    if "review_id" not in v2.columns:
        print(
            "\nWARNING: V2 calibration has no review_id; "
            "same-sample verification skipped."
        )
        return

    v2_ids = set(
        v2["review_id"]
        .astype(str)
        .tolist()
    )

    v3_ids = set(
        work_df["review_id"]
        .astype(str)
        .tolist()
    )

    same = (
        v2_ids
        ==
        v3_ids
    )

    print(
        "\nSame calibration rows as V2:",
        same,
    )

    if not same:
        only_v2 = sorted(
            v2_ids - v3_ids
        )[:10]

        only_v3 = sorted(
            v3_ids - v2_ids
        )[:10]

        raise RuntimeError(
            "V3 calibration rows do not match V2.\n"
            f"Only V2: {only_v2}\n"
            f"Only V3: {only_v3}"
        )


# ============================================================
# Position assignment
# ============================================================

def stable_bit(
    review_id: str,
    seed: int,
    salt: str,
) -> bool:
    digest = hashlib.sha256(
        (
            f"{seed}|"
            f"{salt}|"
            f"{review_id}"
        ).encode(
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


def make_pass1_assignment(
    row: pd.Series,
    seed: int,
) -> dict:
    """
    Pass 1 is balanced/deterministic across A/B.
    Pass 2 always uses the exact reverse.
    """
    swap = stable_bit(
        str(
            row["review_id"]
        ),
        seed,
        "pairwise_pass1",
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
        }

    return {
        "candidate_a_model": "MADLAD",
        "candidate_a_text": madlad,
        "candidate_b_model": "OPUS",
        "candidate_b_text": opus,
    }


def reverse_assignment(
    assignment: dict,
) -> dict:
    return {
        "candidate_a_model": assignment[
            "candidate_b_model"
        ],
        "candidate_a_text": assignment[
            "candidate_b_text"
        ],
        "candidate_b_model": assignment[
            "candidate_a_model"
        ],
        "candidate_b_text": assignment[
            "candidate_a_text"
        ],
    }


def make_adjudication_assignment(
    row: pd.Series,
    seed: int,
) -> dict:
    """
    Independent X/Y order for adjudication.
    This is NOT inherited from Pass 1 or Pass 2.
    """
    swap = stable_bit(
        str(
            row["review_id"]
        ),
        seed,
        "adjudication_xy",
    )

    opus = str(
        row["opus_prediction"]
    )

    madlad = str(
        row["madlad_prediction"]
    )

    if not swap:
        return {
            "candidate_x_model": "OPUS",
            "candidate_x_text": opus,
            "candidate_y_model": "MADLAD",
            "candidate_y_text": madlad,
        }

    return {
        "candidate_x_model": "MADLAD",
        "candidate_x_text": madlad,
        "candidate_y_model": "OPUS",
        "candidate_y_text": opus,
    }


# ============================================================
# Pairwise prompt
# ============================================================

PAIRWISE_SYSTEM_PROMPT = """
You are a strict bilingual Chinese-English translation evaluator.

Compare two translations of the SAME source sentence.

SOURCE is authoritative.
REFERENCE is auxiliary only and may itself be imperfect.

Judge semantic fidelity to SOURCE, not lexical overlap with REFERENCE.

Evaluate A and B independently BEFORE choosing the winner.

Quality labels:
GOOD  = no substantive translation error.
MINOR = small non-critical issue; factual proposition remains intact.
MAJOR = substantive semantic error, important omission/addition, wrong role,
        entity, number, negation, time/order, or location/direction relation.

Winner consistency:
- A=MAJOR and B!=MAJOR -> winner B
- B=MAJOR and A!=MAJOR -> winner A
- both MAJOR -> BOTH_BAD
- same quality and essentially equivalent -> TIE
- when neither is MAJOR, choose A or B only if one is clearly more faithful

Equivalent number formats and valid paraphrases are not errors.

Return exactly ONE valid JSON object.
No markdown.
No chain-of-thought.
No text outside JSON.
""".strip()


PAIRWISE_SCHEMA = """
Use exactly this JSON schema:

{
  "a_quality": "GOOD|MINOR|MAJOR",
  "b_quality": "GOOD|MINOR|MAJOR",
  "winner": "A|B|TIE|BOTH_BAD",
  "confidence": "HIGH|MEDIUM|LOW",
  "primary_error_a": "NONE|MEANING|OMISSION|ADDITION|ENTITY|NUMBER|NEGATION|TIME|LOCATION|ROLE|FLUENCY|OTHER",
  "primary_error_b": "NONE|MEANING|OMISSION|ADDITION|ENTITY|NUMBER|NEGATION|TIME|LOCATION|ROLE|FLUENCY|OTHER",
  "reason": "brief evidence-based explanation"
}
""".strip()


def build_pairwise_user_prompt(
    row: pd.Series,
    assignment: dict,
) -> str:
    return f"""
PAIRWISE TEACHER JUDGMENT

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

{PAIRWISE_SCHEMA}
""".strip()


# ============================================================
# Adjudication prompt
# ============================================================

ADJUDICATION_SYSTEM_PROMPT = """
You are performing an independent final quality adjudication of two
Chinese-English translations.

SOURCE is authoritative.
REFERENCE is auxiliary only and may itself be imperfect.

Do NOT choose a winner.
Judge Translation X and Translation Y independently.

Quality labels:
GOOD  = no substantive translation error.
MINOR = small non-critical issue; factual proposition remains intact.
MAJOR = substantive semantic error or important omission/addition.

Equivalent number formats, valid paraphrases, and plausible transliterations
must not be penalized merely because wording differs from the reference.

Return exactly ONE valid JSON object.
No markdown.
No chain-of-thought.
No text outside JSON.
""".strip()


ADJUDICATION_SCHEMA = """
Use exactly this JSON schema:

{
  "x_quality": "GOOD|MINOR|MAJOR",
  "y_quality": "GOOD|MINOR|MAJOR",
  "confidence": "HIGH|MEDIUM|LOW",
  "primary_error_x": "NONE|MEANING|OMISSION|ADDITION|ENTITY|NUMBER|NEGATION|TIME|LOCATION|ROLE|FLUENCY|OTHER",
  "primary_error_y": "NONE|MEANING|OMISSION|ADDITION|ENTITY|NUMBER|NEGATION|TIME|LOCATION|ROLE|FLUENCY|OTHER",
  "reason": "brief evidence-based explanation"
}
""".strip()


def build_adjudication_user_prompt(
    row: pd.Series,
    assignment: dict,
) -> str:
    return f"""
INDEPENDENT FINAL ADJUDICATION

DIRECTION:
{row["direction"]}

SOURCE:
{row["source_text"]}

REFERENCE — auxiliary only:
{row["reference_text"]}

TRANSLATION X:
{assignment["candidate_x_text"]}

TRANSLATION Y:
{assignment["candidate_y_text"]}

{ADJUDICATION_SCHEMA}
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

    tokenizer.padding_side = "left"

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

    # Deterministic judging.
    # Also remove irrelevant warnings from Qwen's generation config.
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
    system_prompt: str,
    user_prompt: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": system_prompt,
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
        max_tokens,
        32,
    )

    first_n = max_tokens // 2
    last_n = max_tokens - first_n

    return tokenizer.decode(
        ids[:first_n]
        +
        ids[-last_n:],
        skip_special_tokens=True,
    )


def build_safe_pairwise_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    max_input_tokens: int,
) -> tuple[
    str,
    dict,
]:
    user_prompt = build_pairwise_user_prompt(
        row,
        assignment,
    )

    chat_prompt = apply_chat_template(
        tokenizer,
        PAIRWISE_SYSTEM_PROMPT,
        user_prompt,
    )

    original_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    meta = {
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
            meta,
        )

    field_budget = 450

    compact_user = f"""
PAIRWISE TEACHER JUDGMENT

DIRECTION:
{row["direction"]}

SOURCE:
{truncate_text_middle(tokenizer, str(row["source_text"]), field_budget)}

REFERENCE — auxiliary only:
{truncate_text_middle(tokenizer, str(row["reference_text"]), field_budget)}

CANDIDATE A:
{truncate_text_middle(tokenizer, str(assignment["candidate_a_text"]), field_budget)}

CANDIDATE B:
{truncate_text_middle(tokenizer, str(assignment["candidate_b_text"]), field_budget)}

{PAIRWISE_SCHEMA}
""".strip()

    chat_prompt = apply_chat_template(
        tokenizer,
        PAIRWISE_SYSTEM_PROMPT,
        compact_user,
    )

    final_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    if final_tokens > max_input_tokens:
        raise RuntimeError(
            "Pairwise prompt still exceeds token limit after safe compression.\n"
            f"review_id={row['review_id']}\n"
            f"tokens={final_tokens}\n"
            f"limit={max_input_tokens}"
        )

    meta["final_prompt_tokens"] = int(
        final_tokens
    )

    meta["prompt_truncated"] = True

    return (
        chat_prompt,
        meta,
    )


def build_safe_adjudication_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    max_input_tokens: int,
) -> tuple[
    str,
    dict,
]:
    user_prompt = build_adjudication_user_prompt(
        row,
        assignment,
    )

    chat_prompt = apply_chat_template(
        tokenizer,
        ADJUDICATION_SYSTEM_PROMPT,
        user_prompt,
    )

    original_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    meta = {
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
            meta,
        )

    field_budget = 450

    compact_user = f"""
INDEPENDENT FINAL ADJUDICATION

DIRECTION:
{row["direction"]}

SOURCE:
{truncate_text_middle(tokenizer, str(row["source_text"]), field_budget)}

REFERENCE — auxiliary only:
{truncate_text_middle(tokenizer, str(row["reference_text"]), field_budget)}

TRANSLATION X:
{truncate_text_middle(tokenizer, str(assignment["candidate_x_text"]), field_budget)}

TRANSLATION Y:
{truncate_text_middle(tokenizer, str(assignment["candidate_y_text"]), field_budget)}

{ADJUDICATION_SCHEMA}
""".strip()

    chat_prompt = apply_chat_template(
        tokenizer,
        ADJUDICATION_SYSTEM_PROMPT,
        compact_user,
    )

    final_tokens = count_tokens(
        tokenizer,
        chat_prompt,
    )

    if final_tokens > max_input_tokens:
        raise RuntimeError(
            "Adjudication prompt still exceeds token limit after safe compression.\n"
            f"review_id={row['review_id']}\n"
            f"tokens={final_tokens}\n"
            f"limit={max_input_tokens}"
        )

    meta["final_prompt_tokens"] = int(
        final_tokens
    )

    meta["prompt_truncated"] = True

    return (
        chat_prompt,
        meta,
    )


# ============================================================
# JSON extraction
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


def normalize_label(
    value: Any,
) -> str:
    return str(
        value
    ).strip().upper()


# ============================================================
# Pairwise parser
# ============================================================

def validate_pairwise_logic(
    obj: dict,
) -> tuple[
    bool,
    str | None,
]:
    a_quality = obj[
        "a_quality"
    ]

    b_quality = obj[
        "b_quality"
    ]

    winner = obj[
        "winner"
    ]

    a_major = (
        a_quality
        ==
        "MAJOR"
    )

    b_major = (
        b_quality
        ==
        "MAJOR"
    )

    if (
        a_major
        and
        not b_major
        and
        winner != "B"
    ):
        return (
            False,
            "A_MAJOR_B_NONMAJOR_WINNER_MUST_BE_B",
        )

    if (
        b_major
        and
        not a_major
        and
        winner != "A"
    ):
        return (
            False,
            "B_MAJOR_A_NONMAJOR_WINNER_MUST_BE_A",
        )

    if (
        a_major
        and
        b_major
        and
        winner != "BOTH_BAD"
    ):
        return (
            False,
            "BOTH_MAJOR_WINNER_MUST_BE_BOTH_BAD",
        )

    if (
        not a_major
        and
        not b_major
        and
        winner == "BOTH_BAD"
    ):
        return (
            False,
            "NONMAJOR_PAIR_CANNOT_BE_BOTH_BAD",
        )

    return (
        True,
        None,
    )


def parse_pairwise_result(
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

    required = [
        "a_quality",
        "b_quality",
        "winner",
        "confidence",
        "primary_error_a",
        "primary_error_b",
        "reason",
    ]

    missing = [
        field
        for field in required
        if field not in obj
    ]

    if missing:
        return (
            None,
            "MISSING_FIELDS:"
            +
            "|".join(
                missing
            ),
        )

    a_quality = normalize_label(
        obj["a_quality"]
    )

    b_quality = normalize_label(
        obj["b_quality"]
    )

    winner = normalize_label(
        obj["winner"]
    )

    confidence = normalize_label(
        obj["confidence"]
    )

    primary_error_a = normalize_label(
        obj["primary_error_a"]
    )

    primary_error_b = normalize_label(
        obj["primary_error_b"]
    )

    reason = str(
        obj["reason"]
    ).strip()

    if a_quality not in VALID_QUALITY:
        return (
            None,
            f"INVALID_A_QUALITY:{a_quality}",
        )

    if b_quality not in VALID_QUALITY:
        return (
            None,
            f"INVALID_B_QUALITY:{b_quality}",
        )

    if winner not in VALID_PAIR_WINNERS:
        return (
            None,
            f"INVALID_WINNER:{winner}",
        )

    if confidence not in VALID_CONFIDENCE:
        return (
            None,
            f"INVALID_CONFIDENCE:{confidence}",
        )

    if primary_error_a not in VALID_PRIMARY_ERRORS:
        return (
            None,
            f"INVALID_PRIMARY_ERROR_A:{primary_error_a}",
        )

    if primary_error_b not in VALID_PRIMARY_ERRORS:
        return (
            None,
            f"INVALID_PRIMARY_ERROR_B:{primary_error_b}",
        )

    if not reason:
        return (
            None,
            "EMPTY_REASON",
        )

    obj = {
        "a_quality": a_quality,
        "b_quality": b_quality,
        "winner": winner,
        "confidence": confidence,
        "primary_error_a": primary_error_a,
        "primary_error_b": primary_error_b,
        "reason": reason,
    }

    logic_ok, logic_error = (
        validate_pairwise_logic(
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
# Adjudication parser
# ============================================================

def parse_adjudication_result(
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

    required = [
        "x_quality",
        "y_quality",
        "confidence",
        "primary_error_x",
        "primary_error_y",
        "reason",
    ]

    missing = [
        field
        for field in required
        if field not in obj
    ]

    if missing:
        return (
            None,
            "MISSING_FIELDS:"
            +
            "|".join(
                missing
            ),
        )

    x_quality = normalize_label(
        obj["x_quality"]
    )

    y_quality = normalize_label(
        obj["y_quality"]
    )

    confidence = normalize_label(
        obj["confidence"]
    )

    primary_error_x = normalize_label(
        obj["primary_error_x"]
    )

    primary_error_y = normalize_label(
        obj["primary_error_y"]
    )

    reason = str(
        obj["reason"]
    ).strip()

    if x_quality not in VALID_QUALITY:
        return (
            None,
            f"INVALID_X_QUALITY:{x_quality}",
        )

    if y_quality not in VALID_QUALITY:
        return (
            None,
            f"INVALID_Y_QUALITY:{y_quality}",
        )

    if confidence not in VALID_CONFIDENCE:
        return (
            None,
            f"INVALID_CONFIDENCE:{confidence}",
        )

    if primary_error_x not in VALID_PRIMARY_ERRORS:
        return (
            None,
            f"INVALID_PRIMARY_ERROR_X:{primary_error_x}",
        )

    if primary_error_y not in VALID_PRIMARY_ERRORS:
        return (
            None,
            f"INVALID_PRIMARY_ERROR_Y:{primary_error_y}",
        )

    if not reason:
        return (
            None,
            "EMPTY_REASON",
        )

    return (
        {
            "x_quality": x_quality,
            "y_quality": y_quality,
            "confidence": confidence,
            "primary_error_x": primary_error_x,
            "primary_error_y": primary_error_y,
            "reason": reason,
        },
        None,
    )


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
    if not prompts:
        return []

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    )

    encoded = {
        key:
            value.to("cuda")
        for key, value
        in encoded.items()
    }

    input_length = (
        encoded["input_ids"]
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

    decoded = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return [
        str(text).strip()
        for text in decoded
    ]


# ============================================================
# Retry prompts
# ============================================================

def build_pairwise_retry_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    validation_error: str,
) -> str:
    retry_system = """
Your previous pairwise translation judgment failed automatic validation.

Re-evaluate from SOURCE.
Keep the JSON schema exact.
Ensure winner is consistent with A/B quality:
A MAJOR only -> B
B MAJOR only -> A
both MAJOR -> BOTH_BAD

Return one JSON object only.
""".strip()

    retry_user = f"""
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

{PAIRWISE_SCHEMA}
""".strip()

    return apply_chat_template(
        tokenizer,
        retry_system,
        retry_user,
    )


def build_adjudication_retry_prompt(
    tokenizer,
    row: pd.Series,
    assignment: dict,
    validation_error: str,
) -> str:
    retry_system = """
Your previous independent translation quality adjudication failed parsing.

Judge X and Y independently from SOURCE.
Do not choose a winner.
Return one valid JSON object only.
""".strip()

    retry_user = f"""
VALIDATION ERROR:
{validation_error}

DIRECTION:
{row["direction"]}

SOURCE:
{row["source_text"]}

REFERENCE — auxiliary only:
{row["reference_text"]}

TRANSLATION X:
{assignment["candidate_x_text"]}

TRANSLATION Y:
{assignment["candidate_y_text"]}

{ADJUDICATION_SCHEMA}
""".strip()

    return apply_chat_template(
        tokenizer,
        retry_system,
        retry_user,
    )


# ============================================================
# Single pass helpers
# ============================================================

def map_pairwise_to_models(
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

    quality_by_model = {
        model_a: parsed[
            "a_quality"
        ],
        model_b: parsed[
            "b_quality"
        ],
    }

    error_by_model = {
        model_a: parsed[
            "primary_error_a"
        ],
        model_b: parsed[
            "primary_error_b"
        ],
    }

    return {
        "winner_ab": winner_ab,
        "winner_model": winner_model,
        "confidence": parsed[
            "confidence"
        ],
        "reason": parsed[
            "reason"
        ],
        "opus_quality": quality_by_model[
            "OPUS"
        ],
        "madlad_quality": quality_by_model[
            "MADLAD"
        ],
        "opus_primary_error": error_by_model[
            "OPUS"
        ],
        "madlad_primary_error": error_by_model[
            "MADLAD"
        ],
    }


def run_pairwise_batch(
    *,
    rows: list[pd.Series],
    assignments: list[dict],
    tokenizer,
    model,
    args,
) -> list[dict]:
    prompts = []
    metas = []

    for row, assignment in zip(
        rows,
        assignments,
    ):
        prompt, meta = (
            build_safe_pairwise_prompt(
                tokenizer,
                row,
                assignment,
                args.max_input_tokens,
            )
        )

        prompts.append(
            prompt
        )

        metas.append(
            meta
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

    results = []

    for (
        row,
        assignment,
        meta,
        raw_output,
    ) in zip(
        rows,
        assignments,
        metas,
        raw_outputs,
    ):
        parsed, error = (
            parse_pairwise_result(
                raw_output
            )
        )

        retry_count = 0
        error_history = []

        if error:
            error_history.append(
                error
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
                build_pairwise_retry_prompt(
                    tokenizer,
                    row,
                    assignment,
                    error or "UNKNOWN",
                )
            )

            if (
                count_tokens(
                    tokenizer,
                    retry_prompt,
                )
                >
                args.max_input_tokens
            ):
                raise RuntimeError(
                    "Pairwise retry prompt exceeds "
                    f"{args.max_input_tokens} tokens "
                    f"for {row['review_id']}"
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

            parsed, error = (
                parse_pairwise_result(
                    raw_output
                )
            )

            if error:
                error_history.append(
                    error
                )

        result = {
            "parse_success": bool(
                parsed is not None
            ),
            "parse_error": (
                ""
                if parsed is not None
                else str(error)
            ),
            "validation_error_history": (
                " || ".join(
                    error_history
                )
            ),
            "retry_count": int(
                retry_count
            ),
            "raw_output": str(
                raw_output
            ),
            "original_prompt_tokens": int(
                meta[
                    "original_prompt_tokens"
                ]
            ),
            "final_prompt_tokens": int(
                meta[
                    "final_prompt_tokens"
                ]
            ),
            "prompt_truncated": bool(
                meta[
                    "prompt_truncated"
                ]
            ),
        }

        if parsed is not None:
            result.update(
                map_pairwise_to_models(
                    parsed,
                    assignment,
                )
            )

        results.append(
            result
        )

    return results


# ============================================================
# Pair consistency
# ============================================================

def pair_consensus(
    pass1: dict,
    pass2: dict,
) -> tuple[
    bool,
    str | None,
    str,
]:
    """
    Returns:
      stable,
      winner_if_stable,
      resolution_reason

    Stable only when both orderings independently map to
    exactly the same model-level outcome.
    """
    if (
        not pass1[
            "parse_success"
        ]
        or
        not pass2[
            "parse_success"
        ]
    ):
        return (
            False,
            None,
            "PAIR_PARSE_FAILURE",
        )

    winner1 = pass1[
        "winner_model"
    ]

    winner2 = pass2[
        "winner_model"
    ]

    if winner1 == winner2:
        return (
            True,
            winner1,
            "DUAL_ORDER_CONSENSUS",
        )

    return (
        False,
        None,
        "DUAL_ORDER_DISAGREEMENT",
    )


# ============================================================
# Adjudication
# ============================================================

def derive_adjudicated_winner(
    parsed: dict,
    assignment: dict,
) -> dict:
    x_model = assignment[
        "candidate_x_model"
    ]

    y_model = assignment[
        "candidate_y_model"
    ]

    quality_by_model = {
        x_model: parsed[
            "x_quality"
        ],
        y_model: parsed[
            "y_quality"
        ],
    }

    error_by_model = {
        x_model: parsed[
            "primary_error_x"
        ],
        y_model: parsed[
            "primary_error_y"
        ],
    }

    opus_quality = quality_by_model[
        "OPUS"
    ]

    madlad_quality = quality_by_model[
        "MADLAD"
    ]

    opus_severity = QUALITY_SEVERITY[
        opus_quality
    ]

    madlad_severity = QUALITY_SEVERITY[
        madlad_quality
    ]

    if (
        opus_quality == "MAJOR"
        and
        madlad_quality == "MAJOR"
    ):
        winner = "BOTH_BAD"

    elif opus_severity < madlad_severity:
        winner = "OPUS"

    elif madlad_severity < opus_severity:
        winner = "MADLAD"

    else:
        # Conservative policy:
        # equal absolute quality => TIE.
        winner = "TIE"

    return {
        "winner_model": winner,
        "confidence": parsed[
            "confidence"
        ],
        "reason": parsed[
            "reason"
        ],
        "opus_quality": opus_quality,
        "madlad_quality": madlad_quality,
        "opus_primary_error": error_by_model[
            "OPUS"
        ],
        "madlad_primary_error": error_by_model[
            "MADLAD"
        ],
    }


def run_adjudication_batch(
    *,
    rows: list[pd.Series],
    assignments: list[dict],
    tokenizer,
    model,
    args,
) -> list[dict]:
    if not rows:
        return []

    prompts = []
    metas = []

    for row, assignment in zip(
        rows,
        assignments,
    ):
        prompt, meta = (
            build_safe_adjudication_prompt(
                tokenizer,
                row,
                assignment,
                args.max_input_tokens,
            )
        )

        prompts.append(
            prompt
        )

        metas.append(
            meta
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

    results = []

    for (
        row,
        assignment,
        meta,
        raw_output,
    ) in zip(
        rows,
        assignments,
        metas,
        raw_outputs,
    ):
        parsed, error = (
            parse_adjudication_result(
                raw_output
            )
        )

        retry_count = 0
        error_history = []

        if error:
            error_history.append(
                error
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
                build_adjudication_retry_prompt(
                    tokenizer,
                    row,
                    assignment,
                    error or "UNKNOWN",
                )
            )

            if (
                count_tokens(
                    tokenizer,
                    retry_prompt,
                )
                >
                args.max_input_tokens
            ):
                raise RuntimeError(
                    "Adjudication retry prompt exceeds "
                    f"{args.max_input_tokens} tokens "
                    f"for {row['review_id']}"
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

            parsed, error = (
                parse_adjudication_result(
                    raw_output
                )
            )

            if error:
                error_history.append(
                    error
                )

        result = {
            "parse_success": bool(
                parsed is not None
            ),
            "parse_error": (
                ""
                if parsed is not None
                else str(error)
            ),
            "validation_error_history": (
                " || ".join(
                    error_history
                )
            ),
            "retry_count": int(
                retry_count
            ),
            "raw_output": str(
                raw_output
            ),
            "original_prompt_tokens": int(
                meta[
                    "original_prompt_tokens"
                ]
            ),
            "final_prompt_tokens": int(
                meta[
                    "final_prompt_tokens"
                ]
            ),
            "prompt_truncated": bool(
                meta[
                    "prompt_truncated"
                ]
            ),
        }

        if parsed is not None:
            result.update(
                derive_adjudicated_winner(
                    parsed,
                    assignment,
                )
            )

        results.append(
            result
        )

    return results


# ============================================================
# Main judging pipeline
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
            ~df["review_id"]
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

        rows = [
            row
            for _, row
            in batch_df.iterrows()
        ]

        pass1_assignments = [
            make_pass1_assignment(
                row,
                args.seed,
            )
            for row in rows
        ]

        pass2_assignments = [
            reverse_assignment(
                assignment
            )
            for assignment
            in pass1_assignments
        ]

        # ----------------------------------------------------
        # Pass 1
        # ----------------------------------------------------

        pass1_results = (
            run_pairwise_batch(
                rows=rows,
                assignments=(
                    pass1_assignments
                ),
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
        )

        # ----------------------------------------------------
        # Pass 2: exact reversed order
        # ----------------------------------------------------

        pass2_results = (
            run_pairwise_batch(
                rows=rows,
                assignments=(
                    pass2_assignments
                ),
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
        )

        # ----------------------------------------------------
        # Determine which rows need independent adjudication
        # ----------------------------------------------------

        adjudication_rows = []
        adjudication_assignments = []
        adjudication_indices = []

        stable_info = {}

        for i, (
            row,
            p1,
            p2,
        ) in enumerate(
            zip(
                rows,
                pass1_results,
                pass2_results,
            )
        ):
            stable, winner, reason = (
                pair_consensus(
                    p1,
                    p2,
                )
            )

            stable_info[
                i
            ] = {
                "stable": stable,
                "winner": winner,
                "reason": reason,
            }

            if not stable:
                adjudication_rows.append(
                    row
                )

                adjudication_assignments.append(
                    make_adjudication_assignment(
                        row,
                        args.seed,
                    )
                )

                adjudication_indices.append(
                    i
                )

        # ----------------------------------------------------
        # Third pass only for dual-order disagreement/failure
        # ----------------------------------------------------

        adjudication_results = (
            run_adjudication_batch(
                rows=adjudication_rows,
                assignments=(
                    adjudication_assignments
                ),
                tokenizer=tokenizer,
                model=model,
                args=args,
            )
            if adjudication_rows
            else []
        )

        adjudication_by_index = {
            idx: (
                assignment,
                result,
            )
            for idx, assignment, result
            in zip(
                adjudication_indices,
                adjudication_assignments,
                adjudication_results,
            )
        }

        batch_final_counter = Counter()
        pair_stable_count = 0
        adjudicated_count = 0
        unresolved_count = 0

        # ----------------------------------------------------
        # Save one complete record per source sample
        # ----------------------------------------------------

        for i, (
            row,
            p1_assignment,
            p2_assignment,
            p1,
            p2,
        ) in enumerate(
            zip(
                rows,
                pass1_assignments,
                pass2_assignments,
                pass1_results,
                pass2_results,
            )
        ):
            info = stable_info[
                i
            ]

            final_resolved = False
            final_winner = (
                "UNRESOLVED"
            )
            final_confidence = ""
            final_reason = ""
            final_resolution = ""
            final_opus_quality = ""
            final_madlad_quality = ""
            final_opus_primary_error = ""
            final_madlad_primary_error = ""

            adjudication_used = (
                not info[
                    "stable"
                ]
            )

            adjudication_assignment = None
            adjudication_result = None

            if info[
                "stable"
            ]:
                pair_stable_count += 1

                final_resolved = True
                final_winner = str(
                    info[
                        "winner"
                    ]
                )

                final_resolution = (
                    "DUAL_ORDER_CONSENSUS"
                )

                # Conservative aggregation of absolute quality:
                # use the worse severity seen across two positions for each model.
                if (
                    p1[
                        "parse_success"
                    ]
                    and
                    p2[
                        "parse_success"
                    ]
                ):
                    opus_candidates = [
                        p1[
                            "opus_quality"
                        ],
                        p2[
                            "opus_quality"
                        ],
                    ]

                    madlad_candidates = [
                        p1[
                            "madlad_quality"
                        ],
                        p2[
                            "madlad_quality"
                        ],
                    ]

                    final_opus_quality = max(
                        opus_candidates,
                        key=lambda q: (
                            QUALITY_SEVERITY[q]
                        ),
                    )

                    final_madlad_quality = max(
                        madlad_candidates,
                        key=lambda q: (
                            QUALITY_SEVERITY[q]
                        ),
                    )

                    # Use error type from the worse-quality pass.
                    p1_opus_sev = QUALITY_SEVERITY[
                        p1[
                            "opus_quality"
                        ]
                    ]
                    p2_opus_sev = QUALITY_SEVERITY[
                        p2[
                            "opus_quality"
                        ]
                    ]

                    final_opus_primary_error = (
                        p1[
                            "opus_primary_error"
                        ]
                        if p1_opus_sev
                        >=
                        p2_opus_sev
                        else
                        p2[
                            "opus_primary_error"
                        ]
                    )

                    p1_madlad_sev = QUALITY_SEVERITY[
                        p1[
                            "madlad_quality"
                        ]
                    ]
                    p2_madlad_sev = QUALITY_SEVERITY[
                        p2[
                            "madlad_quality"
                        ]
                    ]

                    final_madlad_primary_error = (
                        p1[
                            "madlad_primary_error"
                        ]
                        if p1_madlad_sev
                        >=
                        p2_madlad_sev
                        else
                        p2[
                            "madlad_primary_error"
                        ]
                    )

                    if (
                        p1[
                            "confidence"
                        ]
                        ==
                        p2[
                            "confidence"
                        ]
                    ):
                        final_confidence = p1[
                            "confidence"
                        ]
                    else:
                        # Conservative confidence aggregation.
                        order = {
                            "LOW": 0,
                            "MEDIUM": 1,
                            "HIGH": 2,
                        }

                        final_confidence = min(
                            [
                                p1[
                                    "confidence"
                                ],
                                p2[
                                    "confidence"
                                ],
                            ],
                            key=lambda x: (
                                order[x]
                            ),
                        )

                    final_reason = (
                        "Pass1 and Pass2 agree after "
                        "A/B reversal. "
                        f"P1: {p1['reason']} "
                        f"P2: {p2['reason']}"
                    )

            else:
                (
                    adjudication_assignment,
                    adjudication_result,
                ) = adjudication_by_index[
                    i
                ]

                if adjudication_result[
                    "parse_success"
                ]:
                    adjudicated_count += 1

                    final_resolved = True
                    final_winner = (
                        adjudication_result[
                            "winner_model"
                        ]
                    )

                    final_confidence = (
                        adjudication_result[
                            "confidence"
                        ]
                    )

                    final_reason = (
                        adjudication_result[
                            "reason"
                        ]
                    )

                    final_resolution = (
                        "INDEPENDENT_ADJUDICATION"
                    )

                    final_opus_quality = (
                        adjudication_result[
                            "opus_quality"
                        ]
                    )

                    final_madlad_quality = (
                        adjudication_result[
                            "madlad_quality"
                        ]
                    )

                    final_opus_primary_error = (
                        adjudication_result[
                            "opus_primary_error"
                        ]
                    )

                    final_madlad_primary_error = (
                        adjudication_result[
                            "madlad_primary_error"
                        ]
                    )

                else:
                    unresolved_count += 1

                    final_resolution = (
                        "ADJUDICATION_PARSE_FAILURE"
                    )

            record = {
                "review_id": str(
                    row[
                        "review_id"
                    ]
                ),
                "sample_id": str(
                    row[
                        "sample_id"
                    ]
                ),
                "pair_id": str(
                    row[
                        "pair_id"
                    ]
                ),
                "direction": str(
                    row[
                        "direction"
                    ]
                ),
                "source_dataset": str(
                    row[
                        "source_dataset"
                    ]
                ),
                "source_text": str(
                    row[
                        "source_text"
                    ]
                ),
                "reference_text": str(
                    row[
                        "reference_text"
                    ]
                ),
                "opus_prediction": str(
                    row[
                        "opus_prediction"
                    ]
                ),
                "madlad_prediction": str(
                    row[
                        "madlad_prediction"
                    ]
                ),
                "teacher_disagreement_score": float(
                    row[
                        "teacher_disagreement_score"
                    ]
                ),

                # Pass 1
                "pass1_a_model": p1_assignment[
                    "candidate_a_model"
                ],
                "pass1_b_model": p1_assignment[
                    "candidate_b_model"
                ],
                "pass1_parse_success": bool(
                    p1[
                        "parse_success"
                    ]
                ),
                "pass1_parse_error": str(
                    p1[
                        "parse_error"
                    ]
                ),
                "pass1_retry_count": int(
                    p1[
                        "retry_count"
                    ]
                ),
                "pass1_prompt_truncated": bool(
                    p1[
                        "prompt_truncated"
                    ]
                ),
                "pass1_winner_ab": (
                    p1.get(
                        "winner_ab",
                        "",
                    )
                ),
                "pass1_winner_model": (
                    p1.get(
                        "winner_model",
                        "",
                    )
                ),
                "pass1_confidence": (
                    p1.get(
                        "confidence",
                        "",
                    )
                ),
                "pass1_opus_quality": (
                    p1.get(
                        "opus_quality",
                        "",
                    )
                ),
                "pass1_madlad_quality": (
                    p1.get(
                        "madlad_quality",
                        "",
                    )
                ),
                "pass1_opus_primary_error": (
                    p1.get(
                        "opus_primary_error",
                        "",
                    )
                ),
                "pass1_madlad_primary_error": (
                    p1.get(
                        "madlad_primary_error",
                        "",
                    )
                ),
                "pass1_reason": (
                    p1.get(
                        "reason",
                        "",
                    )
                ),
                "pass1_validation_error_history": str(
                    p1[
                        "validation_error_history"
                    ]
                ),
                "pass1_raw_output": str(
                    p1[
                        "raw_output"
                    ]
                ),

                # Pass 2
                "pass2_a_model": p2_assignment[
                    "candidate_a_model"
                ],
                "pass2_b_model": p2_assignment[
                    "candidate_b_model"
                ],
                "pass2_parse_success": bool(
                    p2[
                        "parse_success"
                    ]
                ),
                "pass2_parse_error": str(
                    p2[
                        "parse_error"
                    ]
                ),
                "pass2_retry_count": int(
                    p2[
                        "retry_count"
                    ]
                ),
                "pass2_prompt_truncated": bool(
                    p2[
                        "prompt_truncated"
                    ]
                ),
                "pass2_winner_ab": (
                    p2.get(
                        "winner_ab",
                        "",
                    )
                ),
                "pass2_winner_model": (
                    p2.get(
                        "winner_model",
                        "",
                    )
                ),
                "pass2_confidence": (
                    p2.get(
                        "confidence",
                        "",
                    )
                ),
                "pass2_opus_quality": (
                    p2.get(
                        "opus_quality",
                        "",
                    )
                ),
                "pass2_madlad_quality": (
                    p2.get(
                        "madlad_quality",
                        "",
                    )
                ),
                "pass2_opus_primary_error": (
                    p2.get(
                        "opus_primary_error",
                        "",
                    )
                ),
                "pass2_madlad_primary_error": (
                    p2.get(
                        "madlad_primary_error",
                        "",
                    )
                ),
                "pass2_reason": (
                    p2.get(
                        "reason",
                        "",
                    )
                ),
                "pass2_validation_error_history": str(
                    p2[
                        "validation_error_history"
                    ]
                ),
                "pass2_raw_output": str(
                    p2[
                        "raw_output"
                    ]
                ),

                # Pair order consistency
                "dual_order_consistent": bool(
                    info[
                        "stable"
                    ]
                ),
                "dual_order_status": str(
                    info[
                        "reason"
                    ]
                ),

                # Adjudication
                "adjudication_used": bool(
                    adjudication_used
                ),
                "adjudication_x_model": (
                    ""
                    if adjudication_assignment
                    is None
                    else adjudication_assignment[
                        "candidate_x_model"
                    ]
                ),
                "adjudication_y_model": (
                    ""
                    if adjudication_assignment
                    is None
                    else adjudication_assignment[
                        "candidate_y_model"
                    ]
                ),
                "adjudication_parse_success": (
                    True
                    if not adjudication_used
                    else bool(
                        adjudication_result[
                            "parse_success"
                        ]
                    )
                ),
                "adjudication_parse_error": (
                    ""
                    if adjudication_result
                    is None
                    else str(
                        adjudication_result[
                            "parse_error"
                        ]
                    )
                ),
                "adjudication_retry_count": (
                    0
                    if adjudication_result
                    is None
                    else int(
                        adjudication_result[
                            "retry_count"
                        ]
                    )
                ),
                "adjudication_prompt_truncated": (
                    False
                    if adjudication_result
                    is None
                    else bool(
                        adjudication_result[
                            "prompt_truncated"
                        ]
                    )
                ),
                "adjudication_raw_output": (
                    ""
                    if adjudication_result
                    is None
                    else str(
                        adjudication_result[
                            "raw_output"
                        ]
                    )
                ),

                # Final
                "final_resolved": bool(
                    final_resolved
                ),
                "final_winner": str(
                    final_winner
                ),
                "final_confidence": str(
                    final_confidence
                ),
                "final_resolution": str(
                    final_resolution
                ),
                "final_opus_quality": str(
                    final_opus_quality
                ),
                "final_madlad_quality": str(
                    final_madlad_quality
                ),
                "final_opus_primary_error": str(
                    final_opus_primary_error
                ),
                "final_madlad_primary_error": str(
                    final_madlad_primary_error
                ),
                "final_reason": str(
                    final_reason
                ),

                "prompt_version": (
                    PROMPT_VERSION
                ),
                "adjudication_version": (
                    ADJUDICATION_VERSION
                ),
            }

            append_jsonl(
                record,
                checkpoint_file,
            )

            existing[
                record[
                    "review_id"
                ]
            ] = record

            processed_this_run += 1

            batch_final_counter[
                record[
                    "final_winner"
                ]
            ] += 1

        total_done = (
            len(completed_ids)
            +
            processed_this_run
        )

        print(
            f"{total_done}/{len(df)}"
            f" | stable {pair_stable_count}/{len(batch_df)}"
            f" | adjudicated {adjudicated_count}"
            f" | unresolved {unresolved_count}"
            f" | {dict(batch_final_counter)}"
        )

    final_rows = []

    for _, row in df.iterrows():
        review_id = str(
            row[
                "review_id"
            ]
        )

        if review_id not in existing:
            raise RuntimeError(
                f"Missing final result for {review_id}"
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
# Diagnostics
# ============================================================

def build_overall_report(
    df: pd.DataFrame,
) -> dict:
    total = len(df)

    resolved = (
        df[
            "final_resolved"
        ]
        ==
        True
    )

    pass1_parse = (
        df[
            "pass1_parse_success"
        ]
        ==
        True
    )

    pass2_parse = (
        df[
            "pass2_parse_success"
        ]
        ==
        True
    )

    dual_consistent = (
        df[
            "dual_order_consistent"
        ]
        ==
        True
    )

    adjudication_used = (
        df[
            "adjudication_used"
        ]
        ==
        True
    )

    adjudication_success = (
        adjudication_used
        &
        (
            df[
                "adjudication_parse_success"
            ]
            ==
            True
        )
    )

    final_counts = (
        df.loc[
            resolved,
            "final_winner",
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "rows": int(
            total
        ),
        "pass1_parse_success": int(
            pass1_parse.sum()
        ),
        "pass1_parse_success_percent": float(
            pass1_parse.mean()
            *
            100
        ),
        "pass2_parse_success": int(
            pass2_parse.sum()
        ),
        "pass2_parse_success_percent": float(
            pass2_parse.mean()
            *
            100
        ),
        "dual_order_consistent_rows": int(
            dual_consistent.sum()
        ),
        "dual_order_consistency_percent": float(
            dual_consistent.mean()
            *
            100
        ),
        "adjudication_used_rows": int(
            adjudication_used.sum()
        ),
        "adjudication_success_rows": int(
            adjudication_success.sum()
        ),
        "final_resolved_rows": int(
            resolved.sum()
        ),
        "final_resolved_percent": float(
            resolved.mean()
            *
            100
        ),
        "unresolved_rows": int(
            (
                ~resolved
            )
            .sum()
        ),
        "pass1_retried_rows": int(
            (
                df[
                    "pass1_retry_count"
                ]
                >
                0
            )
            .sum()
        ),
        "pass2_retried_rows": int(
            (
                df[
                    "pass2_retry_count"
                ]
                >
                0
            )
            .sum()
        ),
        "adjudication_retried_rows": int(
            (
                df[
                    "adjudication_retry_count"
                ]
                >
                0
            )
            .sum()
        ),
        "prompt_truncated_rows_any_stage": int(
            (
                df[
                    [
                        "pass1_prompt_truncated",
                        "pass2_prompt_truncated",
                        "adjudication_prompt_truncated",
                    ]
                ]
                .astype(bool)
                .any(
                    axis=1
                )
            )
            .sum()
        ),
        "final_winner_counts": {
            winner: int(
                final_counts.get(
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


def build_stratified_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        direction,
        source_dataset,
    ), part in df.groupby(
        [
            "direction",
            "source_dataset",
        ]
    ):
        resolved = (
            part[
                part[
                    "final_resolved"
                ]
                ==
                True
            ]
            .copy()
        )

        counts = (
            resolved[
                "final_winner"
            ]
            .value_counts()
            .to_dict()
        )

        n_resolved = len(
            resolved
        )

        row = {
            "direction": direction,
            "source_dataset": source_dataset,
            "rows": int(
                len(part)
            ),
            "resolved_rows": int(
                n_resolved
            ),
            "resolved_percent": float(
                n_resolved
                /
                len(part)
                *
                100
            ),
            "dual_order_consistency_percent": float(
                part[
                    "dual_order_consistent"
                ]
                .astype(bool)
                .mean()
                *
                100
            ),
            "adjudication_used_percent": float(
                part[
                    "adjudication_used"
                ]
                .astype(bool)
                .mean()
                *
                100
            ),
        }

        for winner in [
            "OPUS",
            "MADLAD",
            "TIE",
            "BOTH_BAD",
        ]:
            count = int(
                counts.get(
                    winner,
                    0,
                )
            )

            row[
                f"{winner.lower()}_count"
            ] = count

            row[
                f"{winner.lower()}_percent_of_resolved"
            ] = (
                float(
                    count
                    /
                    n_resolved
                    *
                    100
                )
                if n_resolved
                else 0.0
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def build_position_bias_report(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Since every row is judged twice with reversed positions, we can measure
    model win rate when located at A vs when located at B across all pairwise
    passes. This is a pure diagnostic; final winner does not directly use it.
    """
    records = []

    for _, row in df.iterrows():
        for pass_name in [
            "pass1",
            "pass2",
        ]:
            parse_success = bool(
                row[
                    f"{pass_name}_parse_success"
                ]
            )

            if not parse_success:
                continue

            a_model = str(
                row[
                    f"{pass_name}_a_model"
                ]
            )

            b_model = str(
                row[
                    f"{pass_name}_b_model"
                ]
            )

            winner_model = str(
                row[
                    f"{pass_name}_winner_model"
                ]
            )

            records.append(
                {
                    "direction": str(
                        row[
                            "direction"
                        ]
                    ),
                    "source_dataset": str(
                        row[
                            "source_dataset"
                        ]
                    ),
                    "pass": pass_name,
                    "a_model": a_model,
                    "b_model": b_model,
                    "winner_model": winner_model,
                }
            )

    long_df = pd.DataFrame(
        records
    )

    if long_df.empty:
        return pd.DataFrame()

    rows = []

    for direction, direction_part in long_df.groupby(
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
                column = (
                    "a_model"
                    if position == "A"
                    else "b_model"
                )

                part = (
                    direction_part[
                        direction_part[
                            column
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
                            "winner_model"
                        ]
                        ==
                        model_name
                    )
                    .sum()
                )

                opponent = (
                    "MADLAD"
                    if model_name == "OPUS"
                    else "OPUS"
                )

                opponent_wins = int(
                    (
                        part[
                            "winner_model"
                        ]
                        ==
                        opponent
                    )
                    .sum()
                )

                ties = int(
                    (
                        part[
                            "winner_model"
                        ]
                        ==
                        "TIE"
                    )
                    .sum()
                )

                both_bad = int(
                    (
                        part[
                            "winner_model"
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


def build_dual_order_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    valid = (
        df[
            (
                df[
                    "pass1_parse_success"
                ]
                ==
                True
            )
            &
            (
                df[
                    "pass2_parse_success"
                ]
                ==
                True
            )
        ]
        .copy()
    )

    if valid.empty:
        return pd.DataFrame()

    matrix = pd.crosstab(
        valid[
            "pass1_winner_model"
        ],
        valid[
            "pass2_winner_model"
        ],
        margins=True,
    )

    return matrix


def save_audit(
    df: pd.DataFrame,
    path: Path,
) -> None:
    lines = []

    categories = [
        (
            "DUAL_ORDER_CONSENSUS",
            df[
                df[
                    "final_resolution"
                ]
                ==
                "DUAL_ORDER_CONSENSUS"
            ]
            .copy()
        ),
        (
            "INDEPENDENT_ADJUDICATION",
            df[
                df[
                    "final_resolution"
                ]
                ==
                "INDEPENDENT_ADJUDICATION"
            ]
            .copy()
        ),
        (
            "UNRESOLVED",
            df[
                df[
                    "final_resolved"
                ]
                ==
                False
            ]
            .copy()
        ),
    ]

    for title, part in categories:
        if part.empty:
            continue

        part = (
            part.sort_values(
                "teacher_disagreement_score",
                ascending=False,
            )
            .head(10)
        )

        lines.extend(
            [
                "",
                "=" * 120,
                title,
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
                    f"SOURCE DATASET: {row['source_dataset']}",
                    f"P1: {row['pass1_a_model']} vs {row['pass1_b_model']} -> {row['pass1_winner_model']}",
                    f"P2: {row['pass2_a_model']} vs {row['pass2_b_model']} -> {row['pass2_winner_model']}",
                    f"DUAL CONSISTENT: {row['dual_order_consistent']}",
                    f"ADJUDICATION USED: {row['adjudication_used']}",
                    f"FINAL: {row['final_winner']}",
                    f"RESOLUTION: {row['final_resolution']}",
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
                    "FINAL REASON:",
                    str(
                        row[
                            "final_reason"
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

    input_df = load_disagreement_set(
        project_root
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

        verify_same_calibration_as_v2(
            project_root,
            work_df,
        )

        mode_dir = (
            "calibration"
        )

        output_name = (
            f"qwen_pairwise_calibration_"
            f"{len(work_df)}_v3"
        )

    else:
        work_df = (
            input_df.copy()
            .reset_index(
                drop=True
            )
        )

        mode_dir = (
            "full"
        )

        output_name = (
            "qwen_pairwise_800_v3"
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
        f"{output_name}_stratified.csv"
    )

    position_file = (
        output_root
        /
        f"{output_name}_position_bias.csv"
    )

    dual_matrix_file = (
        output_root
        /
        f"{output_name}_dual_order_matrix.csv"
    )

    audit_file = (
        output_root
        /
        f"{output_name}_audit.txt"
    )

    report_file = (
        output_root
        /
        f"{output_name}_report.json"
    )

    v3_files = [
        checkpoint_file,
        parquet_file,
        csv_file,
        summary_file,
        position_file,
        dual_matrix_file,
        audit_file,
        report_file,
    ]

    # NEVER remove V1/V2 outputs.
    if args.overwrite:
        for path in v3_files:
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
            "\nExisting V3 output found:\n"
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
        "STEP 17C V3 - DUAL-ORDER PAIRWISE JUDGE"
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
        "Batch size:",
        args.batch_size,
    )

    print(
        "Max input tokens:",
        args.max_input_tokens,
    )

    print(
        "Max new tokens:",
        args.max_new_tokens,
    )

    print(
        "Max retries:",
        args.max_retries,
    )

    print(
        "\nV3 policy:"
    )

    print(
        "- Same calibration rows as V2"
    )

    print(
        "- Every sample judged twice"
    )

    print(
        "- Pass 2 exactly reverses A/B"
    )

    print(
        "- Model-level agreement -> direct consensus"
    )

    print(
        "- Any disagreement/parse failure -> independent adjudication"
    )

    print(
        "- Adjudication does NOT ask Qwen to pick a winner"
    )

    print(
        "- Final adjudication winner is derived from absolute quality in code"
    )

    print(
        "- Equal adjudication quality -> conservative TIE"
    )

    print(
        "- Validation samples are NEVER KD training data"
    )

    tokenizer, model = load_qwen(
        Path(
            args.model_path
        )
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

    overall = build_overall_report(
        result_df
    )

    stratified_df = (
        build_stratified_report(
            result_df
        )
    )

    position_df = (
        build_position_bias_report(
            result_df
        )
    )

    dual_matrix_df = (
        build_dual_order_matrix(
            result_df
        )
    )

    stratified_df.to_csv(
        summary_file,
        index=False,
        encoding="utf-8-sig",
    )

    position_df.to_csv(
        position_file,
        index=False,
        encoding="utf-8-sig",
    )

    dual_matrix_df.to_csv(
        dual_matrix_file,
        encoding="utf-8-sig",
    )

    save_audit(
        result_df,
        audit_file,
    )

    report = {
        "step": "17C",
        "step_version": STEP_VERSION,
        "prompt_version": PROMPT_VERSION,
        "adjudication_version": ADJUDICATION_VERSION,
        "mode": args.mode,
        "rows": int(
            len(result_df)
        ),
        "policy": {
            "same_calibration_sampling_as_v1_v2": True,
            "dual_order_pairwise": True,
            "second_pass_exact_reverse": True,
            "pairwise_quality_labels": [
                "GOOD",
                "MINOR",
                "MAJOR",
            ],
            "pairwise_position_consistency_required_for_direct_accept": True,
            "independent_adjudication_on_disagreement": True,
            "adjudication_requests_winner_from_model": False,
            "adjudication_winner_derived_from_quality": True,
            "equal_adjudication_quality_becomes_tie": True,
            "allowed_for_kd_training": False,
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
            "max_retries": int(
                args.max_retries
            ),
            "do_sample": False,
            "seed": int(
                args.seed
            ),
        },
        "overall": overall,
        "outputs": {
            "results": str(
                parquet_file
            ),
            "stratified": str(
                summary_file
            ),
            "position_bias": str(
                position_file
            ),
            "dual_order_matrix": str(
                dual_matrix_file
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
            "V3_CALIBRATION_COMPLETE_REVIEW_REQUIRED"
            if args.mode
            ==
            "calibration"
            else
            "V3_FULL_COMPLETE"
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
        "STEP 17C V3 RESULT"
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
        overall[
            "rows"
        ],
    )

    print(
        "\nPass 1 parse:"
    )

    print(
        f"{overall['pass1_parse_success']}/"
        f"{overall['rows']} "
        f"({overall['pass1_parse_success_percent']:.2f}%)"
    )

    print(
        "\nPass 2 parse:"
    )

    print(
        f"{overall['pass2_parse_success']}/"
        f"{overall['rows']} "
        f"({overall['pass2_parse_success_percent']:.2f}%)"
    )

    print(
        "\nDual-order consistency:"
    )

    print(
        f"{overall['dual_order_consistent_rows']}/"
        f"{overall['rows']} "
        f"({overall['dual_order_consistency_percent']:.2f}%)"
    )

    print(
        "\nAdjudication:"
    )

    print(
        "Used:",
        overall[
            "adjudication_used_rows"
        ],
    )

    print(
        "Successful:",
        overall[
            "adjudication_success_rows"
        ],
    )

    print(
        "\nFinal resolution:"
    )

    print(
        f"{overall['final_resolved_rows']}/"
        f"{overall['rows']} "
        f"({overall['final_resolved_percent']:.2f}%)"
    )

    print(
        "Unresolved:",
        overall[
            "unresolved_rows"
        ],
    )

    print(
        "\nRetries:"
    )

    print(
        "Pass1 retried rows:",
        overall[
            "pass1_retried_rows"
        ],
    )

    print(
        "Pass2 retried rows:",
        overall[
            "pass2_retried_rows"
        ],
    )

    print(
        "Adjudication retried rows:",
        overall[
            "adjudication_retried_rows"
        ],
    )

    print(
        "\nPrompt truncated rows (any stage):",
        overall[
            "prompt_truncated_rows_any_stage"
        ],
    )

    print(
        "\nFinal winners:"
    )

    for winner, count in (
        overall[
            "final_winner_counts"
        ]
        .items()
    ):
        print(
            f"{winner}: {count}"
        )

    print(
        "\nStratified final distribution:"
    )

    if not stratified_df.empty:
        print(
            stratified_df
            .round(3)
            .to_string(
                index=False
            )
        )

    print(
        "\nDual-order transition matrix:"
    )

    if not dual_matrix_df.empty:
        print(
            dual_matrix_df
            .to_string()
        )

    print(
        "\nPairwise position-bias diagnostic:"
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
