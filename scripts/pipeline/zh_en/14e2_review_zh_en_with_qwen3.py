from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Versions
# ============================================================

STEP_VERSION = "14E2_V5"

PROMPT_VERSION = "ZH_EN_JUDGE_V5_STRUCTURED_FINAL"

JUDGE_MODEL_NAME = "Qwen3-8B"


# ============================================================
# Final labels
# ============================================================

VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}


# ============================================================
# Structured semantic fields
# ============================================================

BOOLEAN_FIELDS = [

    "uncertain",

    "core_meaning_preserved",

    "main_action_state_match",

    "participant_roles_match",

    "location_direction_match",

    "time_event_order_match",

    "quantity_match",

    "entity_match",

    "negation_event_status_match",

    "important_information_preserved",

    "unsupported_information_added",

    "minor_semantic_issue",

    "fluency_or_grammar_affects_understanding",
]


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14E-2 V5 - Structured semantic review "
            "for ZH-EN using local Qwen3-8B."
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
        "--calibration_size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--calibration_risk_size",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=1536,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=260,
    )

    parser.add_argument(
        "--parse_retries",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--no_reuse_calibration",
        dest="reuse_calibration",
        action="store_false",
    )

    parser.set_defaults(
        reuse_calibration=True
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Deterministic sampling
# ============================================================

def stable_hash(
    value: str,
    seed: int,
) -> str:

    text = (
        f"{seed}\n"
        f"{value}"
    )

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def allocate_proportional(
    df: pd.DataFrame,
    total_size: int,
    column: str,
) -> dict[str, int]:

    if total_size < 0:

        raise ValueError(
            "total_size must be >= 0"
        )

    if total_size > len(df):

        raise ValueError(
            f"Requested {total_size} rows, "
            f"but only {len(df)} are available."
        )

    if total_size == 0:

        return {}

    counts = (
        df[column]
        .astype(str)
        .value_counts()
        .sort_index()
    )

    total_rows = int(
        counts.sum()
    )

    allocations = {}

    fractions = {}

    for key, count in counts.items():

        exact = (
            total_size
            *
            int(count)
            /
            total_rows
        )

        floor_value = int(
            math.floor(
                exact
            )
        )

        allocations[
            str(key)
        ] = floor_value

        fractions[
            str(key)
        ] = (
            exact
            -
            floor_value
        )

    remaining = (
        total_size
        -
        sum(
            allocations.values()
        )
    )

    ordered = sorted(
        fractions.keys(),
        key=lambda key: (
            -fractions[key],
            key,
        ),
    )

    for key in ordered[
        :remaining
    ]:

        allocations[
            key
        ] += 1

    return allocations


def deterministic_stratified_sample(
    df: pd.DataFrame,
    sample_size: int,
    seed: int,
    stratum_column: str = "source_dataset",
) -> pd.DataFrame:

    if sample_size == 0:

        return (
            df
            .head(0)
            .copy()
        )

    if sample_size >= len(df):

        return (
            df
            .copy()
            .reset_index(
                drop=True
            )
        )

    allocations = allocate_proportional(
        df=df,
        total_size=sample_size,
        column=stratum_column,
    )

    parts = []

    for (
        stratum,
        count,
    ) in allocations.items():

        part = (
            df[
                df[
                    stratum_column
                ]
                .astype(str)
                ==
                stratum
            ]
            .copy()
        )

        part[
            "_sample_hash"
        ] = [

            stable_hash(
                str(review_id),
                seed,
            )

            for review_id
            in part[
                "review_id"
            ]
            .astype(str)
        ]

        part = (
            part
            .sort_values(
                [
                    "_sample_hash",
                    "review_id",
                ]
            )
            .head(
                count
            )
            .copy()
        )

        parts.append(
            part
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                stratum_column,
                "_sample_hash",
                "review_id",
            ]
        )
        .drop(
            columns=[
                "_sample_hash"
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    if len(result) != sample_size:

        raise RuntimeError(
            "\nCalibration sample size mismatch.\n"
            f"Expected: {sample_size}\n"
            f"Found: {len(result)}"
        )

    return result


def build_calibration_subset(
    review_df: pd.DataFrame,
    calibration_size: int,
    calibration_risk_size: int,
    seed: int,
) -> pd.DataFrame:

    if calibration_size > len(
        review_df
    ):

        raise ValueError(
            "Calibration size exceeds dataset size."
        )

    if (
        calibration_risk_size
        >
        calibration_size
    ):

        raise ValueError(
            "calibration_risk_size exceeds calibration_size."
        )

    risk = (
        review_df[
            review_df[
                "review_group"
            ]
            ==
            "RISK_REVIEW"
        ]
        .copy()
    )

    auto = (
        review_df[
            review_df[
                "review_group"
            ]
            ==
            "AUTO_ACCEPT_AUDIT"
        ]
        .copy()
    )

    auto_size = (
        calibration_size
        -
        calibration_risk_size
    )

    risk_sample = (
        deterministic_stratified_sample(
            risk,
            calibration_risk_size,
            seed,
        )
    )

    auto_sample = (
        deterministic_stratified_sample(
            auto,
            auto_size,
            seed + 1,
        )
    )

    calibration = pd.concat(
        [
            risk_sample,
            auto_sample,
        ],
        ignore_index=True,
    )

    calibration[
        "calibration_selected"
    ] = True

    calibration = (
        calibration
        .sort_values(
            [
                "review_group",
                "source_dataset",
                "review_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            calibration
        )
        !=
        calibration_size
    ):

        raise RuntimeError(
            "Calibration final size mismatch."
        )

    return calibration


# ============================================================
# Structured blind judge prompt
# ============================================================

SYSTEM_PROMPT = r"""
You are an independent bilingual Chinese-English translation
quality analyst.

You will receive one English sentence and one Chinese sentence.

Do NOT assign PASS, MINOR, or FAIL yourself.

Instead, independently compare the factual and semantic content
and return structured semantic checks.

Be tolerant of differences in expression, but strict about
differences in meaning.

Normal translation differences are not errors by themselves:

- word order differences;
- English-Chinese grammatical restructuring;
- colloquial versus formal wording;
- equivalent number formats;
- equivalent date/time formats;
- natural question or negation restructuring;
- reasonable proper-name transliterations;
- slightly awkward wording that remains understandable.

Examples of equivalent formats:

160 million <-> 1.6亿
9:00 p.m. <-> 晚上9点
39th article <-> 第39条
1997 <-> 一九九七

Evaluate these dimensions independently:

1. core_meaning_preserved
Does the overall factual proposition remain the same?

2. main_action_state_match
Do the main action and/or state match?

3. participant_roles_match
Are subject/object/agent/patient roles preserved?

4. location_direction_match
Are locations, directions, inside/outside, to/from and similar
relations preserved?

5. time_event_order_match
Are time information, event order and event sequence preserved?

6. quantity_match
Are important quantities, dates, percentages and numerical
facts semantically equivalent?
Equivalent written formats count as matching.

7. entity_match
Are important people, organizations, places and named entities
preserved?
Reasonable transliteration variation counts as matching.

8. negation_event_status_match
Are negation, possibility, expectation, success/failure,
intention, completion and event-status meanings preserved?

9. important_information_preserved
Is important information from the English source retained?
Tiny stylistic details do not count as important omissions.

10. unsupported_information_added
Does the Chinese translation introduce an important factual
claim not supported by the English source?

11. minor_semantic_issue
Is there a real but non-critical semantic defect while all
critical factual meaning remains correct?

12. fluency_or_grammar_affects_understanding
Does grammar or fluency genuinely damage understanding?
Do NOT set this true merely because a more elegant translation
could be written.

13. uncertain
Set true only if you genuinely cannot judge reliably.

IMPORTANT:

Do not confuse style problems with meaning problems.

A colloquial but understandable translation may still have all
semantic dimensions marked correct.

If an important factual relation changes, such as:
- action/state;
- location;
- direction;
- participant role;
- number;
- entity;
- event status;
then mark the corresponding semantic check false.

Return exactly ONE valid JSON object.
No markdown.
No text outside JSON.
No chain-of-thought.

Schema:

{
  "uncertain": false,
  "core_meaning_preserved": true,
  "main_action_state_match": true,
  "participant_roles_match": true,
  "location_direction_match": true,
  "time_event_order_match": true,
  "quantity_match": true,
  "entity_match": true,
  "negation_event_status_match": true,
  "important_information_preserved": true,
  "unsupported_information_added": false,
  "minor_semantic_issue": false,
  "fluency_or_grammar_affects_understanding": false,
  "reason": "brief evidence-based reason"
}
""".strip()


def build_user_prompt(
    row: pd.Series,
) -> str:

    en = str(
        row[
            "en"
        ]
    ).strip()

    zh = str(
        row[
            "zh"
        ]
    ).strip()

    return (
        "ENGLISH:\n"
        f"{en}\n\n"
        "CHINESE:\n"
        f"{zh}\n\n"
        "Evaluate the semantic dimensions and return only "
        "the required JSON object."
    )


def render_prompt(
    tokenizer,
    row: pd.Series,
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
                build_user_prompt(
                    row
                ),
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
# Parsing
# ============================================================

THINK_RE = re.compile(
    r"<think>.*?</think>",
    flags=(
        re.DOTALL
        |
        re.IGNORECASE
    ),
)


def strip_model_wrappers(
    text: str,
) -> str:

    text = str(
        text
    ).strip()

    text = THINK_RE.sub(
        "",
        text,
    ).strip()

    if text.startswith(
        "```"
    ):

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    return text.strip()


def extract_json_object(
    text: str,
) -> dict:

    text = strip_model_wrappers(
        text
    )

    try:

        value = json.loads(
            text
        )

        if isinstance(
            value,
            dict,
        ):

            return value

    except Exception:
        pass

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start >= 0
        and
        end > start
    ):

        candidate = text[
            start:
            end + 1
        ]

        value = json.loads(
            candidate
        )

        if isinstance(
            value,
            dict,
        ):

            return value

    raise ValueError(
        "No valid JSON object found."
    )


def coerce_bool(
    value,
) -> bool:

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        int,
    ):

        if value in (
            0,
            1,
        ):

            return bool(
                value
            )

    if isinstance(
        value,
        str,
    ):

        normalized = (
            value
            .strip()
            .lower()
        )

        if normalized in {
            "true",
            "yes",
            "1",
        }:

            return True

        if normalized in {
            "false",
            "no",
            "0",
        }:

            return False

    raise ValueError(
        f"Cannot convert to bool: {value!r}"
    )


# ============================================================
# Deterministic label mapping
# ============================================================

def derive_final_label(
    values: dict,
) -> tuple[
    str,
    list[str],
]:

    failed_dimensions = []

    if values[
        "uncertain"
    ]:

        return (
            "UNCERTAIN",
            failed_dimensions,
        )

    critical_match_fields = [

        "core_meaning_preserved",

        "main_action_state_match",

        "participant_roles_match",

        "location_direction_match",

        "time_event_order_match",

        "quantity_match",

        "entity_match",

        "negation_event_status_match",

        "important_information_preserved",
    ]

    for field in (
        critical_match_fields
    ):

        if not values[
            field
        ]:

            failed_dimensions.append(
                field
            )

    if values[
        "unsupported_information_added"
    ]:

        failed_dimensions.append(
            "unsupported_information_added"
        )

    # ========================================================
    # FAIL:
    # any critical factual / semantic mismatch
    # ========================================================

    if failed_dimensions:

        return (
            "FAIL",
            failed_dimensions,
        )

    # ========================================================
    # MINOR:
    # no critical mismatch, but a real limited issue remains
    # ========================================================

    if (
        values[
            "minor_semantic_issue"
        ]
        or
        values[
            "fluency_or_grammar_affects_understanding"
        ]
    ):

        return (
            "MINOR",
            failed_dimensions,
        )

    # ========================================================
    # PASS
    # ========================================================

    return (
        "PASS",
        failed_dimensions,
    )


def normalize_structured_json(
    value: dict,
) -> dict:

    normalized = {}

    for field in BOOLEAN_FIELDS:

        if field not in value:

            raise ValueError(
                f"Missing field: {field}"
            )

        normalized[
            field
        ] = coerce_bool(
            value[
                field
            ]
        )

    reason = str(
        value.get(
            "reason",
            ""
        )
    ).strip()

    if len(
        reason
    ) > 1000:

        reason = reason[
            :1000
        ]

    (
        final_label,
        failed_dimensions,
    ) = derive_final_label(
        normalized
    )

    # ========================================================
    # Deterministic error flags
    # ========================================================

    error_meaning = bool(
        (
            not normalized[
                "core_meaning_preserved"
            ]
        )
        or
        (
            not normalized[
                "main_action_state_match"
            ]
        )
    )

    error_omission = bool(
        not normalized[
            "important_information_preserved"
        ]
    )

    error_addition = bool(
        normalized[
            "unsupported_information_added"
        ]
    )

    error_number = bool(
        not normalized[
            "quantity_match"
        ]
    )

    error_entity = bool(
        not normalized[
            "entity_match"
        ]
    )

    error_negation = bool(
        not normalized[
            "negation_event_status_match"
        ]
    )

    error_grammar = bool(
        normalized[
            "fluency_or_grammar_affects_understanding"
        ]
    )

    error_fluency = bool(
        normalized[
            "fluency_or_grammar_affects_understanding"
        ]
    )

    semantic_equivalent = None

    if final_label in {
        "PASS",
        "MINOR",
    }:

        semantic_equivalent = True

    elif final_label == "FAIL":

        semantic_equivalent = False

    return {

        # ----------------------------
        # Final deterministic label
        # ----------------------------

        "judge_label":
            final_label,

        "judge_semantic_equivalent":
            semantic_equivalent,

        "judge_major_error":
            bool(
                final_label
                ==
                "FAIL"
            ),

        "judge_minor_error":
            bool(
                final_label
                ==
                "MINOR"
            ),

        # ----------------------------
        # Structured dimensions
        # ----------------------------

        "judge_uncertain":
            normalized[
                "uncertain"
            ],

        "judge_core_meaning_preserved":
            normalized[
                "core_meaning_preserved"
            ],

        "judge_main_action_state_match":
            normalized[
                "main_action_state_match"
            ],

        "judge_participant_roles_match":
            normalized[
                "participant_roles_match"
            ],

        "judge_location_direction_match":
            normalized[
                "location_direction_match"
            ],

        "judge_time_event_order_match":
            normalized[
                "time_event_order_match"
            ],

        "judge_quantity_match":
            normalized[
                "quantity_match"
            ],

        "judge_entity_match":
            normalized[
                "entity_match"
            ],

        "judge_negation_event_status_match":
            normalized[
                "negation_event_status_match"
            ],

        "judge_important_information_preserved":
            normalized[
                "important_information_preserved"
            ],

        "judge_unsupported_information_added":
            normalized[
                "unsupported_information_added"
            ],

        "judge_minor_semantic_issue":
            normalized[
                "minor_semantic_issue"
            ],

        "judge_fluency_or_grammar_affects_understanding":
            normalized[
                "fluency_or_grammar_affects_understanding"
            ],

        # ----------------------------
        # Derived error flags
        # ----------------------------

        "judge_error_meaning":
            error_meaning,

        "judge_error_omission":
            error_omission,

        "judge_error_addition":
            error_addition,

        "judge_error_number":
            error_number,

        "judge_error_entity":
            error_entity,

        "judge_error_negation":
            error_negation,

        "judge_error_grammar":
            error_grammar,

        "judge_error_fluency":
            error_fluency,

        "judge_failed_dimensions":
            "|".join(
                failed_dimensions
            ),

        "judge_reason":
            reason,
    }


def parse_response(
    text: str,
) -> dict:

    value = extract_json_object(
        text
    )

    return normalize_structured_json(
        value
    )


# ============================================================
# Checkpoint helpers
# ============================================================

def load_checkpoint(
    checkpoint_file: Path,
) -> dict[str, dict]:

    completed = {}

    if not checkpoint_file.exists():

        return completed

    with open(
        checkpoint_file,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            try:

                record = json.loads(
                    line
                )

            except Exception:

                continue

            review_id = str(
                record.get(
                    "review_id",
                    ""
                )
            )

            if review_id:

                # Reject stale checkpoints generated
                # using older prompts.
                if (
                    record.get(
                        "judge_prompt_version"
                    )
                    !=
                    PROMPT_VERSION
                ):

                    continue

                completed[
                    review_id
                ] = record

    return completed


def append_checkpoint_records(
    checkpoint_file: Path,
    records: list[dict],
):

    checkpoint_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        checkpoint_file,
        "a",
        encoding="utf-8",
    ) as f:

        for record in records:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            f.write(
                "\n"
            )

        f.flush()


# ============================================================
# Calibration reuse
# ============================================================

def load_calibration_reuse(
    calibration_results_file: Path,
) -> dict[str, dict]:

    if not calibration_results_file.exists():

        return {}

    df = pd.read_parquet(
        calibration_results_file
    )

    required = {

        "review_id",

        "judge_label",

        "judge_parse_success",

        "judge_prompt_version",
    }

    if not required.issubset(
        set(
            df.columns
        )
    ):

        return {}

    reusable = {}

    for _, row in df.iterrows():

        if not bool(
            row[
                "judge_parse_success"
            ]
        ):

            continue

        if (
            str(
                row[
                    "judge_prompt_version"
                ]
            )
            !=
            PROMPT_VERSION
        ):

            continue

        record = {}

        for column in df.columns:

            if (
                column.startswith(
                    "judge_"
                )
            ):

                value = row[
                    column
                ]

                if pd.isna(
                    value
                ):

                    value = None

                elif hasattr(
                    value,
                    "item",
                ):

                    try:

                        value = value.item()

                    except Exception:

                        pass

                record[
                    column
                ] = value

        record[
            "review_id"
        ] = str(
            row[
                "review_id"
            ]
        )

        record[
            "judge_reused_from_calibration"
        ] = True

        reusable[
            record[
                "review_id"
            ]
        ] = record

    return reusable


# ============================================================
# Model loading
# ============================================================

def load_judge_model(
    model_path: Path,
):

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    print(
        "\nGPU:"
    )

    print(
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                model_path
            ),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    tokenizer.padding_side = (
        "left"
    )

    if (
        tokenizer.pad_token_id
        is None
    ):

        tokenizer.pad_token_id = (
            tokenizer.eos_token_id
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
            str(
                model_path
            ),
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        .to(
            "cuda"
        )
    )

    model.eval()

    model.config.use_cache = True

    try:

        model.generation_config.do_sample = (
            False
        )

        model.generation_config.temperature = (
            None
        )

        model.generation_config.top_p = (
            None
        )

        model.generation_config.top_k = (
            None
        )

    except Exception:

        pass

    print(
        "Model loaded."
    )

    return (
        tokenizer,
        model,
    )


# ============================================================
# Prompt preflight
# ============================================================

def check_prompt_lengths(
    tokenizer,
    selected_df: pd.DataFrame,
    max_input_tokens: int,
):

    print(
        "\nChecking judge prompt lengths..."
    )

    lengths = []

    longest_review_id = None

    longest_length = -1

    for _, row in (
        selected_df.iterrows()
    ):

        prompt = render_prompt(
            tokenizer,
            row,
        )

        ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )[
            "input_ids"
        ]

        length = len(
            ids
        )

        lengths.append(
            length
        )

        if (
            length
            >
            longest_length
        ):

            longest_length = length

            longest_review_id = str(
                row[
                    "review_id"
                ]
            )

    min_length = min(
        lengths
    )

    max_length = max(
        lengths
    )

    mean_length = (
        sum(
            lengths
        )
        /
        len(
            lengths
        )
    )

    print(
        "Prompt tokens min:",
        min_length
    )

    print(
        "Prompt tokens mean:",
        f"{mean_length:.2f}"
    )

    print(
        "Prompt tokens max:",
        max_length
    )

    print(
        "Longest review_id:",
        longest_review_id
    )

    print(
        "Configured max_input_tokens:",
        max_input_tokens
    )

    if (
        max_length
        >
        max_input_tokens
    ):

        raise RuntimeError(
            "\nPrompt preflight failed.\n"
            f"Longest prompt: {max_length}\n"
            f"Limit: {max_input_tokens}\n"
            f"Review: {longest_review_id}\n\n"
            "Refusing to silently truncate."
        )


# ============================================================
# Batched generation
# ============================================================

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
):

    # ========================================================
    # Permanent safety protection:
    # NEVER silently truncate judge input.
    # ========================================================

    token_lengths = []

    for prompt in prompts:

        token_lengths.append(
            len(
                tokenizer(
                    prompt,
                    add_special_tokens=False,
                )[
                    "input_ids"
                ]
            )
        )

    max_prompt_tokens = max(
        token_lengths
    )

    if (
        max_prompt_tokens
        >
        max_input_tokens
    ):

        raise RuntimeError(
            "\nJudge input exceeds safe token limit.\n"
            f"Longest prompt tokens: {max_prompt_tokens}\n"
            f"max_input_tokens: {max_input_tokens}\n"
            "Refusing to silently truncate."
        )

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    inputs = {

        key:
            value.to(
                "cuda"
            )

        for (
            key,
            value,
        )
        in inputs.items()
    }

    input_width = int(
        inputs[
            "input_ids"
        ].shape[
            1
        ]
    )

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():

        output_ids = model.generate(

            **inputs,

            do_sample=False,

            num_beams=1,

            max_new_tokens=
                max_new_tokens,

            use_cache=True,

            pad_token_id=
                tokenizer.pad_token_id,

            eos_token_id=
                tokenizer.eos_token_id,
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    generated_ids = output_ids[
        :,
        input_width:
    ]

    texts = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return (
        [
            str(
                text
            ).strip()

            for text
            in texts
        ],

        elapsed,
    )


# ============================================================
# Judge batch
# ============================================================

def judge_batch(
    batch_df: pd.DataFrame,
    model,
    tokenizer,
    max_input_tokens: int,
    max_new_tokens: int,
    parse_retries: int,
):

    prompts = [

        render_prompt(
            tokenizer,
            row,
        )

        for _, row
        in batch_df.iterrows()
    ]

    (
        outputs,
        elapsed,
    ) = generate_batch(

        model=model,

        tokenizer=tokenizer,

        prompts=prompts,

        max_input_tokens=
            max_input_tokens,

        max_new_tokens=
            max_new_tokens,
    )

    per_sample_latency = (
        elapsed
        /
        max(
            len(
                batch_df
            ),
            1,
        )
    )

    records = []

    failed_indexes = []

    for local_index, (
        (_, row),
        raw_output,
    ) in enumerate(
        zip(
            batch_df.iterrows(),
            outputs,
        )
    ):

        try:

            parsed = parse_response(
                raw_output
            )

            parse_success = True

        except Exception:

            parsed = None

            parse_success = False

            failed_indexes.append(
                local_index
            )

        record = {

            "review_id":
                str(
                    row[
                        "review_id"
                    ]
                ),

            "judge_step_version":
                STEP_VERSION,

            "judge_prompt_version":
                PROMPT_VERSION,

            "judge_model":
                JUDGE_MODEL_NAME,

            "judge_parse_success":
                parse_success,

            "judge_parse_attempts":
                1,

            "judge_raw_response":
                raw_output,

            "judge_latency_seconds":
                float(
                    per_sample_latency
                ),

            "judge_reused_from_calibration":
                False,

            "judge_timestamp_utc":
                datetime.now(
                    timezone.utc
                ).isoformat(),
        }

        if parsed is not None:

            record.update(
                parsed
            )

        records.append(
            record
        )

    # ========================================================
    # Retry invalid JSON
    # ========================================================

    for _ in range(
        parse_retries
    ):

        if not failed_indexes:

            break

        retry_prompts = []

        retry_record_indexes = []

        for index in failed_indexes:

            original_prompt = prompts[
                index
            ]

            retry_prompt = (
                original_prompt
                +
                "\n\nYour previous response was not valid "
                "JSON. Return exactly ONE JSON object using "
                "the required schema. No markdown and no "
                "additional text."
            )

            retry_prompts.append(
                retry_prompt
            )

            retry_record_indexes.append(
                index
            )

        (
            retry_outputs,
            retry_elapsed,
        ) = generate_batch(

            model=model,

            tokenizer=tokenizer,

            prompts=retry_prompts,

            max_input_tokens=
                max_input_tokens,

            max_new_tokens=
                max_new_tokens,
        )

        retry_latency = (
            retry_elapsed
            /
            max(
                len(
                    retry_prompts
                ),
                1,
            )
        )

        still_failed = []

        for (
            record_index,
            retry_output,
        ) in zip(
            retry_record_indexes,
            retry_outputs,
        ):

            record = records[
                record_index
            ]

            record[
                "judge_parse_attempts"
            ] += 1

            record[
                "judge_latency_seconds"
            ] += float(
                retry_latency
            )

            record[
                "judge_raw_response"
            ] = retry_output

            try:

                parsed = parse_response(
                    retry_output
                )

                record.update(
                    parsed
                )

                record[
                    "judge_parse_success"
                ] = True

            except Exception:

                record[
                    "judge_parse_success"
                ] = False

                still_failed.append(
                    record_index
                )

        failed_indexes = (
            still_failed
        )

    # ========================================================
    # Parse failure fallback
    # ========================================================

    for record in records:

        if not record[
            "judge_parse_success"
        ]:

            record.update({

                "judge_label":
                    "UNCERTAIN",

                "judge_semantic_equivalent":
                    None,

                "judge_major_error":
                    False,

                "judge_minor_error":
                    False,

                "judge_uncertain":
                    True,

                "judge_core_meaning_preserved":
                    None,

                "judge_main_action_state_match":
                    None,

                "judge_participant_roles_match":
                    None,

                "judge_location_direction_match":
                    None,

                "judge_time_event_order_match":
                    None,

                "judge_quantity_match":
                    None,

                "judge_entity_match":
                    None,

                "judge_negation_event_status_match":
                    None,

                "judge_important_information_preserved":
                    None,

                "judge_unsupported_information_added":
                    None,

                "judge_minor_semantic_issue":
                    None,

                "judge_fluency_or_grammar_affects_understanding":
                    None,

                "judge_error_meaning":
                    False,

                "judge_error_omission":
                    False,

                "judge_error_addition":
                    False,

                "judge_error_number":
                    False,

                "judge_error_entity":
                    False,

                "judge_error_negation":
                    False,

                "judge_error_grammar":
                    False,

                "judge_error_fluency":
                    False,

                "judge_failed_dimensions":
                    "",

                "judge_reason":
                    "JSON_PARSE_FAILED",
            })

    return records


# ============================================================
# Report tables
# ============================================================

def build_report_tables(
    result_df: pd.DataFrame,
):

    label_report = (
        result_df[
            "judge_label"
        ]
        .value_counts()
        .rename_axis(
            "label"
        )
        .reset_index(
            name="count"
        )
    )

    label_report[
        "percent"
    ] = (
        label_report[
            "count"
        ]
        /
        len(
            result_df
        )
        *
        100
    )

    group_report = (
        result_df
        .groupby(
            [
                "review_group",
                "judge_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    group_totals = (
        group_report
        .groupby(
            "review_group"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )

    group_report[
        "percent_within_group"
    ] = (
        group_report[
            "count"
        ]
        /
        group_totals
        *
        100
    )

    source_report = (
        result_df
        .groupby(
            [
                "source_dataset",
                "judge_label",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    source_totals = (
        source_report
        .groupby(
            "source_dataset"
        )[
            "count"
        ]
        .transform(
            "sum"
        )
    )

    source_report[
        "percent_within_source"
    ] = (
        source_report[
            "count"
        ]
        /
        source_totals
        *
        100
    )

    return (
        label_report,
        group_report,
        source_report,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    model_path = Path(
        "/root/autodl-tmp/models/Qwen3-8B"
    )

    review_root = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14e_qwen_review"
    )

    input_file = (
        review_root
        /
        "qwen_review_input_v1.parquet"
    )

    calibration_dir = (
        review_root
        /
        "calibration"
    )

    full_dir = (
        review_root
        /
        "full_review"
    )

    calibration_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration_input_file = (
        calibration_dir
        /
        "calibration_input_500_v1.parquet"
    )

    calibration_results_file = (
        calibration_dir
        /
        "qwen_calibration_500_v1.parquet"
    )

    calibration_csv_file = (
        calibration_dir
        /
        "qwen_calibration_500_v1.csv"
    )

    calibration_checkpoint = (
        calibration_dir
        /
        "qwen_calibration_checkpoint_v1.jsonl"
    )

    calibration_report_file = (
        calibration_dir
        /
        "calibration_report_v1.json"
    )

    full_results_file = (
        full_dir
        /
        "qwen_review_results_v1.parquet"
    )

    full_csv_file = (
        full_dir
        /
        "qwen_review_results_v1.csv"
    )

    full_checkpoint = (
        full_dir
        /
        "qwen_review_checkpoint_v1.jsonl"
    )

    full_report_file = (
        full_dir
        /
        "qwen_review_report_v1.json"
    )

    if args.mode == "calibration":

        output_file = (
            calibration_results_file
        )

        output_csv = (
            calibration_csv_file
        )

        checkpoint_file = (
            calibration_checkpoint
        )

        report_file = (
            calibration_report_file
        )

    else:

        output_file = (
            full_results_file
        )

        output_csv = (
            full_csv_file
        )

        checkpoint_file = (
            full_checkpoint
        )

        report_file = (
            full_report_file
        )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14E-2 V5 - STRUCTURED QWEN3-8B REVIEW"
    )

    print(
        "=" * 110
    )

    print(
        "\nStep version:",
        STEP_VERSION
    )

    print(
        "Prompt version:",
        PROMPT_VERSION
    )

    print(
        "Mode:",
        args.mode
    )

    if not input_file.exists():

        raise FileNotFoundError(
            input_file
        )

    if not model_path.exists():

        raise FileNotFoundError(
            model_path
        )

    if (
        output_file.exists()
        and
        not args.overwrite
    ):

        raise RuntimeError(
            "\nFinal output already exists:\n"
            f"{output_file}\n\n"
            "Use --overwrite if rebuilding intentionally."
        )

    if args.overwrite:

        for path in [

            output_file,
            output_csv,
            checkpoint_file,
            report_file,

        ]:

            if path.exists():

                path.unlink()

    # ========================================================
    # Frozen input
    # ========================================================

    review_df = pd.read_parquet(
        input_file
    )

    required = {

        "review_id",

        "pair_id",

        "en",

        "zh",

        "source_dataset",

        "review_group",

        "risk_flags",
    }

    missing = (
        required
        -
        set(
            review_df.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    if (
        review_df[
            "review_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate review_id."
        )

    print(
        "\nFrozen review rows:",
        len(
            review_df
        )
    )

    print(
        "\nReview groups:"
    )

    print(
        review_df[
            "review_group"
        ]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Mode selection
    # ========================================================

    if args.mode == "calibration":

        selected_df = (
            build_calibration_subset(

                review_df=
                    review_df,

                calibration_size=
                    args.calibration_size,

                calibration_risk_size=
                    args.calibration_risk_size,

                seed=
                    args.seed,
            )
        )

        selected_df.to_parquet(
            calibration_input_file,
            index=False,
        )

    else:

        selected_df = (
            review_df
            .copy()
            .reset_index(
                drop=True
            )
        )

    print(
        "\nSelected rows:",
        len(
            selected_df
        )
    )

    # ========================================================
    # Prompt preflight BEFORE loading 8B model
    # ========================================================

    print(
        "\nLoading tokenizer for prompt preflight..."
    )

    preflight_tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                model_path
            ),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    check_prompt_lengths(
        preflight_tokenizer,
        selected_df,
        args.max_input_tokens,
    )

    del preflight_tokenizer

    # ========================================================
    # Existing checkpoint
    # ========================================================

    completed = load_checkpoint(
        checkpoint_file
    )

    print(
        "\nValid checkpoint rows:",
        len(
            completed
        )
    )

    # ========================================================
    # Reuse V5 calibration only
    # ========================================================

    reused_calibration = 0

    if (
        args.mode == "full"
        and
        args.reuse_calibration
    ):

        reusable = load_calibration_reuse(
            calibration_results_file
        )

        valid_ids = set(
            selected_df[
                "review_id"
            ]
            .astype(str)
        )

        for (
            review_id,
            record,
        ) in reusable.items():

            if (
                review_id
                not in
                valid_ids
            ):

                continue

            if (
                review_id
                in
                completed
            ):

                continue

            completed[
                review_id
            ] = record

            reused_calibration += 1

        print(
            "Calibration rows reused:",
            reused_calibration
        )

    completed_ids = set(
        completed.keys()
    )

    pending_df = (
        selected_df[
            ~selected_df[
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

    # ========================================================
    # Inference
    # ========================================================

    if len(
        pending_df
    ) > 0:

        (
            tokenizer,
            model,
        ) = load_judge_model(
            model_path
        )

        total_pending = len(
            pending_df
        )

        for start in range(
            0,
            total_pending,
            args.batch_size,
        ):

            batch = (
                pending_df
                .iloc[
                    start:
                    start
                    +
                    args.batch_size
                ]
                .copy()
            )

            records = judge_batch(

                batch_df=
                    batch,

                model=
                    model,

                tokenizer=
                    tokenizer,

                max_input_tokens=
                    args.max_input_tokens,

                max_new_tokens=
                    args.max_new_tokens,

                parse_retries=
                    args.parse_retries,
            )

            append_checkpoint_records(
                checkpoint_file,
                records,
            )

            for record in records:

                completed[
                    record[
                        "review_id"
                    ]
                ] = record

            done = min(
                start
                +
                args.batch_size,
                total_pending,
            )

            parse_success_count = sum(

                bool(
                    record[
                        "judge_parse_success"
                    ]
                )

                for record
                in records
            )

            labels = Counter(

                record[
                    "judge_label"
                ]

                for record
                in records
            )

            print(
                f"{done}/{total_pending} "
                f"| parse "
                f"{parse_success_count}/"
                f"{len(records)} "
                f"| {dict(labels)}"
            )

        del model
        del tokenizer

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # Assemble results
    # ========================================================

    result_records = []

    missing_ids = []

    for review_id in (
        selected_df[
            "review_id"
        ]
        .astype(str)
    ):

        record = completed.get(
            review_id
        )

        if record is None:

            missing_ids.append(
                review_id
            )

            continue

        result_records.append(
            record
        )

    if missing_ids:

        raise RuntimeError(
            "\nMissing judge records.\n"
            f"Count: {len(missing_ids)}"
        )

    judge_df = pd.DataFrame(
        result_records
    )

    judge_columns = [

        column

        for column
        in judge_df.columns

        if (
            column
            ==
            "review_id"
            or
            column.startswith(
                "judge_"
            )
        )
    ]

    judge_df = judge_df[
        judge_columns
    ].copy()

    result_df = selected_df.merge(
        judge_df,
        on="review_id",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # Reports
    # ========================================================

    (
        label_report,
        group_report,
        source_report,
    ) = build_report_tables(
        result_df
    )

    parse_success = int(
        result_df[
            "judge_parse_success"
        ]
        .fillna(
            False
        )
        .astype(bool)
        .sum()
    )

    parse_rate = (
        parse_success
        /
        max(
            len(
                result_df
            ),
            1,
        )
        *
        100
    )

    semantic_usable = int(
        result_df[
            "judge_label"
        ]
        .isin(
            [
                "PASS",
                "MINOR",
            ]
        )
        .sum()
    )

    semantic_usable_rate = (
        semantic_usable
        /
        max(
            len(
                result_df
            ),
            1,
        )
        *
        100
    )

    auto_part = result_df[
        result_df[
            "review_group"
        ]
        ==
        "AUTO_ACCEPT_AUDIT"
    ]

    auto_fail_count = int(
        (
            auto_part[
                "judge_label"
            ]
            ==
            "FAIL"
        )
        .sum()
    )

    auto_fail_rate = (
        auto_fail_count
        /
        max(
            len(
                auto_part
            ),
            1,
        )
        *
        100
    )

    # ========================================================
    # Save
    # ========================================================

    result_df.to_parquet(
        output_file,
        index=False,
    )

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    label_report_file = (
        output_file.parent
        /
        "label_report_v1.csv"
    )

    group_report_file = (
        output_file.parent
        /
        "review_group_report_v1.csv"
    )

    source_report_file = (
        output_file.parent
        /
        "source_report_v1.csv"
    )

    label_report.to_csv(
        label_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    group_report.to_csv(
        group_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    source_report.to_csv(
        source_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    label_counts = {

        str(key):
            int(value)

        for (
            key,
            value,
        )
        in (
            result_df[
                "judge_label"
            ]
            .value_counts()
            .items()
        )
    }

    report = {

        "step":
            "14E-2",

        "step_version":
            STEP_VERSION,

        "prompt_version":
            PROMPT_VERSION,

        "mode":
            args.mode,

        "judge_model":
            JUDGE_MODEL_NAME,

        "judge_policy":
            (
                "Structured semantic dimensions "
                "with deterministic Python label mapping."
            ),

        "selected_rows":
            int(
                len(
                    result_df
                )
            ),

        "generation": {

            "batch_size":
                int(
                    args.batch_size
                ),

            "max_input_tokens":
                int(
                    args.max_input_tokens
                ),

            "max_new_tokens":
                int(
                    args.max_new_tokens
                ),

            "parse_retries":
                int(
                    args.parse_retries
                ),

            "silent_truncation":
                False,
        },

        "labels":
            label_counts,

        "parse": {

            "success":
                parse_success,

            "total":
                int(
                    len(
                        result_df
                    )
                ),

            "success_percent":
                float(
                    parse_rate
                ),
        },

        "semantic_usable": {

            "pass_plus_minor":
                semantic_usable,

            "percent":
                float(
                    semantic_usable_rate
                ),
        },

        "auto_accept_audit": {

            "rows":
                int(
                    len(
                        auto_part
                    )
                ),

            "fail":
                auto_fail_count,

            "fail_percent":
                float(
                    auto_fail_rate
                ),
        },

        "calibration_reused":
            int(
                reused_calibration
            ),

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            (
                "V5_CALIBRATION_COMPLETE"
                if args.mode
                ==
                "calibration"
                else
                "V5_FULL_REVIEW_COMPLETE"
            ),
    }

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print("\n")
    print(
        "=" * 110
    )

    print(
        "STEP 14E-2 V5 RESULT"
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
        f"{parse_success}/"
        f"{len(result_df)} "
        f"({parse_rate:.2f}%)"
    )

    print(
        "\nLabels:"
    )

    print(
        label_report
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nReview-group distribution:"
    )

    print(
        group_report
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nSource distribution:"
    )

    print(
        source_report
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nPASS + MINOR:"
    )

    print(
        semantic_usable,
        f"({semantic_usable_rate:.2f}%)"
    )

    print(
        "\nAUTO_ACCEPT audit:"
    )

    print(
        "Rows:",
        len(
            auto_part
        )
    )

    print(
        "FAIL:",
        auto_fail_count,
        f"({auto_fail_rate:.2f}%)"
    )

    print(
        "\nOutput:"
    )

    print(
        output_file
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


if __name__ == "__main__":

    main()