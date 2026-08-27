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

STEP_VERSION = "14E2_V1"

PROMPT_VERSION = "ZH_EN_JUDGE_V2"

JUDGE_MODEL_NAME = "Qwen3-8B"


# ============================================================
# Labels
# ============================================================

VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}

ERROR_KEYS = [
    "meaning",
    "omission",
    "addition",
    "number",
    "entity",
    "negation",
    "grammar",
    "fluency",
]


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14E-2 - Review ZH-EN human parallel "
            "data with local Qwen3-8B."
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
        help=(
            "Number of RISK_REVIEW rows in calibration. "
            "Remaining calibration rows come from "
            "AUTO_ACCEPT_AUDIT."
        ),
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
        default=1024,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--parse_retries",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help=(
            "Append each completed batch to checkpoint. "
            "1 means every batch."
        ),
    )

    parser.add_argument(
        "--no_reuse_calibration",
        dest="reuse_calibration",
        action="store_false",
        help=(
            "In full mode, do not reuse matching "
            "calibration results."
        ),
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
# Stable deterministic sampling
# ============================================================

def stable_hash(
    value: str,
    seed: int,
) -> str:

    content = (
        f"{seed}\n"
        f"{value}"
    )

    return hashlib.sha256(
        content.encode(
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
            f"Requested {total_size}, "
            f"but only {len(df)} rows available."
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

        base = int(
            math.floor(
                exact
            )
        )

        allocations[
            str(key)
        ] = base

        fractions[
            str(key)
        ] = (
            exact
            -
            base
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

    allocations = (
        allocate_proportional(
            df=df,
            total_size=
                sample_size,
            column=
                stratum_column,
        )
    )

    sampled = []

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
            "_calibration_hash"
        ] = [

            stable_hash(
                value=
                    str(
                        review_id
                    ),
                seed=
                    seed,
            )

            for review_id
            in (
                part[
                    "review_id"
                ]
            )
        ]

        part = (
            part
            .sort_values(
                [
                    "_calibration_hash",
                    "review_id",
                ]
            )
            .head(
                count
            )
            .copy()
        )

        sampled.append(
            part
        )

    result = (
        pd.concat(
            sampled,
            ignore_index=True,
        )
    )

    result = (
        result
        .sort_values(
            [
                stratum_column,
                "_calibration_hash",
                "review_id",
            ]
        )
        .drop(
            columns=[
                "_calibration_hash"
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            result
        )
        !=
        sample_size
    ):

        raise RuntimeError(
            "\nCalibration sampling size mismatch.\n"
            f"Expected: {sample_size}\n"
            f"Found: {len(result)}"
        )

    return result


# ============================================================
# Build calibration subset
# ============================================================

def build_calibration_subset(
    review_df: pd.DataFrame,
    calibration_size: int,
    calibration_risk_size: int,
    seed: int,
) -> pd.DataFrame:

    if (
        calibration_size
        >
        len(
            review_df
        )
    ):

        raise ValueError(
            "\nCalibration size is larger "
            "than review dataset."
        )

    if (
        calibration_risk_size
        >
        calibration_size
    ):

        raise ValueError(
            "calibration_risk_size cannot "
            "exceed calibration_size."
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

    if (
        calibration_risk_size
        >
        len(
            risk
        )
    ):

        raise RuntimeError(
            "\nNot enough RISK_REVIEW rows "
            "for calibration."
        )

    if (
        auto_size
        >
        len(
            auto
        )
    ):

        raise RuntimeError(
            "\nNot enough AUTO_ACCEPT_AUDIT "
            "rows for calibration."
        )

    risk_sample = (
        deterministic_stratified_sample(
            df=risk,
            sample_size=
                calibration_risk_size,
            seed=
                seed,
        )
    )

    auto_sample = (
        deterministic_stratified_sample(
            df=auto,
            sample_size=
                auto_size,
            seed=
                seed + 1,
        )
    )

    calibration = (
        pd.concat(
            [
                risk_sample,
                auto_sample,
            ],
            ignore_index=True,
        )
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
# Judge prompt
# ============================================================

SYSTEM_PROMPT = r"""
You are an independent bilingual Chinese-English translation
quality judge.

You will receive one English sentence and one Chinese sentence.

Your ONLY task is to determine whether they are valid parallel
translations of the same underlying meaning.

Judge the translation itself.
Do not infer anything from how the sample was selected.

============================================================
CORE PRINCIPLE
============================================================

Semantic equivalence is more important than literal wording.

English and Chinese often express the same meaning using
different:

- grammar
- word order
- tense/aspect realization
- negation structure
- question structure
- number formatting
- date/time formatting
- transliteration style
- punctuation
- units or conventional written forms

These differences are NOT translation errors by themselves.

============================================================
IMPORTANT VALID VARIATIONS
============================================================

The following kinds of differences may be completely correct:

1997
<-> 一九九七

4.7 million
<-> 470万

10.30 as a time
<-> 10点30分

9:00 p.m.
<-> 晚上9点

two years later
<-> 2年后

39th article
<-> 第39条

barely know
<-> 不是很熟悉

unharmed
<-> 没有受到伤害

different
<-> 不同

Isn't that mine?
<-> 那是我的吗？

Proper names may use reasonable Chinese transliterations.
Do NOT mark a translation wrong merely because you personally
prefer a different transliteration unless the rendered name
clearly refers to a different entity.

============================================================
LABEL POLICY
============================================================

PASS

Use PASS when the core meaning is correctly preserved and
there is no actual translation defect.

PASS still applies when:

- wording is somewhat literal but understandable;
- number/date/time formatting differs naturally;
- word order differs naturally;
- Chinese is slightly less elegant than ideal;
- proper nouns use a reasonable transliteration;
- grammar is different because of normal Chinese-English
  structural differences.

Do NOT use MINOR merely because you can imagine a smoother or
more elegant translation.

------------------------------------------------------------

MINOR

Use MINOR only when there is a REAL but non-critical defect.

Examples:

- a small piece of information is weakened or imprecise;
- a minor modifier is omitted;
- wording creates a genuine small ambiguity;
- a grammatical problem slightly harms interpretation;
- a clearly incorrect but non-critical name rendering occurs;
- fluency is sufficiently poor that it genuinely degrades the
  translation, not merely because another phrasing is nicer.

The core meaning must still remain correct.

------------------------------------------------------------

FAIL

Use FAIL when there is a substantive translation error.

Examples:

- wrong meaning;
- important omission;
- important unsupported addition;
- wrong number or date;
- wrong entity;
- reversed relation;
- mistranslated action;
- source and target are substantially unrelated.

------------------------------------------------------------

UNCERTAIN

Use UNCERTAIN only when there is not enough information to
judge reliably.

============================================================
ERROR FLAGS
============================================================

Only mark an error flag true when an actual error exists.

Do NOT mark:

number=true

simply because equivalent numbers use different formats.

Do NOT mark:

entity=true

simply because a proper name uses a different but plausible
transliteration.

Do NOT mark:

fluency=true

just because you can write a more natural sentence.

============================================================
OUTPUT
============================================================

Return ONE valid JSON object only.

No markdown.
No commentary outside JSON.
No chain-of-thought.

Schema:

{
  "label": "PASS|MINOR|FAIL|UNCERTAIN",
  "semantic_equivalent": true,
  "major_error": false,
  "minor_error": false,
  "errors": {
    "meaning": false,
    "omission": false,
    "addition": false,
    "number": false,
    "entity": false,
    "negation": false,
    "grammar": false,
    "fluency": false
  },
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
        "Independently judge whether these two sentences "
        "are valid parallel translations. "
        "Return only the required JSON object."
    )


# ============================================================
# Chat template
# ============================================================

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

        return (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    except TypeError:

        # Compatibility fallback for older
        # tokenizer/chat-template versions.

        return (
            tokenizer
            .apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )


# ============================================================
# JSON parsing
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

    # First attempt:
    # response is already pure JSON.
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

    # Second attempt:
    # extract outermost {...}
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
    default: bool = False,
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

    return default


def normalize_judge_json(
    value: dict,
) -> dict:

    label = (
        str(
            value.get(
                "label",
                ""
            )
        )
        .strip()
        .upper()
    )

    if label not in VALID_LABELS:

        raise ValueError(
            f"Invalid label: {label}"
        )

    errors_raw = value.get(
        "errors",
        {}
    )

    if not isinstance(
        errors_raw,
        dict,
    ):

        errors_raw = {}

    errors = {

        key:
            coerce_bool(
                errors_raw.get(
                    key,
                    False
                )
            )

        for key
        in ERROR_KEYS
    }

    # Canonicalize the high-level
    # interpretation by label.

    if label == "PASS":

        semantic_equivalent = True
        major_error = False
        minor_error = False

    elif label == "MINOR":

        semantic_equivalent = True
        major_error = False
        minor_error = True

    elif label == "FAIL":

        semantic_equivalent = False
        major_error = True
        minor_error = False

    else:

        semantic_equivalent = None
        major_error = False
        minor_error = False

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

    return {

        "judge_label":
            label,

        "judge_semantic_equivalent":
            semantic_equivalent,

        "judge_major_error":
            major_error,

        "judge_minor_error":
            minor_error,

        "judge_error_meaning":
            errors[
                "meaning"
            ],

        "judge_error_omission":
            errors[
                "omission"
            ],

        "judge_error_addition":
            errors[
                "addition"
            ],

        "judge_error_number":
            errors[
                "number"
            ],

        "judge_error_entity":
            errors[
                "entity"
            ],

        "judge_error_negation":
            errors[
                "negation"
            ],

        "judge_error_grammar":
            errors[
                "grammar"
            ],

        "judge_error_fluency":
            errors[
                "fluency"
            ],

        "judge_reason":
            reason,
    }


def parse_response(
    text: str,
) -> dict:

    value = extract_json_object(
        text
    )

    return normalize_judge_json(
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
# Calibration reuse for full review
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

        record = {}

        for column in df.columns:

            if column.startswith(
                "judge_"
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

    # Prevent irrelevant warnings from
    # sampling configuration where possible.

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
# Batched generation
# ============================================================

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[
    list[str],
    float,
]:

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=
            max_input_tokens,
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

        output_ids = (
            model.generate(
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
        )

    torch.cuda.synchronize()

    elapsed = (
        time.perf_counter()
        -
        start
    )

    generated_ids = (
        output_ids[
            :,
            input_width:
        ]
    )

    texts = (
        tokenizer
        .batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
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
# Judge one batch
# ============================================================

def judge_batch(
    batch_df: pd.DataFrame,
    model,
    tokenizer,
    max_input_tokens: int,
    max_new_tokens: int,
    parse_retries: int,
) -> list[dict]:

    prompts = [

        render_prompt(
            tokenizer,
            row,
        )

        for _, row
        in batch_df.iterrows()
    ]

    (
        raw_outputs,
        elapsed,
    ) = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=
            prompts,
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
            raw_outputs,
        )
    ):

        try:

            parsed = (
                parse_response(
                    raw_output
                )
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
    # Parse retry
    # ========================================================

    for retry_index in range(
        parse_retries
    ):

        if not failed_indexes:
            break

        retry_prompts = []

        retry_record_indexes = []

        for index in failed_indexes:

            previous_output = (
                records[
                    index
                ][
                    "judge_raw_response"
                ]
            )

            retry_prompt = (
                prompts[
                    index
                ]
                +
                "\n\nIMPORTANT: Your previous response "
                "could not be parsed as valid JSON. "
                "Return ONE JSON object only. "
                "No markdown, no comments, no reasoning.\n"
                "Previous invalid response:\n"
                +
                previous_output[
                    :2000
                ]
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
            prompts=
                retry_prompts,
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

            record = (
                records[
                    record_index
                ]
            )

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

                parsed = (
                    parse_response(
                        retry_output
                    )
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
    # Final fallback for unparseable rows
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

                "judge_reason":
                    "JSON_PARSE_FAILED",
            })

    return records


# ============================================================
# Result summary
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

    # ========================================================
    # Header
    # ========================================================

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14E-2 - QWEN3-8B QUALITY REVIEW"
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

    print(
        "\nInput:"
    )

    print(
        input_file
    )

    print(
        "\nModel:"
    )

    print(
        model_path
    )

    # ========================================================
    # Validate inputs
    # ========================================================

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
            "Use --overwrite only if you "
            "intentionally want to rebuild it."
        )

    # ========================================================
    # Overwrite handling
    # ========================================================

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
    # Load frozen review input
    # ========================================================

    review_df = (
        pd.read_parquet(
            input_file
        )
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
            "Review input missing columns: "
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
            "Duplicate review_id in input."
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
    # Select mode input
    # ========================================================

    if args.mode == "calibration":

        selected_df = (
            build_calibration_subset(
                review_df=
                    review_df,

                calibration_size=
                    args
                    .calibration_size,

                calibration_risk_size=
                    args
                    .calibration_risk_size,

                seed=
                    args
                    .seed,
            )
        )

        selected_df.to_parquet(
            calibration_input_file,
            index=False,
        )

        print(
            "\nCalibration composition:"
        )

        print(
            selected_df
            .groupby(
                [
                    "review_group",
                    "source_dataset",
                ]
            )
            .size()
            .to_string()
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
    # Existing checkpoint
    # ========================================================

    completed = (
        load_checkpoint(
            checkpoint_file
        )
    )

    print(
        "\nExisting checkpoint rows:",
        len(
            completed
        )
    )

    # ========================================================
    # Reuse calibration in full mode
    # ========================================================

    reused_calibration = 0

    if (
        args.mode == "full"
        and
        args.reuse_calibration
    ):

        reusable = (
            load_calibration_reuse(
                calibration_results_file
            )
        )

        selected_ids = set(
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
                selected_ids
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

    # ========================================================
    # Pending rows
    # ========================================================

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
    # Judge pending rows
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

        batch_number = 0

        for start in range(
            0,
            total_pending,
            args.batch_size,
        ):

            batch_number += 1

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
                    args
                    .max_input_tokens,

                max_new_tokens=
                    args
                    .max_new_tokens,

                parse_retries=
                    args
                    .parse_retries,
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
                f"{parse_success_count}/{len(records)} "
                f"| {dict(labels)}"
            )

        del model
        del tokenizer

        gc.collect()

        torch.cuda.empty_cache()

    # ========================================================
    # Assemble final result
    # ========================================================

    result_records = []

    missing_result_ids = []

    for review_id in (
        selected_df[
            "review_id"
        ]
        .astype(str)
    ):

        record = (
            completed
            .get(
                review_id
            )
        )

        if record is None:

            missing_result_ids.append(
                review_id
            )

            continue

        result_records.append(
            record
        )

    if missing_result_ids:

        raise RuntimeError(
            "\nMissing judge results.\n"
            f"Count: {len(missing_result_ids)}\n"
            f"Examples: {missing_result_ids[:10]}"
        )

    judge_df = pd.DataFrame(
        result_records
    )

    # Avoid duplicated metadata columns.
    judge_columns = [

        column

        for column
        in judge_df.columns

        if column
        ==
        "review_id"

        or
        column.startswith(
            "judge_"
        )
    ]

    judge_df = (
        judge_df[
            judge_columns
        ]
        .copy()
    )

    result_df = (
        selected_df
        .merge(
            judge_df,
            on="review_id",
            how="left",
            validate="one_to_one",
        )
    )

    if (
        len(
            result_df
        )
        !=
        len(
            selected_df
        )
    ):

        raise RuntimeError(
            "Final result size mismatch."
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

    semantic_equivalent_count = int(
        (
            result_df[
                "judge_label"
            ]
            .isin(
                [
                    "PASS",
                    "MINOR",
                ]
            )
        )
        .sum()
    )

    semantic_equivalent_rate = (
        semantic_equivalent_count
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

    auto_part = (
        result_df[
            result_df[
                "review_group"
            ]
            ==
            "AUTO_ACCEPT_AUDIT"
        ]
    )

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
    # Save final result
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
        output_file
        .parent
        /
        "label_report_v1.csv"
    )

    group_report_file = (
        output_file
        .parent
        /
        "review_group_report_v1.csv"
    )

    source_report_file = (
        output_file
        .parent
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

    # ========================================================
    # JSON report
    # ========================================================

    label_counts = {

        str(
            key
        ):
            int(
                value
            )

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

        "judge_model_path":
            str(
                model_path
            ),

        "input_file":
            str(
                input_file
            ),

        "selected_rows":
            int(
                len(
                    selected_df
                )
            ),

        "generation": {

            "batch_size":
                int(
                    args.batch_size
                ),

            "max_input_tokens":
                int(
                    args
                    .max_input_tokens
                ),

            "max_new_tokens":
                int(
                    args
                    .max_new_tokens
                ),

            "do_sample":
                False,

            "num_beams":
                1,

            "parse_retries":
                int(
                    args
                    .parse_retries
                ),
        },

        "calibration": {

            "calibration_size":
                (
                    int(
                        args
                        .calibration_size
                    )
                    if args.mode
                    ==
                    "calibration"
                    else None
                ),

            "calibration_risk_size":
                (
                    int(
                        args
                        .calibration_risk_size
                    )
                    if args.mode
                    ==
                    "calibration"
                    else None
                ),

            "seed":
                int(
                    args.seed
                ),

            "reused_in_full":
                int(
                    reused_calibration
                ),
        },

        "parse": {

            "success":
                int(
                    parse_success
                ),

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

        "labels":
            label_counts,

        "semantic_equivalent": {

            "pass_plus_minor":
                int(
                    semantic_equivalent_count
                ),

            "percent":
                float(
                    semantic_equivalent_rate
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
                int(
                    auto_fail_count
                ),

            "fail_percent":
                float(
                    auto_fail_rate
                ),
        },

        "outputs": {

            "result_parquet":
                str(
                    output_file
                ),

            "result_csv":
                str(
                    output_csv
                ),

            "checkpoint":
                str(
                    checkpoint_file
                ),

            "label_report":
                str(
                    label_report_file
                ),

            "group_report":
                str(
                    group_report_file
                ),

            "source_report":
                str(
                    source_report_file
                ),
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            (
                "CALIBRATION_COMPLETE_REVIEW_REQUIRED"
                if args.mode
                ==
                "calibration"
                else
                "FULL_QWEN_REVIEW_COMPLETE"
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
        "STEP 14E-2 RESULT"
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
        semantic_equivalent_count,
        f"({semantic_equivalent_rate:.2f}%)"
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

    if args.mode == "calibration":

        print("\n")
        print(
            "IMPORTANT:"
        )

        print(
            "Calibration completed."
        )

        print(
            "Inspect label distribution and "
            "sample judgments before running full mode."
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