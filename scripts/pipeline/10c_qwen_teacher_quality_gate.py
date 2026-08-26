from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# ============================================================
# Constants
# ============================================================

PROMPT_VERSION = "teacher_quality_gate_v1"

VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}

VALID_USEFULNESS = {
    "HIGH",
    "MEDIUM",
    "LOW",
    "REJECT",
}

ERROR_TYPES = [
    "omission",
    "addition",
    "mistranslation",
    "number_error",
    "time_error",
    "entity_error",
    "negation_error",
]

DEFAULT_QWEN_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

DEFAULT_LIMIT = 500
DEFAULT_SEED = 2026


# ============================================================
# Args
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "STEP 10C - Qwen3-8B Teacher Quality Gate"
        )
    )

    parser.add_argument(
        "--qwen_path",
        type=str,
        default=str(DEFAULT_QWEN_PATH),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=(
            "500 = calibration. "
            "0 = full 20K."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--progress_every",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Stable hash
# ============================================================

def stable_hash(
    *parts,
    length: int = 64,
) -> str:

    payload = "|".join(
        str(x)
        for x in parts
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:length]


# ============================================================
# JSONL
# ============================================================

def append_jsonl(
    path: Path,
    rows: list[dict],
):

    if not rows:
        return

    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
            )

            f.write("\n")

        f.flush()
        os.fsync(
            f.fileno()
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

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                rows.append(
                    json.loads(
                        line
                    )
                )

            except Exception:

                print(
                    f"[WARNING] Broken JSONL line: "
                    f"{line_no}"
                )

    return rows


class CheckpointBuffer:

    def __init__(
        self,
        path: Path,
        flush_every: int,
    ):

        self.path = path

        self.flush_every = max(
            1,
            flush_every,
        )

        self.rows = []

    def add(
        self,
        rows: list[dict],
    ):

        self.rows.extend(
            rows
        )

        if len(
            self.rows
        ) >= self.flush_every:

            self.flush()

    def flush(self):

        if not self.rows:
            return

        append_jsonl(
            self.path,
            self.rows,
        )

        self.rows.clear()


# ============================================================
# Deterministic stratified sampling
# ============================================================

def stratified_sample_exact(
    df: pd.DataFrame,
    target_n: int,
    seed: int,
) -> pd.DataFrame:

    if len(df) < target_n:

        raise RuntimeError(
            f"Need {target_n}, "
            f"available {len(df)}."
        )

    work = df.copy()

    strata = [
        "quality_tier",
        "data_source",
        "length_bucket",
    ]

    for col in strata:

        if col not in work.columns:
            work[col] = "UNKNOWN"

        work[col] = (
            work[col]
            .fillna("UNKNOWN")
            .astype(str)
        )

    groups = (
        work
        .groupby(
            strata,
            dropna=False,
        )
        .size()
        .reset_index(
            name="available"
        )
    )

    total = int(
        groups[
            "available"
        ].sum()
    )

    groups[
        "raw_quota"
    ] = (
        groups[
            "available"
        ]
        /
        total
        *
        target_n
    )

    groups[
        "take"
    ] = (
        groups[
            "raw_quota"
        ]
        .apply(
            math.floor
        )
        .astype(int)
    )

    groups[
        "fraction"
    ] = (
        groups[
            "raw_quota"
        ]
        -
        groups[
            "take"
        ]
    )

    remaining = (
        target_n
        -
        int(
            groups[
                "take"
            ].sum()
        )
    )

    while remaining > 0:

        eligible = (
            groups[
                groups[
                    "take"
                ]
                <
                groups[
                    "available"
                ]
            ]
            .sort_values(
                [
                    "fraction",
                    "available",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
        )

        if eligible.empty:

            raise RuntimeError(
                "Cannot allocate sample."
            )

        for idx in eligible.index:

            if remaining <= 0:
                break

            groups.at[
                idx,
                "take",
            ] += 1

            remaining -= 1

    parts = []

    for group in groups.itertuples(
        index=False
    ):

        take = int(
            group.take
        )

        if take <= 0:
            continue

        mask = (
            work[
                "quality_tier"
            ].eq(
                group.quality_tier
            )
            &
            work[
                "data_source"
            ].eq(
                group.data_source
            )
            &
            work[
                "length_bucket"
            ].eq(
                group.length_bucket
            )
        )

        part = (
            work[
                mask
            ]
            .copy()
        )

        part[
            "_sample_key"
        ] = (
            part[
                "candidate_id"
            ]
            .map(
                lambda x:
                    stable_hash(
                        seed,
                        x,
                    )
            )
        )

        part = (
            part
            .sort_values(
                [
                    "_sample_key",
                    "candidate_id",
                ]
            )
            .head(
                take
            )
            .drop(
                columns=[
                    "_sample_key"
                ]
            )
        )

        parts.append(
            part
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    if len(result) != target_n:

        raise RuntimeError(
            f"Expected {target_n}, "
            f"got {len(result)}."
        )

    return result


# ============================================================
# Select calibration / full dataset
# ============================================================

def select_judge_dataset(
    df: pd.DataFrame,
    limit: int,
    seed: int,
) -> pd.DataFrame:

    # Full 20K
    if limit == 0:

        return (
            df.copy()
            .sort_values(
                [
                    "direction",
                    "source_word_count",
                    "candidate_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    if limit <= 0:

        raise ValueError(
            "--limit must be 0 or > 0"
        )

    if limit % 2 != 0:

        raise ValueError(
            "--limit must be even."
        )

    per_direction = (
        limit // 2
    )

    en_uz = df[
        df[
            "direction"
        ]
        ==
        "en_uz"
    ].copy()

    uz_en = df[
        df[
            "direction"
        ]
        ==
        "uz_en"
    ].copy()

    selected_en_uz = (
        stratified_sample_exact(
            en_uz,
            per_direction,
            seed,
        )
    )

    selected_uz_en = (
        stratified_sample_exact(
            uz_en,
            per_direction,
            seed + 1,
        )
    )

    result = pd.concat(
        [
            selected_en_uz,
            selected_uz_en,
        ],
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                "direction",
                "source_word_count",
                "candidate_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# Judge prompt
# ============================================================

SYSTEM_PROMPT = """
You are a strict bilingual translation quality judge.

Your job is NOT to judge whether two translations are literally identical.
Different wording is allowed if the meaning is faithfully preserved.

You are evaluating a MADLAD teacher translation that may later be used
to train a smaller translation model.

Judge the TEACHER TRANSLATION against:
1. the SOURCE sentence;
2. the HUMAN/REAL REFERENCE.

Important rules:
- Semantic faithfulness matters more than wording similarity.
- A teacher translation can be PASS even when it differs from the reference.
- Do not punish valid paraphrases or synonyms.
- Numbers, dates, times, entities and negation must be preserved accurately.
- Do not treat the human reference as the only possible correct translation.
- For English-to-Uzbek, the teacher output is normalized to Latin Uzbek.
- Be conservative with FAIL: use FAIL only for substantive translation errors.
- MINOR means essentially correct with a small linguistic/style/grammar issue.
- UNCERTAIN means you genuinely cannot judge reliably.

Labels:
PASS:
  Correct, faithful and usable as a training target.

MINOR:
  Core meaning is correct but there is a small non-critical issue.

FAIL:
  There is a substantive semantic or factual translation error.

UNCERTAIN:
  Reliable judgment is not possible.

Teacher usefulness:
HIGH:
  Correct and provides a useful alternative expression compared with reference.

MEDIUM:
  Correct and somewhat useful, but only modest additional variation.

LOW:
  Correct but almost redundant, awkward, or adds little training value.

REJECT:
  Should not be used as a teacher target.

Error fields must be booleans:
omission
addition
mistranslation
number_error
time_error
entity_error
negation_error

Return ONLY one JSON object.
Do not return Markdown.
Do not explain outside the JSON.

Required schema:

{
  "label": "PASS|MINOR|FAIL|UNCERTAIN",
  "confidence": 0.0,
  "semantic_equivalent": true,
  "teacher_usefulness": "HIGH|MEDIUM|LOW|REJECT",
  "errors": {
    "omission": false,
    "addition": false,
    "mistranslation": false,
    "number_error": false,
    "time_error": false,
    "entity_error": false,
    "negation_error": false
  },
  "reason": "brief reason"
}
""".strip()


def build_user_prompt(
    row,
) -> str:

    direction = str(
        row[
            "direction"
        ]
    )

    if direction == "en_uz":

        direction_text = (
            "English -> Uzbek (Latin)"
        )

    else:

        direction_text = (
            "Uzbek (Latin) -> English"
        )

    return f"""
DIRECTION:
{direction_text}

SOURCE:
{row["source_text"]}

HUMAN REFERENCE:
{row["real_reference"]}

TEACHER TRANSLATION:
{row["teacher_prediction"]}

Evaluate whether the TEACHER TRANSLATION is a good additional
training target for this SOURCE.
""".strip()


# ============================================================
# Chat template
# ============================================================

def render_prompt(
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

    # Qwen3 supports enable_thinking=False
    try:

        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    except TypeError:

        # Compatibility fallback
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ============================================================
# JSON extraction
# ============================================================

def extract_json_object(
    text: str,
) -> dict | None:

    if text is None:
        return None

    text = str(text).strip()

    # remove common markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    # direct parse
    try:

        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict,
        ):

            return obj

    except Exception:
        pass

    # balanced JSON object extraction
    start = text.find(
        "{"
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(text),
    ):

        char = text[index]

        if in_string:

            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = (
                    text[
                        start:
                        index + 1
                    ]
                )

                try:

                    obj = json.loads(
                        candidate
                    )

                    if isinstance(
                        obj,
                        dict,
                    ):

                        return obj

                except Exception:
                    return None

    return None


# ============================================================
# Normalize judge result
# ============================================================

def normalize_bool(
    value: Any,
) -> bool:

    if isinstance(
        value,
        bool,
    ):

        return value

    if isinstance(
        value,
        (int, float),
    ):

        return bool(
            value
        )

    value = str(
        value
    ).strip().lower()

    return value in {
        "true",
        "1",
        "yes",
        "y",
    }


def normalize_judge_object(
    obj: dict,
) -> dict | None:

    if not isinstance(
        obj,
        dict,
    ):

        return None

    label = str(
        obj.get(
            "label",
            "",
        )
    ).strip().upper()

    if label not in VALID_LABELS:
        return None

    usefulness = str(
        obj.get(
            "teacher_usefulness",
            "",
        )
    ).strip().upper()

    if usefulness not in VALID_USEFULNESS:
        return None

    try:

        confidence = float(
            obj.get(
                "confidence",
                0.0,
            )
        )

    except Exception:

        return None

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    semantic_equivalent = (
        normalize_bool(
            obj.get(
                "semantic_equivalent",
                False,
            )
        )
    )

    raw_errors = obj.get(
        "errors",
        {}
    )

    if not isinstance(
        raw_errors,
        dict,
    ):

        raw_errors = {}

    errors = {

        error_type:
            normalize_bool(
                raw_errors.get(
                    error_type,
                    False,
                )
            )

        for error_type
        in ERROR_TYPES
    }

    reason = str(
        obj.get(
            "reason",
            "",
        )
    ).strip()

    # Safety consistency
    if label == "FAIL":

        if usefulness != "REJECT":

            usefulness = "REJECT"

    if label == "UNCERTAIN":

        usefulness = "REJECT"

    if label == "PASS":

        semantic_equivalent = True

    return {

        "label":
            label,

        "confidence":
            confidence,

        "semantic_equivalent":
            semantic_equivalent,

        "teacher_usefulness":
            usefulness,

        "errors":
            errors,

        "reason":
            reason,
    }


# ============================================================
# Generation
# ============================================================

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    device,
    max_input_length: int,
    max_new_tokens: int,
) -> tuple[list[str], float]:

    tokenizer.padding_side = (
        "left"
    )

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    )

    encoded = {

        key:
            value.to(
                device
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

    torch.cuda.synchronize()

    start = (
        time.perf_counter()
    )

    with torch.inference_mode():

        generated = model.generate(
            **encoded,

            max_new_tokens=
                max_new_tokens,

            do_sample=False,

            num_beams=1,

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

    generated_only = (
        generated[
            :,
            input_length:
        ]
    )

    texts = (
        tokenizer
        .batch_decode(
            generated_only,
            skip_special_tokens=True,
        )
    )

    return (
        texts,
        elapsed,
    )


# ============================================================
# Retry one failed case
# ============================================================

def retry_single(
    model,
    tokenizer,
    row,
    device,
    max_input_length,
    max_new_tokens,
) -> tuple[
    dict | None,
    str,
    float,
]:

    user_prompt = (
        build_user_prompt(
            row
        )
    )

    retry_prompt = (
        user_prompt
        +
        "\n\nIMPORTANT: Return ONLY valid JSON "
        "matching the required schema. "
        "No prose and no markdown."
    )

    rendered = (
        render_prompt(
            tokenizer,
            retry_prompt,
        )
    )

    outputs, elapsed = (
        generate_batch(
            model,
            tokenizer,
            [
                rendered
            ],
            device,
            max_input_length,
            max_new_tokens,
        )
    )

    raw = outputs[0]

    obj = extract_json_object(
        raw
    )

    normalized = (
        normalize_judge_object(
            obj
        )
        if obj is not None
        else None
    )

    return (
        normalized,
        raw,
        elapsed,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_file = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10b_teacher_generation"
        / "teacher_predictions_20k_v1.parquet"
    )

    qwen_path = Path(
        args.qwen_path
    )

    # ========================================================
    # Output scope
    # ========================================================

    if args.limit == 0:

        scope_name = "full_20k"

    else:

        scope_name = (
            f"calibration_{args.limit}"
        )

    output_dir = (
        project_root
        / "data"
        / "distillation"
        / "en_uz"
        / "v1"
        / "10c_qwen_quality_gate"
        / scope_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_file = (
        output_dir
        / "qwen_judge_checkpoint.jsonl"
    )

    selected_file = (
        output_dir
        / "selected_candidates.parquet"
    )

    output_parquet = (
        output_dir
        / "qwen_judge_results.parquet"
    )

    output_csv = (
        output_dir
        / "qwen_judge_results.csv"
    )

    report_file = (
        output_dir
        / "judge_report.json"
    )

    config_file = (
        output_dir
        / "judge_config.json"
    )

    # ========================================================
    # Header
    # ========================================================

    print("=" * 110)
    print("EN-UZ STUDENT PIPELINE")
    print("STEP 10C - QWEN3-8B TEACHER QUALITY GATE")
    print("=" * 110)

    print(
        "\nQwen model:"
    )

    print(
        qwen_path
    )

    print(
        "\nInput:"
    )

    print(
        input_file
    )

    print(
        "\nMode:",
        (
            "FULL 20K"
            if args.limit == 0
            else f"CALIBRATION {args.limit}"
        )
    )

    # ========================================================
    # Check files
    # ========================================================

    if not input_file.exists():

        raise FileNotFoundError(
            input_file
        )

    if not qwen_path.exists():

        raise FileNotFoundError(
            qwen_path
        )

    required_model_files = [
        "config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]

    for filename in required_model_files:

        if not (
            qwen_path
            /
            filename
        ).exists():

            raise FileNotFoundError(
                qwen_path
                /
                filename
            )

    # ========================================================
    # CUDA
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA unavailable."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "\nGPU:",
        torch.cuda.get_device_name(
            0
        )
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    # ========================================================
    # Load Teacher data
    # ========================================================

    df = pd.read_parquet(
        input_file
    )

    print(
        "\nTeacher rows:",
        len(df)
    )

    required_columns = [
        "candidate_id",
        "direction",
        "source_text",
        "real_reference",
        "teacher_prediction",
        "generation_status",
        "quality_tier",
        "data_source",
        "length_bucket",
        "source_word_count",
    ]

    missing = [

        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if len(df) != 20_000:

        raise RuntimeError(
            f"Expected 20000 rows, "
            f"got {len(df)}."
        )

    if (
        df[
            "candidate_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate candidate_id."
        )

    if (
        df[
            "generation_status"
        ]
        .astype(str)
        .ne("OK")
        .any()
    ):

        raise RuntimeError(
            "Non-OK Teacher output exists."
        )

    # ========================================================
    # Select 500 / Full
    # ========================================================

    selected = (
        select_judge_dataset(
            df,
            args.limit,
            args.seed,
        )
    )

    print(
        "\nSelected:",
        len(
            selected
        )
    )

    print(
        "\nDirection:"
    )

    print(
        selected[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    if args.limit > 0:

        expected_each = (
            args.limit // 2
        )

        counts = (
            selected[
                "direction"
            ]
            .value_counts()
            .to_dict()
        )

        if (
            counts.get(
                "en_uz",
                0,
            )
            !=
            expected_each
            or
            counts.get(
                "uz_en",
                0,
            )
            !=
            expected_each
        ):

            raise RuntimeError(
                "Calibration direction imbalance."
            )

    selected.to_parquet(
        selected_file,
        index=False,
    )

    # ========================================================
    # Judge identity
    # ========================================================

    candidate_id_digest = (
        stable_hash(
            *selected[
                "candidate_id"
            ].astype(str).tolist()
        )
    )

    judge_identity = {

        "prompt_version":
            PROMPT_VERSION,

        "qwen_path":
            str(
                qwen_path.resolve()
            ),

        "input_file":
            str(
                input_file.resolve()
            ),

        "selected_rows":
            len(
                selected
            ),

        "candidate_digest":
            candidate_id_digest,

        "max_input_length":
            args.max_input_length,

        "max_new_tokens":
            args.max_new_tokens,

        "thinking":
            False,

        "do_sample":
            False,
    }

    # ========================================================
    # Overwrite
    # ========================================================

    if args.overwrite:

        print(
            "\n[OVERWRITE] "
            "Removing previous judge outputs..."
        )

        for path in [
            checkpoint_file,
            output_parquet,
            output_csv,
            report_file,
            config_file,
        ]:

            if path.exists():
                path.unlink()

    # ========================================================
    # Config compatibility
    # ========================================================

    if (
        config_file.exists()
        and
        not args.overwrite
    ):

        with open(
            config_file,
            "r",
            encoding="utf-8",
        ) as f:

            previous = json.load(
                f
            )

        if (
            previous.get(
                "judge_identity"
            )
            !=
            judge_identity
        ):

            raise RuntimeError(
                "Existing checkpoint belongs to a "
                "different judge configuration.\n"
                "Use --overwrite only if intentional."
            )

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "judge_identity":
                    judge_identity,

                "runtime": {
                    "batch_size":
                        args.batch_size,

                    "checkpoint_every":
                        args.checkpoint_every,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Resume
    # ========================================================

    existing_rows = (
        load_jsonl(
            checkpoint_file
        )
    )

    result_map = {}

    for row in existing_rows:

        cid = str(
            row.get(
                "candidate_id",
                "",
            )
        )

        if cid:

            result_map[
                cid
            ] = row

    completed_ids = set(
        result_map.keys()
    )

    selected_ids = set(
        selected[
            "candidate_id"
        ]
        .astype(str)
    )

    unknown = (
        completed_ids
        -
        selected_ids
    )

    if unknown:

        raise RuntimeError(
            "Checkpoint contains unknown IDs."
        )

    print(
        "\nAlready judged:",
        len(
            completed_ids
        )
    )

    print(
        "Pending:",
        len(selected)
        -
        len(
            completed_ids
        )
    )

    pending = (
        selected[
            ~selected[
                "candidate_id"
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

    # ========================================================
    # Load Qwen
    # ========================================================

    print(
        "\nLoading Qwen tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            str(
                qwen_path
            ),
            local_files_only=True,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    print(
        "Tokenizer:",
        type(
            tokenizer
        ).__name__
    )

    print(
        "\nLoading Qwen3-8B FP16..."
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            str(
                qwen_path
            ),
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=True,
        )
        .to(
            device
        )
    )

    model.eval()

    model.config.use_cache = True

    # ============================================================
    # Deterministic judge generation
    # Qwen's generation_config.json contains sampling defaults:
    # temperature=0.6, top_p=0.95, top_k=20.
    #
    # Step10C uses greedy deterministic decoding, so disable all
    # sampling-only parameters explicitly.
    # ============================================================

    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    print(
        "Model:",
        type(
            model
        ).__name__
    )

    print(
        "GPU allocated:",
        f"{torch.cuda.memory_allocated()/1024**3:.2f} GB"
    )

    # ========================================================
    # Judge
    # ========================================================

    checkpoint_buffer = (
        CheckpointBuffer(
            checkpoint_file,
            args.checkpoint_every,
        )
    )

    run_start = (
        time.perf_counter()
    )

    processed_this_run = 0
    parse_retry_count = 0
    parse_fail_count = 0

    print("\n")
    print("=" * 110)
    print("QWEN QUALITY JUDGING")
    print("=" * 110)

    try:

        for start in range(
            0,
            len(
                pending
            ),
            args.batch_size,
        ):

            end = min(
                start
                +
                args.batch_size,
                len(
                    pending
                ),
            )

            batch = (
                pending
                .iloc[
                    start:end
                ]
            )

            prompts = []

            for _, row in batch.iterrows():

                user_prompt = (
                    build_user_prompt(
                        row
                    )
                )

                prompts.append(
                    render_prompt(
                        tokenizer,
                        user_prompt,
                    )
                )

            outputs, elapsed = (
                generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    device=device,
                    max_input_length=
                        args.max_input_length,
                    max_new_tokens=
                        args.max_new_tokens,
                )
            )

            latency_per_sample = (
                elapsed
                /
                len(batch)
            )

            batch_results = []

            for (
                (_, row),
                raw_output,
            ) in zip(
                batch.iterrows(),
                outputs,
            ):

                obj = extract_json_object(
                    raw_output
                )

                parsed = (
                    normalize_judge_object(
                        obj
                    )
                    if obj is not None
                    else None
                )

                retry_used = False
                retry_raw = ""
                retry_seconds = 0.0

                # --------------------------------------------
                # Retry malformed JSON once
                # --------------------------------------------

                if parsed is None:

                    retry_used = True
                    parse_retry_count += 1

                    (
                        parsed,
                        retry_raw,
                        retry_seconds,
                    ) = retry_single(
                        model=model,
                        tokenizer=tokenizer,
                        row=row,
                        device=device,
                        max_input_length=
                            args.max_input_length,
                        max_new_tokens=
                            args.max_new_tokens,
                    )

                # --------------------------------------------
                # Still failed
                # --------------------------------------------

                if parsed is None:

                    parse_fail_count += 1

                    parsed = {

                        "label":
                            "UNCERTAIN",

                        "confidence":
                            0.0,

                        "semantic_equivalent":
                            False,

                        "teacher_usefulness":
                            "REJECT",

                        "errors": {
                            error:
                                False
                            for error
                            in ERROR_TYPES
                        },

                        "reason":
                            "JUDGE_PARSE_ERROR",
                    }

                    parse_status = (
                        "PARSE_ERROR"
                    )

                else:

                    parse_status = "OK"

                errors = parsed[
                    "errors"
                ]

                result = {

                    "candidate_id":
                        str(
                            row[
                                "candidate_id"
                            ]
                        ),

                    "direction":
                        str(
                            row[
                                "direction"
                            ]
                        ),

                    "source_text":
                        str(
                            row[
                                "source_text"
                            ]
                        ),

                    "real_reference":
                        str(
                            row[
                                "real_reference"
                            ]
                        ),

                    "teacher_prediction":
                        str(
                            row[
                                "teacher_prediction"
                            ]
                        ),

                    "quality_tier":
                        str(
                            row[
                                "quality_tier"
                            ]
                        ),

                    "data_source":
                        str(
                            row[
                                "data_source"
                            ]
                        ),

                    "length_bucket":
                        str(
                            row[
                                "length_bucket"
                            ]
                        ),

                    "teacher_label":
                        parsed[
                            "label"
                        ],

                    "teacher_confidence":
                        float(
                            parsed[
                                "confidence"
                            ]
                        ),

                    "semantic_equivalent":
                        bool(
                            parsed[
                                "semantic_equivalent"
                            ]
                        ),

                    "teacher_usefulness":
                        parsed[
                            "teacher_usefulness"
                        ],

                    "omission":
                        bool(
                            errors[
                                "omission"
                            ]
                        ),

                    "addition":
                        bool(
                            errors[
                                "addition"
                            ]
                        ),

                    "mistranslation":
                        bool(
                            errors[
                                "mistranslation"
                            ]
                        ),

                    "number_error":
                        bool(
                            errors[
                                "number_error"
                            ]
                        ),

                    "time_error":
                        bool(
                            errors[
                                "time_error"
                            ]
                        ),

                    "entity_error":
                        bool(
                            errors[
                                "entity_error"
                            ]
                        ),

                    "negation_error":
                        bool(
                            errors[
                                "negation_error"
                            ]
                        ),

                    "judge_reason":
                        parsed[
                            "reason"
                        ],

                    "judge_parse_status":
                        parse_status,

                    "judge_retry_used":
                        retry_used,

                    "judge_raw_output":
                        str(
                            raw_output
                        ),

                    "judge_retry_raw_output":
                        str(
                            retry_raw
                        ),

                    "judge_seconds":
                        float(
                            latency_per_sample
                            +
                            retry_seconds
                        ),

                    "prompt_version":
                        PROMPT_VERSION,
                }

                batch_results.append(
                    result
                )

                result_map[
                    result[
                        "candidate_id"
                    ]
                ] = result

            checkpoint_buffer.add(
                batch_results
            )

            processed_this_run += (
                len(
                    batch_results
                )
            )

            completed = len(
                result_map
            )

            if (
                processed_this_run
                %
                args.progress_every
                <
                args.batch_size
                or
                completed
                ==
                len(
                    selected
                )
            ):

                run_elapsed = (
                    time.perf_counter()
                    -
                    run_start
                )

                speed = (
                    processed_this_run
                    /
                    run_elapsed
                    if run_elapsed > 0
                    else 0.0
                )

                remaining = (
                    len(selected)
                    -
                    completed
                )

                eta = (
                    remaining
                    /
                    speed
                    if speed > 0
                    else 0.0
                )

                print(
                    f"Judged "
                    f"{completed}/{len(selected)} "
                    f"| batch={len(batch)} "
                    f"| {speed:.2f}/s "
                    f"| ETA={eta/60:.1f} min "
                    f"| retry={parse_retry_count} "
                    f"| parse_fail={parse_fail_count} "
                    f"| GPU={torch.cuda.memory_allocated()/1024**3:.2f}GB"
                )

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTED] "
            "Saving checkpoint..."
        )

        checkpoint_buffer.flush()

        print(
            "Checkpoint saved."
        )

        raise

    except Exception:

        checkpoint_buffer.flush()
        raise

    finally:

        checkpoint_buffer.flush()

    # ========================================================
    # Release model
    # ========================================================

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    # ========================================================
    # Reload checkpoint
    # ========================================================

    rows = load_jsonl(
        checkpoint_file
    )

    final_map = {}

    for row in rows:

        cid = str(
            row.get(
                "candidate_id",
                "",
            )
        )

        if cid:

            final_map[
                cid
            ] = row

    result_df = pd.DataFrame(
        list(
            final_map.values()
        )
    )

    if len(
        result_df
    ) != len(
        selected
    ):

        raise RuntimeError(
            "\nJudge incomplete.\n"
            f"Expected: {len(selected)}\n"
            f"Found   : {len(result_df)}\n"
            "Run again to resume."
        )

    # ========================================================
    # Restore selected ordering
    # ========================================================

    order_map = {

        str(cid):
            i

        for i, cid
        in enumerate(
            selected[
                "candidate_id"
            ]
            .astype(str)
        )
    }

    result_df[
        "_order"
    ] = (
        result_df[
            "candidate_id"
        ]
        .astype(str)
        .map(
            order_map
        )
    )

    result_df = (
        result_df
        .sort_values(
            "_order"
        )
        .drop(
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Metrics
    # ========================================================

    total = len(
        result_df
    )

    parse_errors = int(
        result_df[
            "judge_parse_status"
        ]
        .ne(
            "OK"
        )
        .sum()
    )

    parse_success_rate = (
        (
            total
            -
            parse_errors
        )
        /
        total
        *
        100
    )

    label_counts = (
        result_df[
            "teacher_label"
        ]
        .value_counts()
    )

    usefulness_counts = (
        result_df[
            "teacher_usefulness"
        ]
        .value_counts()
    )

    avg_confidence = float(
        result_df[
            "teacher_confidence"
        ]
        .mean()
    )

    avg_latency = float(
        result_df[
            "judge_seconds"
        ]
        .mean()
    )

    semantic_rate = float(
        result_df[
            "semantic_equivalent"
        ]
        .mean()
        *
        100
    )

    # ========================================================
    # Calibration decision
    # ========================================================

    warnings = []

    if parse_success_rate < 99.0:

        warnings.append(
            "JSON_PARSE_RATE_BELOW_99"
        )

    pass_rate = (
        float(
            (
                result_df[
                    "teacher_label"
                ]
                ==
                "PASS"
            ).mean()
            *
            100
        )
    )

    fail_rate = (
        float(
            (
                result_df[
                    "teacher_label"
                ]
                ==
                "FAIL"
            ).mean()
            *
            100
        )
    )

    if pass_rate > 97:

        warnings.append(
            "PASS_RATE_EXTREMELY_HIGH"
        )

    if fail_rate > 80:

        warnings.append(
            "FAIL_RATE_EXTREMELY_HIGH"
        )

    if parse_success_rate >= 99.0:

        calibration_status = (
            "CALIBRATION_PASS"
        )

    else:

        calibration_status = (
            "CALIBRATION_REVIEW"
        )

    # ========================================================
    # Save
    # ========================================================

    result_df.to_parquet(
        output_parquet,
        index=False,
    )

    result_df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    error_counts = {

        error:
            int(
                result_df[
                    error
                ].sum()
            )

        for error in ERROR_TYPES
    }

    report = {

        "step":
            "10C",

        "scope":
            scope_name,

        "prompt_version":
            PROMPT_VERSION,

        "qwen_path":
            str(
                qwen_path
            ),

        "samples":
            total,

        "direction":
            {
                str(k): int(v)
                for k, v
                in result_df[
                    "direction"
                ]
                .value_counts()
                .items()
            },

        "labels":
            {
                str(k): int(v)
                for k, v
                in label_counts.items()
            },

        "usefulness":
            {
                str(k): int(v)
                for k, v
                in usefulness_counts.items()
            },

        "errors":
            error_counts,

        "parse_success_rate":
            parse_success_rate,

        "avg_confidence":
            avg_confidence,

        "semantic_equivalent_rate":
            semantic_rate,

        "avg_judge_seconds":
            avg_latency,

        "pass_rate":
            pass_rate,

        "fail_rate":
            fail_rate,

        "warnings":
            warnings,

        "status":
            calibration_status,
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
    print("=" * 110)
    print("STEP 10C RESULT")
    print("=" * 110)

    print(
        "\nSamples:",
        total
    )

    print(
        "\nLabel distribution:"
    )

    print(
        result_df[
            "teacher_label"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nLabel distribution by direction:"
    )

    print(
        pd.crosstab(
            result_df[
                "direction"
            ],
            result_df[
                "teacher_label"
            ],
        ).to_string()
    )

    print(
        "\nTeacher usefulness:"
    )

    print(
        result_df[
            "teacher_usefulness"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nError types:"
    )

    for error in ERROR_TYPES:

        count = int(
            result_df[
                error
            ].sum()
        )

        print(
            f"{error:18s}: "
            f"{count:4d} "
            f"({count/total*100:.2f}%)"
        )

    print(
        "\nParse success:",
        f"{parse_success_rate:.2f}%"
    )

    print(
        "Average confidence:",
        f"{avg_confidence:.3f}"
    )

    print(
        "Semantic equivalent:",
        f"{semantic_rate:.2f}%"
    )

    print(
        "Average latency:",
        f"{avg_latency:.3f}s/sample"
    )

    if warnings:

        print(
            "\nWarnings:"
        )

        for warning in warnings:

            print(
                "-",
                warning
            )

    print("\n")
    print("=" * 110)

    print(
        "STATUS:",
        calibration_status
    )

    print("=" * 110)

    print(
        "\nResults:"
    )

    print(
        output_parquet
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )


if __name__ == "__main__":

    main()