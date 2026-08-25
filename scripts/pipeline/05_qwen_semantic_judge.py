from __future__ import annotations

from pathlib import Path
import json
import math
import time

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "qwen_review_input.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# 最终正式文件
DETAIL_FILE = (
    OUTPUT_DIR
    / "qwen_semantic_judge_details.parquet"
)

DETAIL_CSV = (
    OUTPUT_DIR
    / "qwen_semantic_judge_details.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "qwen_semantic_judge_summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "qwen_semantic_judge_summary.json"
)

# 新版高速断点文件
CHECKPOINT_JSONL = (
    OUTPUT_DIR
    / "qwen_semantic_judge_checkpoint.jsonl"
)


# ============================================================
# 2. Performance config
# ============================================================

# V100 32GB 推荐先从4开始。
#
# 如果实际显存很空，可以后面试6/8。
# OOM时脚本会自动缩小。
BATCH_SIZE = 4

# 输入本身并不长，1536 足够
MAX_INPUT_LENGTH = 1536

# 我们只需要一个短 JSON
MAX_NEW_TOKENS = 160

# 每多少个 batch 打一次详细进度
PRINT_EVERY_BATCHES = 5

# parse失败时最多额外修复一次
ENABLE_JSON_REPAIR = True


# ============================================================
# 3. Environment
# ============================================================

print("=" * 100)
print("EN-UZ PIPELINE")
print("STEP 05B - QWEN3 SEMANTIC JUDGE [BATCH OPTIMIZED]")
print("=" * 100)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA unavailable."
    )

DEVICE_NAME = torch.cuda.get_device_name(0)

TOTAL_VRAM_GB = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / 1024**3
)

print("CUDA:", True)
print("GPU :", DEVICE_NAME)
print(
    "VRAM:",
    f"{TOTAL_VRAM_GB:.2f} GB",
)

print(
    "PyTorch:",
    torch.__version__,
)


# ============================================================
# 4. Read review data
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{INPUT_FILE}"
    )


df = pd.read_parquet(
    INPUT_FILE
)


required_columns = [
    "review_id",
    "review_type",
    "normalized_pair_id",
    "source_text_normalized",
    "target_text_normalized",
    "risk_flags",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing columns: "
        f"{missing_columns}"
    )


df = (
    df
    .drop_duplicates(
        subset=[
            "review_id"
        ]
    )
    .reset_index(
        drop=True
    )
)


df["review_id"] = (
    df["review_id"]
    .astype(str)
)


print(
    "\nTotal review rows:",
    len(df)
)

print(
    "\nReview distribution:"
)

print(
    df[
        "review_type"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 5. Load tokenizer
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
)


# Batch generation for decoder-only models:
# LEFT padding is important.
tokenizer.padding_side = "left"


if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# 6. Load Qwen3-8B
# ============================================================

print(
    "Loading Qwen3-8B..."
)

model = (
    AutoModelForCausalLM
    .from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
)


model.eval()


model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None
model.generation_config.use_cache = True


MODEL_DEVICE = next(
    model.parameters()
).device


print(
    "Model loaded."
)

print(
    "GPU allocated:",
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
)

print(
    "GPU reserved :",
    f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB",
)


# ============================================================
# 7. Prompt
# ============================================================

def build_prompt(
    english: str,
    uzbek: str,
    risk_flags: str,
) -> str:

    if (
        risk_flags is None
        or
        pd.isna(risk_flags)
        or
        not str(risk_flags).strip()
    ):
        risk_flags = "NONE"

    else:
        risk_flags = str(
            risk_flags
        ).strip()


    return f"""
Evaluate whether this English-Uzbek pair expresses the same meaning.

ENGLISH:
{english}

UZBEK:
{uzbek}

AUTOMATIC RISK FLAGS:
{risk_flags}

The automatic flags may be wrong. Judge the actual semantics.

Labels:

PASS
- Meanings are essentially equivalent.
- Suitable as a high-quality training pair.

MINOR
- Core meaning is preserved.
- Only a small omission, addition, lexical or grammatical issue.

FAIL
- Important meaning differs.
- Significant mistranslation, contradiction, omission,
  wrong number, entity, time, date or negation.

UNCERTAIN
- Cannot confidently judge.

Do not require word-for-word translation.
Natural paraphrases are acceptable.
Uzbek apostrophe/spelling variants alone are not errors.

Return ONLY this JSON structure:

{{
  "label": "PASS",
  "semantic_consistent": true,
  "omission": false,
  "addition": false,
  "mistranslation": false,
  "number_error": false,
  "time_error": false,
  "entity_error": false,
  "negation_error": false,
  "confidence": 0.95,
  "reason": "Brief reason."
}}
""".strip()


# ============================================================
# 8. Build chat prompt
# ============================================================

def build_chat_text(
    row,
) -> str:

    prompt = build_prompt(

        english=str(
            row[
                "source_text_normalized"
            ]
        ),

        uzbek=str(
            row[
                "target_text_normalized"
            ]
        ),

        risk_flags=row.get(
            "risk_flags",
            "",
        ),
    )


    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict but fair "
                "English-Uzbek parallel-corpus auditor. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]


    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ============================================================
# 9. JSON parsing
# ============================================================

def extract_json(
    text: str,
):

    text = str(
        text
    ).strip()


    try:

        return json.loads(
            text
        )

    except Exception:
        pass


    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )


    if (
        start < 0
        or
        end < 0
        or
        end <= start
    ):

        return None


    candidate = text[
        start:end + 1
    ]


    try:

        return json.loads(
            candidate
        )

    except Exception:

        return None


# ============================================================
# 10. Normalize fields
# ============================================================

VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}


def parse_bool(
    value,
) -> bool:

    if isinstance(
        value,
        bool,
    ):

        return value


    return (
        str(value)
        .strip()
        .lower()
        ==
        "true"
    )


def normalize_result(
    parsed,
):

    if not isinstance(
        parsed,
        dict,
    ):

        return None


    label = (
        str(
            parsed.get(
                "label",
                "",
            )
        )
        .strip()
        .upper()
    )


    if label not in VALID_LABELS:

        return None


    try:

        confidence = float(
            parsed.get(
                "confidence",
                0.0,
            )
        )

    except Exception:

        confidence = 0.0


    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


    return {

        "judge_label":
            label,

        "semantic_consistent":
            parse_bool(
                parsed.get(
                    "semantic_consistent",
                    False,
                )
            ),

        "judge_omission":
            parse_bool(
                parsed.get(
                    "omission",
                    False,
                )
            ),

        "judge_addition":
            parse_bool(
                parsed.get(
                    "addition",
                    False,
                )
            ),

        "judge_mistranslation":
            parse_bool(
                parsed.get(
                    "mistranslation",
                    False,
                )
            ),

        "judge_number_error":
            parse_bool(
                parsed.get(
                    "number_error",
                    False,
                )
            ),

        "judge_time_error":
            parse_bool(
                parsed.get(
                    "time_error",
                    False,
                )
            ),

        "judge_entity_error":
            parse_bool(
                parsed.get(
                    "entity_error",
                    False,
                )
            ),

        "judge_negation_error":
            parse_bool(
                parsed.get(
                    "negation_error",
                    False,
                )
            ),

        "judge_confidence":
            confidence,

        "judge_reason":
            str(
                parsed.get(
                    "reason",
                    "",
                )
            ).strip(),
    }


# ============================================================
# 11. Optimized batch generation
# ============================================================

@torch.inference_mode()
def generate_batch(
    batch_df: pd.DataFrame,
):

    prompts = [
        build_chat_text(
            row
        )

        for _, row
        in batch_df.iterrows()
    ]


    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )


    inputs = {
        key: value.to(
            MODEL_DEVICE
        )

        for key, value
        in inputs.items()
    }


    # 所有 batch 输入都被 pad 到相同长度
    prompt_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    torch.cuda.synchronize()

    start_time = (
        time.perf_counter()
    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=
            MAX_NEW_TOKENS,

        do_sample=False,

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
        start_time
    )


    generated_tokens = outputs[
        :,
        prompt_length:
    ]


    responses = (
        tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )
    )


    return (
        responses,
        elapsed,
    )


# ============================================================
# 12. Repair malformed JSON
# ============================================================

@torch.inference_mode()
def repair_json(
    raw_response: str,
):

    repair_prompt = f"""
Convert the following output into ONE valid JSON object.

Do not change its intended judgement.
Do not explain anything.
Return JSON only.

OUTPUT:
{raw_response}

Required keys:

label
semantic_consistent
omission
addition
mistranslation
number_error
time_error
entity_error
negation_error
confidence
reason
""".strip()


    messages = [
        {
            "role":
                "system",

            "content":
                "Return valid JSON only.",
        },

        {
            "role":
                "user",

            "content":
                repair_prompt,
        },
    ]


    prompt = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )


    inputs = {
        key: value.to(
            MODEL_DEVICE
        )

        for key, value
        in inputs.items()
    }


    prompt_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    outputs = model.generate(

        **inputs,

        max_new_tokens=120,

        do_sample=False,

        use_cache=True,

        pad_token_id=
            tokenizer.pad_token_id,

        eos_token_id=
            tokenizer.eos_token_id,
    )


    generated = outputs[0][
        prompt_length:
    ]


    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


    return response


# ============================================================
# 13. Default uncertain result
# ============================================================

def make_uncertain_result(
    raw_response: str,
    latency: float,
):

    return {

        "judge_label":
            "UNCERTAIN",

        "semantic_consistent":
            False,

        "judge_omission":
            False,

        "judge_addition":
            False,

        "judge_mistranslation":
            False,

        "judge_number_error":
            False,

        "judge_time_error":
            False,

        "judge_entity_error":
            False,

        "judge_negation_error":
            False,

        "judge_confidence":
            0.0,

        "judge_reason":
            (
                "Judge output could not "
                "be parsed."
            ),

        "judge_parse_success":
            False,

        "judge_latency":
            latency,

        "judge_raw_response":
            raw_response,
    }


# ============================================================
# 14. Append checkpoint
# ============================================================

def append_jsonl(
    rows: list[dict],
):

    with open(
        CHECKPOINT_JSONL,
        "a",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    default=str,
                )
            )

            file.write(
                "\n"
            )


# ============================================================
# 15. Read JSONL checkpoint
# ============================================================

def read_jsonl_checkpoint():

    rows = []


    if not CHECKPOINT_JSONL.exists():

        return rows


    with open(
        CHECKPOINT_JSONL,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

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
                    "[WARNING] Broken "
                    "checkpoint line skipped."
                )


    return rows


# ============================================================
# 16. Resume compatibility
#
# Supports:
#
# 1. New JSONL checkpoint
# 2. Old sequential Parquet result
# ============================================================

def load_existing_results():

    # --------------------------------------------------------
    # Prefer new checkpoint
    # --------------------------------------------------------

    if CHECKPOINT_JSONL.exists():

        rows = (
            read_jsonl_checkpoint()
        )


        print(
            "\nCheckpoint detected:"
        )

        print(
            "Completed:",
            len(rows)
        )


        return rows


    # --------------------------------------------------------
    # Migrate old Parquet result
    # --------------------------------------------------------

    if DETAIL_FILE.exists():

        print(
            "\nOld Parquet result detected."
        )

        old_df = pd.read_parquet(
            DETAIL_FILE
        )


        rows = old_df.to_dict(
            orient="records"
        )


        # 转成新的追加式 checkpoint
        with open(
            CHECKPOINT_JSONL,
            "w",
            encoding="utf-8",
        ) as file:

            for row in rows:

                file.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        default=str,
                    )
                )

                file.write(
                    "\n"
                )


        print(
            "Migrated existing progress:",
            len(rows)
        )


        return rows


    return []


# ============================================================
# 17. Load progress
# ============================================================

result_rows = (
    load_existing_results()
)


# 防止 JSONL 中偶发重复
result_by_id = {}


for row in result_rows:

    review_id = str(
        row[
            "review_id"
        ]
    )

    result_by_id[
        review_id
    ] = row


result_rows = list(
    result_by_id.values()
)


completed_ids = set(
    result_by_id.keys()
)


print(
    "\nAlready completed:",
    len(
        completed_ids
    )
)


# ============================================================
# 18. Pending
# ============================================================

pending_df = df[
    ~df[
        "review_id"
    ].isin(
        completed_ids
    )
].copy()


pending_df = (
    pending_df
    .reset_index(
        drop=True
    )
)


print(
    "Pending:",
    len(
        pending_df
    )
)


if len(
    pending_df
) > 0:

    print(
        "Initial batch size:",
        BATCH_SIZE
    )


# ============================================================
# 19. Main batch loop
# ============================================================

print("\n")
print("=" * 100)
print("START BATCH SEMANTIC JUDGE")
print("=" * 100)


current_batch_size = (
    BATCH_SIZE
)

processed_this_run = 0

run_start_time = (
    time.perf_counter()
)

batch_index = 0


while processed_this_run < len(
    pending_df
):

    batch_index += 1


    start_index = (
        processed_this_run
    )


    end_index = min(
        start_index
        +
        current_batch_size,

        len(
            pending_df
        ),
    )


    batch_df = (
        pending_df
        .iloc[
            start_index:end_index
        ]
        .copy()
    )


    # ========================================================
    # Generate
    # ========================================================

    try:

        (
            responses,
            batch_latency,
        ) = generate_batch(
            batch_df
        )


    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()


        if current_batch_size <= 1:

            raise


        current_batch_size = max(
            1,
            current_batch_size // 2,
        )


        print(
            "\n[CUDA OOM]"
        )

        print(
            "Reducing batch size to:",
            current_batch_size
        )

        continue


    # ========================================================
    # Parse every response
    # ========================================================

    batch_results = []


    per_sample_latency = (
        batch_latency
        /
        len(
            batch_df
        )
    )


    for (
        (_, row),
        raw_response,
    ) in zip(
        batch_df.iterrows(),
        responses,
    ):

        parsed = extract_json(
            raw_response
        )


        normalized = normalize_result(
            parsed
        )


        # ====================================================
        # Optional JSON repair
        # ====================================================

        if (
            normalized is None
            and
            ENABLE_JSON_REPAIR
        ):

            repaired_response = (
                repair_json(
                    raw_response
                )
            )


            repaired_json = (
                extract_json(
                    repaired_response
                )
            )


            normalized = (
                normalize_result(
                    repaired_json
                )
            )


            if normalized is not None:

                raw_response = (
                    repaired_response
                )


        # ====================================================
        # Still failed
        # ====================================================

        if normalized is None:

            judge_result = (
                make_uncertain_result(
                    raw_response=
                        raw_response,

                    latency=
                        per_sample_latency,
                )
            )


        else:

            judge_result = {

                **normalized,

                "judge_parse_success":
                    True,

                "judge_latency":
                    per_sample_latency,

                "judge_raw_response":
                    raw_response,
            }


        output_row = {

            **row.to_dict(),

            **judge_result,
        }


        batch_results.append(
            output_row
        )


    # ========================================================
    # Append checkpoint
    # ========================================================

    append_jsonl(
        batch_results
    )


    result_rows.extend(
        batch_results
    )


    processed_this_run += len(
        batch_df
    )


    # ========================================================
    # Progress
    # ========================================================

    total_finished = (
        len(
            completed_ids
        )
        +
        processed_this_run
    )


    total_required = len(
        df
    )


    elapsed_total = (
        time.perf_counter()
        -
        run_start_time
    )


    speed = (
        processed_this_run
        /
        elapsed_total

        if elapsed_total > 0

        else 0.0
    )


    remaining = (
        len(
            pending_df
        )
        -
        processed_this_run
    )


    eta_seconds = (
        remaining
        /
        speed

        if speed > 0

        else 0.0
    )


    if (
        batch_index
        %
        PRINT_EVERY_BATCHES
        ==
        0
        or
        processed_this_run
        ==
        len(
            pending_df
        )
    ):

        print()

        print(
            f"[{total_finished}/"
            f"{total_required}]"
        )

        print(
            "Batch size:",
            len(
                batch_df
            )
        )

        print(
            "Batch time:",
            f"{batch_latency:.2f}s"
        )

        print(
            "Effective avg:",
            f"{per_sample_latency:.2f}s/sample"
        )

        print(
            "Throughput:",
            f"{speed:.3f} samples/s"
        )

        print(
            "ETA:",
            f"{eta_seconds / 3600:.2f} hours"
        )

        print(
            "GPU allocated:",
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        print(
            "Last labels:",
            [
                row[
                    "judge_label"
                ]

                for row in batch_results
            ],
        )


# ============================================================
# 20. Reload checkpoint as source of truth
# ============================================================

print("\n")
print(
    "Reloading final checkpoint..."
)


result_rows = (
    read_jsonl_checkpoint()
)


result_by_id = {

    str(
        row[
            "review_id"
        ]
    ):
        row

    for row
    in result_rows
}


result_df = pd.DataFrame(
    list(
        result_by_id.values()
    )
)


# 按原始 review_id 顺序排序
review_order = {

    review_id:
        index

    for index, review_id
    in enumerate(
        df[
            "review_id"
        ].astype(str)
    )
}


result_df[
    "_sort_index"
] = (
    result_df[
        "review_id"
    ]
    .astype(str)
    .map(
        review_order
    )
)


result_df = (
    result_df
    .sort_values(
        "_sort_index"
    )
    .drop(
        columns=[
            "_sort_index"
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 21. Final Parquet + CSV
#
# 这里只在全部结束后写一次
# ============================================================

print(
    "Saving final Parquet..."
)

result_df.to_parquet(
    DETAIL_FILE,
    index=False,
)


print(
    "Saving final CSV..."
)

result_df.to_csv(
    DETAIL_CSV,
    index=False,
    encoding="utf-8-sig",
)


print("\n")
print("=" * 100)
print("JUDGE COMPLETE")
print("=" * 100)

print(
    "Total judged:",
    len(
        result_df
    )
)


# ============================================================
# 22. Summary
# ============================================================

def build_summary(
    part: pd.DataFrame,
    review_type: str,
):

    total = len(
        part
    )


    counts = (
        part[
            "judge_label"
        ]
        .value_counts()
        .to_dict()
    )


    def get_count(
        label,
    ):

        return int(
            counts.get(
                label,
                0,
            )
        )


    pass_count = (
        get_count(
            "PASS"
        )
    )

    minor_count = (
        get_count(
            "MINOR"
        )
    )

    fail_count = (
        get_count(
            "FAIL"
        )
    )

    uncertain_count = (
        get_count(
            "UNCERTAIN"
        )
    )


    def rate(
        count,
    ):

        if total == 0:
            return 0.0

        return (
            count
            /
            total
            *
            100
        )


    def bool_rate(
        column,
    ):

        if total == 0:
            return 0.0

        return float(
            part[
                column
            ]
            .fillna(False)
            .astype(bool)
            .mean()
            *
            100
        )


    return {

        "review_type":
            review_type,

        "total":
            total,

        "PASS":
            pass_count,

        "MINOR":
            minor_count,

        "FAIL":
            fail_count,

        "UNCERTAIN":
            uncertain_count,

        "pass_rate":
            rate(
                pass_count
            ),

        "pass_minor_rate":
            rate(
                pass_count
                +
                minor_count
            ),

        "fail_rate":
            rate(
                fail_count
            ),

        "uncertain_rate":
            rate(
                uncertain_count
            ),

        "number_error_rate":
            bool_rate(
                "judge_number_error"
            ),

        "time_error_rate":
            bool_rate(
                "judge_time_error"
            ),

        "entity_error_rate":
            bool_rate(
                "judge_entity_error"
            ),

        "negation_error_rate":
            bool_rate(
                "judge_negation_error"
            ),

        "avg_confidence":
            float(
                part[
                    "judge_confidence"
                ]
                .mean()
            )
            if total > 0
            else 0.0,

        "avg_latency":
            float(
                part[
                    "judge_latency"
                ]
                .mean()
            )
            if total > 0
            else 0.0,
    }


summary_rows = []


summary_rows.append(
    build_summary(
        result_df,
        "ALL",
    )
)


for review_type in [
    "RISK_REVIEW",
    "AUTO_ACCEPT_AUDIT",
]:

    part = result_df[
        result_df[
            "review_type"
        ]
        ==
        review_type
    ]


    summary_rows.append(
        build_summary(
            part,
            review_type,
        )
    )


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 23. AUTO_ACCEPT audit
# ============================================================

audit_df = result_df[
    result_df[
        "review_type"
    ]
    ==
    "AUTO_ACCEPT_AUDIT"
]


audit_total = len(
    audit_df
)


audit_fail = int(
    (
        audit_df[
            "judge_label"
        ]
        ==
        "FAIL"
    ).sum()
)


audit_minor = int(
    (
        audit_df[
            "judge_label"
        ]
        ==
        "MINOR"
    ).sum()
)


audit_uncertain = int(
    (
        audit_df[
            "judge_label"
        ]
        ==
        "UNCERTAIN"
    ).sum()
)


audit_fail_rate = (

    audit_fail
    /
    audit_total
    *
    100

    if audit_total > 0

    else 0.0
)


# ============================================================
# 24. Summary JSON
# ============================================================

summary_payload = {

    "pipeline":
        "en_uz",

    "step":
        "05_qwen_semantic_judge_batch",

    "judge_model":
        "Qwen3-8B",

    "batch_size_initial":
        BATCH_SIZE,

    "total_reviewed":
        len(
            result_df
        ),

    "audit": {

        "total":
            audit_total,

        "minor":
            audit_minor,

        "fail":
            audit_fail,

        "uncertain":
            audit_uncertain,

        "fail_rate_percent":
            audit_fail_rate,
    },

    "summary":
        summary_df.to_dict(
            orient="records"
        ),
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        summary_payload,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 25. Risk flag results
# ============================================================

print("\n")
print("=" * 100)
print("RISK FLAG RESULTS")
print("=" * 100)


risk_df = result_df[
    result_df[
        "review_type"
    ]
    ==
    "RISK_REVIEW"
]


all_flags = set()


for value in (
    risk_df[
        "risk_flags"
    ]
    .fillna("")
):

    for flag in str(
        value
    ).split("|"):

        flag = (
            flag.strip()
        )

        if flag:

            all_flags.add(
                flag
            )


for flag in sorted(
    all_flags
):

    mask = (
        risk_df[
            "risk_flags"
        ]
        .fillna("")
        .str.split("|")
        .apply(
            lambda values:
                flag in values
        )
    )


    part = risk_df[
        mask
    ]


    total = len(
        part
    )


    fail_count = int(
        (
            part[
                "judge_label"
            ]
            ==
            "FAIL"
        ).sum()
    )


    minor_count = int(
        (
            part[
                "judge_label"
            ]
            ==
            "MINOR"
        ).sum()
    )


    pass_count = int(
        (
            part[
                "judge_label"
            ]
            ==
            "PASS"
        ).sum()
    )


    print()

    print(
        flag
    )

    print(
        "  total:",
        total
    )

    print(
        "  PASS:",
        pass_count
    )

    print(
        "  MINOR:",
        minor_count
    )

    print(
        "  FAIL:",
        fail_count
    )

    print(
        "  FAIL rate:",
        f"{fail_count / total * 100:.2f}%"
    )


# ============================================================
# 26. Final summary
# ============================================================

print("\n")
print("=" * 130)
print("SEMANTIC QUALITY SUMMARY")
print("=" * 130)


display_columns = [

    "review_type",
    "total",

    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",

    "pass_rate",
    "pass_minor_rate",
    "fail_rate",

    "number_error_rate",
    "time_error_rate",
    "entity_error_rate",
    "negation_error_rate",

    "avg_confidence",
    "avg_latency",
]


print(
    summary_df[
        display_columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("AUTO_ACCEPT AUDIT")
print("=" * 100)


print(
    "Audit size:",
    audit_total
)

print(
    "MINOR:",
    audit_minor
)

print(
    "FAIL:",
    audit_fail
)

print(
    "UNCERTAIN:",
    audit_uncertain
)

print(
    "FAIL rate:",
    f"{audit_fail_rate:.2f}%"
)


print("\n")
print("=" * 100)
print("OUTPUT FILES")
print("=" * 100)


print(
    "Checkpoint:"
)

print(
    CHECKPOINT_JSONL
)


print(
    "\nDetails:"
)

print(
    DETAIL_FILE
)


print(
    "\nCSV:"
)

print(
    DETAIL_CSV
)


print(
    "\nSummary:"
)

print(
    SUMMARY_FILE
)


print(
    "\nJSON:"
)

print(
    SUMMARY_JSON
)


print("\nDone.")