from __future__ import annotations

from pathlib import Path
import json
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


# ============================================================
# 2. Config
# ============================================================

MAX_INPUT_LENGTH = 1536
MAX_NEW_TOKENS = 220

MAX_RETRIES = 2

SAVE_EVERY = 10


# ============================================================
# 3. Environment
# ============================================================

print("=" * 100)
print("EN-UZ PIPELINE")
print("STEP 05B - QWEN3 SEMANTIC JUDGE")
print("=" * 100)

print(
    "CUDA:",
    torch.cuda.is_available(),
)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )


print(
    "GPU:",
    torch.cuda.get_device_name(0),
)


# ============================================================
# 4. Load review input
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到：\n{INPUT_FILE}"
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


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        f"缺少字段：{missing}"
    )


df = (
    df
    .drop_duplicates(
        subset=[
            "review_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nReview rows:",
    len(df)
)


print(
    "\nReview type:"
)

print(
    df[
        "review_type"
    ]
    .value_counts()
)


# ============================================================
# 5. Load Qwen3-8B
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


print(
    "Model loaded."
)


print(
    "GPU allocated:",
    round(
        torch.cuda.memory_allocated()
        / 1024 ** 3,
        2,
    ),
    "GB",
)


# ============================================================
# 6. Prompt
# ============================================================

def build_prompt(
    english: str,
    uzbek: str,
    risk_flags: str,
):

    risk_flags = (
        str(risk_flags)
        if pd.notna(risk_flags)
        else ""
    )


    return f"""
You are independently auditing an English-Uzbek parallel corpus.

Determine whether the English sentence and the Uzbek sentence
express the same meaning.

ENGLISH:
{english}

UZBEK:
{uzbek}

AUTOMATIC RISK FLAGS:
{risk_flags if risk_flags else "NONE"}

Important:

1. The automatic risk flags may be wrong.
2. Judge the actual semantic relationship yourself.
3. Do not require word-for-word translation.
4. Natural paraphrases are acceptable.
5. Minor stylistic differences are acceptable.
6. Uzbek spelling and apostrophe variants alone are not errors.
7. Pay special attention to:
   - negation
   - numbers
   - dates
   - time
   - locations
   - people
   - entities
   - missing information
   - added information
   - contradictions

Choose exactly one quality label:

PASS
- Both sentences preserve essentially the same meaning.
- Suitable as a high-quality parallel training pair.

MINOR
- Core meaning is preserved.
- There is a small omission, addition, lexical problem,
  or grammatical issue.
- Still potentially usable, but lower quality than PASS.

FAIL
- Important meaning differs.
- There is a significant mistranslation, mismatch,
  contradiction, missing content, wrong number,
  wrong entity, wrong time, or negation error.

UNCERTAIN
- The pair cannot be judged confidently.

Return ONLY valid JSON:

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
  "reason": "Short explanation in English."
}}

No Markdown.
No text outside JSON.
""".strip()


# ============================================================
# 7. JSON extraction
# ============================================================

def extract_json(
    text: str,
):

    text = str(text).strip()


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
        start == -1
        or
        end == -1
        or
        end <= start
    ):

        return None


    try:

        return json.loads(
            text[
                start:end + 1
            ]
        )

    except Exception:

        return None


# ============================================================
# 8. Normalize output
# ============================================================

VALID_LABELS = {
    "PASS",
    "MINOR",
    "FAIL",
    "UNCERTAIN",
}


def as_bool(
    value,
):

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
    result,
):

    if not isinstance(
        result,
        dict,
    ):

        return None


    label = (
        str(
            result.get(
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
            result.get(
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
            as_bool(
                result.get(
                    "semantic_consistent",
                    False,
                )
            ),

        "judge_omission":
            as_bool(
                result.get(
                    "omission",
                    False,
                )
            ),

        "judge_addition":
            as_bool(
                result.get(
                    "addition",
                    False,
                )
            ),

        "judge_mistranslation":
            as_bool(
                result.get(
                    "mistranslation",
                    False,
                )
            ),

        "judge_number_error":
            as_bool(
                result.get(
                    "number_error",
                    False,
                )
            ),

        "judge_time_error":
            as_bool(
                result.get(
                    "time_error",
                    False,
                )
            ),

        "judge_entity_error":
            as_bool(
                result.get(
                    "entity_error",
                    False,
                )
            ),

        "judge_negation_error":
            as_bool(
                result.get(
                    "negation_error",
                    False,
                )
            ),

        "judge_confidence":
            confidence,

        "judge_reason":
            str(
                result.get(
                    "reason",
                    "",
                )
            ).strip(),
    }


# ============================================================
# 9. Single inference
# ============================================================

@torch.inference_mode()
def judge_once(
    english: str,
    uzbek: str,
    risk_flags: str,
):

    prompt = build_prompt(
        english=english,
        uzbek=uzbek,
        risk_flags=risk_flags,
    )


    messages = [

        {
            "role":
                "system",

            "content":
                (
                    "You are a strict but fair "
                    "English-Uzbek parallel corpus auditor. "
                    "Return JSON only."
                ),
        },

        {
            "role":
                "user",

            "content":
                prompt,
        },
    ]


    text = (
        tokenizer
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )


    device = next(
        model.parameters()
    ).device


    inputs = {
        key: value.to(
            device
        )
        for key, value
        in inputs.items()
    }


    torch.cuda.synchronize()


    start = time.perf_counter()


    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


    torch.cuda.synchronize()


    latency = (
        time.perf_counter()
        -
        start
    )


    generated = outputs[0][
        inputs[
            "input_ids"
        ].shape[1]:
    ]


    response = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


    parsed = extract_json(
        response
    )


    return (
        parsed,
        response,
        latency,
    )


# ============================================================
# 10. Retry
# ============================================================

def judge_pair(
    row,
):

    english = str(
        row[
            "source_text_normalized"
        ]
    )


    uzbek = str(
        row[
            "target_text_normalized"
        ]
    )


    risk_flags = (
        row.get(
            "risk_flags",
            "",
        )
    )


    last_response = ""
    last_latency = 0.0


    for attempt in range(
        MAX_RETRIES
    ):

        (
            parsed,
            raw_response,
            latency,
        ) = judge_once(
            english=english,
            uzbek=uzbek,
            risk_flags=risk_flags,
        )


        last_response = (
            raw_response
        )

        last_latency = (
            latency
        )


        normalized = (
            normalize_result(
                parsed
            )
        )


        if normalized is not None:

            normalized[
                "judge_parse_success"
            ] = True

            normalized[
                "judge_latency"
            ] = latency

            normalized[
                "judge_raw_response"
            ] = raw_response

            return normalized


        print(
            f"  Parse failed "
            f"{attempt + 1}/{MAX_RETRIES}"
        )


    # ========================================================
    # Parsing failure
    # ========================================================

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
            "Judge output could not be parsed.",

        "judge_parse_success":
            False,

        "judge_latency":
            last_latency,

        "judge_raw_response":
            last_response,
    }


# ============================================================
# 11. Resume
# ============================================================

if DETAIL_FILE.exists():

    print(
        "\nExisting result found."
    )


    old_df = pd.read_parquet(
        DETAIL_FILE
    )


    completed_ids = set(
        old_df[
            "review_id"
        ].astype(str)
    )


    result_rows = (
        old_df
        .to_dict(
            orient="records"
        )
    )


    print(
        "Resume completed:",
        len(
            completed_ids
        )
    )


else:

    completed_ids = set()

    result_rows = []


# ============================================================
# 12. Run Judge
# ============================================================

print("\n")
print("=" * 100)
print("START SEMANTIC JUDGE")
print("=" * 100)


total = len(df)


for index, row in df.iterrows():

    review_id = str(
        row[
            "review_id"
        ]
    )


    if review_id in completed_ids:

        continue


    print(
        f"\n"
        f"[{index + 1}/{total}] "
        f"{row['review_type']} "
        f"| {row['risk_flags']}"
    )


    judge_result = judge_pair(
        row
    )


    output_row = {

        **row.to_dict(),

        **judge_result,
    }


    result_rows.append(
        output_row
    )


    completed_ids.add(
        review_id
    )


    print(
        "  Label:",
        judge_result[
            "judge_label"
        ],
    )


    print(
        "  Confidence:",
        judge_result[
            "judge_confidence"
        ],
    )


    # ========================================================
    # Incremental save
    # ========================================================

    completed_count = len(
        result_rows
    )


    if (
        completed_count % SAVE_EVERY == 0
        or
        completed_count == total
    ):

        temp_df = pd.DataFrame(
            result_rows
        )


        temp_df.to_parquet(
            DETAIL_FILE,
            index=False,
        )


        temp_df.to_csv(
            DETAIL_CSV,
            index=False,
            encoding="utf-8-sig",
        )


        print(
            f"  Saved: "
            f"{completed_count}/{total}"
        )


# ============================================================
# 13. Final result
# ============================================================

result_df = pd.DataFrame(
    result_rows
)


result_df.to_parquet(
    DETAIL_FILE,
    index=False,
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
# 14. Summary helper
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


    def count(
        label,
    ):

        return int(
            counts.get(
                label,
                0,
            )
        )


    pass_count = (
        count(
            "PASS"
        )
    )


    minor_count = (
        count(
            "MINOR"
        )
    )


    fail_count = (
        count(
            "FAIL"
        )
    )


    uncertain_count = (
        count(
            "UNCERTAIN"
        )
    )


    def rate(
        value,
    ):

        if total == 0:
            return 0.0

        return (
            value
            /
            total
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
            float(
                part[
                    "judge_number_error"
                ]
                .astype(bool)
                .mean()
                *
                100
            )
            if total > 0
            else 0.0,

        "time_error_rate":
            float(
                part[
                    "judge_time_error"
                ]
                .astype(bool)
                .mean()
                *
                100
            )
            if total > 0
            else 0.0,

        "entity_error_rate":
            float(
                part[
                    "judge_entity_error"
                ]
                .astype(bool)
                .mean()
                *
                100
            )
            if total > 0
            else 0.0,

        "negation_error_rate":
            float(
                part[
                    "judge_negation_error"
                ]
                .astype(bool)
                .mean()
                *
                100
            )
            if total > 0
            else 0.0,

        "avg_confidence":
            float(
                part[
                    "judge_confidence"
                ].mean()
            )
            if total > 0
            else 0.0,

        "avg_latency":
            float(
                part[
                    "judge_latency"
                ].mean()
            )
            if total > 0
            else 0.0,
    }


# ============================================================
# 15. Build summaries
# ============================================================

summary_rows = []


# ============================================================
# ALL
# ============================================================

summary_rows.append(
    build_summary(
        result_df,
        "ALL",
    )
)


# ============================================================
# Review type
# ============================================================

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
# 16. Audit confidence interval information
#
# 简单记录 n 和错误数。
#
# 下一步再正式解释统计置信区间。
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
# 17. JSON report
# ============================================================

summary_json = {

    "pipeline":
        "en_uz",

    "step":
        "05_qwen_semantic_judge",

    "judge_model":
        "Qwen3-8B",

    "total_reviewed":
        len(
            result_df
        ),

    "risk_review_count":
        int(
            (
                result_df[
                    "review_type"
                ]
                ==
                "RISK_REVIEW"
            ).sum()
        ),

    "auto_accept_audit_count":
        audit_total,

    "auto_accept_audit_fail":
        audit_fail,

    "auto_accept_audit_fail_rate_percent":
        audit_fail_rate,

    "summary":
        summary_df.to_dict(
            orient="records"
        ),
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary_json,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 18. Risk flag specific summary
# ============================================================

print("\n")
print("=" * 100)
print("RISK FLAG RESULTS")
print("=" * 100)


risk_review_df = result_df[
    result_df[
        "review_type"
    ]
    ==
    "RISK_REVIEW"
]


risk_flags = set()


for value in (
    risk_review_df[
        "risk_flags"
    ]
    .fillna("")
):

    for flag in str(
        value
    ).split("|"):

        flag = flag.strip()

        if flag:

            risk_flags.add(
                flag
            )


for flag in sorted(
    risk_flags
):

    mask = (
        risk_review_df[
            "risk_flags"
        ]
        .fillna("")
        .str.split("|")
        .apply(
            lambda values:
            flag in values
        )
    )


    part = risk_review_df[
        mask
    ]


    if len(part) == 0:

        continue


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


    print()

    print(
        flag
    )

    print(
        "  total:",
        len(part)
    )

    print(
        "  PASS:",
        int(
            (
                part[
                    "judge_label"
                ]
                ==
                "PASS"
            ).sum()
        )
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
        "  fail rate:",
        f"{fail_count / len(part) * 100:.2f}%"
    )


# ============================================================
# 19. Final output
# ============================================================

print("\n")
print("=" * 120)
print("SEMANTIC QUALITY SUMMARY")
print("=" * 120)


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
    "FAIL:",
    audit_fail
)

print(
    "FAIL rate:",
    f"{audit_fail_rate:.2f}%"
)


print("\n")
print("=" * 100)
print("FILES")
print("=" * 100)


print(
    "Details:"
)

print(
    DETAIL_FILE
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


print(
    "\nDone."
)