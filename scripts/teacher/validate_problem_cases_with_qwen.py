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
# 1. 配置
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

MODEL_PATH = Path(
    "/root/autodl-tmp/models/Qwen3-8B"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
    / "qwen_judge"
    / "judge_details.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
    / "qwen_judge"
    / "second_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "problem_cases_reviewed.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "problem_cases_summary.json"
)

PROBLEM_GRADES = {
    "C",
    "D",
    "F",
    "PARSE_ERROR",
}

MAX_NEW_TOKENS = 300


# ============================================================
# 2. 检查环境
# ============================================================

print("=" * 80)
print("Second-stage Problem Case Review")
print("=" * 80)

print("CUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)


# ============================================================
# 3. 读取第一阶段结果
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        INPUT_FILE
    )

df = pd.read_csv(
    INPUT_FILE
)

problem_df = df[
    df["grade"].isin(
        PROBLEM_GRADES
    )
].copy()

problem_df = (
    problem_df
    .reset_index(drop=True)
)

print(
    "\n总 Judge 数量:",
    len(df)
)

print(
    "异常样本数量:",
    len(problem_df)
)

print(
    "\n方向分布:"
)

print(
    problem_df[
        "direction"
    ].value_counts()
)


# ============================================================
# 4. 加载 Qwen3-8B
# ============================================================

print(
    "\nLoading Qwen3 tokenizer..."
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
        / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 5. 第二阶段 Judge Prompt
# ============================================================

def build_prompt(
    direction,
    source,
    reference,
    candidate,
    first_grade,
    first_reason,
):

    if direction == "uz_en":

        src_lang = "Uzbek"
        tgt_lang = "English"

    elif direction == "en_uz":

        src_lang = "English"
        tgt_lang = "Uzbek"

    else:

        src_lang = "Unknown"
        tgt_lang = "Unknown"


    return f"""
You are performing a SECOND-PASS audit of a machine translation benchmark.

The first evaluator flagged this example as potentially problematic.

Translation direction:
{src_lang} -> {tgt_lang}

SOURCE:
{source}

HUMAN REFERENCE:
{reference}

MODEL CANDIDATE:
{candidate}

FIRST-PASS GRADE:
{first_grade}

FIRST-PASS REASON:
{first_reason}


You must independently answer TWO separate questions.

QUESTION 1:
Is the HUMAN REFERENCE itself a semantically correct translation
of the SOURCE?

QUESTION 2:
Is the MODEL CANDIDATE a semantically correct translation
of the SOURCE?

Do NOT assume that the human reference is always correct.

Possible final labels:

MODEL_ERROR
- source/reference are reasonably consistent
- candidate contains a meaningful mistranslation, omission,
  addition, entity error, number error, time error,
  or negation error

DATASET_ERROR
- source/reference pair itself is incorrect or clearly mismatched
- candidate may actually match the source better than the reference

AMBIGUOUS
- source/reference/candidate cannot be confidently resolved
- multiple reasonable interpretations exist

JUDGE_ERROR
- the first-pass evaluator flagged the sample incorrectly
- both the reference and candidate are semantically acceptable

Important:
- Judge meaning, not exact wording.
- Paraphrases are acceptable.
- For Uzbek, spelling and Latin/Cyrillic differences alone
  are not translation errors.
- Pay special attention to negation, numbers, time,
  people, locations, objects and named entities.

Return ONLY valid JSON:

{{
  "source_reference_consistent": true,
  "source_candidate_consistent": true,
  "final_label": "JUDGE_ERROR",
  "model_error_type": "NONE",
  "confidence": 0.95,
  "explanation": "Short explanation in English."
}}

final_label must be one of:

MODEL_ERROR
DATASET_ERROR
AMBIGUOUS
JUDGE_ERROR

model_error_type must be one of:

NONE
MISTRANSLATION
OMISSION
ADDITION
ENTITY_ERROR
NUMBER_ERROR
TIME_ERROR
NEGATION_ERROR
OTHER
""".strip()


# ============================================================
# 6. JSON提取
# ============================================================

def extract_json(text):

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

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
            text[start:end + 1]
        )

    except Exception:

        return None


# ============================================================
# 7. 单条推理
# ============================================================

@torch.inference_mode()
def review_case(row):

    prompt = build_prompt(
        direction=row["direction"],
        source=row["source"],
        reference=row["reference"],
        candidate=row["candidate"],
        first_grade=row["grade"],
        first_reason=row.get(
            "reason",
            "",
        ),
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict independent "
                "machine translation benchmark auditor. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = (
        tokenizer.apply_chat_template(
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
        max_length=2048,
    )

    device = next(
        model.parameters()
    ).device

    inputs = {
        k: v.to(device)
        for k, v
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
        - start
    )

    new_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    result = extract_json(
        response
    )

    return (
        result,
        response,
        latency,
    )


# ============================================================
# 8. 输出标准化
# ============================================================

VALID_LABELS = {
    "MODEL_ERROR",
    "DATASET_ERROR",
    "AMBIGUOUS",
    "JUDGE_ERROR",
}

VALID_ERROR_TYPES = {
    "NONE",
    "MISTRANSLATION",
    "OMISSION",
    "ADDITION",
    "ENTITY_ERROR",
    "NUMBER_ERROR",
    "TIME_ERROR",
    "NEGATION_ERROR",
    "OTHER",
}


def normalize_result(
    result,
):

    if not isinstance(
        result,
        dict,
    ):
        return None

    label = str(
        result.get(
            "final_label",
            "",
        )
    ).upper().strip()

    if label not in VALID_LABELS:
        return None

    error_type = str(
        result.get(
            "model_error_type",
            "NONE",
        )
    ).upper().strip()

    if (
        error_type
        not in VALID_ERROR_TYPES
    ):
        error_type = "OTHER"


    def get_bool(name):

        value = result.get(
            name,
            False,
        )

        if isinstance(
            value,
            bool,
        ):
            return value

        return (
            str(value).lower()
            ==
            "true"
        )


    try:

        confidence = float(
            result.get(
                "confidence",
                0,
            )
        )

    except Exception:

        confidence = 0.0


    confidence = max(
        0.0,
        min(
            confidence,
            1.0,
        ),
    )


    return {
        "source_reference_consistent":
            get_bool(
                "source_reference_consistent"
            ),

        "source_candidate_consistent":
            get_bool(
                "source_candidate_consistent"
            ),

        "final_label":
            label,

        "model_error_type":
            error_type,

        "second_confidence":
            confidence,

        "second_explanation":
            str(
                result.get(
                    "explanation",
                    "",
                )
            ).strip(),
    }


# ============================================================
# 9. 断点恢复
# ============================================================

if OUTPUT_FILE.exists():

    old_df = pd.read_csv(
        OUTPUT_FILE
    )

    completed = set(
        zip(
            old_df["sample_id"],
            old_df["direction"],
        )
    )

    output_rows = (
        old_df
        .to_dict(
            orient="records"
        )
    )

    print(
        "\nResume mode:",
        len(output_rows),
        "already completed"
    )

else:

    completed = set()
    output_rows = []


# ============================================================
# 10. 第二阶段正式审核
# ============================================================

for idx, row in problem_df.iterrows():

    key = (
        row["sample_id"],
        row["direction"],
    )

    if key in completed:
        continue

    print()
    print(
        f"[{idx + 1}/{len(problem_df)}]"
    )

    print(
        "Direction:",
        row["direction"]
    )

    print(
        "First grade:",
        row["grade"]
    )


    normalized = None
    raw_response = ""
    latency = 0


    # 最多重试两次
    for attempt in range(2):

        result, raw, latency = (
            review_case(row)
        )

        raw_response = raw

        normalized = (
            normalize_result(
                result
            )
        )

        if normalized is not None:
            break

        print(
            "Parse failed, retry:",
            attempt + 1
        )


    if normalized is None:

        normalized = {
            "source_reference_consistent":
                False,

            "source_candidate_consistent":
                False,

            "final_label":
                "AMBIGUOUS",

            "model_error_type":
                "OTHER",

            "second_confidence":
                0.0,

            "second_explanation":
                "Second-pass output could not be parsed.",
        }


    output_row = {
        **row.to_dict(),
        **normalized,

        "second_latency":
            latency,

        "second_raw_response":
            raw_response,
    }

    output_rows.append(
        output_row
    )

    completed.add(
        key
    )


    # 每条保存
    pd.DataFrame(
        output_rows
    ).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "Final:",
        normalized[
            "final_label"
        ]
    )

    print(
        "Error type:",
        normalized[
            "model_error_type"
        ]
    )

    print(
        "Confidence:",
        normalized[
            "second_confidence"
        ]
    )


# ============================================================
# 11. 最终统计
# ============================================================

review_df = pd.DataFrame(
    output_rows
)

print("\n")
print("=" * 80)
print("SECOND REVIEW RESULT")
print("=" * 80)


def direction_summary(
    direction,
):

    part = review_df[
        review_df[
            "direction"
        ]
        ==
        direction
    ]

    total = len(part)

    counts = (
        part["final_label"]
        .value_counts()
        .to_dict()
    )

    print()
    print(direction.upper())
    print("-" * 40)

    print(
        "Total:",
        total
    )

    print(
        "Labels:",
        counts
    )

    if total > 0:

        model_errors = counts.get(
            "MODEL_ERROR",
            0,
        )

        dataset_errors = counts.get(
            "DATASET_ERROR",
            0,
        )

        judge_errors = counts.get(
            "JUDGE_ERROR",
            0,
        )

        ambiguous = counts.get(
            "AMBIGUOUS",
            0,
        )

        print(
            "MODEL_ERROR rate:",
            f"{model_errors / total * 100:.2f}%"
        )

        print(
            "DATASET_ERROR rate:",
            f"{dataset_errors / total * 100:.2f}%"
        )

        print(
            "JUDGE_ERROR rate:",
            f"{judge_errors / total * 100:.2f}%"
        )

        print(
            "AMBIGUOUS rate:",
            f"{ambiguous / total * 100:.2f}%"
        )

    return {
        "total":
            total,

        "labels":
            counts,
    }


uz_en_summary = (
    direction_summary(
        "uz_en"
    )
)

en_uz_summary = (
    direction_summary(
        "en_uz"
    )
)


# ============================================================
# 12. MODEL_ERROR 类型
# ============================================================

model_error_df = review_df[
    review_df[
        "final_label"
    ]
    ==
    "MODEL_ERROR"
]

print("\n")
print("=" * 80)
print("MODEL ERROR TYPES")
print("=" * 80)

print(
    model_error_df[
        "model_error_type"
    ]
    .value_counts()
)


# ============================================================
# 13. 保存 Summary
# ============================================================

summary = {
    "total_problem_cases":
        len(review_df),

    "uz_en":
        uz_en_summary,

    "en_uz":
        en_uz_summary,

    "model_error_types":
        model_error_df[
            "model_error_type"
        ]
        .value_counts()
        .to_dict(),
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n")
print("Saved:")
print(OUTPUT_FILE)
print(SUMMARY_FILE)

print("\nDone.")