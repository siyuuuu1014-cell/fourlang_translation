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
    / "challenge_v1"
    / "challenge_v1_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
    / "challenge_v1"
    / "qwen_judge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "challenge_judge_details.csv"
)

CATEGORY_SUMMARY_FILE = (
    OUTPUT_DIR
    / "challenge_category_summary.csv"
)

SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    / "challenge_judge_summary.json"
)

MAX_NEW_TOKENS = 256


# ============================================================
# 2. 环境
# ============================================================

print("=" * 90)
print("Qwen3-8B Judge - MADLAD Challenge Benchmark")
print("=" * 90)

print(
    "CUDA:",
    torch.cuda.is_available()
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA 不可用。"
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3,
        2,
    ),
    "GB",
)


# ============================================================
# 3. 输入数据
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到 Challenge Prediction：\n"
        f"{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

required_columns = [
    "challenge_id",
    "category",
    "en",
    "uz",
    "uz_en_prediction",
    "en_uz_prediction_latin",
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"缺少字段：{missing}\n"
        f"当前字段：{df.columns.tolist()}"
    )


df = df.dropna(
    subset=required_columns
).copy()

for col in required_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )

df = df.reset_index(
    drop=True
)

print(
    "\nChallenge samples:",
    len(df)
)

print(
    "Total judgements:",
    len(df) * 2
)

print(
    "\nCategories:"
)

print(
    df["category"]
    .value_counts()
)


# ============================================================
# 4. 加载 Qwen3-8B
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

# 清理默认采样设置
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

print(
    "\nModel loaded successfully."
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
# 5. 评分标准
# ============================================================

GRADE_DESCRIPTION = """
A = Fully correct.
The candidate preserves all important meaning.

B = Correct / acceptable.
The wording differs from the reference,
but the core meaning is fully preserved.

C = Minor error.
A small mistranslation, omission or addition exists,
but the main meaning remains correct and usable.

D = Major error.
Important meaning is mistranslated, omitted or added,
or an important number, entity, time or negation is wrong.

F = Failed translation.
The translation is largely unrelated,
contradictory or unusable.
""".strip()


# ============================================================
# 6. Prompt
# ============================================================

def build_prompt(
    source_language: str,
    target_language: str,
    source: str,
    reference: str,
    candidate: str,
    category: str,
):

    return f"""
You are an independent professional machine translation evaluator.

This sample belongs to the following challenge category:

{category}

Evaluate the CANDIDATE translation using both
the SOURCE and the human REFERENCE.

Source language:
{source_language}

Target language:
{target_language}

SOURCE:
{source}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

GRADING STANDARD:

{GRADE_DESCRIPTION}

Judge semantic correctness rather than exact wording.

A valid paraphrase must not be penalized.

Pay special attention to:

- negation
- numbers
- dates
- times
- amounts
- locations
- people
- named entities
- objects
- omissions
- hallucinated additions

For Uzbek:

- Latin Uzbek spelling variants may exist.
- Do not penalize a translation only because its spelling
  differs slightly from the reference.
- Judge whether the actual meaning is preserved.

Return ONLY one valid JSON object:

{{
  "grade": "A",
  "semantic_correct": true,
  "omission": false,
  "addition": false,
  "number_error": false,
  "time_error": false,
  "entity_error": false,
  "negation_error": false,
  "confidence": 0.95,
  "reason": "Short explanation in English."
}}

Rules:

grade:
A, B, C, D or F

confidence:
number between 0 and 1

Do not output Markdown.
Do not output anything outside the JSON.
""".strip()


# ============================================================
# 7. JSON 提取
# ============================================================

def extract_json(text: str):

    text = text.strip()

    try:
        return json.loads(
            text
        )

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
# 8. 标准化 Judge 结果
# ============================================================

VALID_GRADES = {
    "A",
    "B",
    "C",
    "D",
    "F",
}


def normalize_result(
    result,
):

    if not isinstance(
        result,
        dict,
    ):
        return None


    grade = str(
        result.get(
            "grade",
            "",
        )
    ).upper().strip()


    if grade not in VALID_GRADES:

        return None


    def bool_value(name):

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
            str(value)
            .lower()
            .strip()
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
            1.0,
            confidence,
        ),
    )


    return {

        "grade":
            grade,

        "semantic_correct":
            bool_value(
                "semantic_correct"
            ),

        "omission":
            bool_value(
                "omission"
            ),

        "addition":
            bool_value(
                "addition"
            ),

        "number_error":
            bool_value(
                "number_error"
            ),

        "time_error":
            bool_value(
                "time_error"
            ),

        "entity_error":
            bool_value(
                "entity_error"
            ),

        "negation_error":
            bool_value(
                "negation_error"
            ),

        "confidence":
            confidence,

        "reason":
            str(
                result.get(
                    "reason",
                    "",
                )
            ).strip(),
    }


# ============================================================
# 9. Judge 推理
# ============================================================

@torch.inference_mode()
def judge_once(
    source_language,
    target_language,
    source,
    reference,
    candidate,
    category,
):

    prompt = build_prompt(
        source_language=source_language,
        target_language=target_language,
        source=source,
        reference=reference,
        candidate=candidate,
        category=category,
    )


    messages = [

        {
            "role":
                "system",

            "content":
                (
                    "You are a strict but fair "
                    "machine translation quality evaluator. "
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


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
        - start
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
# 10. 最多重试两次
# ============================================================

def run_judge(
    source_language,
    target_language,
    source,
    reference,
    candidate,
    category,
):

    raw_response = ""
    latency = 0.0


    for attempt in range(2):

        result, raw_response, latency = (
            judge_once(
                source_language,
                target_language,
                source,
                reference,
                candidate,
                category,
            )
        )


        normalized = normalize_result(
            result
        )


        if normalized is not None:

            normalized[
                "parse_success"
            ] = True

            normalized[
                "judge_latency"
            ] = latency

            normalized[
                "raw_response"
            ] = raw_response

            return normalized


        print(
            f"  JSON parse failed "
            f"({attempt + 1}/2)"
        )


    return {

        "grade":
            "PARSE_ERROR",

        "semantic_correct":
            False,

        "omission":
            False,

        "addition":
            False,

        "number_error":
            False,

        "time_error":
            False,

        "entity_error":
            False,

        "negation_error":
            False,

        "confidence":
            0.0,

        "reason":
            "Judge output could not be parsed.",

        "parse_success":
            False,

        "judge_latency":
            latency,

        "raw_response":
            raw_response,
    }


# ============================================================
# 11. 断点续跑
# ============================================================

if DETAIL_FILE.exists():

    old_df = pd.read_csv(
        DETAIL_FILE
    )

    completed = set(
        zip(
            old_df[
                "challenge_id"
            ],
            old_df[
                "direction"
            ],
        )
    )

    result_rows = (
        old_df
        .to_dict(
            orient="records"
        )
    )

    print(
        "\nResume mode:"
    )

    print(
        "Already completed:",
        len(result_rows)
    )

else:

    completed = set()

    result_rows = []


# ============================================================
# 12. 开始 600 次 Judge
# ============================================================

print("\n")
print("=" * 90)
print("START CHALLENGE JUDGE")
print("=" * 90)


for index, row in df.iterrows():

    challenge_id = row[
        "challenge_id"
    ]

    category = row[
        "category"
    ]


    # --------------------------------------------------------
    # UZ -> EN
    # --------------------------------------------------------

    key = (
        challenge_id,
        "uz_en",
    )


    if key not in completed:

        print(
            f"\n"
            f"[{index + 1}/{len(df)}] "
            f"{category} | UZ -> EN"
        )


        judge_result = run_judge(
            source_language="Uzbek",
            target_language="English",

            source=row["uz"],

            reference=row["en"],

            candidate=row[
                "uz_en_prediction"
            ],

            category=category,
        )


        result_rows.append({

            "challenge_id":
                challenge_id,

            "category":
                category,

            "direction":
                "uz_en",

            "source_language":
                "uz",

            "target_language":
                "en",

            "source":
                row["uz"],

            "reference":
                row["en"],

            "candidate":
                row[
                    "uz_en_prediction"
                ],

            **judge_result,
        })


        completed.add(
            key
        )


        pd.DataFrame(
            result_rows
        ).to_csv(
            DETAIL_FILE,
            index=False,
            encoding="utf-8-sig",
        )


        print(
            "  Grade:",
            judge_result["grade"],
            "| Confidence:",
            judge_result["confidence"]
        )


    # --------------------------------------------------------
    # EN -> UZ
    # --------------------------------------------------------

    key = (
        challenge_id,
        "en_uz",
    )


    if key not in completed:

        print(
            f"[{index + 1}/{len(df)}] "
            f"{category} | EN -> UZ"
        )


        judge_result = run_judge(
            source_language="English",
            target_language="Uzbek",

            source=row["en"],

            reference=row["uz"],

            candidate=row[
                "en_uz_prediction_latin"
            ],

            category=category,
        )


        result_rows.append({

            "challenge_id":
                challenge_id,

            "category":
                category,

            "direction":
                "en_uz",

            "source_language":
                "en",

            "target_language":
                "uz",

            "source":
                row["en"],

            "reference":
                row["uz"],

            "candidate":
                row[
                    "en_uz_prediction_latin"
                ],

            **judge_result,
        })


        completed.add(
            key
        )


        pd.DataFrame(
            result_rows
        ).to_csv(
            DETAIL_FILE,
            index=False,
            encoding="utf-8-sig",
        )


        print(
            "  Grade:",
            judge_result["grade"],
            "| Confidence:",
            judge_result["confidence"]
        )


# ============================================================
# 13. 最终数据
# ============================================================

result_df = pd.DataFrame(
    result_rows
)

print("\n")
print("=" * 90)
print("JUDGE COMPLETED")
print("=" * 90)

print(
    "Total:",
    len(result_df)
)


# ============================================================
# 14. Summary
# ============================================================

def calculate_summary(
    part,
):

    total = len(part)

    counts = (
        part["grade"]
        .value_counts()
        .to_dict()
    )


    def grade_count(grade):

        return int(
            counts.get(
                grade,
                0,
            )
        )


    A = grade_count("A")
    B = grade_count("B")
    C = grade_count("C")
    D = grade_count("D")
    F = grade_count("F")
    P = grade_count(
        "PARSE_ERROR"
    )


    if total > 0:

        ab_rate = (
            (A + B)
            / total
            * 100
        )

        c_rate = (
            C
            / total
            * 100
        )

        severe_rate = (
            (D + F)
            / total
            * 100
        )

    else:

        ab_rate = 0.0
        c_rate = 0.0
        severe_rate = 0.0


    def error_rate(column):

        if total == 0:
            return 0.0

        return float(
            part[column]
            .fillna(False)
            .astype(bool)
            .mean()
            * 100
        )


    return {

        "total":
            total,

        "A":
            A,

        "B":
            B,

        "C":
            C,

        "D":
            D,

        "F":
            F,

        "parse_error":
            P,

        "A_B_rate":
            ab_rate,

        "C_rate":
            c_rate,

        "D_F_rate":
            severe_rate,

        "omission_rate":
            error_rate(
                "omission"
            ),

        "addition_rate":
            error_rate(
                "addition"
            ),

        "number_error_rate":
            error_rate(
                "number_error"
            ),

        "time_error_rate":
            error_rate(
                "time_error"
            ),

        "entity_error_rate":
            error_rate(
                "entity_error"
            ),

        "negation_error_rate":
            error_rate(
                "negation_error"
            ),

        "avg_confidence":
            float(
                part[
                    "confidence"
                ].mean()
            ),

        "avg_latency":
            float(
                part[
                    "judge_latency"
                ].mean()
            ),
    }


# ============================================================
# 15. PASS 判定
# ============================================================

def teacher_decision(
    summary,
):

    ab = summary[
        "A_B_rate"
    ]

    severe = summary[
        "D_F_rate"
    ]


    if (
        ab >= 90
        and
        severe <= 5
    ):

        return "PASS"


    if (
        ab >= 80
        and
        severe <= 10
    ):

        return "CONDITIONAL_PASS"


    return "FAIL"


# ============================================================
# 16. 按方向 + category 汇总
# ============================================================

summary_rows = []


for direction in [
    "uz_en",
    "en_uz",
]:

    # 整体
    overall = result_df[
        result_df[
            "direction"
        ]
        ==
        direction
    ]

    summary = calculate_summary(
        overall
    )

    summary_rows.append({
        "direction":
            direction,

        "category":
            "ALL",

        **summary,

        "decision":
            teacher_decision(
                summary
            ),
    })


    # 各类别
    categories = sorted(
        result_df[
            "category"
        ].unique()
    )


    for category in categories:

        part = result_df[
            (
                result_df[
                    "direction"
                ]
                ==
                direction
            )
            &
            (
                result_df[
                    "category"
                ]
                ==
                category
            )
        ]


        summary = calculate_summary(
            part
        )


        summary_rows.append({

            "direction":
                direction,

            "category":
                category,

            **summary,

            "decision":
                teacher_decision(
                    summary
                ),
        })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 17. 保存
# ============================================================

summary_df.to_csv(
    CATEGORY_SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


summary_json = {

    "judge_model":
        "Qwen3-8B",

    "translation_model":
        "MADLAD-400-3B",

    "challenge_samples":
        len(df),

    "total_judgements":
        len(result_df),

    "results":
        summary_df.to_dict(
            orient="records"
        ),
}


with open(
    SUMMARY_JSON_FILE,
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
# 18. Terminal 输出
# ============================================================

print("\n")
print("=" * 130)
print("CHALLENGE JUDGE SUMMARY")
print("=" * 130)


display_columns = [

    "direction",
    "category",
    "total",

    "A_B_rate",
    "C_rate",
    "D_F_rate",

    "number_error_rate",
    "time_error_rate",
    "entity_error_rate",
    "negation_error_rate",

    "decision",
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
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

print(
    "Details:",
    DETAIL_FILE
)

print(
    "Category Summary:",
    CATEGORY_SUMMARY_FILE
)

print(
    "JSON:",
    SUMMARY_JSON_FILE
)

print(
    "\nDone."
)