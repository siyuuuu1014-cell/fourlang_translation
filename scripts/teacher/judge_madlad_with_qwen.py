from pathlib import Path
import json
import re
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
    / "tatoeba_500_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
    / "qwen_judge"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLE_FILE = (
    OUTPUT_DIR
    / "judge_sample_100.csv"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "judge_details.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "judge_summary.json"
)

SAMPLE_SIZE = 100

SEED = 2026

MAX_NEW_TOKENS = 256


# ============================================================
# 2. 环境检查
# ============================================================

print("=" * 80)
print("Qwen3-8B Judge for MADLAD-400-3B")
print("=" * 80)

print("\nCUDA:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA 不可用，请检查 GPU 环境。"
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
        / 1024 ** 3,
        2,
    ),
    "GB",
)


# ============================================================
# 3. 检查输入文件
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到 MADLAD 预测文件：\n"
        f"{INPUT_FILE}"
    )

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"找不到 Qwen3-8B：\n"
        f"{MODEL_PATH}"
    )


# ============================================================
# 4. 读取 MADLAD 预测结果
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("\nInput rows:", len(df))
print("Columns:", df.columns.tolist())


required_columns = [
    "en",
    "uz",
    "uz_en_prediction",
    "en_uz_prediction_raw",
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


# ============================================================
# 5. 清洗
# ============================================================

df = df.dropna(
    subset=[
        "en",
        "uz",
        "uz_en_prediction",
        "en_uz_prediction_latin",
    ]
).copy()

for col in required_columns:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )

df = df[
    (df["en"] != "")
    &
    (df["uz"] != "")
    &
    (df["uz_en_prediction"] != "")
    &
    (df["en_uz_prediction_latin"] != "")
].copy()

df = df.reset_index(drop=True)

print(
    "Valid rows:",
    len(df)
)


# ============================================================
# 6. 固定随机抽100条
#
# 如果 SAMPLE_FILE 已存在，
# 就直接读取，确保多次实验用的是同样的100条。
# ============================================================

if SAMPLE_FILE.exists():

    print(
        "\nUsing existing sample:"
    )

    print(
        SAMPLE_FILE
    )

    sample_df = pd.read_csv(
        SAMPLE_FILE
    )

else:

    actual_sample_size = min(
        SAMPLE_SIZE,
        len(df),
    )

    sample_df = (
        df.sample(
            n=actual_sample_size,
            random_state=SEED,
        )
        .reset_index(drop=True)
    )

    sample_df.insert(
        0,
        "sample_id",
        range(
            1,
            len(sample_df) + 1,
        ),
    )

    sample_df.to_csv(
        SAMPLE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\nSample saved:"
    )

    print(
        SAMPLE_FILE
    )


if "sample_id" not in sample_df.columns:

    sample_df.insert(
        0,
        "sample_id",
        range(
            1,
            len(sample_df) + 1,
        ),
    )


print(
    "\nJudge sample size:",
    len(sample_df)
)


# ============================================================
# 7. 加载 Qwen3-8B
# ============================================================

print("\nLoading Qwen3 tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


print(
    "Loading Qwen3-8B..."
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=True,
)

model.eval()


# 清理默认采样配置，避免 warning
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None


print(
    "\nQwen3 loaded successfully."
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
# 8. Judge Grade 定义
# ============================================================

GRADE_DESCRIPTION = """
A = Fully correct.
    The candidate preserves all important meaning.
    Differences in wording or grammar style are acceptable.

B = Correct / acceptable.
    The candidate expresses the same core meaning,
    but wording differs noticeably from the reference.
    No important information is lost or changed.

C = Minor error.
    There is a small translation problem,
    awkward expression, or minor omission/addition,
    but the main meaning remains understandable.

D = Major error.
    Important meaning is mistranslated, omitted, added,
    or important entities/numbers/time/negation are incorrect.

F = Failed translation.
    The candidate is mostly unrelated, seriously wrong,
    or fails to translate the source.
"""


# ============================================================
# 9. 构造 Judge Prompt
# ============================================================

def build_judge_prompt(
    source_language,
    target_language,
    source,
    reference,
    candidate,
):

    return f"""
You are an independent professional machine translation evaluator.

Evaluate the candidate translation by comparing:

1. the SOURCE text,
2. the human REFERENCE translation,
3. the CANDIDATE translation.

Source language: {source_language}
Target language: {target_language}

SOURCE:
{source}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Use the following grading standard:

{GRADE_DESCRIPTION}

Important evaluation rules:

- Judge SEMANTIC correctness, not exact wording.
- A valid paraphrase should NOT be penalized.
- Pay special attention to:
  - numbers
  - dates
  - times
  - money
  - locations
  - names
  - entities
  - negation
  - omitted information
  - hallucinated information

For Uzbek:
- Latin Uzbek spelling variants may exist.
- Do not penalize a candidate merely because its wording
  differs from the reference if the meaning is equivalent.

Return ONLY one valid JSON object.

Required JSON format:

{{
  "grade": "A",
  "semantic_correct": true,
  "omission": false,
  "addition": false,
  "number_error": false,
  "entity_error": false,
  "negation_error": false,
  "confidence": 0.98,
  "reason": "Brief explanation in English."
}}

Requirements:

- grade must be one of A, B, C, D, F.
- all error fields must be true or false.
- confidence must be between 0 and 1.
- reason must be concise.
- do not output markdown.
- do not output anything outside the JSON object.
""".strip()


# ============================================================
# 10. JSON解析
# ============================================================

def extract_json(text):

    text = text.strip()

    # 最理想情况
    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass


    # 找第一个 { 到最后一个 }
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


    candidate = text[
        start:end + 1
    ]

    try:

        return json.loads(
            candidate
        )

    except json.JSONDecodeError:

        return None


# ============================================================
# 11. Judge单条推理
# ============================================================

@torch.inference_mode()
def judge_translation(
    source_language,
    target_language,
    source,
    reference,
    candidate,
):

    prompt = build_judge_prompt(
        source_language,
        target_language,
        source,
        reference,
        candidate,
    )


    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict but fair "
                "machine translation quality evaluator. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
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
        key: value.to(device)
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


    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
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
# 12. 结果校验
# ============================================================

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


    if grade not in {
        "A",
        "B",
        "C",
        "D",
        "F",
    }:

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

        if isinstance(
            value,
            str,
        ):

            return (
                value.lower()
                ==
                "true"
            )

        return bool(
            value
        )


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
# 13. 单条审核，失败自动重试一次
# ============================================================

def run_judge(
    source_language,
    target_language,
    source,
    reference,
    candidate,
):

    for attempt in range(2):

        result, raw, latency = (
            judge_translation(
                source_language,
                target_language,
                source,
                reference,
                candidate,
            )
        )


        normalized = normalize_result(
            result
        )


        if normalized is not None:

            normalized[
                "judge_latency"
            ] = latency

            normalized[
                "raw_response"
            ] = raw

            normalized[
                "parse_success"
            ] = True

            return normalized


        print(
            f"JSON parse failed. "
            f"Retry {attempt + 1}/2"
        )


    return {
        "grade": "PARSE_ERROR",

        "semantic_correct":
            False,

        "omission":
            False,

        "addition":
            False,

        "number_error":
            False,

        "entity_error":
            False,

        "negation_error":
            False,

        "confidence":
            0.0,

        "reason":
            "Judge output could not be parsed.",

        "judge_latency":
            0.0,

        "raw_response":
            raw,

        "parse_success":
            False,
    }


# ============================================================
# 14. 恢复机制
#
# 如果之前运行中断，
# 再运行时跳过已经完成的结果。
# ============================================================

if DETAIL_FILE.exists():

    result_df = pd.read_csv(
        DETAIL_FILE
    )

    completed_keys = set(
        zip(
            result_df[
                "sample_id"
            ],
            result_df[
                "direction"
            ],
        )
    )

    result_rows = (
        result_df
        .to_dict(
            orient="records"
        )
    )

    print(
        "\nResume mode."
    )

    print(
        "Already judged:",
        len(result_rows)
    )

else:

    completed_keys = set()

    result_rows = []


# ============================================================
# 15. 正式审核
# ============================================================

TOTAL_JUDGEMENTS = (
    len(sample_df) * 2
)

print("\n")
print("=" * 80)
print("START JUDGING")
print("=" * 80)

print(
    "Samples:",
    len(sample_df)
)

print(
    "Total judgements:",
    TOTAL_JUDGEMENTS
)


for row_index, row in sample_df.iterrows():

    sample_id = int(
        row[
            "sample_id"
        ]
    )


    # ========================================================
    # A. Uzbek -> English
    # ========================================================

    direction = "uz_en"

    key = (
        sample_id,
        direction,
    )


    if key not in completed_keys:

        print(
            f"\n"
            f"[{sample_id}/{len(sample_df)}] "
            f"UZ -> EN"
        )


        judge_result = run_judge(
            source_language="Uzbek",
            target_language="English",

            source=row["uz"],

            reference=row["en"],

            candidate=row[
                "uz_en_prediction"
            ],
        )


        result_rows.append({
            "sample_id":
                sample_id,

            "direction":
                direction,

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


        completed_keys.add(
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
            "Grade:",
            judge_result[
                "grade"
            ],
            "| confidence:",
            judge_result[
                "confidence"
            ],
        )


    # ========================================================
    # B. English -> Uzbek
    # ========================================================

    direction = "en_uz"

    key = (
        sample_id,
        direction,
    )


    if key not in completed_keys:

        print(
            f"[{sample_id}/{len(sample_df)}] "
            f"EN -> UZ"
        )


        judge_result = run_judge(
            source_language="English",
            target_language="Uzbek",

            source=row["en"],

            reference=row["uz"],

            # 这里用 Latin 转写后的版本
            candidate=row[
                "en_uz_prediction_latin"
            ],
        )


        result_rows.append({
            "sample_id":
                sample_id,

            "direction":
                direction,

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


        completed_keys.add(
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
            "Grade:",
            judge_result[
                "grade"
            ],
            "| confidence:",
            judge_result[
                "confidence"
            ],
        )


# ============================================================
# 16. 读取最终结果
# ============================================================

result_df = pd.DataFrame(
    result_rows
)

print("\n")
print("=" * 80)
print("JUDGING COMPLETED")
print("=" * 80)

print(
    "Total results:",
    len(result_df)
)


# ============================================================
# 17. Summary函数
# ============================================================

def calculate_summary(
    direction_df,
):

    total = len(
        direction_df
    )


    grade_counts = (
        direction_df[
            "grade"
        ]
        .value_counts()
        .to_dict()
    )


    def count_grade(
        grade,
    ):

        return int(
            grade_counts.get(
                grade,
                0,
            )
        )


    A = count_grade("A")
    B = count_grade("B")
    C = count_grade("C")
    D = count_grade("D")
    F = count_grade("F")

    parse_errors = count_grade(
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

        ab_rate = 0
        c_rate = 0
        severe_rate = 0


    def bool_rate(
        column,
    ):

        if total == 0:

            return 0.0

        return float(
            direction_df[
                column
            ]
            .fillna(False)
            .astype(bool)
            .mean()
            * 100
        )


    return {
        "total":
            total,

        "grade_counts": {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "F": F,
            "PARSE_ERROR":
                parse_errors,
        },

        "A_B_rate_percent":
            ab_rate,

        "C_rate_percent":
            c_rate,

        "D_F_rate_percent":
            severe_rate,

        "omission_rate_percent":
            bool_rate(
                "omission"
            ),

        "addition_rate_percent":
            bool_rate(
                "addition"
            ),

        "number_error_rate_percent":
            bool_rate(
                "number_error"
            ),

        "entity_error_rate_percent":
            bool_rate(
                "entity_error"
            ),

        "negation_error_rate_percent":
            bool_rate(
                "negation_error"
            ),

        "average_confidence":
            float(
                direction_df[
                    "confidence"
                ].mean()
            ),

        "average_judge_latency":
            float(
                direction_df[
                    "judge_latency"
                ].mean()
            ),
    }


# ============================================================
# 18. 分方向统计
# ============================================================

uz_en_df = result_df[
    result_df[
        "direction"
    ]
    ==
    "uz_en"
].copy()


en_uz_df = result_df[
    result_df[
        "direction"
    ]
    ==
    "en_uz"
].copy()


uz_en_summary = calculate_summary(
    uz_en_df
)

en_uz_summary = calculate_summary(
    en_uz_df
)


# ============================================================
# 19. Teacher PASS判断
#
# 第一版标准：
#
# A+B >= 90%
# D+F <= 5%
# ============================================================

def teacher_decision(
    summary,
):

    ab_rate = summary[
        "A_B_rate_percent"
    ]

    severe_rate = summary[
        "D_F_rate_percent"
    ]


    if (
        ab_rate >= 90
        and
        severe_rate <= 5
    ):

        return "PASS"


    if (
        ab_rate >= 80
        and
        severe_rate <= 10
    ):

        return "CONDITIONAL_PASS"


    return "FAIL"


uz_en_summary[
    "teacher_decision"
] = teacher_decision(
    uz_en_summary
)


en_uz_summary[
    "teacher_decision"
] = teacher_decision(
    en_uz_summary
)


# ============================================================
# 20. 最终 Summary
# ============================================================

final_summary = {
    "judge_model":
        "Qwen3-8B",

    "translation_model":
        "MADLAD-400-3B",

    "sample_size":
        len(sample_df),

    "judgements":
        len(result_df),

    "uz_en":
        uz_en_summary,

    "en_uz":
        en_uz_summary,
}


with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 21. Terminal显示
# ============================================================

def print_summary(
    title,
    summary,
):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Total:",
        summary[
            "total"
        ]
    )

    print(
        "Grades:",
        summary[
            "grade_counts"
        ]
    )

    print(
        f"A+B rate: "
        f"{summary['A_B_rate_percent']:.2f}%"
    )

    print(
        f"C rate: "
        f"{summary['C_rate_percent']:.2f}%"
    )

    print(
        f"D+F severe error rate: "
        f"{summary['D_F_rate_percent']:.2f}%"
    )

    print(
        f"Omission rate: "
        f"{summary['omission_rate_percent']:.2f}%"
    )

    print(
        f"Addition rate: "
        f"{summary['addition_rate_percent']:.2f}%"
    )

    print(
        f"Number error rate: "
        f"{summary['number_error_rate_percent']:.2f}%"
    )

    print(
        f"Entity error rate: "
        f"{summary['entity_error_rate_percent']:.2f}%"
    )

    print(
        f"Negation error rate: "
        f"{summary['negation_error_rate_percent']:.2f}%"
    )

    print(
        f"Average confidence: "
        f"{summary['average_confidence']:.4f}"
    )

    print(
        f"Average Judge latency: "
        f"{summary['average_judge_latency']:.4f}s"
    )

    print(
        "Teacher decision:",
        summary[
            "teacher_decision"
        ]
    )


print_summary(
    "UZ -> EN",
    uz_en_summary,
)

print_summary(
    "EN -> UZ",
    en_uz_summary,
)


print("\n")
print("=" * 80)
print("FILES")
print("=" * 80)

print(
    "Sample :",
    SAMPLE_FILE
)

print(
    "Details:",
    DETAIL_FILE
)

print(
    "Summary:",
    SUMMARY_FILE
)

print("\nDone.")