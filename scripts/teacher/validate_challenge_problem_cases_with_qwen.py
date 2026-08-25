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
# 1. Config
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
    / "qwen_judge"
    / "challenge_judge_details.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
    / "challenge_v1"
    / "qwen_judge"
    / "second_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "challenge_problem_cases_reviewed.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "challenge_second_review_summary.csv"
)

SUMMARY_JSON_FILE = (
    OUTPUT_DIR
    / "challenge_second_review_summary.json"
)

MAX_NEW_TOKENS = 320

TOTAL_PER_DIRECTION = 300


# ============================================================
# 2. Environment
# ============================================================

print("=" * 90)
print("Qwen3 Challenge Second Review")
print("=" * 90)

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
# 3. Read first-pass Judge results
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到：{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    "\nFirst-pass rows:",
    len(df),
)

print(
    "Columns:",
    df.columns.tolist(),
)


# ============================================================
# 4. Normalize boolean columns
# ============================================================

RISK_COLUMNS = [
    "number_error",
    "time_error",
    "entity_error",
    "negation_error",
]


def normalize_bool_series(series):

    return (
        series
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    )


for col in RISK_COLUMNS:

    if col not in df.columns:

        df[col] = False

    else:

        df[col] = (
            normalize_bool_series(
                df[col]
            )
        )


# ============================================================
# 5. Select suspicious cases
# ============================================================

problem_grade_mask = (
    df["grade"]
    .astype(str)
    .str.upper()
    .isin(
        [
            "C",
            "D",
            "F",
            "PARSE_ERROR",
        ]
    )
)

risk_mask = False

for col in RISK_COLUMNS:

    risk_mask = (
        risk_mask
        |
        df[col]
    )


problem_df = df[
    problem_grade_mask
    |
    risk_mask
].copy()


problem_df = (
    problem_df
    .drop_duplicates(
        subset=[
            "challenge_id",
            "direction",
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\nSuspicious cases:",
    len(problem_df),
)

print(
    "\nBy direction:"
)

print(
    problem_df[
        "direction"
    ].value_counts()
)

print(
    "\nBy category:"
)

print(
    problem_df[
        [
            "direction",
            "category",
        ]
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 6. Load Qwen3
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
    "Model loaded successfully."
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
# 7. Second-review prompt
# ============================================================

def build_prompt(row):

    direction = row[
        "direction"
    ]

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
You are performing an independent SECOND-PASS audit
of a machine translation benchmark.

The first evaluator marked this translation as suspicious.

Challenge category:
{row["category"]}

Translation direction:
{src_lang} -> {tgt_lang}

SOURCE:
{row["source"]}

HUMAN REFERENCE:
{row["reference"]}

MADLAD CANDIDATE:
{row["candidate"]}

FIRST-PASS GRADE:
{row["grade"]}

FIRST-PASS REASON:
{row.get("reason", "")}

FIRST-PASS ERROR FLAGS:

number_error:
{row.get("number_error", False)}

time_error:
{row.get("time_error", False)}

entity_error:
{row.get("entity_error", False)}

negation_error:
{row.get("negation_error", False)}


Independently evaluate two questions.

QUESTION 1:
Is the HUMAN REFERENCE a semantically correct translation
of the SOURCE?

QUESTION 2:
Is the MADLAD CANDIDATE a semantically correct translation
of the SOURCE?

Do NOT assume that the human reference is always correct.

Final labels:

MODEL_ERROR
- Reference is reasonably consistent with the source.
- Candidate contains a real translation error.

DATASET_ERROR
- Source/reference pair itself is incorrect or mismatched.
- Candidate may actually represent the source better.

JUDGE_ERROR
- First-pass evaluator was overly strict or incorrect.
- Candidate is semantically acceptable.

AMBIGUOUS
- Cannot confidently determine whether the reference
  or candidate is correct.

Model error types:

NONE
MISTRANSLATION
OMISSION
ADDITION
ENTITY_ERROR
NUMBER_ERROR
TIME_ERROR
NEGATION_ERROR
OTHER

Important rules:

- Judge meaning, not exact wording.
- Valid paraphrases are correct.
- For Uzbek, Latin/Cyrillic spelling differences alone
  are not translation errors.
- Pay special attention to negation, numbers, time,
  entities and omitted information.

Return ONLY one valid JSON object:

{{
  "source_reference_consistent": true,
  "source_candidate_consistent": true,
  "final_label": "JUDGE_ERROR",
  "model_error_type": "NONE",
  "high_risk": false,
  "confidence": 0.95,
  "explanation": "Short explanation in English."
}}

final_label must be exactly one of:

MODEL_ERROR
DATASET_ERROR
JUDGE_ERROR
AMBIGUOUS

Do not output Markdown.
Do not output anything outside JSON.
""".strip()


# ============================================================
# 8. JSON extraction
# ============================================================

def extract_json(text):

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
            text[
                start:end + 1
            ]
        )

    except Exception:

        return None


# ============================================================
# 9. Normalize result
# ============================================================

VALID_LABELS = {
    "MODEL_ERROR",
    "DATASET_ERROR",
    "JUDGE_ERROR",
    "AMBIGUOUS",
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


def normalize_result(result):

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


    def as_bool(name):

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
            .strip()
            .lower()
            ==
            "true"
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

        "source_reference_consistent":
            as_bool(
                "source_reference_consistent"
            ),

        "source_candidate_consistent":
            as_bool(
                "source_candidate_consistent"
            ),

        "final_label":
            label,

        "model_error_type":
            error_type,

        "high_risk":
            as_bool(
                "high_risk"
            ),

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
# 10. Inference
# ============================================================

@torch.inference_mode()
def review_once(row):

    prompt = build_prompt(
        row
    )


    messages = [

        {
            "role":
                "system",

            "content":
                (
                    "You are an independent "
                    "machine translation benchmark auditor. "
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

    start = (
        time.perf_counter()
    )


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


    generated_tokens = outputs[0][
        inputs[
            "input_ids"
        ].shape[1]:
    ]


    response = tokenizer.decode(
        generated_tokens,
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
# 11. Retry wrapper
# ============================================================

def review_case(row):

    raw_response = ""
    latency = 0.0


    for attempt in range(2):

        parsed, raw_response, latency = (
            review_once(
                row
            )
        )


        normalized = (
            normalize_result(
                parsed
            )
        )


        if normalized is not None:

            normalized[
                "second_latency"
            ] = latency

            normalized[
                "second_raw_response"
            ] = raw_response

            normalized[
                "second_parse_success"
            ] = True

            return normalized


        print(
            f"  Parse failed "
            f"{attempt + 1}/2"
        )


    return {

        "source_reference_consistent":
            False,

        "source_candidate_consistent":
            False,

        "final_label":
            "AMBIGUOUS",

        "model_error_type":
            "OTHER",

        "high_risk":
            False,

        "second_confidence":
            0.0,

        "second_explanation":
            (
                "Second-pass output "
                "could not be parsed."
            ),

        "second_latency":
            latency,

        "second_raw_response":
            raw_response,

        "second_parse_success":
            False,
    }


# ============================================================
# 12. Resume
# ============================================================

if OUTPUT_FILE.exists():

    old_df = pd.read_csv(
        OUTPUT_FILE
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
        "\nResume mode:",
        len(result_rows),
        "completed"
    )

else:

    completed = set()
    result_rows = []


# ============================================================
# 13. Run second review
# ============================================================

print("\n")
print("=" * 90)
print("START SECOND REVIEW")
print("=" * 90)


for index, row in problem_df.iterrows():

    key = (
        row[
            "challenge_id"
        ],
        row[
            "direction"
        ],
    )


    if key in completed:

        continue


    print(
        f"\n[{index + 1}/{len(problem_df)}]"
    )

    print(
        row["direction"],
        "|",
        row["category"],
        "| first:",
        row["grade"],
    )


    result = review_case(
        row
    )


    output_row = {

        **row.to_dict(),

        **result,
    }


    result_rows.append(
        output_row
    )


    completed.add(
        key
    )


    pd.DataFrame(
        result_rows
    ).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        "  Final:",
        result[
            "final_label"
        ]
    )

    print(
        "  Type:",
        result[
            "model_error_type"
        ]
    )

    print(
        "  High risk:",
        result[
            "high_risk"
        ]
    )

    print(
        "  Confidence:",
        result[
            "second_confidence"
        ]
    )


# ============================================================
# 14. Read second-review results
# ============================================================

review_df = pd.DataFrame(
    result_rows
)


print("\n")
print("=" * 90)
print("SECOND REVIEW COMPLETED")
print("=" * 90)

print(
    "Reviewed:",
    len(review_df)
)

all_judge_df = df.copy()
# ============================================================
# 15. Corrected statistics
# ============================================================

summary_rows = []


categories = sorted(
    df[
        "category"
    ].dropna().unique()
)


for direction in [
    "uz_en",
    "en_uz",
]:

    direction_total = len(
        df[
            df[
                "direction"
            ]
            ==
            direction
        ]
    )


    # 如果第一轮正好300条
    if direction_total == 0:

        direction_total = (
            TOTAL_PER_DIRECTION
        )


    direction_review = review_df[
        review_df[
            "direction"
        ]
        ==
        direction
    ]


    confirmed_errors = direction_review[
        direction_review[
            "final_label"
        ]
        ==
        "MODEL_ERROR"
    ]


    confirmed_high_risk = (
        confirmed_errors[
            confirmed_errors[
                "high_risk"
            ]
            ==
            True
        ]
    )


    summary_rows.append({

        "direction":
            direction,

        "category":
            "ALL",

        "total_samples":
            TOTAL_PER_DIRECTION,

        "suspicious_cases":
            len(
                direction_review
            ),

        "confirmed_model_errors":
            len(
                confirmed_errors
            ),

        "confirmed_model_error_rate":
            (
                len(
                    confirmed_errors
                )
                /
                TOTAL_PER_DIRECTION
                *
                100
            ),

        "high_risk_errors":
            len(
                confirmed_high_risk
            ),

        "high_risk_error_rate":
            (
                len(
                    confirmed_high_risk
                )
                /
                TOTAL_PER_DIRECTION
                *
                100
            ),

        "dataset_errors":
            int(
                (
                    direction_review[
                        "final_label"
                    ]
                    ==
                    "DATASET_ERROR"
                ).sum()
            ),

        "judge_errors":
            int(
                (
                    direction_review[
                        "final_label"
                    ]
                    ==
                    "JUDGE_ERROR"
                ).sum()
            ),

        "ambiguous":
            int(
                (
                    direction_review[
                        "final_label"
                    ]
                    ==
                    "AMBIGUOUS"
                ).sum()
            ),
    })


    # ========================================================
    # Category statistics
    # ========================================================

    for category in categories:

        category_total = len(
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
                        "category"
                    ]
                    ==
                    category
                )
            ]
        )


        # challenge 每类理论上50
        if category_total == 0:

            category_total = 50


        part = direction_review[
            direction_review[
                "category"
            ]
            ==
            category
        ]


        model_errors = part[
            part[
                "final_label"
            ]
            ==
            "MODEL_ERROR"
        ]


        high_risk_errors = (
            model_errors[
                model_errors[
                    "high_risk"
                ]
                ==
                True
            ]
        )


        summary_rows.append({

            "direction":
                direction,

            "category":
                category,

            "total_samples":
                category_total,

            "suspicious_cases":
                len(part),

            "confirmed_model_errors":
                len(
                    model_errors
                ),

            "confirmed_model_error_rate":
                (
                    len(
                        model_errors
                    )
                    /
                    category_total
                    *
                    100
                ),

            "high_risk_errors":
                len(
                    high_risk_errors
                ),

            "high_risk_error_rate":
                (
                    len(
                        high_risk_errors
                    )
                    /
                    category_total
                    *
                    100
                ),

            "dataset_errors":
                int(
                    (
                        part[
                            "final_label"
                        ]
                        ==
                        "DATASET_ERROR"
                    ).sum()
                ),

            "judge_errors":
                int(
                    (
                        part[
                            "final_label"
                        ]
                        ==
                        "JUDGE_ERROR"
                    ).sum()
                ),

            "ambiguous":
                int(
                    (
                        part[
                            "final_label"
                        ]
                        ==
                        "AMBIGUOUS"
                    ).sum()
                ),
        })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 16. Teacher decision
# ============================================================

def decision(row):

    error_rate = row[
        "confirmed_model_error_rate"
    ]

    high_risk = row[
        "high_risk_error_rate"
    ]


    if (
        error_rate <= 5
        and
        high_risk <= 2
    ):

        return "PASS"


    if (
        error_rate <= 10
        and
        high_risk <= 5
    ):

        return "CONDITIONAL_PASS"


    return "FAIL"


summary_df[
    "decision"
] = summary_df.apply(
    decision,
    axis=1,
)


# ============================================================
# 17. Save summary
# ============================================================

summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


summary_json = {

    "translation_model":
        "MADLAD-400-3B",

    "judge_model":
        "Qwen3-8B",

    "challenge_version":
        "v1",

    "reviewed_cases":
        len(review_df),

    "summary":
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
# 18. Error types
# ============================================================

model_error_df = review_df[
    review_df[
        "final_label"
    ]
    ==
    "MODEL_ERROR"
]


print("\n")
print("=" * 90)
print("MODEL ERROR TYPES")
print("=" * 90)

print(
    model_error_df[
        [
            "direction",
            "model_error_type",
        ]
    ]
    .value_counts()
)


# ============================================================
# 19. Final table
# ============================================================

print("\n")
print("=" * 120)
print("CORRECTED CHALLENGE RESULT")
print("=" * 120)


columns = [

    "direction",
    "category",
    "total_samples",

    "suspicious_cases",

    "confirmed_model_errors",
    "confirmed_model_error_rate",

    "high_risk_errors",
    "high_risk_error_rate",

    "dataset_errors",
    "judge_errors",
    "ambiguous",

    "decision",
]


print(
    summary_df[
        columns
    ]
    .round(2)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 90)
print("FILES")
print("=" * 90)

print(
    "Details:",
    OUTPUT_FILE
)

print(
    "Summary:",
    SUMMARY_FILE
)

print(
    "JSON:",
    SUMMARY_JSON_FILE
)

print(
    "\nDone."
)