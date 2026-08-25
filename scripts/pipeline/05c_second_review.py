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

STEP05B_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "05_qwen_review"
    / "results"
)

INPUT_FILE = (
    STEP05B_DIR
    / "qwen_semantic_judge_details.parquet"
)

OUTPUT_DIR = (
    STEP05B_DIR
    / "second_review"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


CHECKPOINT_FILE = (
    OUTPUT_DIR
    / "second_review_checkpoint.jsonl"
)

DETAIL_FILE = (
    OUTPUT_DIR
    / "second_review_details.parquet"
)

DETAIL_CSV = (
    OUTPUT_DIR
    / "second_review_details.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "second_review_summary.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "second_review_summary.json"
)

CONFIRMED_FAIL_FILE = (
    OUTPUT_DIR
    / "confirmed_fail.parquet"
)

RECOVERED_FILE = (
    OUTPUT_DIR
    / "recovered_pairs.parquet"
)

QUARANTINE_FILE = (
    OUTPUT_DIR
    / "quarantine.parquet"
)


# ============================================================
# 2. Config
# ============================================================

BATCH_SIZE = 4

MAX_INPUT_LENGTH = 1536

MAX_NEW_TOKENS = 160

ENABLE_JSON_REPAIR = True

PRINT_EVERY_BATCHES = 5


# ============================================================
# 3. Environment
# ============================================================

print("=" * 100)
print("EN-UZ PIPELINE")
print("STEP 05C - SECOND SEMANTIC REVIEW")
print("=" * 100)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA unavailable."
    )


print(
    "CUDA:",
    True
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
# 4. Load Step05B results
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到 Step05B 结果：\n"
        f"{INPUT_FILE}"
    )


all_df = pd.read_parquet(
    INPUT_FILE
)


required_columns = [
    "review_id",
    "review_type",
    "normalized_pair_id",
    "source_text_normalized",
    "target_text_normalized",
    "judge_label",
]


missing = [

    column

    for column
    in required_columns

    if column not in all_df.columns
]


if missing:

    raise ValueError(
        f"缺少字段：{missing}"
    )


# ============================================================
# 5. Only FAIL + UNCERTAIN
# ============================================================

problem_df = all_df[
    all_df[
        "judge_label"
    ]
    .astype(str)
    .str.upper()
    .isin(
        [
            "FAIL",
            "UNCERTAIN",
        ]
    )
].copy()


problem_df = (
    problem_df
    .drop_duplicates(
        subset=[
            "review_id",
        ]
    )
    .reset_index(
        drop=True
    )
)


problem_df[
    "first_judge_label"
] = (
    problem_df[
        "judge_label"
    ]
    .astype(str)
    .str.upper()
)


print(
    "\nStep05B total:",
    len(all_df)
)

print(
    "Second review candidates:",
    len(problem_df)
)


print(
    "\nFirst-pass labels:"
)

print(
    problem_df[
        "first_judge_label"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# 6. Load tokenizer
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

tokenizer.padding_side = "left"

if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# 7. Load model
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


MODEL_DEVICE = (
    next(
        model.parameters()
    ).device
)


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
# 8. Independent second-review prompt
#
# IMPORTANT:
#
# 不告诉模型：
#
# first judge label
# first judge reason
#
# 尽量减少 anchoring。
# ============================================================

def build_prompt(
    english: str,
    uzbek: str,
) -> str:

    return f"""
You are independently reviewing an English-Uzbek parallel sentence pair.

Determine whether these two sentences express the same meaning.

ENGLISH:
{english}

UZBEK:
{uzbek}

Evaluate this pair independently.

Do NOT assume that the pair is correct or incorrect.

Judge semantic equivalence, not literal word overlap.

Natural paraphrases are acceptable.

Minor Uzbek spelling or apostrophe variation alone is not an error.

Pay special attention to:

- missing information
- added information
- mistranslation
- numbers
- dates
- times
- people
- locations
- named entities
- negation
- contradictions

Choose exactly one label:

PASS
- The two sentences preserve essentially the same meaning.
- Suitable as a high-quality parallel training pair.

MINOR
- The core meaning is preserved.
- There is only a small lexical, grammatical,
  omission or addition issue.

FAIL
- Important meaning differs.
- The pair is unsuitable as a parallel training pair.

UNCERTAIN
- The relationship cannot be judged confidently.

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
  "high_risk": false,
  "confidence": 0.95,
  "reason": "Brief explanation."
}}

No Markdown.
No additional text.
""".strip()


# ============================================================
# 9. Chat template
# ============================================================

def build_chat_text(
    row,
):

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
    )


    messages = [

        {
            "role":
                "system",

            "content":
                (
                    "You are an independent "
                    "English-Uzbek translation auditor. "
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


    return tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True,

        enable_thinking=False,
    )


# ============================================================
# 10. JSON extraction
# ============================================================

def extract_json(
    text,
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


    try:

        return json.loads(
            text[
                start:end + 1
            ]
        )

    except Exception:

        return None


# ============================================================
# 11. Normalize result
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

        "second_label":
            label,

        "second_semantic_consistent":
            as_bool(
                result.get(
                    "semantic_consistent",
                    False,
                )
            ),

        "second_omission":
            as_bool(
                result.get(
                    "omission",
                    False,
                )
            ),

        "second_addition":
            as_bool(
                result.get(
                    "addition",
                    False,
                )
            ),

        "second_mistranslation":
            as_bool(
                result.get(
                    "mistranslation",
                    False,
                )
            ),

        "second_number_error":
            as_bool(
                result.get(
                    "number_error",
                    False,
                )
            ),

        "second_time_error":
            as_bool(
                result.get(
                    "time_error",
                    False,
                )
            ),

        "second_entity_error":
            as_bool(
                result.get(
                    "entity_error",
                    False,
                )
            ),

        "second_negation_error":
            as_bool(
                result.get(
                    "negation_error",
                    False,
                )
            ),

        "second_high_risk":
            as_bool(
                result.get(
                    "high_risk",
                    False,
                )
            ),

        "second_confidence":
            confidence,

        "second_reason":
            str(
                result.get(
                    "reason",
                    "",
                )
            ).strip(),
    }


# ============================================================
# 12. Batch generation
# ============================================================

@torch.inference_mode()
def generate_batch(
    batch_df,
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

        max_length=
            MAX_INPUT_LENGTH,
    )


    inputs = {

        key:
            value.to(
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


    torch.cuda.synchronize()

    start = (
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


    latency = (
        time.perf_counter()
        -
        start
    )


    generated = outputs[
        :,
        prompt_length:
    ]


    responses = (
        tokenizer.batch_decode(

            generated,

            skip_special_tokens=True,
        )
    )


    return (
        responses,
        latency,
    )


# ============================================================
# 13. JSON repair
# ============================================================

@torch.inference_mode()
def repair_json(
    response,
):

    prompt = f"""
Convert the following output into one valid JSON object.

Do not change the intended judgement.

Return JSON only.

OUTPUT:

{response}
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

        max_length=1024,
    )


    inputs = {

        key:
            value.to(
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


    return tokenizer.decode(

        generated,

        skip_special_tokens=True,

    ).strip()


# ============================================================
# 14. Default uncertain
# ============================================================

def uncertain_result(
    raw_response,
    latency,
):

    return {

        "second_label":
            "UNCERTAIN",

        "second_semantic_consistent":
            False,

        "second_omission":
            False,

        "second_addition":
            False,

        "second_mistranslation":
            False,

        "second_number_error":
            False,

        "second_time_error":
            False,

        "second_entity_error":
            False,

        "second_negation_error":
            False,

        "second_high_risk":
            False,

        "second_confidence":
            0.0,

        "second_reason":
            "Second review output could not be parsed.",

        "second_parse_success":
            False,

        "second_latency":
            latency,

        "second_raw_response":
            raw_response,
    }


# ============================================================
# 15. Resolution logic
# ============================================================

def resolve_result(
    first_label,
    second_label,
):

    first_label = (
        str(first_label)
        .upper()
        .strip()
    )

    second_label = (
        str(second_label)
        .upper()
        .strip()
    )


    # ========================================================
    # First FAIL
    # ========================================================

    if first_label == "FAIL":

        if second_label == "FAIL":

            return "CONFIRMED_FAIL"


        if second_label == "PASS":

            return "JUDGE_ERROR"


        if second_label == "MINOR":

            return "DOWNGRADED_MINOR"


        return "QUARANTINE"


    # ========================================================
    # First UNCERTAIN
    # ========================================================

    if first_label == "UNCERTAIN":

        if second_label == "PASS":

            return "RESOLVED_PASS"


        if second_label == "MINOR":

            return "RESOLVED_MINOR"


        if second_label == "FAIL":

            return "CONFIRMED_FAIL"


        return "QUARANTINE"


    return "QUARANTINE"


# ============================================================
# 16. Checkpoint helpers
# ============================================================

def append_checkpoint(
    rows,
):

    with open(

        CHECKPOINT_FILE,

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


def load_checkpoint():

    if not CHECKPOINT_FILE.exists():

        return []


    rows = []


    with open(

        CHECKPOINT_FILE,

        "r",

        encoding="utf-8",

    ) as file:

        for line in file:

            line = (
                line.strip()
            )

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
                    "[WARNING] Broken checkpoint line skipped."
                )


    return rows


# ============================================================
# 17. Resume
# ============================================================

old_rows = (
    load_checkpoint()
)


old_map = {

    str(
        row[
            "review_id"
        ]
    ):
        row

    for row
    in old_rows
}


completed_ids = set(
    old_map.keys()
)


pending_df = problem_df[
    ~problem_df[
        "review_id"
    ]
    .astype(str)
    .isin(
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
    "\nAlready completed:",
    len(
        completed_ids
    )
)

print(
    "Pending:",
    len(
        pending_df
    )
)


# ============================================================
# 18. Main loop
# ============================================================

print("\n")
print("=" * 100)
print("START SECOND REVIEW")
print("=" * 100)


current_batch_size = (
    BATCH_SIZE
)

processed = 0

batch_index = 0

start_all = (
    time.perf_counter()
)


while processed < len(
    pending_df
):

    batch_index += 1


    end = min(

        processed
        +
        current_batch_size,

        len(
            pending_df
        ),
    )


    batch_df = (
        pending_df
        .iloc[
            processed:end
        ]
        .copy()
    )


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
            "\nCUDA OOM"
        )

        print(
            "Reduce batch size to:",
            current_batch_size
        )

        continue


    per_sample_latency = (

        batch_latency
        /
        len(batch_df)
    )


    batch_results = []


    for (
        (_, row),
        response,
    ) in zip(

        batch_df.iterrows(),
        responses,
    ):

        parsed = (
            extract_json(
                response
            )
        )


        normalized = (
            normalize_result(
                parsed
            )
        )


        # ====================================================
        # Optional repair
        # ====================================================

        if (
            normalized is None
            and
            ENABLE_JSON_REPAIR
        ):

            repaired = (
                repair_json(
                    response
                )
            )


            parsed = (
                extract_json(
                    repaired
                )
            )


            normalized = (
                normalize_result(
                    parsed
                )
            )


            if normalized is not None:

                response = (
                    repaired
                )


        # ====================================================
        # Parse failed
        # ====================================================

        if normalized is None:

            second = (
                uncertain_result(
                    raw_response=
                        response,

                    latency=
                        per_sample_latency,
                )
            )


        else:

            second = {

                **normalized,

                "second_parse_success":
                    True,

                "second_latency":
                    per_sample_latency,

                "second_raw_response":
                    response,
            }


        resolution = (
            resolve_result(

                first_label=
                    row[
                        "first_judge_label"
                    ],

                second_label=
                    second[
                        "second_label"
                    ],
            )
        )


        output_row = {

            **row.to_dict(),

            **second,

            "resolution":
                resolution,
        }


        batch_results.append(
            output_row
        )


    append_checkpoint(
        batch_results
    )


    processed += len(
        batch_df
    )


    # ========================================================
    # Progress
    # ========================================================

    elapsed = (
        time.perf_counter()
        -
        start_all
    )


    speed = (

        processed
        /
        elapsed

        if elapsed > 0

        else 0
    )


    remaining = (
        len(
            pending_df
        )
        -
        processed
    )


    eta = (

        remaining
        /
        speed

        if speed > 0

        else 0
    )


    if (
        batch_index
        %
        PRINT_EVERY_BATCHES
        ==
        0
        or
        processed
        ==
        len(
            pending_df
        )
    ):

        print()

        print(
            f"{processed}/"
            f"{len(pending_df)}"
        )

        print(
            "Batch size:",
            len(batch_df)
        )

        print(
            "Batch time:",
            f"{batch_latency:.2f}s"
        )

        print(
            "Avg/sample:",
            f"{per_sample_latency:.2f}s"
        )

        print(
            "Throughput:",
            f"{speed:.3f} samples/s"
        )

        print(
            "ETA:",
            f"{eta / 60:.1f} min"
        )

        print(
            "Last resolutions:",
            [
                row[
                    "resolution"
                ]
                for row
                in batch_results
            ]
        )


# ============================================================
# 19. Build final dataframe
# ============================================================

final_rows = (
    load_checkpoint()
)


final_map = {

    str(
        row[
            "review_id"
        ]
    ):
        row

    for row
    in final_rows
}


result_df = pd.DataFrame(
    list(
        final_map.values()
    )
)


# ============================================================
# 20. Save details
# ============================================================

result_df.to_parquet(
    DETAIL_FILE,
    index=False,
)


result_df.to_csv(
    DETAIL_CSV,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 21. Split by resolution
# ============================================================

confirmed_fail_df = result_df[
    result_df[
        "resolution"
    ]
    ==
    "CONFIRMED_FAIL"
].copy()


recovered_df = result_df[
    result_df[
        "resolution"
    ]
    .isin(
        [
            "JUDGE_ERROR",
            "DOWNGRADED_MINOR",
            "RESOLVED_PASS",
            "RESOLVED_MINOR",
        ]
    )
].copy()


quarantine_df = result_df[
    result_df[
        "resolution"
    ]
    ==
    "QUARANTINE"
].copy()


confirmed_fail_df.to_parquet(
    CONFIRMED_FAIL_FILE,
    index=False,
)


recovered_df.to_parquet(
    RECOVERED_FILE,
    index=False,
)


quarantine_df.to_parquet(
    QUARANTINE_FILE,
    index=False,
)


# ============================================================
# 22. Summary
# ============================================================

resolution_counts = (
    result_df[
        "resolution"
    ]
    .value_counts()
)


summary_df = (
    resolution_counts
    .rename_axis(
        "resolution"
    )
    .reset_index(
        name="count"
    )
)


summary_df[
    "percent"
] = (
    summary_df[
        "count"
    ]
    /
    len(result_df)
    *
    100
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 23. First -> Second cross table
# ============================================================

cross_table = pd.crosstab(

    result_df[
        "first_judge_label"
    ],

    result_df[
        "second_label"
    ],
)


# ============================================================
# 24. Error types
# ============================================================

confirmed_error_stats = {

    "omission":
        int(
            confirmed_fail_df[
                "second_omission"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "addition":
        int(
            confirmed_fail_df[
                "second_addition"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "mistranslation":
        int(
            confirmed_fail_df[
                "second_mistranslation"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "number_error":
        int(
            confirmed_fail_df[
                "second_number_error"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "time_error":
        int(
            confirmed_fail_df[
                "second_time_error"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "entity_error":
        int(
            confirmed_fail_df[
                "second_entity_error"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

    "negation_error":
        int(
            confirmed_fail_df[
                "second_negation_error"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),
}


# ============================================================
# 25. JSON report
# ============================================================

report = {

    "pipeline":
        "en_uz",

    "step":
        "05c_second_review",

    "judge_model":
        "Qwen3-8B",

    "first_pass_problem_cases":
        len(
            problem_df
        ),

    "second_reviewed":
        len(
            result_df
        ),

    "confirmed_fail":
        len(
            confirmed_fail_df
        ),

    "recovered":
        len(
            recovered_df
        ),

    "quarantine":
        len(
            quarantine_df
        ),

    "resolution_distribution":
        summary_df.to_dict(
            orient="records"
        ),

    "confirmed_error_types":
        confirmed_error_stats,
}


with open(
    SUMMARY_JSON,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 26. Terminal output
# ============================================================

print("\n")
print("=" * 110)
print("SECOND REVIEW RESULT")
print("=" * 110)


print(
    "\nFirst-pass problem cases:",
    len(
        problem_df
    )
)

print(
    "Second reviewed:",
    len(
        result_df
    )
)


print(
    "\nResolution:"
)

print(
    summary_df
    .round(2)
    .to_string(
        index=False
    )
)


print(
    "\nFirst -> Second:"
)

print(
    cross_table.to_string()
)


print(
    "\nConfirmed error types:"
)

for key, value in (
    confirmed_error_stats.items()
):

    print(
        f"{key:<20}: "
        f"{value}"
    )


print("\nFinal:")

print(
    "CONFIRMED_FAIL:",
    len(
        confirmed_fail_df
    )
)

print(
    "RECOVERED:",
    len(
        recovered_df
    )
)

print(
    "QUARANTINE:",
    len(
        quarantine_df
    )
)


print(
    "\nFiles:"
)

print(
    DETAIL_FILE
)

print(
    SUMMARY_FILE
)

print(
    CONFIRMED_FAIL_FILE
)

print(
    RECOVERED_FILE
)

print(
    QUARANTINE_FILE
)

print(
    SUMMARY_JSON
)


print(
    "\nDone."
)