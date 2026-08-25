from pathlib import Path
import re
import time
import unicodedata

import pandas as pd
import torch
import sacrebleu

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
    / "data"
    / "benchmark"
    / "en_uz"
    / "challenge_v1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "qwen3_8b"
    / "challenge_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "qwen3_en_uz_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "qwen3_en_uz_metrics.csv"
)

MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 192


# ============================================================
# 2. Environment
# ============================================================

print("=" * 90)
print("Qwen3-8B EN -> UZ Challenge Benchmark")
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
# 3. Input
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Challenge file not found:\n"
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
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Columns: {df.columns.tolist()}"
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
    "\nSamples:",
    len(df),
)

print(
    "\nCategories:"
)

print(
    df["category"]
    .value_counts()
)


# ============================================================
# 4. Load Qwen3
# ============================================================

print(
    "\nLoading tokenizer..."
)

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


# 禁止采样，保证实验可复现
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None


print(
    "\nModel loaded."
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
# 5. Text normalization
# ============================================================

def normalize_uzbek_text(
    text: str,
) -> str:

    text = str(text).strip()

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Apostrophe normalization
    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("ʻ", "'")
        .replace("`", "'")
        .replace("´", "'")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        re.search(
            r"[\u0400-\u04FF]",
            str(text),
        )
    )


# ============================================================
# 6. Translation prompt
# ============================================================

def build_translation_messages(
    english_text: str,
):

    return [
        {
            "role": "system",
            "content": (
                "You are a professional English to Uzbek translator. "
                "Translate accurately into modern standard Uzbek. "
                "Always write Uzbek using the Latin alphabet. "
                "Preserve all numbers, times, dates, names, locations, "
                "negation and factual details exactly. "
                "Do not explain the translation. "
                "Return only the Uzbek translation."
            ),
        },
        {
            "role": "user",
            "content": (
                "Translate the following English text into Uzbek "
                "using the Latin alphabet only:\n\n"
                f"{english_text}"
            ),
        },
    ]


# ============================================================
# 7. Translation
# ============================================================

@torch.inference_mode()
def translate_en_to_uz(
    english_text: str,
):

    messages = build_translation_messages(
        english_text
    )


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
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


    generated = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    translation = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


    translation = normalize_uzbek_text(
        translation
    )


    return (
        translation,
        latency,
    )


# ============================================================
# 8. Warmup
# ============================================================

print(
    "\nWarming up..."
)

warmup_translation, _ = (
    translate_en_to_uz(
        "I am a student."
    )
)

print(
    "Warmup output:",
    warmup_translation
)


# ============================================================
# 9. Resume
# ============================================================

if PREDICTION_FILE.exists():

    old_df = pd.read_csv(
        PREDICTION_FILE
    )

    if "challenge_id" not in old_df.columns:

        completed_ids = set()

        result_rows = []

    else:

        completed_ids = set(
            old_df[
                "challenge_id"
            ].astype(str)
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
        len(completed_ids)
    )

else:

    completed_ids = set()

    result_rows = []


# ============================================================
# 10. Run inference
# ============================================================

print("\n")
print("=" * 90)
print("START QWEN3 EN -> UZ INFERENCE")
print("=" * 90)


total = len(df)


for index, row in df.iterrows():

    challenge_id = str(
        row["challenge_id"]
    )


    if challenge_id in completed_ids:

        continue


    prediction, latency = (
        translate_en_to_uz(
            row["en"]
        )
    )


    result_rows.append({

        **row.to_dict(),

        "qwen3_en_uz_prediction":
            prediction,

        "qwen3_en_uz_latency":
            latency,

        "qwen3_has_cyrillic":
            contains_cyrillic(
                prediction
            ),
    })


    completed_ids.add(
        challenge_id
    )


    # 每条保存，防止中断
    pd.DataFrame(
        result_rows
    ).to_csv(
        PREDICTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    print(
        f"[{index + 1}/{total}] "
        f"{row['category']} | "
        f"{latency:.3f}s"
    )


# ============================================================
# 11. Load complete predictions
# ============================================================

prediction_df = pd.DataFrame(
    result_rows
)


prediction_df[
    "qwen3_en_uz_prediction"
] = (
    prediction_df[
        "qwen3_en_uz_prediction"
    ]
    .astype(str)
    .map(
        normalize_uzbek_text
    )
)


prediction_df["uz_normalized"] = (
    prediction_df["uz"]
    .astype(str)
    .map(
        normalize_uzbek_text
    )
)


print("\n")
print("=" * 90)
print("INFERENCE COMPLETE")
print("=" * 90)

print(
    "Rows:",
    len(prediction_df)
)

print(
    "Cyrillic outputs:",
    int(
        prediction_df[
            "qwen3_has_cyrillic"
        ].sum()
    ),
)


# ============================================================
# 12. Metrics
# ============================================================

def calculate_metrics(
    predictions,
    references,
):

    predictions = [
        normalize_uzbek_text(x)
        for x in predictions
    ]

    references = [
        normalize_uzbek_text(x)
        for x in references
    ]


    bleu = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score


    chrf = sacrebleu.corpus_chrf(
        predictions,
        [references],
        word_order=2,
    ).score


    return {
        "bleu":
            float(bleu),

        "chrf++":
            float(chrf),
    }


# ============================================================
# 13. Overall + per-category
# ============================================================

metric_rows = []


def add_metric_row(
    category,
    part,
):

    metrics = calculate_metrics(

        predictions=part[
            "qwen3_en_uz_prediction"
        ].tolist(),

        references=part[
            "uz_normalized"
        ].tolist(),
    )


    metric_rows.append({

        "category":
            category,

        "size":
            len(part),

        "bleu":
            metrics[
                "bleu"
            ],

        "chrf++":
            metrics[
                "chrf++"
            ],

        "avg_latency":
            float(
                part[
                    "qwen3_en_uz_latency"
                ].mean()
            ),

        "cyrillic_count":
            int(
                part[
                    "qwen3_has_cyrillic"
                ].sum()
            ),

        "cyrillic_rate":
            float(
                part[
                    "qwen3_has_cyrillic"
                ].mean()
                * 100
            ),
    })


# Overall
add_metric_row(
    "ALL",
    prediction_df,
)


# Categories
for category in sorted(
    prediction_df[
        "category"
    ].unique()
):

    part = prediction_df[
        prediction_df[
            "category"
        ]
        ==
        category
    ]


    add_metric_row(
        category,
        part,
    )


metrics_df = pd.DataFrame(
    metric_rows
)


# ============================================================
# 14. Save metrics
# ============================================================

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. Display
# ============================================================

print("\n")
print("=" * 100)
print("QWEN3 EN -> UZ CHALLENGE RESULT")
print("=" * 100)


for _, row in metrics_df.iterrows():

    print()

    print(
        f"[{row['category']}]"
    )

    print(
        f"Samples       : "
        f"{int(row['size'])}"
    )

    print(
        f"BLEU          : "
        f"{row['bleu']:.4f}"
    )

    print(
        f"chrF++        : "
        f"{row['chrf++']:.4f}"
    )

    print(
        f"Avg latency   : "
        f"{row['avg_latency']:.4f}s"
    )

    print(
        f"Cyrillic rate : "
        f"{row['cyrillic_rate']:.2f}%"
    )


print("\n")
print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)


print(
    metrics_df[
        [
            "category",
            "size",
            "bleu",
            "chrf++",
            "avg_latency",
            "cyrillic_rate",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(
    "Predictions:",
    PREDICTION_FILE
)

print(
    "Metrics:",
    METRICS_FILE
)

print(
    "\nDone."
)