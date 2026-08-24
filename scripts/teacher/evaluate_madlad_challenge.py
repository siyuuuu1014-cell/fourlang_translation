from pathlib import Path
import time

import pandas as pd
import torch
import sacrebleu

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# 1. 配置
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

MODEL_NAME = "google/madlad400-3b-mt"

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
    / "madlad400_3b"
    / "challenge_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "challenge_v1_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "challenge_v1_metrics.csv"
)


# ============================================================
# 2. 参数
# ============================================================

NUM_BEAMS = 5

MAX_INPUT_LENGTH = 256

MAX_NEW_TOKENS = 128


# ============================================================
# 3. 环境检查
# ============================================================

print("=" * 80)
print("MADLAD-400-3B Challenge Benchmark")
print("=" * 80)

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
        / 1024 ** 3,
        2,
    ),
    "GB",
)


# ============================================================
# 4. 读取 Challenge
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"找不到 Challenge 文件：\n"
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
        f"缺少字段：{missing}\n"
        f"当前字段：{df.columns.tolist()}"
    )


df = df.dropna(
    subset=[
        "en",
        "uz",
        "category",
    ]
).copy()


for col in [
    "en",
    "uz",
    "category",
]:

    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


df = df.reset_index(
    drop=True
)


print(
    "\nChallenge rows:",
    len(df)
)

print(
    "\nCategory distribution:"
)

print(
    df["category"]
    .value_counts()
)


# ============================================================
# 5. 加载 MADLAD
# ============================================================

print(
    "\nLoading tokenizer..."
)

tokenizer = (
    AutoTokenizer
    .from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
)


print(
    "Loading MADLAD-400-3B..."
)

model = (
    AutoModelForSeq2SeqLM
    .from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
)


model.eval()


print(
    "\nModel loaded."
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
# 6. Uzbek Cyrillic -> Latin
#
# 注意：
# 当前仍然是 Benchmark 实验用转写。
# 后续 Student 数据正式生成时再换更完善版本。
# ============================================================

CYR_TO_LAT = {
    "А": "A", "а": "a",
    "Б": "B", "б": "b",
    "В": "V", "в": "v",
    "Г": "G", "г": "g",
    "Д": "D", "д": "d",
    "Е": "E", "е": "e",
    "Ё": "Yo", "ё": "yo",
    "Ж": "J", "ж": "j",
    "З": "Z", "з": "z",
    "И": "I", "и": "i",
    "Й": "Y", "й": "y",
    "К": "K", "к": "k",
    "Л": "L", "л": "l",
    "М": "M", "м": "m",
    "Н": "N", "н": "n",
    "О": "O", "о": "o",
    "П": "P", "п": "p",
    "Р": "R", "р": "r",
    "С": "S", "с": "s",
    "Т": "T", "т": "t",
    "У": "U", "у": "u",
    "Ф": "F", "ф": "f",
    "Х": "X", "х": "x",
    "Ц": "Ts", "ц": "ts",
    "Ч": "Ch", "ч": "ch",
    "Ш": "Sh", "ш": "sh",
    "Щ": "Sh", "щ": "sh",
    "Ъ": "'", "ъ": "'",
    "Ь": "", "ь": "",
    "Э": "E", "э": "e",
    "Ю": "Yu", "ю": "yu",
    "Я": "Ya", "я": "ya",

    # Uzbek 特有字符
    "Қ": "Q", "қ": "q",
    "Ғ": "G'", "ғ": "g'",
    "Ҳ": "H", "ҳ": "h",
    "Ў": "O'", "ў": "o'",
}


def cyrillic_to_latin(
    text: str,
) -> str:

    return "".join(
        CYR_TO_LAT.get(
            ch,
            ch,
        )
        for ch in str(text)
    )


# ============================================================
# 7. 翻译函数
# ============================================================

LANG_TOKEN = {
    "en": "<2en>",
    "uz": "<2uz>",
}


@torch.inference_mode()
def translate(
    text: str,
    target_language: str,
):

    if target_language not in LANG_TOKEN:

        raise ValueError(
            f"Unsupported language: "
            f"{target_language}"
        )


    prompt = (
        f"{LANG_TOKEN[target_language]} "
        f"{text}"
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )


    # 输入 embedding 所在设备
    input_device = (
        model
        .get_input_embeddings()
        .weight
        .device
    )


    inputs = {
        key: value.to(
            input_device
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
        num_beams=NUM_BEAMS,
        max_new_tokens=MAX_NEW_TOKENS,
        early_stopping=True,
    )


    torch.cuda.synchronize()

    latency = (
        time.perf_counter()
        - start
    )


    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    ).strip()


    return (
        result,
        latency,
    )


# ============================================================
# 8. GPU预热
# ============================================================

print(
    "\nWarming up..."
)

_ = translate(
    "Men talabaman.",
    "en",
)

print(
    "Warmup completed."
)


# ============================================================
# 9. 断点续跑
# ============================================================

if PREDICTION_FILE.exists():

    print(
        "\nExisting prediction file found."
    )

    old_df = pd.read_csv(
        PREDICTION_FILE
    )

    if (
        len(old_df) == len(df)
        and
        "uz_en_prediction"
        in old_df.columns
        and
        "en_uz_prediction_latin"
        in old_df.columns
    ):

        print(
            "Prediction file appears complete."
        )

        prediction_df = old_df

        NEED_INFERENCE = False

    else:

        print(
            "Prediction file incomplete. "
            "Will rerun inference."
        )

        NEED_INFERENCE = True

else:

    NEED_INFERENCE = True


# ============================================================
# 10. 正式翻译
# ============================================================

if NEED_INFERENCE:

    results = []

    total = len(df)


    print("\n")
    print("=" * 80)
    print("START INFERENCE")
    print("=" * 80)

    print(
        "Samples:",
        total
    )

    print(
        "Translations:",
        total * 2
    )


    for index, row in df.iterrows():

        # ----------------------------------------------------
        # UZ -> EN
        # ----------------------------------------------------

        uz_en_prediction, uz_en_latency = (
            translate(
                row["uz"],
                "en",
            )
        )


        # ----------------------------------------------------
        # EN -> UZ
        # ----------------------------------------------------

        en_uz_raw, en_uz_latency = (
            translate(
                row["en"],
                "uz",
            )
        )


        en_uz_latin = (
            cyrillic_to_latin(
                en_uz_raw
            )
        )


        result = {
            **row.to_dict(),

            "uz_en_prediction":
                uz_en_prediction,

            "uz_en_latency":
                uz_en_latency,

            "en_uz_prediction_raw":
                en_uz_raw,

            "en_uz_prediction_latin":
                en_uz_latin,

            "en_uz_latency":
                en_uz_latency,
        }


        results.append(
            result
        )


        # 每10条保存一次
        if (
            (index + 1) % 10 == 0
            or
            (index + 1) == total
        ):

            pd.DataFrame(
                results
            ).to_csv(
                PREDICTION_FILE,
                index=False,
                encoding="utf-8-sig",
            )


            print(
                f"{index + 1}/{total}"
            )


    prediction_df = pd.DataFrame(
        results
    )


print(
    "\nPrediction rows:",
    len(prediction_df)
)


# ============================================================
# 11. 指标函数
# ============================================================

def calculate_metrics(
    predictions,
    references,
):

    predictions = [
        str(x).strip()
        for x in predictions
    ]

    references = [
        str(x).strip()
        for x in references
    ]


    bleu = (
        sacrebleu.corpus_bleu(
            predictions,
            [references],
        ).score
    )


    chrf = (
        sacrebleu.corpus_chrf(
            predictions,
            [references],
            word_order=2,
        ).score
    )


    return {
        "bleu":
            float(bleu),

        "chrf++":
            float(chrf),
    }


# ============================================================
# 12. 整体指标
# ============================================================

metric_rows = []


def add_metric_row(
    category,
    part,
):

    uz_en = calculate_metrics(
        predictions=part[
            "uz_en_prediction"
        ].tolist(),

        references=part[
            "en"
        ].tolist(),
    )


    en_uz = calculate_metrics(
        predictions=part[
            "en_uz_prediction_latin"
        ].tolist(),

        references=part[
            "uz"
        ].tolist(),
    )


    metric_rows.append({
        "category":
            category,

        "size":
            len(part),

        "uz_en_bleu":
            uz_en["bleu"],

        "uz_en_chrf++":
            uz_en["chrf++"],

        "en_uz_bleu":
            en_uz["bleu"],

        "en_uz_chrf++":
            en_uz["chrf++"],

        "uz_en_avg_latency":
            float(
                part[
                    "uz_en_latency"
                ].mean()
            ),

        "en_uz_avg_latency":
            float(
                part[
                    "en_uz_latency"
                ].mean()
            ),
    })


# 总体
add_metric_row(
    "ALL",
    prediction_df,
)


# 每个类别
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
# 13. 保存指标
# ============================================================

metrics_df.to_csv(
    METRICS_FILE,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. Terminal显示
# ============================================================

print("\n")
print("=" * 100)
print("CHALLENGE BENCHMARK RESULT")
print("=" * 100)


for _, row in metrics_df.iterrows():

    print()
    print(
        f"[{row['category']}]"
    )

    print(
        f"Samples            : "
        f"{int(row['size'])}"
    )

    print(
        f"UZ -> EN BLEU      : "
        f"{row['uz_en_bleu']:.4f}"
    )

    print(
        f"UZ -> EN chrF++    : "
        f"{row['uz_en_chrf++']:.4f}"
    )

    print(
        f"EN -> UZ BLEU      : "
        f"{row['en_uz_bleu']:.4f}"
    )

    print(
        f"EN -> UZ chrF++    : "
        f"{row['en_uz_chrf++']:.4f}"
    )

    print(
        f"UZ -> EN latency   : "
        f"{row['uz_en_avg_latency']:.4f}s"
    )

    print(
        f"EN -> UZ latency   : "
        f"{row['en_uz_avg_latency']:.4f}s"
    )


# ============================================================
# 15. 方便观察的表格
# ============================================================

print("\n")
print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)

display_columns = [
    "category",
    "size",
    "uz_en_bleu",
    "uz_en_chrf++",
    "en_uz_bleu",
    "en_uz_chrf++",
]

print(
    metrics_df[
        display_columns
    ].round(4).to_string(
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