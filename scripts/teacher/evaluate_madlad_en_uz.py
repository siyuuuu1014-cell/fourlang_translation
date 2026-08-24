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
# Config
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
)

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "en_uz"
    / "tatoeba_en_uz_500.csv"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "teacher"
    / "madlad400_3b"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_NAME = "google/madlad400-3b-mt"


# ============================================================
# Environment
# ============================================================

print("=" * 80)
print("Environment")
print("=" * 80)

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# Load
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    local_files_only=True,
)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    local_files_only=True,
)

model.eval()

print("Model loaded.")


# ============================================================
# Uzbek Cyrillic -> Latin normalization
#
# 注意：
# 这个主要用于 EN->UZ 指标归一化，
# 不是最终产品级 transliterator。
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


def cyrillic_to_latin(text: str) -> str:
    return "".join(
        CYR_TO_LAT.get(ch, ch)
        for ch in text
    )


# ============================================================
# Translation
# ============================================================

LANG_TOKEN = {
    "en": "<2en>",
    "uz": "<2uz>",
}


@torch.inference_mode()
def translate(
    text: str,
    tgt_lang: str,
    num_beams: int = 5,
    max_new_tokens: int = 128,
):
    prompt = (
        f"{LANG_TOKEN[tgt_lang]} {text}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    input_device = (
        model.encoder.embed_tokens.weight.device
    )

    inputs = {
        k: v.to(input_device)
        for k, v in inputs.items()
    }

    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(
        **inputs,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        early_stopping=True,
    )

    torch.cuda.synchronize()
    latency = (
        time.perf_counter()
        - start
    )

    prediction = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return prediction, latency


# ============================================================
# Dataset
# ============================================================

df = pd.read_csv(
    BENCHMARK_FILE
)

print("\nBenchmark:", len(df))


# ============================================================
# Warmup
# ============================================================

print("Warming up GPU...")

translate(
    "Men talabaman.",
    "en",
    num_beams=1,
)


# ============================================================
# UZ -> EN
# ============================================================

print("\n" + "=" * 80)
print("UZ -> EN")
print("=" * 80)

uz_en_predictions = []
uz_en_latencies = []

for index, row in df.iterrows():

    prediction, latency = translate(
        row["uz"],
        "en",
    )

    uz_en_predictions.append(
        prediction
    )

    uz_en_latencies.append(
        latency
    )

    if (
        index + 1
    ) % 25 == 0:

        print(
            f"{index + 1}/{len(df)}"
        )


df["uz_en_prediction"] = (
    uz_en_predictions
)

df["uz_en_latency"] = (
    uz_en_latencies
)


# ============================================================
# EN -> UZ
# ============================================================

print("\n" + "=" * 80)
print("EN -> UZ")
print("=" * 80)

en_uz_raw_predictions = []
en_uz_latin_predictions = []
en_uz_latencies = []

for index, row in df.iterrows():

    prediction, latency = translate(
        row["en"],
        "uz",
    )

    latin_prediction = (
        cyrillic_to_latin(
            prediction
        )
    )

    en_uz_raw_predictions.append(
        prediction
    )

    en_uz_latin_predictions.append(
        latin_prediction
    )

    en_uz_latencies.append(
        latency
    )

    if (
        index + 1
    ) % 25 == 0:

        print(
            f"{index + 1}/{len(df)}"
        )


df["en_uz_prediction_raw"] = (
    en_uz_raw_predictions
)

df["en_uz_prediction_latin"] = (
    en_uz_latin_predictions
)

df["en_uz_latency"] = (
    en_uz_latencies
)


# ============================================================
# Metrics
# ============================================================

references_en = (
    df["en"].tolist()
)

references_uz = (
    df["uz"].tolist()
)


uz_en_bleu = (
    sacrebleu.corpus_bleu(
        uz_en_predictions,
        [references_en],
    )
)

uz_en_chrf = (
    sacrebleu.corpus_chrf(
        uz_en_predictions,
        [references_en],
        word_order=2,
    )
)


en_uz_bleu = (
    sacrebleu.corpus_bleu(
        en_uz_latin_predictions,
        [references_uz],
    )
)

en_uz_chrf = (
    sacrebleu.corpus_chrf(
        en_uz_latin_predictions,
        [references_uz],
        word_order=2,
    )
)


print("\n")
print("=" * 80)
print("RESULT")
print("=" * 80)

print(
    "UZ -> EN BLEU:",
    round(
        uz_en_bleu.score,
        4
    )
)

print(
    "UZ -> EN chrF++:",
    round(
        uz_en_chrf.score,
        4
    )
)

print()

print(
    "EN -> UZ BLEU:",
    round(
        en_uz_bleu.score,
        4
    )
)

print(
    "EN -> UZ chrF++:",
    round(
        en_uz_chrf.score,
        4
    )
)

print()

print(
    "UZ -> EN avg latency:",
    round(
        sum(uz_en_latencies)
        /
        len(uz_en_latencies),
        4
    )
)

print(
    "EN -> UZ avg latency:",
    round(
        sum(en_uz_latencies)
        /
        len(en_uz_latencies),
        4
    )
)


# ============================================================
# Save
# ============================================================

OUTPUT_FILE = (
    RESULT_DIR
    / "tatoeba_500_predictions.csv"
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)

print(
    "\nSaved:",
    OUTPUT_FILE
)