from pathlib import Path
import json

import numpy as np
import pandas as pd
import sacrebleu


# ============================================================
# 1. 配置
# ============================================================

PROJECT_ROOT = Path(
    "/root/autodl-tmp/fourlang_translation"
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
    / "stability"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 42
NUM_GROUPS = 5


# ============================================================
# 2. 加载数据
# ============================================================

print("=" * 80)
print("MADLAD-400-3B Teacher Stability Analysis")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到文件：{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nRows:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# 3. 检查字段
# ============================================================

required_columns = [
    "en",
    "uz",
    "uz_en_prediction",
    "en_uz_prediction_latin",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"缺少字段：{missing_columns}\n"
        f"当前字段：{df.columns.tolist()}"
    )


# ============================================================
# 4. 清洗
# ============================================================

before_clean = len(df)

df = df.dropna(
    subset=required_columns
).copy()

for col in required_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )

df = df[
    (df["en"] != "")
    & (df["uz"] != "")
    & (df["uz_en_prediction"] != "")
    & (df["en_uz_prediction_latin"] != "")
].copy()

df = df.reset_index(drop=True)

print("\n原始样本数:", before_clean)
print("有效样本数:", len(df))
print("被过滤数量:", before_clean - len(df))

if len(df) < NUM_GROUPS:
    raise ValueError(
        "样本数量不足，无法进行五组稳定性分析。"
    )


# ============================================================
# 5. 固定随机种子打乱
# ============================================================

shuffled_df = (
    df.sample(
        frac=1.0,
        random_state=SEED,
    )
    .reset_index(drop=True)
)

shuffled_file = (
    OUTPUT_DIR
    / "stability_order.csv"
)

shuffled_df.to_csv(
    shuffled_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. 分成五组
#
# 关键修复：
# 先切分行号，再 iloc，
# 保证 group_df 始终是 pandas DataFrame。
# ============================================================

index_groups = np.array_split(
    np.arange(len(shuffled_df)),
    NUM_GROUPS,
)

groups = [
    shuffled_df.iloc[index_group].copy()
    for index_group in index_groups
]

print("\n分组情况:")

for i, group_df in enumerate(
    groups,
    start=1,
):
    print(
        f"Group {i}: {len(group_df)} samples"
    )


# ============================================================
# 7. 指标函数
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
        "bleu": float(bleu),
        "chrf++": float(chrf),
    }


# ============================================================
# 8. 五组计算
# ============================================================

group_results = []

print("\n")
print("=" * 80)
print("GROUP RESULTS")
print("=" * 80)

for group_id, group_df in enumerate(
    groups,
    start=1,
):

    # UZ -> EN
    uz_en_metrics = calculate_metrics(
        predictions=group_df[
            "uz_en_prediction"
        ].tolist(),

        references=group_df[
            "en"
        ].tolist(),
    )

    # EN -> UZ
    en_uz_metrics = calculate_metrics(
        predictions=group_df[
            "en_uz_prediction_latin"
        ].tolist(),

        references=group_df[
            "uz"
        ].tolist(),
    )

    row = {
        "group": group_id,
        "size": len(group_df),

        "uz_en_bleu":
            uz_en_metrics["bleu"],

        "uz_en_chrf++":
            uz_en_metrics["chrf++"],

        "en_uz_bleu":
            en_uz_metrics["bleu"],

        "en_uz_chrf++":
            en_uz_metrics["chrf++"],
    }

    group_results.append(row)

    print()
    print(f"Group {group_id}")
    print("-" * 50)

    print(
        f"UZ -> EN BLEU : "
        f"{row['uz_en_bleu']:.4f}"
    )

    print(
        f"UZ -> EN chrF++: "
        f"{row['uz_en_chrf++']:.4f}"
    )

    print(
        f"EN -> UZ BLEU : "
        f"{row['en_uz_bleu']:.4f}"
    )

    print(
        f"EN -> UZ chrF++: "
        f"{row['en_uz_chrf++']:.4f}"
    )


group_results_df = pd.DataFrame(
    group_results
)


# ============================================================
# 9. 保存五组指标
# ============================================================

group_result_file = (
    OUTPUT_DIR
    / "group_metrics.csv"
)

group_results_df.to_csv(
    group_result_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 10. 汇总统计
# ============================================================

metric_columns = [
    "uz_en_bleu",
    "uz_en_chrf++",
    "en_uz_bleu",
    "en_uz_chrf++",
]

summary_rows = []

for metric in metric_columns:

    values = (
        group_results_df[metric]
        .astype(float)
        .to_numpy()
    )

    mean_value = float(
        np.mean(values)
    )

    std_value = float(
        np.std(
            values,
            ddof=1,
        )
    )

    min_value = float(
        np.min(values)
    )

    max_value = float(
        np.max(values)
    )

    range_value = (
        max_value
        - min_value
    )

    if mean_value != 0:
        cv = (
            std_value
            / mean_value
            * 100
        )
    else:
        cv = np.nan

    summary_rows.append({
        "metric": metric,
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "range": range_value,
        "cv_percent": cv,
    })


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 11. 稳定性等级
# ============================================================

def stability_level(std_value):

    if std_value < 2:
        return "VERY_STABLE"

    if std_value < 3:
        return "STABLE"

    if std_value < 5:
        return "MODERATE"

    return "UNSTABLE"


summary_df["stability"] = (
    summary_df["std"]
    .apply(stability_level)
)


# ============================================================
# 12. 输出完整 Summary
# ============================================================

print("\n")
print("=" * 80)
print("STABILITY SUMMARY")
print("=" * 80)

for _, row in summary_df.iterrows():

    print()
    print(row["metric"])
    print("-" * 50)

    print(
        f"Mean      : {row['mean']:.4f}"
    )

    print(
        f"Std       : {row['std']:.4f}"
    )

    print(
        f"Min       : {row['min']:.4f}"
    )

    print(
        f"Max       : {row['max']:.4f}"
    )

    print(
        f"Range     : {row['range']:.4f}"
    )

    print(
        f"CV        : "
        f"{row['cv_percent']:.2f}%"
    )

    print(
        f"Stability : "
        f"{row['stability']}"
    )


# ============================================================
# 13. 两个方向最终判断
# ============================================================

uz_en_row = summary_df[
    summary_df["metric"]
    ==
    "uz_en_chrf++"
].iloc[0]

en_uz_row = summary_df[
    summary_df["metric"]
    ==
    "en_uz_chrf++"
].iloc[0]


print("\n")
print("=" * 80)
print("FINAL STABILITY DECISION")
print("=" * 80)

print()

print("UZ -> EN")
print("-" * 30)

print(
    f"chrF++ mean : "
    f"{uz_en_row['mean']:.4f}"
)

print(
    f"chrF++ std  : "
    f"{uz_en_row['std']:.4f}"
)

print(
    f"chrF++ range: "
    f"{uz_en_row['range']:.4f}"
)

print(
    f"Stability   : "
    f"{uz_en_row['stability']}"
)


print()

print("EN -> UZ")
print("-" * 30)

print(
    f"chrF++ mean : "
    f"{en_uz_row['mean']:.4f}"
)

print(
    f"chrF++ std  : "
    f"{en_uz_row['std']:.4f}"
)

print(
    f"chrF++ range: "
    f"{en_uz_row['range']:.4f}"
)

print(
    f"Stability   : "
    f"{en_uz_row['stability']}"
)


# ============================================================
# 14. 保存 Summary
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "stability_summary.csv"
)

summary_df.to_csv(
    summary_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. JSON
# ============================================================

final_summary = {
    "experiment":
        "MADLAD-400-3B Tatoeba Stability",

    "seed":
        SEED,

    "num_samples":
        len(shuffled_df),

    "num_groups":
        NUM_GROUPS,

    "group_sizes":
        [
            len(group)
            for group in groups
        ],

    "uz_en": {
        "chrf_mean":
            float(uz_en_row["mean"]),

        "chrf_std":
            float(uz_en_row["std"]),

        "chrf_min":
            float(uz_en_row["min"]),

        "chrf_max":
            float(uz_en_row["max"]),

        "chrf_range":
            float(uz_en_row["range"]),

        "stability":
            uz_en_row["stability"],
    },

    "en_uz": {
        "chrf_mean":
            float(en_uz_row["mean"]),

        "chrf_std":
            float(en_uz_row["std"]),

        "chrf_min":
            float(en_uz_row["min"]),

        "chrf_max":
            float(en_uz_row["max"]),

        "chrf_range":
            float(en_uz_row["range"]),

        "stability":
            en_uz_row["stability"],
    },
}


json_file = (
    OUTPUT_DIR
    / "stability_summary.json"
)

with open(
    json_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        final_summary,
        f,
        ensure_ascii=False,
        indent=2,
    )


print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print("Groups :", group_result_file)
print("Summary:", summary_file)
print("JSON   :", json_file)

print("\nDone.")