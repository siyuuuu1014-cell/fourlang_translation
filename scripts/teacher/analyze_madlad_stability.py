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
# 3. 检查所需字段
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
        "缺少以下字段："
        f"{missing_columns}\n"
        f"当前字段：{df.columns.tolist()}"
    )


# ============================================================
# 4. 基础清洗
# ============================================================

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

print("\n有效样本数:", len(df))

if len(df) < NUM_GROUPS:
    raise ValueError(
        "有效样本太少，无法分成 5 组。"
    )


# ============================================================
# 5. 固定随机种子重新打乱
#
# 原因：
# 避免原 CSV 中的数据排序影响 Group1~Group5 的结果。
# random_state 固定以后，实验可复现。
# ============================================================

shuffled_df = (
    df.sample(
        frac=1.0,
        random_state=SEED,
    )
    .reset_index(drop=True)
)

# 保存本次稳定性实验实际使用的数据顺序
shuffled_file = (
    OUTPUT_DIR
    / "tatoeba_500_stability_order.csv"
)

shuffled_df.to_csv(
    shuffled_file,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 6. 分成 5 组
#
# np.array_split 可以保证即使不是严格500条，
# 也会尽量均匀分组。
# ============================================================

groups = np.array_split(
    shuffled_df,
    NUM_GROUPS,
)

print("\n分组情况:")

for i, group in enumerate(
    groups,
    start=1,
):
    print(
        f"Group {i}: {len(group)} samples"
    )


# ============================================================
# 7. 单方向指标计算函数
# ============================================================

def calculate_metrics(
    predictions,
    references,
):
    """
    计算 corpus-level BLEU 和 chrF++。
    """

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
# 8. 计算 5 组 UZ -> EN 和 EN -> UZ
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

    # --------------------------------------------------------
    # UZ -> EN
    # --------------------------------------------------------

    uz_en_metrics = calculate_metrics(
        predictions=group_df[
            "uz_en_prediction"
        ].tolist(),

        references=group_df[
            "en"
        ].tolist(),
    )

    # --------------------------------------------------------
    # EN -> UZ
    # 注意：
    # 使用 Latin 转写后的 prediction
    # --------------------------------------------------------

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
    print("-" * 40)

    print(
        "UZ -> EN BLEU :",
        f"{row['uz_en_bleu']:.4f}"
    )

    print(
        "UZ -> EN chrF++:",
        f"{row['uz_en_chrf++']:.4f}"
    )

    print(
        "EN -> UZ BLEU :",
        f"{row['en_uz_bleu']:.4f}"
    )

    print(
        "EN -> UZ chrF++:",
        f"{row['en_uz_chrf++']:.4f}"
    )


group_results_df = pd.DataFrame(
    group_results
)


# ============================================================
# 9. 保存每组结果
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
# 10. 统计 mean / std / min / max / range / CV
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
        .values
    )

    mean_value = float(
        np.mean(values)
    )

    # ddof=1:
    # 使用样本标准差，与 pandas .std() 一致
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
        max_value - min_value
    )

    if mean_value != 0:
        cv = (
            std_value
            /
            mean_value
            *
            100
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
# 11. 稳定性判断
#
# 主要根据 chrF++ 的 5组标准差判断。
#
# < 2   非常稳定
# 2~3   稳定
# 3~5   一般，需要检查
# > 5   不稳定
# ============================================================

def stability_level(std_value):

    if std_value < 2:
        return "VERY_STABLE"

    elif std_value < 3:
        return "STABLE"

    elif std_value < 5:
        return "MODERATE"

    else:
        return "UNSTABLE"


summary_df["stability"] = (
    summary_df["std"]
    .apply(stability_level)
)


# ============================================================
# 12. 输出 Summary
# ============================================================

print("\n")
print("=" * 80)
print("STABILITY SUMMARY")
print("=" * 80)

for _, row in summary_df.iterrows():

    print()
    print(row["metric"])
    print("-" * 40)

    print(
        f"Mean       : {row['mean']:.4f}"
    )

    print(
        f"Std        : {row['std']:.4f}"
    )

    print(
        f"Min        : {row['min']:.4f}"
    )

    print(
        f"Max        : {row['max']:.4f}"
    )

    print(
        f"Range      : {row['range']:.4f}"
    )

    print(
        f"CV         : "
        f"{row['cv_percent']:.2f}%"
    )

    print(
        "Stability  :",
        row["stability"]
    )


# ============================================================
# 13. 专门输出两个方向 chrF++ 判断
# ============================================================

uz_en_chrf_row = summary_df[
    summary_df["metric"]
    ==
    "uz_en_chrf++"
].iloc[0]

en_uz_chrf_row = summary_df[
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
print(
    "chrF++ mean:",
    f"{uz_en_chrf_row['mean']:.4f}"
)

print(
    "chrF++ std :",
    f"{uz_en_chrf_row['std']:.4f}"
)

print(
    "判断:",
    uz_en_chrf_row[
        "stability"
    ]
)

print()

print("EN -> UZ")
print(
    "chrF++ mean:",
    f"{en_uz_chrf_row['mean']:.4f}"
)

print(
    "chrF++ std :",
    f"{en_uz_chrf_row['std']:.4f}"
)

print(
    "判断:",
    en_uz_chrf_row[
        "stability"
    ]
)


# ============================================================
# 14. 保存 summary CSV
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
# 15. 保存 JSON Summary
# ============================================================

final_summary = {
    "experiment":
        "MADLAD-400-3B Tatoeba 500 Stability",

    "seed":
        SEED,

    "num_samples":
        len(shuffled_df),

    "num_groups":
        NUM_GROUPS,

    "uz_en": {
        "chrf_mean":
            float(
                uz_en_chrf_row[
                    "mean"
                ]
            ),

        "chrf_std":
            float(
                uz_en_chrf_row[
                    "std"
                ]
            ),

        "chrf_min":
            float(
                uz_en_chrf_row[
                    "min"
                ]
            ),

        "chrf_max":
            float(
                uz_en_chrf_row[
                    "max"
                ]
            ),

        "stability":
            uz_en_chrf_row[
                "stability"
            ],
    },

    "en_uz": {
        "chrf_mean":
            float(
                en_uz_chrf_row[
                    "mean"
                ]
            ),

        "chrf_std":
            float(
                en_uz_chrf_row[
                    "std"
                ]
            ),

        "chrf_min":
            float(
                en_uz_chrf_row[
                    "min"
                ]
            ),

        "chrf_max":
            float(
                en_uz_chrf_row[
                    "max"
                ]
            ),

        "stability":
            en_uz_chrf_row[
                "stability"
            ],
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


# ============================================================
# 16. 完成
# ============================================================

print("\n")
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "Group metrics:",
    group_result_file
)

print(
    "Summary:",
    summary_file
)

print(
    "JSON:",
    json_file
)

print(
    "Shuffled benchmark:",
    shuffled_file
)

print("\nDone.")