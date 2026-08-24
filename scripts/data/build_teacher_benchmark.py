from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\dev\projects\fourlang_translation")

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "en_uz"
    / "tatoeba_en_uz_latin.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "en_uz"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

print("原始列：", df.columns.tolist())
print("原始数量：", len(df))

# 兼容你之前不同版本的字段名
if {"en", "uz"}.issubset(df.columns):
    df = df[["en", "uz"]].copy()

elif {"src_en", "tgt_uz"}.issubset(df.columns):
    df = df[["src_en", "tgt_uz"]].copy()
    df.columns = ["en", "uz"]

else:
    raise ValueError(
        f"无法识别字段：{df.columns.tolist()}"
    )

df = df.dropna()
df["en"] = df["en"].astype(str).str.strip()
df["uz"] = df["uz"].astype(str).str.strip()

df = df[
    (df["en"] != "")
    &
    (df["uz"] != "")
].drop_duplicates()

benchmark_df = df.sample(
    n=min(500, len(df)),
    random_state=42
).reset_index(drop=True)

output_file = OUTPUT_DIR / "tatoeba_en_uz_500.csv"

benchmark_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("Benchmark 数量：", len(benchmark_df))
print("保存到：", output_file)
print(benchmark_df.head())