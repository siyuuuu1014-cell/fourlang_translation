from __future__ import annotations

from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# 1. 项目配置
# ============================================================

from pathlib import Path

# 当前脚本：
# D:\dev\projects\fourlang_translation\scripts\data\build_en_uz_challenge_benchmark.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "en_uz"
)

BENCHMARK_DIR = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "en_uz"
)

OUTPUT_DIR = BENCHMARK_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "challenge_v1.csv"
)

REPORT_FILE = (
    OUTPUT_DIR
    / "challenge_v1_report.csv"
)

print("PROJECT_ROOT =", PROJECT_ROOT)
print("CLEAN_DIR    =", CLEAN_DIR)
print("CLEAN exists =", CLEAN_DIR.exists())
# ============================================================
# 2. 实验参数
# ============================================================

SEED = 2026

SAMPLES_PER_CATEGORY = 50

CATEGORIES = [
    "normal",
    "negation",
    "number",
    "time_date",
    "entity",
    "long",
]

TARGET_TOTAL = (
    SAMPLES_PER_CATEGORY
    * len(CATEGORIES)
)


# ============================================================
# 3. 基础文本函数
# ============================================================

def normalize_text(text: str) -> str:
    """
    用于去重和比较。

    不修改最终保存文本，只产生规范化 key。
    """

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def pair_key(
    en: str,
    uz: str,
) -> tuple[str, str]:

    return (
        normalize_text(en).lower(),
        normalize_text(uz).lower(),
    )


def contains_cyrillic(
    text: str,
) -> bool:

    return bool(
        re.search(
            r"[\u0400-\u04FF]",
            str(text),
        )
    )


def word_count(
    text: str,
) -> int:

    return len(
        re.findall(
            r"\b[\w'’\-]+\b",
            str(text),
            flags=re.UNICODE,
        )
    )


# ============================================================
# 4. 自动识别 EN / UZ 字段
# ============================================================

EN_COLUMN_CANDIDATES = [
    "en",
    "english",
    "src_en",
    "source_en",
    "en_text",
    "text_en",
]

UZ_COLUMN_CANDIDATES = [
    "uz",
    "uzbek",
    "tgt_uz",
    "target_uz",
    "uz_text",
    "text_uz",
]


def find_column(
    columns,
    candidates,
):

    lower_map = {
        str(col).lower().strip(): col
        for col in columns
    }

    for candidate in candidates:

        if candidate in lower_map:

            return lower_map[candidate]

    return None


# ============================================================
# 5. 读取单个数据文件
# ============================================================

def read_parallel_file(
    file_path: Path,
) -> pd.DataFrame | None:

    suffix = file_path.suffix.lower()

    try:

        if suffix == ".csv":

            df = pd.read_csv(
                file_path,
                low_memory=False,
            )

        elif suffix in {
            ".tsv",
            ".txt",
        }:

            df = pd.read_csv(
                file_path,
                sep="\t",
                low_memory=False,
            )

        elif suffix == ".parquet":

            df = pd.read_parquet(
                file_path
            )

        else:

            return None

    except Exception as exc:

        print(
            f"[SKIP] 读取失败：{file_path}"
        )

        print(
            "       ",
            repr(exc),
        )

        return None


    en_col = find_column(
        df.columns,
        EN_COLUMN_CANDIDATES,
    )

    uz_col = find_column(
        df.columns,
        UZ_COLUMN_CANDIDATES,
    )


    if (
        en_col is None
        or
        uz_col is None
    ):

        print(
            f"[SKIP] 未找到 en/uz 字段："
            f"{file_path}"
        )

        print(
            "       columns =",
            df.columns.tolist(),
        )

        return None


    result = df[
        [
            en_col,
            uz_col,
        ]
    ].copy()

    result.columns = [
        "en",
        "uz",
    ]

    result["source_file"] = str(
        file_path.relative_to(
            PROJECT_ROOT
        )
    )

    return result


# ============================================================
# 6. 加载 clean/en_uz 下所有可用数据
# ============================================================

def load_all_clean_data() -> pd.DataFrame:

    if not CLEAN_DIR.exists():

        raise FileNotFoundError(
            f"找不到目录：{CLEAN_DIR}"
        )


    files = []

    for pattern in [
        "**/*.csv",
        "**/*.tsv",
        "**/*.parquet",
    ]:

        files.extend(
            CLEAN_DIR.glob(pattern)
        )


    files = sorted(
        set(files)
    )


    print("=" * 80)
    print("扫描清洗数据")
    print("=" * 80)

    print(
        "找到文件数量：",
        len(files),
    )


    all_frames = []


    for file_path in files:

        print(
            "\n读取：",
            file_path.relative_to(
                PROJECT_ROOT
            ),
        )

        part = read_parallel_file(
            file_path
        )

        if part is None:
            continue

        print(
            "有效原始行：",
            len(part),
        )

        all_frames.append(
            part
        )


    if not all_frames:

        raise RuntimeError(
            "没有找到包含 en / uz 字段的清洗数据。"
        )


    df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    return df


# ============================================================
# 7. 基础清洗
# ============================================================

def clean_parallel_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print("基础清洗")
    print("=" * 80)

    print(
        "合并后数量：",
        len(df),
    )


    df = df.dropna(
        subset=[
            "en",
            "uz",
        ]
    ).copy()


    df["en"] = (
        df["en"]
        .astype(str)
        .map(normalize_text)
    )

    df["uz"] = (
        df["uz"]
        .astype(str)
        .map(normalize_text)
    )


    # 空文本
    df = df[
        (df["en"] != "")
        &
        (df["uz"] != "")
    ].copy()


    # 去掉明显过短
    df = df[
        (df["en"].str.len() >= 2)
        &
        (df["uz"].str.len() >= 2)
    ].copy()


    # 本 Challenge 先只使用 Latin Uzbek
    before = len(df)

    df = df[
        ~df["uz"].apply(
            contains_cyrillic
        )
    ].copy()

    print(
        "去除 Cyrillic Uzbek：",
        before - len(df),
    )


    # 基础长度限制
    df["en_words"] = (
        df["en"]
        .apply(word_count)
    )

    df["uz_words"] = (
        df["uz"]
        .apply(word_count)
    )


    df = df[
        df["en_words"].between(
            1,
            80,
        )
        &
        df["uz_words"].between(
            1,
            80,
        )
    ].copy()


    # pair key
    df["pair_key"] = [
        pair_key(en, uz)
        for en, uz
        in zip(
            df["en"],
            df["uz"],
        )
    ]


    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "pair_key",
        ],
        keep="first",
    ).copy()

    print(
        "删除重复句对：",
        before - len(df),
    )

    print(
        "基础清洗后：",
        len(df),
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# 8. 读取以前 Benchmark，防止数据泄漏
# ============================================================

def load_previous_benchmark_keys():

    keys = set()


    if not BENCHMARK_DIR.exists():

        return keys


    for file_path in BENCHMARK_DIR.glob(
        "*.csv"
    ):

        # 不读取当前将要生成的文件
        if file_path.name in {
            OUTPUT_FILE.name,
            REPORT_FILE.name,
        }:

            continue


        try:

            old_df = pd.read_csv(
                file_path,
                low_memory=False,
            )

        except Exception:

            continue


        en_col = find_column(
            old_df.columns,
            EN_COLUMN_CANDIDATES,
        )

        uz_col = find_column(
            old_df.columns,
            UZ_COLUMN_CANDIDATES,
        )


        if (
            en_col is None
            or
            uz_col is None
        ):

            continue


        for en, uz in zip(
            old_df[en_col],
            old_df[uz_col],
        ):

            if (
                pd.isna(en)
                or
                pd.isna(uz)
            ):
                continue

            keys.add(
                pair_key(
                    en,
                    uz,
                )
            )


    return keys


# ============================================================
# 9. Challenge 分类规则
# ============================================================

# ------------------------------------------------------------
# 9.1 Negation
# ------------------------------------------------------------

EN_NEGATION_PATTERN = re.compile(
    r"\b("
    r"not|no|never|none|nothing|nobody|nowhere|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|"
    r"can't|cannot|couldn't|won't|wouldn't|"
    r"shouldn't|mustn't|haven't|hasn't|hadn't"
    r")\b",
    flags=re.IGNORECASE,
)


UZ_NEGATION_PATTERN = re.compile(
    r"("
    r"\bemas\b|"
    r"\byo['‘’ʻ`]q\b|"
    r"\bhech\b|"
    r"\bemasdi\b|"
    r"\bemasman\b|"
    r"\bemasmiz\b|"
    r"\bemaslar\b|"
    r"\bbo['‘’ʻ`]lmay|"
    r"\bqilmay|"
    r"\bxohlamay|"
    r"\bkelmay|"
    r"\bbormay"
    r")",
    flags=re.IGNORECASE,
)


def is_negation(
    row,
) -> bool:

    return bool(
        EN_NEGATION_PATTERN.search(
            row["en"]
        )
        or
        UZ_NEGATION_PATTERN.search(
            row["uz"]
        )
    )


# ------------------------------------------------------------
# 9.2 Time / Date
# ------------------------------------------------------------

TIME_NUMBER_PATTERN = re.compile(
    r"\b(?:[01]?\d|2[0-3])"
    r"[:.][0-5]\d\b"
)

DATE_PATTERN = re.compile(
    r"\b("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r")\b"
)

EN_TIME_WORDS = re.compile(
    r"\b("
    r"today|tomorrow|yesterday|"
    r"morning|afternoon|evening|night|noon|midnight|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|"
    r"july|august|september|october|november|december|"
    r"o'clock|a\.m\.|p\.m\.|am|pm"
    r")\b",
    flags=re.IGNORECASE,
)

UZ_TIME_WORDS = re.compile(
    r"\b("
    r"bugun|ertaga|kecha|"
    r"ertalab|tushdan keyin|kechqurun|kechasi|"
    r"soat|daqiqa|"
    r"dushanba|seshanba|chorshanba|payshanba|"
    r"juma|shanba|yakshanba"
    r")\b",
    flags=re.IGNORECASE,
)


def is_time_date(
    row,
) -> bool:

    combined = (
        row["en"]
        + " "
        + row["uz"]
    )

    return bool(
        TIME_NUMBER_PATTERN.search(
            combined
        )
        or
        DATE_PATTERN.search(
            combined
        )
        or
        EN_TIME_WORDS.search(
            row["en"]
        )
        or
        UZ_TIME_WORDS.search(
            row["uz"]
        )
    )


# ------------------------------------------------------------
# 9.3 Number / Money / Unit
# ------------------------------------------------------------

DIGIT_PATTERN = re.compile(
    r"\d"
)

MONEY_UNIT_PATTERN = re.compile(
    r"("
    r"\$|€|£|¥|"
    r"\bUSD\b|\bEUR\b|\bRUB\b|\bCNY\b|\bUZS\b|"
    r"\bdollar(?:s)?\b|"
    r"\beuro\b|"
    r"\bruble(?:s)?\b|"
    r"\byuan\b|"
    r"\bkg\b|\bkm\b|\bcm\b|\bmm\b|\bml\b|\bl\b|"
    r"\bpercent\b|%"
    r")",
    flags=re.IGNORECASE,
)


def is_number(
    row,
) -> bool:

    combined = (
        row["en"]
        + " "
        + row["uz"]
    )

    return bool(
        DIGIT_PATTERN.search(
            combined
        )
        or
        MONEY_UNIT_PATTERN.search(
            combined
        )
    )


# ------------------------------------------------------------
# 9.4 Entity / Location / Proper Name
# ------------------------------------------------------------

ENTITY_LEXICON = {
    "tashkent",
    "samarkand",
    "bukhara",
    "uzbekistan",
    "russia",
    "moscow",
    "china",
    "beijing",
    "shanghai",
    "england",
    "london",
    "america",
    "american",
    "united states",
    "europe",
    "asia",
    "india",
    "japan",
    "tokyo",
    "korea",
    "paris",
    "france",
    "germany",
    "berlin",
    "google",
    "microsoft",
    "apple",
    "amazon",
}


def has_capitalized_entity(
    text: str,
) -> bool:

    tokens = re.findall(
        r"\b[A-Za-z][A-Za-z'’-]*\b",
        text,
    )

    # 第一个词首字母大写通常只是句首，
    # 所以主要检查第二个词以后。
    for token in tokens[1:]:

        if (
            len(token) >= 2
            and
            token[0].isupper()
            and
            token[1:].islower()
        ):

            return True

    return False


def is_entity(
    row,
) -> bool:

    en_lower = (
        row["en"].lower()
    )

    if any(
        entity in en_lower
        for entity in ENTITY_LEXICON
    ):

        return True

    return has_capitalized_entity(
        row["en"]
    )


# ------------------------------------------------------------
# 9.5 Long sentence
# ------------------------------------------------------------

def is_long(
    row,
) -> bool:

    return (
        row["en_words"] >= 15
        or
        row["uz_words"] >= 15
    )


# ------------------------------------------------------------
# 9.6 Normal
# ------------------------------------------------------------

def is_normal(
    row,
) -> bool:

    # 普通句子不要太长或太短
    if not (
        4 <= row["en_words"] <= 14
        and
        4 <= row["uz_words"] <= 14
    ):

        return False

    # Normal 尽量不包含专项挑战特征
    if is_negation(row):
        return False

    if is_time_date(row):
        return False

    if is_number(row):
        return False

    if is_entity(row):
        return False

    if is_long(row):
        return False

    return True


# ============================================================
# 10. 为所有数据打候选标签
# ============================================================

def add_candidate_flags(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\n")
    print("=" * 80)
    print("识别 Challenge 类型")
    print("=" * 80)


    df = df.copy()


    df["is_negation"] = df.apply(
        is_negation,
        axis=1,
    )

    df["is_time_date"] = df.apply(
        is_time_date,
        axis=1,
    )

    df["is_number"] = df.apply(
        is_number,
        axis=1,
    )

    df["is_entity"] = df.apply(
        is_entity,
        axis=1,
    )

    df["is_long"] = df.apply(
        is_long,
        axis=1,
    )

    df["is_normal"] = df.apply(
        is_normal,
        axis=1,
    )


    for category in CATEGORIES:

        count = int(
            df[
                f"is_{category}"
            ].sum()
        )

        print(
            f"{category:<12}: "
            f"{count}"
        )


    return df


# ============================================================
# 11. 每类抽样，并保证类别间不重复
# ============================================================

def select_challenge_samples(
    df: pd.DataFrame,
) -> pd.DataFrame:

    selected_parts = []

    used_keys = set()


    # 优先抽特殊类型，
    # normal 最后。
    selection_order = [
        "negation",
        "time_date",
        "number",
        "entity",
        "long",
        "normal",
    ]


    rng = np.random.RandomState(
        SEED
    )


    print("\n")
    print("=" * 80)
    print("抽取 Challenge Benchmark")
    print("=" * 80)


    for category in selection_order:

        candidate_df = df[
            df[
                f"is_{category}"
            ]
        ].copy()


        candidate_df = candidate_df[
            ~candidate_df[
                "pair_key"
            ].isin(
                used_keys
            )
        ].copy()


        # 为保证每次可复现
        candidate_df = candidate_df.sample(
            frac=1.0,
            random_state=(
                SEED
                + selection_order.index(
                    category
                )
            ),
        )


        n = min(
            SAMPLES_PER_CATEGORY,
            len(candidate_df),
        )


        chosen = (
            candidate_df
            .head(n)
            .copy()
        )


        chosen["category"] = (
            category
        )


        selected_parts.append(
            chosen
        )


        used_keys.update(
            chosen[
                "pair_key"
            ].tolist()
        )


        print(
            f"{category:<12}: "
            f"{n}/{SAMPLES_PER_CATEGORY}"
        )


        if n < SAMPLES_PER_CATEGORY:

            print(
                f"  [WARNING] "
                f"{category} 样本不足。"
            )


    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )


    # 最后统一随机打乱，
    # 避免 CSV 按类别排列。
    result = result.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(
        drop=True
    )


    result.insert(
        0,
        "challenge_id",
        [
            f"challenge_{i:04d}"
            for i in range(
                1,
                len(result) + 1,
            )
        ],
    )


    return result


# ============================================================
# 12. 主程序
# ============================================================

def main():

    print("=" * 80)
    print("EN-UZ Challenge Benchmark Builder")
    print("=" * 80)

    print(
        "目标数量：",
        TARGET_TOTAL,
    )


    # --------------------------------------------------------
    # 加载
    # --------------------------------------------------------

    df = load_all_clean_data()


    # --------------------------------------------------------
    # 清洗
    # --------------------------------------------------------

    df = clean_parallel_data(
        df
    )


    # --------------------------------------------------------
    # 排除已有 benchmark
    # --------------------------------------------------------

    old_keys = (
        load_previous_benchmark_keys()
    )

    print("\n")
    print("=" * 80)
    print("排除已有 Benchmark")
    print("=" * 80)

    print(
        "历史 Benchmark pair：",
        len(old_keys),
    )


    before = len(df)

    df = df[
        ~df[
            "pair_key"
        ].isin(
            old_keys
        )
    ].copy()


    print(
        "排除历史 Benchmark：",
        before - len(df),
    )

    print(
        "剩余候选：",
        len(df),
    )


    # --------------------------------------------------------
    # 打标签
    # --------------------------------------------------------

    df = add_candidate_flags(
        df
    )


    # --------------------------------------------------------
    # 抽样
    # --------------------------------------------------------

    challenge_df = (
        select_challenge_samples(
            df
        )
    )


    # ========================================================
    # 13. 最终只保留需要字段
    # ========================================================

    final_columns = [
        "challenge_id",
        "category",
        "en",
        "uz",
        "source_file",
        "en_words",
        "uz_words",
    ]


    challenge_df = challenge_df[
        final_columns
    ].copy()


    # --------------------------------------------------------
    # 保存 Challenge
    # --------------------------------------------------------

    challenge_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # --------------------------------------------------------
    # 生成统计报告
    # --------------------------------------------------------

    report_df = (
        challenge_df[
            "category"
        ]
        .value_counts()
        .rename_axis(
            "category"
        )
        .reset_index(
            name="count"
        )
    )

    report_df["percent"] = (
        report_df["count"]
        / len(challenge_df)
        * 100
    )


    report_df.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # 14. 最终检查
    # ========================================================

    duplicate_count = (
        challenge_df
        .duplicated(
            subset=[
                "en",
                "uz",
            ]
        )
        .sum()
    )


    cyrillic_count = (
        challenge_df["uz"]
        .apply(
            contains_cyrillic
        )
        .sum()
    )


    print("\n")
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    print(
        "Challenge 数量：",
        len(challenge_df),
    )

    print(
        "目标数量：",
        TARGET_TOTAL,
    )

    print(
        "重复句对：",
        duplicate_count,
    )

    print(
        "Cyrillic Uzbek：",
        cyrillic_count,
    )


    print("\n类别分布：")

    print(
        report_df.to_string(
            index=False
        )
    )


    print("\n输出文件：")

    print(
        OUTPUT_FILE
    )

    print(
        REPORT_FILE
    )


    print("\n前10条：")

    print(
        challenge_df[
            [
                "challenge_id",
                "category",
                "en",
                "uz",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()