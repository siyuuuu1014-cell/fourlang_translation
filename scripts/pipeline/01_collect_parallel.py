from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
import unicodedata

import pandas as pd


# ============================================================
# 1. Project paths
# ============================================================

# 当前脚本：
# fourlang_translation/scripts/pipeline/01_collect_parallel.py
#
# parents[2] =
# fourlang_translation
PROJECT_ROOT = Path(__file__).resolve().parents[2]


INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean"
    / "en_uz"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "pipeline"
    / "en_uz"
    / "01_collected"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


OUTPUT_PARQUET = (
    OUTPUT_DIR
    / "parallel_collected.parquet"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "parallel_collected.csv"
)

SOURCE_REPORT_FILE = (
    OUTPUT_DIR
    / "source_report.csv"
)

MANIFEST_FILE = (
    OUTPUT_DIR
    / "manifest.json"
)


# ============================================================
# 2. Configuration
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".parquet",
}


# ============================================================
# 3. Possible column names
# ============================================================

EN_COLUMN_CANDIDATES = [
    "en",
    "english",
    "source_en",
    "src_en",
    "en_text",
    "text_en",
]

UZ_COLUMN_CANDIDATES = [
    "uz",
    "uzbek",
    "target_uz",
    "tgt_uz",
    "uz_text",
    "text_uz",
]


# ============================================================
# 4. Text normalization
#
# IMPORTANT:
# 此处只做非常轻的 normalization。
#
# 更正式的 Uzbek Latin normalization
# 放到 02_normalize.py。
# ============================================================

def basic_normalize(text: str) -> str:

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# 5. Find column
# ============================================================

def find_column(
    columns,
    candidates,
):

    normalized_map = {
        str(col).strip().lower(): col
        for col in columns
    }

    for candidate in candidates:

        if candidate in normalized_map:

            return normalized_map[
                candidate
            ]

    return None


# ============================================================
# 6. Detect source name
#
# 根据文件路径标记数据来源。
#
# 后续可以看到：
# tatoeba
# opus
# hplt
# public_5k_v1
# public_5k_v2
# exp1
# exp2
# unknown
# ============================================================

def detect_source(
    file_path: Path,
) -> str:

    path_lower = str(
        file_path
    ).lower()


    source_rules = [

        (
            "tatoeba",
            [
                "tatoeba",
            ],
        ),

        (
            "opus",
            [
                "opus",
                "opus100",
                "opus-100",
            ],
        ),

        (
            "hplt",
            [
                "hplt",
            ],
        ),

        (
            "public_5k_v1",
            [
                "public_5k_v1",
            ],
        ),

        (
            "public_5k_v2",
            [
                "public_5k_v2",
            ],
        ),

        (
            "exp1",
            [
                "exp1",
            ],
        ),

        (
            "exp2",
            [
                "exp2",
            ],
        ),
    ]


    for source_name, keywords in source_rules:

        for keyword in keywords:

            if keyword in path_lower:

                return source_name


    return "unknown"


# ============================================================
# 7. Data type
#
# 当前 clean/en_uz 里的内容先统一视作：
# parallel_existing
#
# 后面再根据 source_quality 细分。
# ============================================================

def detect_data_type(
    source_name: str,
) -> str:

    return "parallel_existing"


# ============================================================
# 8. Read one file
# ============================================================

def read_file(
    file_path: Path,
) -> pd.DataFrame | None:

    suffix = (
        file_path
        .suffix
        .lower()
    )


    try:

        if suffix == ".csv":

            df = pd.read_csv(
                file_path,
                low_memory=False,
            )


        elif suffix == ".tsv":

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

        print()
        print(
            "[READ ERROR]",
            file_path,
        )

        print(
            repr(exc)
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

        print()
        print(
            "[SKIP] No EN/UZ columns:"
        )

        print(
            file_path
        )

        print(
            "Columns:",
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
        "source_text",
        "target_text",
    ]


    source_name = detect_source(
        file_path
    )


    result[
        "src_lang"
    ] = "en"


    result[
        "tgt_lang"
    ] = "uz"


    result[
        "data_source"
    ] = source_name


    result[
        "data_type"
    ] = detect_data_type(
        source_name
    )


    result[
        "source_file"
    ] = str(
        file_path.relative_to(
            PROJECT_ROOT
        )
    )


    return result


# ============================================================
# 9. Pair ID
#
# 给每条句对稳定 ID。
#
# 以后即使 CSV 行号变化，
# pair_id 不会改变。
# ============================================================

def make_pair_id(
    source_text: str,
    target_text: str,
) -> str:

    payload = (
        basic_normalize(
            source_text
        ).lower()
        +
        "\u241f"
        +
        basic_normalize(
            target_text
        ).lower()
    )


    return hashlib.sha1(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# 10. Find files
# ============================================================

def find_input_files():

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"找不到输入目录：\n"
            f"{INPUT_DIR}"
        )


    files = [

        path

        for path in INPUT_DIR.rglob("*")

        if (
            path.is_file()
            and
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]


    return sorted(
        files
    )


# ============================================================
# 11. Main
# ============================================================

def main():

    print("=" * 90)
    print("EN-UZ PIPELINE")
    print("STEP 01 - COLLECT PARALLEL DATA")
    print("=" * 90)


    print(
        "\nPROJECT_ROOT:"
    )

    print(
        PROJECT_ROOT
    )


    print(
        "\nINPUT_DIR:"
    )

    print(
        INPUT_DIR
    )


    # ========================================================
    # Find files
    # ========================================================

    files = find_input_files()


    print(
        "\n找到候选文件：",
        len(files)
    )


    all_frames = []

    report_rows = []


    # ========================================================
    # Read files
    # ========================================================

    for file_index, file_path in enumerate(
        files,
        start=1,
    ):

        print()
        print(
            f"[{file_index}/{len(files)}]"
        )

        print(
            file_path.relative_to(
                PROJECT_ROOT
            )
        )


        part = read_file(
            file_path
        )


        if part is None:

            continue


        raw_rows = len(
            part
        )


        print(
            "Rows:",
            raw_rows
        )


        source_name = (
            part[
                "data_source"
            ]
            .iloc[0]
            if raw_rows > 0
            else
            detect_source(
                file_path
            )
        )


        report_rows.append({

            "source_file":
                str(
                    file_path.relative_to(
                        PROJECT_ROOT
                    )
                ),

            "data_source":
                source_name,

            "rows":
                raw_rows,
        })


        all_frames.append(
            part
        )


    # ========================================================
    # Ensure data exists
    # ========================================================

    if not all_frames:

        raise RuntimeError(
            "没有找到可用 EN-UZ 平行数据。"
        )


    combined_df = pd.concat(
        all_frames,
        ignore_index=True,
    )


    print("\n")
    print("=" * 90)
    print("COLLECT RESULT")
    print("=" * 90)


    print(
        "合并总行数：",
        len(combined_df)
    )


    # ========================================================
    # Basic cleaning
    #
    # 注意：
    # 这里只删除空值。
    #
    # 不在这一步做：
    # 去重
    # 长度过滤
    # Cyrillic过滤
    # benchmark排除
    #
    # 全部放到后续 Pipeline。
    # ========================================================

    before = len(
        combined_df
    )


    combined_df = (
        combined_df
        .dropna(
            subset=[
                "source_text",
                "target_text",
            ]
        )
        .copy()
    )


    combined_df[
        "source_text"
    ] = (
        combined_df[
            "source_text"
        ]
        .astype(str)
        .map(
            basic_normalize
        )
    )


    combined_df[
        "target_text"
    ] = (
        combined_df[
            "target_text"
        ]
        .astype(str)
        .map(
            basic_normalize
        )
    )


    combined_df = combined_df[
        (
            combined_df[
                "source_text"
            ]
            !=
            ""
        )
        &
        (
            combined_df[
                "target_text"
            ]
            !=
            ""
        )
    ].copy()


    removed_empty = (
        before
        -
        len(combined_df)
    )


    print(
        "删除空值：",
        removed_empty
    )


    # ========================================================
    # Pair IDs
    # ========================================================

    print(
        "\nGenerating pair IDs..."
    )


    combined_df[
        "pair_id"
    ] = [

        make_pair_id(
            src,
            tgt,
        )

        for src, tgt in zip(
            combined_df[
                "source_text"
            ],
            combined_df[
                "target_text"
            ],
        )
    ]


    # ========================================================
    # Original row ID
    #
    # 用于追踪 Pipeline 中每条数据。
    # ========================================================

    combined_df.insert(
        0,
        "row_id",
        range(
            1,
            len(combined_df) + 1,
        ),
    )


    # ========================================================
    # Standard column order
    # ========================================================

    columns = [

        "row_id",
        "pair_id",

        "src_lang",
        "tgt_lang",

        "source_text",
        "target_text",

        "data_source",
        "data_type",
        "source_file",
    ]


    combined_df = combined_df[
        columns
    ]


    # ========================================================
    # Source report
    # ========================================================

    report_df = pd.DataFrame(
        report_rows
    )


    source_summary = (
        combined_df[
            "data_source"
        ]
        .value_counts()
        .rename_axis(
            "data_source"
        )
        .reset_index(
            name="rows"
        )
    )


    print("\n")
    print("=" * 90)
    print("SOURCE DISTRIBUTION")
    print("=" * 90)


    print(
        source_summary.to_string(
            index=False
        )
    )


    # ========================================================
    # Save Parquet
    # ========================================================

    print(
        "\nSaving Parquet..."
    )


    combined_df.to_parquet(
        OUTPUT_PARQUET,
        index=False,
    )


    # ========================================================
    # Save CSV
    # ========================================================

    print(
        "Saving CSV..."
    )


    combined_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Save source report
    # ========================================================

    source_summary.to_csv(
        SOURCE_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    # ========================================================
    # Manifest
    # ========================================================

    manifest = {

        "pipeline":
            "en_uz",

        "step":
            "01_collect_parallel",

        "input_directory":
            str(
                INPUT_DIR.relative_to(
                    PROJECT_ROOT
                )
            ),

        "input_files":
            len(
                files
            ),

        "rows_before_empty_filter":
            before,

        "removed_empty":
            removed_empty,

        "rows_after_collection":
            len(
                combined_df
            ),

        "unique_pair_ids":
            int(
                combined_df[
                    "pair_id"
                ].nunique()
            ),

        "duplicate_pair_ids":
            int(
                len(
                    combined_df
                )
                -
                combined_df[
                    "pair_id"
                ].nunique()
            ),

        "sources":
            source_summary.to_dict(
                orient="records"
            ),
    }


    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Final report
    # ========================================================

    print("\n")
    print("=" * 90)
    print("STEP 01 COMPLETE")
    print("=" * 90)


    print(
        "最终行数：",
        len(combined_df)
    )


    print(
        "Unique pair_id：",
        combined_df[
            "pair_id"
        ].nunique()
    )


    print(
        "重复 pair_id：",
        len(combined_df)
        -
        combined_df[
            "pair_id"
        ].nunique()
    )


    print("\nFiles:")

    print(
        OUTPUT_PARQUET
    )

    print(
        OUTPUT_CSV
    )

    print(
        SOURCE_REPORT_FILE
    )

    print(
        MANIFEST_FILE
    )


    print("\nSample:")

    print(
        combined_df[
            [
                "row_id",
                "data_source",
                "source_text",
                "target_text",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":

    main()