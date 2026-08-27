from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


DATASET_ID = "openlanguagedata/flores_plus"

EN_CONFIG = "eng_Latn"
ZH_CONFIG = "cmn_Hans"


def load_language(
    config_name: str,
    split: str,
    revision: str,
) -> pd.DataFrame:

    print(
        f"\nLoading {config_name} / {split}"
    )

    ds = load_dataset(
        DATASET_ID,
        config_name,
        split=split,
        revision=revision,
    )

    df = ds.to_pandas()

    required = {
        "id",
        "text",
        "split",
    }

    missing = (
        required
        -
        set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{config_name} missing columns: "
            f"{sorted(missing)}"
        )

    print(
        f"{config_name}: {len(df)} rows"
    )

    return df


def build_parallel(
    en_df: pd.DataFrame,
    zh_df: pd.DataFrame,
    split: str,
) -> pd.DataFrame:

    en = en_df.copy()
    zh = zh_df.copy()

    en["id"] = (
        en["id"]
        .astype(str)
    )

    zh["id"] = (
        zh["id"]
        .astype(str)
    )

    # ----------------------------------------
    # Integrity
    # ----------------------------------------

    if en["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate English IDs detected."
        )

    if zh["id"].duplicated().any():
        raise RuntimeError(
            "Duplicate Chinese IDs detected."
        )

    en_ids = set(
        en["id"]
    )

    zh_ids = set(
        zh["id"]
    )

    if en_ids != zh_ids:
        raise RuntimeError(
            "English / Chinese FLORES IDs differ.\n"
            f"EN only: {len(en_ids - zh_ids)}\n"
            f"ZH only: {len(zh_ids - en_ids)}"
        )

    # ----------------------------------------
    # English
    # ----------------------------------------

    en_columns = {
        "id":
            "flores_id",

        "text":
            "en",
    }

    en_keep = [
        "id",
        "text",
    ]

    for optional in [
        "url",
        "domain",
        "topic",
        "has_image",
        "has_hyperlink",
        "last_updated",
    ]:
        if optional in en.columns:
            en_keep.append(
                optional
            )

    en = (
        en[
            en_keep
        ]
        .rename(
            columns=en_columns
        )
    )

    # ----------------------------------------
    # Chinese
    # ----------------------------------------

    zh = (
        zh[
            [
                "id",
                "text",
            ]
        ]
        .rename(
            columns={
                "id":
                    "flores_id",

                "text":
                    "zh",
            }
        )
    )

    # ----------------------------------------
    # Pair
    # ----------------------------------------

    paired = en.merge(
        zh,
        on="flores_id",
        how="inner",
        validate="one_to_one",
    )

    paired["split"] = split

    paired["pair_id"] = (
        "floresplus_"
        +
        split
        +
        "_"
        +
        paired[
            "flores_id"
        ].astype(str)
    )

    paired["dataset"] = (
        "FLORES+"
    )

    paired["en_config"] = (
        EN_CONFIG
    )

    paired["zh_config"] = (
        ZH_CONFIG
    )

    # ----------------------------------------
    # Empty check
    # ----------------------------------------

    paired["en"] = (
        paired["en"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    paired["zh"] = (
        paired["zh"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    empty_en = int(
        (
            paired["en"] == ""
        ).sum()
    )

    empty_zh = int(
        (
            paired["zh"] == ""
        ).sum()
    )

    if (
        empty_en > 0
        or
        empty_zh > 0
    ):
        raise RuntimeError(
            "Empty text detected.\n"
            f"EN empty: {empty_en}\n"
            f"ZH empty: {empty_zh}"
        )

    # ----------------------------------------
    # Exact duplicate pair
    # ----------------------------------------

    duplicated_pairs = int(
        paired[
            [
                "zh",
                "en",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicated_pairs:
        raise RuntimeError(
            "Duplicate translation pairs detected: "
            f"{duplicated_pairs}"
        )

    first_columns = [
        "pair_id",
        "flores_id",
        "zh",
        "en",
        "dataset",
        "split",
        "zh_config",
        "en_config",
    ]

    remaining = [
        column
        for column in paired.columns
        if column not in first_columns
    ]

    paired = paired[
        first_columns
        +
        remaining
    ]

    return (
        paired
        .sort_values(
            "flores_id"
        )
        .reset_index(
            drop=True
        )
    )


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    output_dir = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)

    print(
        "ZH-EN BASELINE PIPELINE"
    )

    print(
        "STEP 13A - BUILD FLORES+ "
        "FROZEN BENCHMARK"
    )

    print("=" * 100)

    # ========================================================
    # Pin exact Hugging Face revision
    # ========================================================

    api = HfApi()

    info = api.dataset_info(
        DATASET_ID
    )

    revision = info.sha

    print(
        "\nDataset:",
        DATASET_ID
    )

    print(
        "Pinned revision:",
        revision
    )

    # ========================================================
    # DEVTEST
    # Final frozen benchmark
    # ========================================================

    en_devtest = load_language(
        EN_CONFIG,
        "devtest",
        revision,
    )

    zh_devtest = load_language(
        ZH_CONFIG,
        "devtest",
        revision,
    )

    devtest = build_parallel(
        en_devtest,
        zh_devtest,
        "devtest",
    )

    # ========================================================
    # DEV
    # Development/debug evaluation only
    # ========================================================

    en_dev = load_language(
        EN_CONFIG,
        "dev",
        revision,
    )

    zh_dev = load_language(
        ZH_CONFIG,
        "dev",
        revision,
    )

    dev = build_parallel(
        en_dev,
        zh_dev,
        "dev",
    )

    # ========================================================
    # Output
    # ========================================================

    devtest_csv = (
        output_dir
        /
        "flores_plus_zh_en_devtest_v1.csv"
    )

    devtest_parquet = (
        output_dir
        /
        "flores_plus_zh_en_devtest_v1.parquet"
    )

    dev_csv = (
        output_dir
        /
        "flores_plus_zh_en_dev_v1.csv"
    )

    dev_parquet = (
        output_dir
        /
        "flores_plus_zh_en_dev_v1.parquet"
    )

    manifest_file = (
        output_dir
        /
        "benchmark_manifest_v1.json"
    )

    devtest.to_csv(
        devtest_csv,
        index=False,
        encoding="utf-8-sig",
    )

    devtest.to_parquet(
        devtest_parquet,
        index=False,
    )

    dev.to_csv(
        dev_csv,
        index=False,
        encoding="utf-8-sig",
    )

    dev.to_parquet(
        dev_parquet,
        index=False,
    )

    # ========================================================
    # Leakage check between dev and devtest
    # ========================================================

    dev_pairs = set(
        zip(
            dev["zh"],
            dev["en"],
        )
    )

    devtest_pairs = set(
        zip(
            devtest["zh"],
            devtest["en"],
        )
    )

    pair_overlap = len(
        dev_pairs
        &
        devtest_pairs
    )

    zh_overlap = len(
        set(dev["zh"])
        &
        set(devtest["zh"])
    )

    en_overlap = len(
        set(dev["en"])
        &
        set(devtest["en"])
    )

    if (
        pair_overlap
        or
        zh_overlap
        or
        en_overlap
    ):
        raise RuntimeError(
            "Unexpected DEV / DEVTEST overlap.\n"
            f"Pair overlap: {pair_overlap}\n"
            f"ZH overlap: {zh_overlap}\n"
            f"EN overlap: {en_overlap}"
        )

    manifest = {

        "benchmark_version":
            "zh_en_v1",

        "dataset":
            DATASET_ID,

        "dataset_revision":
            revision,

        "language_pair":
            "zh_en",

        "languages": {
            "zh":
                ZH_CONFIG,

            "en":
                EN_CONFIG,
        },

        "frozen_final_benchmark": {
            "split":
                "devtest",

            "pairs":
                int(
                    len(devtest)
                ),

            "directed_samples":
                int(
                    len(devtest) * 2
                ),

            "csv":
                str(devtest_csv),

            "parquet":
                str(devtest_parquet),

            "policy":
                (
                    "Evaluation only. "
                    "Must never enter training, "
                    "teacher generation, "
                    "distillation, or replay."
                ),
        },

        "development_evaluation": {
            "split":
                "dev",

            "pairs":
                int(
                    len(dev)
                ),

            "directed_samples":
                int(
                    len(dev) * 2
                ),

            "csv":
                str(dev_csv),

            "parquet":
                str(dev_parquet),

            "policy":
                (
                    "Development/debug evaluation only. "
                    "Do not use for model training."
                ),
        },

        "leakage_between_dev_and_devtest": {
            "pair":
                pair_overlap,

            "zh":
                zh_overlap,

            "en":
                en_overlap,
        },

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "status":
            "FROZEN_BENCHMARK_READY",
    }

    with open(
        manifest_file,
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
    # Console
    # ========================================================

    print("\n")
    print("=" * 100)

    print(
        "STEP 13A RESULT"
    )

    print("=" * 100)

    print(
        "\nDEV pairs:",
        len(dev)
    )

    print(
        "DEVTEST pairs:",
        len(devtest)
    )

    print(
        "\nDEVTEST directed samples:",
        len(devtest) * 2
    )

    print(
        "\nDEV / DEVTEST leakage:"
    )

    print(
        "Pair:",
        pair_overlap
    )

    print(
        "ZH  :",
        zh_overlap
    )

    print(
        "EN  :",
        en_overlap
    )

    print(
        "\nFrozen benchmark:"
    )

    print(
        devtest_parquet
    )

    print(
        "\nManifest:"
    )

    print(
        manifest_file
    )

    print(
        "\nSTATUS:"
    )

    print(
        "FROZEN_BENCHMARK_READY"
    )


if __name__ == "__main__":
    main()