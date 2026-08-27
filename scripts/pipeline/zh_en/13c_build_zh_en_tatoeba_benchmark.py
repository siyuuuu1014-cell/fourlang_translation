from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Fixed external benchmark
# ============================================================

TATOEBA_TEST_URL = (
    "https://object.pouta.csc.fi/"
    "Tatoeba-MT-models/eng-zho/"
    "opus-2020-07-17.test.txt"
)

TATOEBA_RELEASE = "opus-2020-07-17"

TARGET_TAG = "cmn_Hans"

MAX_PAIRS = 1000


# ============================================================
# Basic text checks
# ============================================================

SOURCE_MARKER_RE = re.compile(
    r"^>>([^<]+)<<\s*(.*)$"
)

CHINESE_RE = re.compile(
    r"[\u4e00-\u9fff]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)


def normalize_text(
    text: str,
) -> str:

    text = str(text)

    text = text.replace(
        "\u00a0",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_bad_artifact(
    text: str,
) -> bool:

    bad_patterns = [

        r"\{\\",          # ASS subtitle tags
        r"\\fn",
        r"\\fs",
        r"\\bord",
        r"\\shad",

        # common mojibake fragments
        "Ã",
        "Â",
        "æ",
        "ç",
        "å¾",
        "ã",
    ]

    for pattern in bad_patterns:

        if pattern in text:

            return True

    return False


def valid_english(
    text: str,
) -> bool:

    text = normalize_text(
        text
    )

    if not text:
        return False

    if not LATIN_RE.search(
        text
    ):
        return False

    if contains_bad_artifact(
        text
    ):
        return False

    return True


def valid_chinese(
    text: str,
) -> bool:

    text = normalize_text(
        text
    )

    if not text:
        return False

    if not CHINESE_RE.search(
        text
    ):
        return False

    if contains_bad_artifact(
        text
    ):
        return False

    return True


# ============================================================
# Download
# ============================================================


def download_if_needed(
    url: str,
    output_file: Path,
):

    if output_file.exists():

        print(
            "Using cached raw file:"
        )

        print(
            output_file
        )

        return

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nDownloading:"
    )

    print(
        url
    )

    urllib.request.urlretrieve(
        url,
        output_file,
    )

    print(
        "Downloaded:"
    )

    print(
        output_file
    )


# ============================================================
# Parse Tatoeba challenge test
#
# Format:
#
# >>cmn_Hans<< English source
# Chinese reference
# Existing historical model hypothesis
#
# We use ONLY:
# source + first reference
#
# Historical hypothesis is ignored.
# ============================================================


def parse_test_file(
    file_path: Path,
) -> pd.DataFrame:

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
        ) as f:

        lines = [
            line.rstrip(
                "\n\r"
            )
            for line in f
        ]

    records = []

    current_tag = None
    current_source = None
    following_lines = []

    def flush_record():

        nonlocal current_tag, current_source, following_lines
        if (
            current_tag
            !=
            TARGET_TAG
        ):

            return

        if not current_source:

            return

        # First non-empty line after source
        # is the human reference.
        reference = None

        for line in following_lines:

            line = normalize_text(
                line
            )

            if line:

                reference = line

                break

        if reference is None:

            return

        records.append({

            "language_tag":
                current_tag,

            "en":
                normalize_text(
                    current_source
                ),

            "zh":
                reference,
        })

    for line in lines:

        marker = SOURCE_MARKER_RE.match(
            line
        )

        if marker:

            # save previous record
            flush_record()

            current_tag = (
                marker
                .group(1)
                .strip()
            )

            current_source = (
                marker
                .group(2)
                .strip()
            )

            following_lines = []

        else:

            if current_source is not None:

                following_lines.append(
                    line
                )

    flush_record()

    return pd.DataFrame(
        records
    )


# ============================================================
# Stable hash
# ============================================================


def stable_hash(
    en: str,
    zh: str,
) -> str:

    value = (
        normalize_text(en)
        +
        "\n"
        +
        normalize_text(zh)
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# FLORES leakage
# ============================================================


def load_flores_protected(
    benchmark_dir: Path,
):

    protected_pairs = set()
    protected_en = set()
    protected_zh = set()

    files = [

        benchmark_dir
        /
        "flores_plus_zh_en_dev_v1.parquet",

        benchmark_dir
        /
        "flores_plus_zh_en_devtest_v1.parquet",
    ]

    for file in files:

        if not file.exists():

            raise FileNotFoundError(
                file
            )

        df = pd.read_parquet(
            file
        )

        for _, row in df.iterrows():

            en = normalize_text(
                row["en"]
            )

            zh = normalize_text(
                row["zh"]
            )

            protected_pairs.add(
                (
                    en,
                    zh,
                )
            )

            protected_en.add(
                en
            )

            protected_zh.add(
                zh
            )

    return (
        protected_pairs,
        protected_en,
        protected_zh,
    )


# ============================================================
# Main
# ============================================================


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    benchmark_dir = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
    )

    external_dir = (
        project_root
        /
        "data"
        /
        "external"
        /
        "zh_en"
        /
        "tatoeba"
    )

    benchmark_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    external_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        external_dir
        /
        "opus-2020-07-17.test.txt"
    )

    output_csv = (
        benchmark_dir
        /
        "tatoeba_zh_en_test_v1.csv"
    )

    output_parquet = (
        benchmark_dir
        /
        "tatoeba_zh_en_test_v1.parquet"
    )

    report_file = (
        benchmark_dir
        /
        "tatoeba_zh_en_test_v1_report.json"
    )

    print(
        "=" * 100
    )

    print(
        "ZH-EN BASELINE PIPELINE"
    )

    print(
        "STEP 13C - BUILD TATOEBA "
        "FROZEN BENCHMARK"
    )

    print(
        "=" * 100
    )

    # ========================================================
    # Download
    # ========================================================

    download_if_needed(
        TATOEBA_TEST_URL,
        raw_file,
    )

    # ========================================================
    # Parse
    # ========================================================

    raw = parse_test_file(
        raw_file
    )

    print(
        "\nParsed cmn_Hans records:",
        len(raw)
    )

    if len(raw) == 0:

        raise RuntimeError(
            "No cmn_Hans records parsed."
        )

    # ========================================================
    # Normalize
    # ========================================================

    raw["en"] = (
        raw["en"]
        .map(
            normalize_text
        )
    )

    raw["zh"] = (
        raw["zh"]
        .map(
            normalize_text
        )
    )

    # ========================================================
    # Quality filtering
    # ========================================================

    valid_mask = (

        raw["en"]
        .map(
            valid_english
        )

        &

        raw["zh"]
        .map(
            valid_chinese
        )
    )

    filtered = (
        raw[
            valid_mask
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    invalid_removed = (
        len(raw)
        -
        len(filtered)
    )

    print(
        "Invalid/artifact removed:",
        invalid_removed
    )

    # ========================================================
    # Exact dedup
    # ========================================================

    before_dedup = len(
        filtered
    )

    filtered = (
        filtered
        .drop_duplicates(
            subset=[
                "en",
                "zh",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    duplicates_removed = (
        before_dedup
        -
        len(filtered)
    )

    print(
        "Exact duplicates removed:",
        duplicates_removed
    )

    # ========================================================
    # FLORES leakage protection
    # ========================================================

    (
        protected_pairs,
        protected_en,
        protected_zh,
    ) = load_flores_protected(
        benchmark_dir
    )

    pair_overlap = 0
    en_overlap = 0
    zh_overlap = 0

    keep_rows = []

    for _, row in filtered.iterrows():

        en = row["en"]
        zh = row["zh"]

        pair_hit = (
            (
                en,
                zh,
            )
            in
            protected_pairs
        )

        en_hit = (
            en
            in
            protected_en
        )

        zh_hit = (
            zh
            in
            protected_zh
        )

        if pair_hit:
            pair_overlap += 1

        if en_hit:
            en_overlap += 1

        if zh_hit:
            zh_overlap += 1

        # strict sentence-level isolation
        if (
            pair_hit
            or
            en_hit
            or
            zh_hit
        ):

            continue

        keep_rows.append(
            row.to_dict()
        )

    clean = pd.DataFrame(
        keep_rows
    )

    print(
        "\nFLORES overlap removed:"
    )

    print(
        "Pair:",
        pair_overlap
    )

    print(
        "EN  :",
        en_overlap
    )

    print(
        "ZH  :",
        zh_overlap
    )

    # ========================================================
    # Stable deterministic selection
    # ========================================================

    clean["stable_hash"] = [

        stable_hash(
            en,
            zh,
        )

        for en, zh
        in zip(
            clean["en"],
            clean["zh"],
        )
    ]

    clean = (
        clean
        .sort_values(
            "stable_hash"
        )
        .reset_index(
            drop=True
        )
    )

    available_clean = len(
        clean
    )

    if available_clean > MAX_PAIRS:

        clean = (
            clean
            .head(
                MAX_PAIRS
            )
            .copy()
        )

    clean = (
        clean
        .reset_index(
            drop=True
        )
    )

    clean["pair_id"] = [

        f"tatoeba_zh_en_{i:05d}"

        for i
        in range(
            len(clean)
        )
    ]

    clean["dataset"] = (
        "Tatoeba-Challenge"
    )

    clean["release"] = (
        TATOEBA_RELEASE
    )

    clean["split"] = (
        "test"
    )

    # Reorder
    clean = clean[
        [
            "pair_id",
            "en",
            "zh",
            "dataset",
            "release",
            "split",
            "language_tag",
            "stable_hash",
        ]
    ]

    # ========================================================
    # Final integrity
    # ========================================================

    assert (
        clean["en"]
        .str.strip()
        .ne("")
        .all()
    )

    assert (
        clean["zh"]
        .str.strip()
        .ne("")
        .all()
    )

    assert not (
        clean[
            [
                "en",
                "zh",
            ]
        ]
        .duplicated()
        .any()
    )

    # ========================================================
    # Save
    # ========================================================

    clean.to_csv(
        output_csv,
        index=False,
        encoding="utf-8-sig",
    )

    clean.to_parquet(
        output_parquet,
        index=False,
    )

    report = {

        "step":
            "13C",

        "benchmark_version":
            "tatoeba_zh_en_test_v1",

        "source":
            "Helsinki-NLP Tatoeba Translation Challenge",

        "source_url":
            TATOEBA_TEST_URL,

        "release":
            TATOEBA_RELEASE,

        "language_filter":
            TARGET_TAG,

        "purpose":
            (
                "Second independent frozen "
                "ZH-EN baseline benchmark."
            ),

        "raw_cmn_hans_records":
            int(
                len(raw)
            ),

        "invalid_or_artifact_removed":
            int(
                invalid_removed
            ),

        "duplicates_removed":
            int(
                duplicates_removed
            ),

        "flores_overlap_detected": {

            "pair":
                int(
                    pair_overlap
                ),

            "en":
                int(
                    en_overlap
                ),

            "zh":
                int(
                    zh_overlap
                ),
        },

        "available_clean_before_limit":
            int(
                available_clean
            ),

        "final_pairs":
            int(
                len(clean)
            ),

        "directed_samples":
            int(
                len(clean)
                *
                2
            ),

        "selection":
            (
                "SHA256 stable deterministic "
                f"selection; max {MAX_PAIRS} pairs"
            ),

        "policy":
            (
                "Evaluation only. "
                "Must never enter training, "
                "teacher generation, "
                "distillation or replay."
            ),

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {

            "csv":
                str(
                    output_csv
                ),

            "parquet":
                str(
                    output_parquet
                ),
        },

        "status":
            "TATOEBA_FROZEN_BENCHMARK_READY",
    }

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Console
    # ========================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        "STEP 13C RESULT"
    )

    print(
        "=" * 100
    )

    print(
        "\nRaw cmn_Hans:",
        len(raw)
    )

    print(
        "Available clean:",
        available_clean
    )

    print(
        "Final frozen pairs:",
        len(clean)
    )

    print(
        "Directed samples:",
        len(clean) * 2
    )

    print(
        "\nFLORES leakage removed:"
    )

    print(
        "Pair:",
        pair_overlap
    )

    print(
        "EN  :",
        en_overlap
    )

    print(
        "ZH  :",
        zh_overlap
    )

    print(
        "\nOutput:"
    )

    print(
        output_parquet
    )

    print(
        "\nReport:"
    )

    print(
        report_file
    )

    print(
        "\nSTATUS:"
    )

    print(
        "TATOEBA_FROZEN_BENCHMARK_READY"
    )


if __name__ == "__main__":

    main()