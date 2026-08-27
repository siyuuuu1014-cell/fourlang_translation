from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
import urllib.error
import urllib.request
import zipfile

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# Source definitions
# ============================================================

SOURCES = {

    "tatoeba": {

        "dataset":
            "Tatoeba",

        "release":
            "v2026-07-08",

        "license":
            "CC BY 2.0 FR",

        "homepage":
            "https://opus.nlpl.eu/datasets/Tatoeba",

        # Verified through OPUS API on 2026-08-27.
        #
        # OPUS canonical pair orientation:
        # cmn -> en
        #
        # Orientation of archive filename does NOT matter.
        # We read both .cmn and .en from the archive and
        # normalize them into our internal fields:
        #
        # en = English
        # zh = Mandarin Chinese
        "url_candidates": [

            (
                "https://object.pouta.csc.fi/"
                "OPUS-Tatoeba/v2026-07-08/"
                "moses/cmn-en.txt.zip"
            ),
        ],

        "target_codes": [
            "cmn",
        ],
    },

    "alt": {

        "dataset":
            "ALT",

        "release":
            "v20191206",

        "license":
            "CC BY 4.0",

        "homepage":
            "https://opus.nlpl.eu/datasets/ALT",

        # Verified through OPUS API on 2026-08-27.
        #
        # OPUS canonical pair:
        # en -> zh
        "url_candidates": [

            (
                "https://object.pouta.csc.fi/"
                "OPUS-ALT/v20191206/"
                "moses/en-zh.txt.zip"
            ),
        ],

        "target_codes": [
            "zh",
        ],
    },
}


CJK_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]"
)

LATIN_RE = re.compile(
    r"[A-Za-z]"
)


# ============================================================
# Args
# ============================================================


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Step 14A - Collect license-audited "
            "ZH-EN human parallel corpora."
        )
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=[
            "tatoeba",
            "alt",
        ],
        choices=sorted(
            SOURCES.keys()
        ),
    )

    parser.add_argument(
        "--max_per_source",
        type=int,
        default=100000,
        help=(
            "Maximum number of pairs retained "
            "from each source after basic filtering. "
            "0 means no limit."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--strict_sources",
        action="store_true",
        help=(
            "Fail if any requested source cannot "
            "be downloaded."
        ),
    )

    return parser.parse_args()


# ============================================================
# Basic normalization
#
# NOTE:
# This is NOT the final Step 14B normalization.
# It only creates stable strings for:
# - empty filtering
# - leakage check
# - exact dedup
# ============================================================


def normalize_basic(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        str(text),
    )

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


def normalized_key(
    text: str,
) -> str:

    return normalize_basic(
        text
    ).casefold()


def pair_key(
    en: str,
    zh: str,
) -> tuple[str, str]:

    return (
        normalized_key(en),
        normalized_key(zh),
    )


# ============================================================
# Stable hash
# ============================================================


def stable_hash(
    en: str,
    zh: str,
    source_dataset: str,
) -> str:

    value = (
        source_dataset
        +
        "\n"
        +
        normalized_key(en)
        +
        "\n"
        +
        normalized_key(zh)
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# Obvious language-shape checks
#
# Only reject extremely obvious invalid pairs.
# More serious quality checks belong to Step 14C/14D.
# ============================================================


def basic_language_shape_ok(
    en: str,
    zh: str,
) -> bool:

    en = normalize_basic(en)
    zh = normalize_basic(zh)

    if not en or not zh:
        return False

    if not LATIN_RE.search(en):
        return False

    if not CJK_RE.search(zh):
        return False

    return True


# ============================================================
# Download OPUS archive
# ============================================================


def download_with_candidates(
    source_name: str,
    config: dict,
    destination_dir: Path,
    overwrite: bool,
) -> tuple[Path, str]:

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached = sorted(
        destination_dir.glob(
            "*.txt.zip"
        )
    )

    if (
        cached
        and
        not overwrite
    ):

        archive = cached[0]

        print(
            f"\n[{source_name}] "
            "Using cached archive:"
        )

        print(
            archive
        )

        return (
            archive,
            "cached",
        )

    errors = []

    for index, url in enumerate(
        config[
            "url_candidates"
        ],
        start=1,
    ):

        archive = (
            destination_dir
            /
            Path(url).name
        )

        temp_file = (
            archive.with_suffix(
                archive.suffix
                +
                ".part"
            )
        )

        print()
        print(
            f"[{source_name}] "
            f"Trying URL {index}:"
        )

        print(
            url
        )

        try:

            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        (
                            "fourlang_translation/"
                            "zh_en_pipeline"
                        )
                },
            )

            with (
                urllib.request.urlopen(
                    request,
                    timeout=60,
                )
                as response
            ):

                with open(
                    temp_file,
                    "wb",
                ) as f:

                    shutil.copyfileobj(
                        response,
                        f,
                    )

            if archive.exists():
                archive.unlink()

            temp_file.rename(
                archive
            )

            print(
                f"[{source_name}] "
                "Download success."
            )

            print(
                archive
            )

            return (
                archive,
                url,
            )

        except Exception as exc:

            if temp_file.exists():
                temp_file.unlink()

            error_message = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            errors.append({
                "url":
                    url,
                "error":
                    error_message,
            })

            print(
                f"[{source_name}] "
                "Download failed:"
            )

            print(
                error_message
            )

    raise RuntimeError(
        "\nAll download candidates failed "
        f"for {source_name}.\n"
        +
        json.dumps(
            errors,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# Find aligned language files inside OPUS Moses zip
# ============================================================


def find_parallel_members(
    archive: Path,
    target_codes: list[str],
) -> tuple[str, str, str]:

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zf:

        members = [
            name
            for name
            in zf.namelist()
            if not name.endswith("/")
        ]

    en_candidates = [
        name
        for name
        in members
        if name.endswith(
            ".en"
        )
    ]

    if len(
        en_candidates
    ) == 0:

        raise RuntimeError(
            "Could not find English file "
            f"inside archive:\n{archive}\n"
            f"Members:\n{members[:50]}"
        )

    target_member = None
    selected_target_code = None

    for target_code in target_codes:

        matches = [
            name
            for name
            in members
            if name.endswith(
                f".{target_code}"
            )
        ]

        if matches:

            target_member = (
                matches[0]
            )

            selected_target_code = (
                target_code
            )

            break

    if target_member is None:

        raise RuntimeError(
            "Could not find Chinese target "
            f"inside archive:\n{archive}\n"
            f"Members:\n{members[:100]}"
        )

    en_member = (
        en_candidates[0]
    )

    print(
        "\nArchive members selected:"
    )

    print(
        "EN:",
        en_member
    )

    print(
        "ZH:",
        target_member
    )

    return (
        en_member,
        target_member,
        selected_target_code,
    )


# ============================================================
# Read aligned Moses files
# ============================================================


def read_parallel_zip(
    archive: Path,
    source_name: str,
    config: dict,
    download_url: str,
) -> pd.DataFrame:

    (
        en_member,
        zh_member,
        target_code,
    ) = find_parallel_members(
        archive,
        config[
            "target_codes"
        ],
    )

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zf:

        with zf.open(
            en_member,
            "r",
        ) as f_en:

            en_lines = [
                line.decode(
                    "utf-8",
                    errors="replace",
                )
                .rstrip(
                    "\r\n"
                )

                for line
                in f_en
            ]

        with zf.open(
            zh_member,
            "r",
        ) as f_zh:

            zh_lines = [
                line.decode(
                    "utf-8",
                    errors="replace",
                )
                .rstrip(
                    "\r\n"
                )

                for line
                in f_zh
            ]

    if len(
        en_lines
    ) != len(
        zh_lines
    ):

        raise RuntimeError(
            "\nParallel line count mismatch.\n"
            f"Archive: {archive}\n"
            f"EN lines: {len(en_lines)}\n"
            f"ZH lines: {len(zh_lines)}"
        )

    rows = []

    for row_id, (
        en,
        zh,
    ) in enumerate(
        zip(
            en_lines,
            zh_lines,
        )
    ):

        rows.append({

            "source_dataset":
                config[
                    "dataset"
                ],

            "source_release":
                config[
                    "release"
                ],

            "source_license":
                config[
                    "license"
                ],

            "source_homepage":
                config[
                    "homepage"
                ],

            "source_download":
                download_url,

            "source_pair_code":
                f"en-{target_code}",

            "source_row_id":
                int(
                    row_id
                ),

            "en_raw":
                en,

            "zh_raw":
                zh,
        })

    df = pd.DataFrame(
        rows
    )

    print()
    print(
        f"[{source_name}] "
        "Raw aligned pairs:",
        len(df)
    )

    return df


# ============================================================
# Basic source cleanup
# ============================================================


def basic_filter_source(
    df: pd.DataFrame,
    max_per_source: int,
) -> tuple[pd.DataFrame, dict]:

    df = df.copy()

    input_rows = len(
        df
    )

    df["en"] = (
        df["en_raw"]
        .map(
            normalize_basic
        )
    )

    df["zh"] = (
        df["zh_raw"]
        .map(
            normalize_basic
        )
    )

    empty_mask = (
        df["en"].eq("")
        |
        df["zh"].eq("")
    )

    empty_rows = int(
        empty_mask.sum()
    )

    df = df[
        ~empty_mask
    ].copy()

    shape_ok = [

        basic_language_shape_ok(
            en,
            zh,
        )

        for en, zh
        in zip(
            df["en"],
            df["zh"],
        )
    ]

    df[
        "basic_language_shape_ok"
    ] = shape_ok

    invalid_shape = int(
        (
            ~df[
                "basic_language_shape_ok"
            ]
        ).sum()
    )

    df = df[
        df[
            "basic_language_shape_ok"
        ]
    ].copy()

    before_exact_dedup = len(
        df
    )

    df = (
        df
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

    source_duplicates = (
        before_exact_dedup
        -
        len(df)
    )

    df[
        "stable_hash"
    ] = [

        stable_hash(
            en,
            zh,
            source_dataset,
        )

        for (
            en,
            zh,
            source_dataset,
        )
        in zip(
            df["en"],
            df["zh"],
            df[
                "source_dataset"
            ],
        )
    ]

    df = (
        df
        .sort_values(
            "stable_hash"
        )
        .reset_index(
            drop=True
        )
    )

    before_limit = len(
        df
    )

    if (
        max_per_source > 0
        and
        len(df)
        >
        max_per_source
    ):

        df = (
            df
            .head(
                max_per_source
            )
            .copy()
            .reset_index(
                drop=True
            )
        )

    report = {

        "input_rows":
            int(
                input_rows
            ),

        "empty_removed":
            int(
                empty_rows
            ),

        "obvious_language_shape_removed":
            int(
                invalid_shape
            ),

        "exact_duplicates_removed":
            int(
                source_duplicates
            ),

        "eligible_before_limit":
            int(
                before_limit
            ),

        "rows_after_limit":
            int(
                len(df)
            ),
    }

    return (
        df,
        report,
    )


# ============================================================
# Protected evaluation sets
# ============================================================


def load_protected_evaluation_sets(
    project_root: Path,
) -> tuple[
    set[tuple[str, str]],
    set[str],
    set[str],
    dict,
]:

    benchmark_dir = (
        project_root
        /
        "data"
        /
        "benchmark"
        /
        "zh_en"
    )

    protected_files = {

        "flores_dev":
            (
                benchmark_dir
                /
                "flores_plus_zh_en_dev_v1.parquet"
            ),

        "flores_devtest":
            (
                benchmark_dir
                /
                "flores_plus_zh_en_devtest_v1.parquet"
            ),

        "tatoeba_frozen":
            (
                benchmark_dir
                /
                "tatoeba_zh_en_test_v1.parquet"
            ),
    }

    pairs = set()
    english = set()
    chinese = set()

    counts = {}

    for (
        name,
        file_path,
    ) in protected_files.items():

        if not file_path.exists():

            raise FileNotFoundError(
                "\nMissing protected benchmark:\n"
                f"{file_path}"
            )

        df = pd.read_parquet(
            file_path
        )

        required = {
            "en",
            "zh",
        }

        missing = (
            required
            -
            set(
                df.columns
            )
        )

        if missing:

            raise RuntimeError(
                f"{name} missing columns: "
                f"{sorted(missing)}"
            )

        counts[
            name
        ] = int(
            len(df)
        )

        for (
            en,
            zh,
        ) in zip(
            df["en"],
            df["zh"],
        ):

            en_key = (
                normalized_key(
                    en
                )
            )

            zh_key = (
                normalized_key(
                    zh
                )
            )

            pairs.add(
                (
                    en_key,
                    zh_key,
                )
            )

            english.add(
                en_key
            )

            chinese.add(
                zh_key
            )

    return (
        pairs,
        english,
        chinese,
        counts,
    )


# ============================================================
# Leakage filter
# ============================================================


def apply_protected_leakage_filter(
    df: pd.DataFrame,
    protected_pairs: set,
    protected_en: set,
    protected_zh: set,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:

    rows = []

    rejected_rows = []

    pair_hits = 0
    en_hits = 0
    zh_hits = 0

    for _, row in df.iterrows():

        en_key = (
            normalized_key(
                row["en"]
            )
        )

        zh_key = (
            normalized_key(
                row["zh"]
            )
        )

        pair_hit = (
            (
                en_key,
                zh_key,
            )
            in
            protected_pairs
        )

        en_hit = (
            en_key
            in
            protected_en
        )

        zh_hit = (
            zh_key
            in
            protected_zh
        )

        if pair_hit:
            pair_hits += 1

        if en_hit:
            en_hits += 1

        if zh_hit:
            zh_hits += 1

        if (
            pair_hit
            or
            en_hit
            or
            zh_hit
        ):

            rejected = (
                row.to_dict()
            )

            rejected[
                "protected_pair_hit"
            ] = pair_hit

            rejected[
                "protected_en_hit"
            ] = en_hit

            rejected[
                "protected_zh_hit"
            ] = zh_hit

            rejected_rows.append(
                rejected
            )

        else:

            rows.append(
                row.to_dict()
            )

    accepted = pd.DataFrame(
        rows
    )

    rejected = pd.DataFrame(
        rejected_rows
    )

    report = {

        "pair_hits":
            int(
                pair_hits
            ),

        "english_sentence_hits":
            int(
                en_hits
            ),

        "chinese_sentence_hits":
            int(
                zh_hits
            ),

        "rejected_rows":
            int(
                len(rejected)
            ),

        "accepted_rows":
            int(
                len(accepted)
            ),
    }

    return (
        accepted,
        rejected,
        report,
    )


# ============================================================
# Main
# ============================================================


def main():

    args = parse_args()

    # script:
    # scripts/pipeline/zh_en/14a_...
    #
    # parents[3] = project root

    project_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    external_root = (
        project_root
        /
        "data"
        /
        "external"
        /
        "zh_en"
    )

    output_dir = (
        project_root
        /
        "data"
        /
        "pipeline"
        /
        "zh_en"
        /
        "14a_collected"
    )

    external_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_output = (
        output_dir
        /
        "parallel_collected_raw_v1.parquet"
    )

    final_output = (
        output_dir
        /
        "parallel_collected_v1.parquet"
    )

    final_csv = (
        output_dir
        /
        "parallel_collected_v1.csv"
    )

    rejected_output = (
        output_dir
        /
        "protected_leakage_rejected_v1.parquet"
    )

    source_report_file = (
        output_dir
        /
        "source_report_v1.csv"
    )

    manifest_file = (
        output_dir
        /
        "manifest_v1.json"
    )

    print(
        "=" * 110
    )

    print(
        "ZH-EN EXP1 PIPELINE"
    )

    print(
        "STEP 14A - COLLECT "
        "LICENSE-AUDITED HUMAN PARALLEL DATA"
    )

    print(
        "=" * 110
    )

    print(
        "\nProject root:"
    )

    print(
        project_root
    )

    print(
        "\nRequested sources:"
    )

    print(
        args.sources
    )

    # ========================================================
    # Protected evaluation sets
    # ========================================================

    (
        protected_pairs,
        protected_en,
        protected_zh,
        protected_counts,
    ) = load_protected_evaluation_sets(
        project_root
    )

    print(
        "\nProtected evaluation sets:"
    )

    for (
        name,
        count,
    ) in protected_counts.items():

        print(
            f"{name}: {count}"
        )

    print(
        "\nProtected unique pairs:",
        len(
            protected_pairs
        )
    )

    print(
        "Protected English sentences:",
        len(
            protected_en
        )
    )

    print(
        "Protected Chinese sentences:",
        len(
            protected_zh
        )
    )

    # ========================================================
    # Sources
    # ========================================================

    collected = []

    source_reports = []

    source_failures = []

    download_records = {}

    for source_name in (
        args.sources
    ):

        print("\n")
        print(
            "=" * 110
        )

        print(
            "SOURCE:",
            source_name.upper()
        )

        print(
            "=" * 110
        )

        config = (
            SOURCES[
                source_name
            ]
        )

        source_dir = (
            external_root
            /
            source_name
        )

        try:

            (
                archive,
                download_url,
            ) = download_with_candidates(

                source_name=
                    source_name,

                config=
                    config,

                destination_dir=
                    source_dir,

                overwrite=
                    args.overwrite,
            )

            raw_df = (
                read_parallel_zip(

                    archive=
                        archive,

                    source_name=
                        source_name,

                    config=
                        config,

                    download_url=
                        download_url,
                )
            )

            (
                filtered_df,
                source_filter_report,
            ) = basic_filter_source(

                raw_df,

                max_per_source=
                    args
                    .max_per_source,
            )

            collected.append(
                filtered_df
            )

            source_report = {

                "source":
                    source_name,

                "dataset":
                    config[
                        "dataset"
                    ],

                "release":
                    config[
                        "release"
                    ],

                "license":
                    config[
                        "license"
                    ],

                "download_url":
                    download_url,

                **source_filter_report,
            }

            source_reports.append(
                source_report
            )

            download_records[
                source_name
            ] = {

                "archive":
                    str(
                        archive
                    ),

                "download_url":
                    download_url,
            }

            print()
            print(
                "Source filter result:"
            )

            for key, value in (
                source_filter_report.items()
            ):

                print(
                    f"{key}: {value}"
                )

        except Exception as exc:

            failure = {

                "source":
                    source_name,

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error":
                    str(
                        exc
                    ),
            }

            source_failures.append(
                failure
            )

            print()
            print(
                "SOURCE FAILED:"
            )

            print(
                json.dumps(
                    failure,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            if args.strict_sources:

                raise

    if not collected:

        raise RuntimeError(
            "No source was collected successfully."
        )

    # ========================================================
    # Combine
    # ========================================================

    combined = pd.concat(
        collected,
        ignore_index=True,
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "COMBINED RAW COLLECTION"
    )

    print(
        "=" * 110
    )

    print(
        "Rows:",
        len(
            combined
        )
    )

    print(
        "\nBy source:"
    )

    print(
        combined[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    combined.to_parquet(
        raw_output,
        index=False,
    )

    # ========================================================
    # Protected benchmark leakage
    # ========================================================

    (
        leakage_clean,
        leakage_rejected,
        leakage_report,
    ) = apply_protected_leakage_filter(

        combined,

        protected_pairs,
        protected_en,
        protected_zh,
    )

    print("\n")
    print(
        "=" * 110
    )

    print(
        "PROTECTED EVALUATION LEAKAGE"
    )

    print(
        "=" * 110
    )

    for key, value in (
        leakage_report.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # Cross-source exact pair dedup
    # ========================================================

    leakage_clean[
        "pair_normalized_key"
    ] = [

        (
            normalized_key(en)
            +
            "\t"
            +
            normalized_key(zh)
        )

        for (
            en,
            zh,
        )
        in zip(
            leakage_clean[
                "en"
            ],
            leakage_clean[
                "zh"
            ],
        )
    ]

    before_cross_source_dedup = (
        len(
            leakage_clean
        )
    )

    # Stable ordering means duplicate pair is retained
    # deterministically.
    leakage_clean = (
        leakage_clean
        .sort_values(
            [
                "stable_hash",
                "source_dataset",
            ]
        )
        .drop_duplicates(
            subset=[
                "pair_normalized_key"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    cross_source_duplicates = (
        before_cross_source_dedup
        -
        len(
            leakage_clean
        )
    )

    # ========================================================
    # Pair IDs
    # ========================================================

    leakage_clean[
        "pair_id"
    ] = [

        f"zh_en_human_{i:07d}"

        for i in range(
            len(
                leakage_clean
            )
        )
    ]

    # Columns for next stage
    first_columns = [

        "pair_id",

        "en",
        "zh",

        "source_dataset",
        "source_release",
        "source_license",
        "source_homepage",
        "source_pair_code",
        "source_row_id",

        "stable_hash",
    ]

    remaining_columns = [

        column

        for column
        in leakage_clean.columns

        if column
        not in first_columns

        and column
        not in {
            "pair_normalized_key"
        }
    ]

    leakage_clean = leakage_clean[
        first_columns
        +
        remaining_columns
    ]

    # ========================================================
    # Final integrity
    # ========================================================

    assert (
        leakage_clean[
            "en"
        ]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    assert (
        leakage_clean[
            "zh"
        ]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
    )

    assert not (
        leakage_clean[
            [
                "en",
                "zh",
            ]
        ]
        .duplicated()
        .any()
    )

    # Recheck protected leakage
    final_pair_hits = 0
    final_en_hits = 0
    final_zh_hits = 0

    for (
        en,
        zh,
    ) in zip(
        leakage_clean[
            "en"
        ],
        leakage_clean[
            "zh"
        ],
    ):

        en_key = (
            normalized_key(
                en
            )
        )

        zh_key = (
            normalized_key(
                zh
            )
        )

        if (
            en_key,
            zh_key,
        ) in protected_pairs:

            final_pair_hits += 1

        if en_key in protected_en:

            final_en_hits += 1

        if zh_key in protected_zh:

            final_zh_hits += 1

    assert (
        final_pair_hits == 0
    )

    assert (
        final_en_hits == 0
    )

    assert (
        final_zh_hits == 0
    )

    # ========================================================
    # Save
    # ========================================================

    leakage_clean.to_parquet(
        final_output,
        index=False,
    )

    leakage_clean.to_csv(
        final_csv,
        index=False,
        encoding="utf-8-sig",
    )

    if len(
        leakage_rejected
    ) > 0:

        leakage_rejected.to_parquet(
            rejected_output,
            index=False,
        )

    source_report_df = pd.DataFrame(
        source_reports
    )

    source_report_df.to_csv(
        source_report_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {

        "step":
            "14A",

        "pipeline":
            "zh_en_exp1_v1",

        "purpose":
            (
                "Collect human parallel data "
                "from sources with explicit "
                "license metadata."
            ),

        "sources_requested":
            args.sources,

        "sources_successful":
            [
                report[
                    "source"
                ]
                for report
                in source_reports
            ],

        "source_failures":
            source_failures,

        "source_reports":
            source_reports,

        "downloads":
            download_records,

        "protected_evaluation_sets":
            protected_counts,

        "protected_policy":
            (
                "FLORES dev/devtest and frozen "
                "Tatoeba benchmark must never "
                "appear in training, validation, "
                "teacher generation, KD or replay."
            ),

        "combined_before_leakage":
            int(
                len(
                    combined
                )
            ),

        "leakage_report":
            leakage_report,

        "cross_source_exact_duplicates_removed":
            int(
                cross_source_duplicates
            ),

        "final_pairs":
            int(
                len(
                    leakage_clean
                )
            ),

        "final_source_distribution":
            {
                str(key):
                    int(value)

                for (
                    key,
                    value,
                )
                in (
                    leakage_clean[
                        "source_dataset"
                    ]
                    .value_counts()
                    .items()
                )
            },

        "final_assertions": {

            "no_empty_en":
                True,

            "no_empty_zh":
                True,

            "no_exact_pair_duplicate":
                True,

            "protected_pair_leakage":
                final_pair_hits,

            "protected_en_leakage":
                final_en_hits,

            "protected_zh_leakage":
                final_zh_hits,
        },

        "license_note":
            (
                "Tatoeba and ALT permit reuse "
                "under their stated Creative "
                "Commons licenses, but attribution "
                "and other license obligations "
                "must still be satisfied before "
                "commercial release."
            ),

        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "outputs": {

            "raw":
                str(
                    raw_output
                ),

            "collected":
                str(
                    final_output
                ),

            "csv":
                str(
                    final_csv
                ),

            "source_report":
                str(
                    source_report_file
                ),

            "protected_rejected":
                (
                    str(
                        rejected_output
                    )
                    if len(
                        leakage_rejected
                    ) > 0
                    else None
                ),
        },

        "status":
            "COLLECTED_READY_FOR_NORMALIZATION",
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
    print(
        "=" * 110
    )

    print(
        "STEP 14A RESULT"
    )

    print(
        "=" * 110
    )

    print(
        "\nCombined before leakage:",
        len(
            combined
        )
    )

    print(
        "Protected leakage rejected:",
        len(
            leakage_rejected
        )
    )

    print(
        "Cross-source duplicates removed:",
        cross_source_duplicates
    )

    print(
        "\nFinal collected pairs:",
        len(
            leakage_clean
        )
    )

    print(
        "\nFinal source distribution:"
    )

    print(
        leakage_clean[
            "source_dataset"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nFinal protected leakage:"
    )

    print(
        "Pair:",
        final_pair_hits
    )

    print(
        "EN  :",
        final_en_hits
    )

    print(
        "ZH  :",
        final_zh_hits
    )

    if source_failures:

        print(
            "\nSource failures:"
        )

        print(
            json.dumps(
                source_failures,
                ensure_ascii=False,
                indent=2,
            )
        )

    print(
        "\nOutput:"
    )

    print(
        final_output
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
        "COLLECTED_READY_FOR_NORMALIZATION"
    )

    if len(
        leakage_clean
    ) < 10000:

        print()
        print(
            "WARNING:"
        )

        print(
            "Collected corpus is below "
            "10,000 pairs."
        )

        print(
            "Do not train yet; add another "
            "license-audited source first."
        )


if __name__ == "__main__":

    main()