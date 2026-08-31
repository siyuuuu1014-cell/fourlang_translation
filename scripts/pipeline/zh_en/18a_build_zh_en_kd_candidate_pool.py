from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STEP_VERSION = "18A_V1"
POLICY_VERSION = "ZH_EN_TEACHER_ROUTING_V1"


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 18A - build ZH<->EN train-only KD candidate pool "
            "from the frozen Step 15A split."
        )
    )

    parser.add_argument(
        "--project_root",
        default=None,
    )

    parser.add_argument(
        "--train_pairs",
        default="data/splits/zh_en/v1/train_pairs_v1.parquet",
    )

    parser.add_argument(
        "--train_directed",
        default="data/splits/zh_en/v1/train_bidirectional_v1.parquet",
    )

    parser.add_argument(
        "--validation_pairs",
        default="data/splits/zh_en/v1/validation_pairs_v1.parquet",
    )

    parser.add_argument(
        "--validation_directed",
        default="data/splits/zh_en/v1/validation_bidirectional_v1.parquet",
    )

    parser.add_argument(
        "--teacher_selection",
        default=(
            "data/teacher_selection/zh_en/v1/"
            "17c_qwen_pairwise/full/qwen_pairwise_800_v3.parquet"
        ),
    )

    parser.add_argument(
        "--routing_policy",
        default=(
            "data/teacher_selection/zh_en/v1/"
            "17d_policy/teacher_routing_policy_v1.json"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="data/distillation/zh_en/v1/18a_candidates",
    )

    parser.add_argument(
        "--benchmark_root",
        action="append",
        default=None,
        help=(
            "Optional benchmark directory. Can be repeated. "
            "If omitted, standard benchmark directories are scanned."
        ),
    )

    parser.add_argument(
        "--allow_no_benchmark",
        action="store_true",
        help="Not recommended. Allows Step18A to continue if no benchmark file is found.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Basic helpers
# ============================================================

def infer_project_root() -> Path:
    # .../fourlang_translation/scripts/pipeline/zh_en/18a_xxx.py
    return Path(__file__).resolve().parents[3]


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} missing:\n{path}")


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u3002": ".",
        "\uff0c": ",",
        "\uff1a": ":",
        "\uff1b": ";",
        "\uff01": "!",
        "\uff1f": "?",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def make_pair_key(en: Any, zh: Any) -> str:
    return normalize_text(en) + "\u241f" + normalize_text(zh)


def sha256_text(*parts: Any) -> str:
    raw = "\u241f".join(str(x) for x in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(obj, list):
            return pd.DataFrame(obj)

        if isinstance(obj, dict):
            for key in ["data", "rows", "items", "examples"]:
                if isinstance(obj.get(key), list):
                    return pd.DataFrame(obj[key])

        raise ValueError(f"Unsupported JSON table structure: {path}")

    raise ValueError(f"Unsupported table type: {path}")


# ============================================================
# Benchmark discovery
# ============================================================

def default_benchmark_roots(project_root: Path) -> list[Path]:
    roots = [
        project_root / "data" / "benchmark" / "zh_en",
        project_root / "data" / "benchmarks" / "zh_en",
        project_root / "data" / "benchmark",
        project_root / "data" / "benchmarks",
    ]

    out = []
    seen = set()

    for root in roots:
        root = root.resolve()

        if root in seen:
            continue

        seen.add(root)

        if root.exists() and root.is_dir():
            out.append(root)

    return out


def looks_like_zh_en_benchmark(path: Path) -> bool:
    low = str(path).lower().replace("\\", "/")

    if "zh_en" in low or "en_zh" in low:
        return True

    if "flores" in low or "tatoeba" in low:
        return True

    return False


def extract_zh_en_pairs(df: pd.DataFrame) -> list[tuple[str, str]]:
    cols = set(df.columns)

    # Pair schema
    if {"en", "zh"}.issubset(cols):
        return [
            (str(en), str(zh))
            for en, zh in zip(df["en"], df["zh"])
        ]

    # Directed schema
    if {"direction", "source_text", "target_text"}.issubset(cols):
        pairs = []

        for _, row in df.iterrows():
            direction = str(row["direction"]).strip().lower()

            if direction == "en_zh":
                pairs.append(
                    (
                        str(row["source_text"]),
                        str(row["target_text"]),
                    )
                )

            elif direction == "zh_en":
                pairs.append(
                    (
                        str(row["target_text"]),
                        str(row["source_text"]),
                    )
                )

        return pairs

    # Common FLORES-style schemas
    en_candidates = [
        "eng_Latn",
        "sentence_eng_Latn",
        "english",
        "en_text",
        "source_en",
    ]

    zh_candidates = [
        "zho_Hans",
        "sentence_zho_Hans",
        "chinese",
        "zh_text",
        "source_zh",
        "target_zh",
    ]

    en_col = next((c for c in en_candidates if c in cols), None)
    zh_col = next((c for c in zh_candidates if c in cols), None)

    if en_col and zh_col:
        return [
            (str(en), str(zh))
            for en, zh in zip(df[en_col], df[zh_col])
        ]

    return []


def discover_benchmark_sets(
    roots: list[Path],
) -> tuple[set[str], set[str], set[str], list[dict]]:
    pair_keys: set[str] = set()
    en_set: set[str] = set()
    zh_set: set[str] = set()
    manifest: list[dict] = []

    supported = {
        ".parquet",
        ".csv",
        ".json",
        ".jsonl",
        ".ndjson",
    }

    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in supported:
                continue

            if not looks_like_zh_en_benchmark(path):
                continue

            rp = path.resolve()

            if rp in seen:
                continue

            seen.add(rp)

            try:
                df = read_table(path)
                pairs = extract_zh_en_pairs(df)

            except Exception as exc:
                manifest.append(
                    {
                        "path": str(rp),
                        "status": "READ_ERROR",
                        "table_rows": None,
                        "zh_en_pairs": 0,
                        "error": repr(exc),
                    }
                )
                continue

            if not pairs:
                manifest.append(
                    {
                        "path": str(rp),
                        "status": "NO_ZH_EN_SCHEMA",
                        "table_rows": int(len(df)),
                        "zh_en_pairs": 0,
                        "error": "",
                    }
                )
                continue

            valid = 0

            for en, zh in pairs:
                en_n = normalize_text(en)
                zh_n = normalize_text(zh)

                if not en_n or not zh_n:
                    continue

                pair_keys.add(make_pair_key(en, zh))
                en_set.add(en_n)
                zh_set.add(zh_n)
                valid += 1

            manifest.append(
                {
                    "path": str(rp),
                    "status": "LOADED",
                    "table_rows": int(len(df)),
                    "zh_en_pairs": int(valid),
                    "error": "",
                }
            )

    return pair_keys, en_set, zh_set, manifest


# ============================================================
# Step17C validation-derived leakage sets
# ============================================================

def teacher_selection_sets(
    df: pd.DataFrame,
) -> tuple[set[str], set[str], set[str], set[str]]:
    pair_ids: set[str] = set()

    if "pair_id" in df.columns:
        pair_ids = set(
            df["pair_id"]
            .dropna()
            .astype(str)
            .tolist()
        )

    pair_keys: set[str] = set()
    en_set: set[str] = set()
    zh_set: set[str] = set()

    required = {
        "direction",
        "source_text",
        "reference_text",
    }

    if required.issubset(df.columns):
        for _, row in df.iterrows():
            direction = str(row["direction"]).strip().lower()

            if direction == "en_zh":
                en = row["source_text"]
                zh = row["reference_text"]

            elif direction == "zh_en":
                en = row["reference_text"]
                zh = row["source_text"]

            else:
                continue

            en_n = normalize_text(en)
            zh_n = normalize_text(zh)

            if en_n:
                en_set.add(en_n)

            if zh_n:
                zh_set.add(zh_n)

            if en_n and zh_n:
                pair_keys.add(make_pair_key(en, zh))

    return pair_ids, pair_keys, en_set, zh_set


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else infer_project_root()
    )

    train_pairs_path = resolve_path(
        project_root,
        args.train_pairs,
    )

    train_directed_path = resolve_path(
        project_root,
        args.train_directed,
    )

    validation_pairs_path = resolve_path(
        project_root,
        args.validation_pairs,
    )

    validation_directed_path = resolve_path(
        project_root,
        args.validation_directed,
    )

    teacher_selection_path = resolve_path(
        project_root,
        args.teacher_selection,
    )

    routing_policy_path = resolve_path(
        project_root,
        args.routing_policy,
    )

    output_dir = resolve_path(
        project_root,
        args.output_dir,
    )

    for path, label in [
        (train_pairs_path, "train_pairs"),
        (train_directed_path, "train_directed"),
        (validation_pairs_path, "validation_pairs"),
        (validation_directed_path, "validation_directed"),
        (teacher_selection_path, "teacher_selection_17c"),
        (routing_policy_path, "routing_policy_17d"),
    ]:
        require_file(path, label)

    print("=" * 110)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18A - BUILD TRAIN-ONLY KD CANDIDATE POOL")
    print("=" * 110)

    print("\nProject root:")
    print(project_root)

    train_pairs = pd.read_parquet(train_pairs_path)
    train_directed = pd.read_parquet(train_directed_path)
    validation_pairs = pd.read_parquet(validation_pairs_path)
    validation_directed = pd.read_parquet(validation_directed_path)
    teacher_selection = pd.read_parquet(teacher_selection_path)

    policy = json.loads(
        routing_policy_path.read_text(
            encoding="utf-8"
        )
    )

    if policy.get("policy_version") != POLICY_VERSION:
        raise RuntimeError(
            "Unexpected routing policy version:\n"
            f"{policy.get('policy_version')}"
        )

    print("\nInput counts:")
    print("train_pairs:", len(train_pairs))
    print("train_directed:", len(train_directed))
    print("validation_pairs:", len(validation_pairs))
    print("validation_directed:", len(validation_directed))
    print("teacher_selection_17c:", len(teacher_selection))

    # --------------------------------------------------------
    # Required schemas
    # --------------------------------------------------------

    pair_required = {
        "pair_id",
        "en",
        "zh",
        "source_dataset",
        "quality_tier",
        "training_weight",
        "split_group_id",
    }

    directed_required = {
        "pair_id",
        "sample_id",
        "source_dataset",
        "quality_tier",
        "training_weight",
        "split_group_id",
        "direction",
        "source_lang",
        "target_lang",
        "source_text",
        "target_text",
    }

    pair_missing = pair_required - set(train_pairs.columns)
    directed_missing = directed_required - set(train_directed.columns)

    if pair_missing:
        raise RuntimeError(
            f"train_pairs missing columns: {sorted(pair_missing)}"
        )

    if directed_missing:
        raise RuntimeError(
            f"train_directed missing columns: {sorted(directed_missing)}"
        )

    # --------------------------------------------------------
    # Structural integrity
    # --------------------------------------------------------

    if train_pairs["pair_id"].duplicated().any():
        raise RuntimeError("Duplicate pair_id in train_pairs.")

    if train_directed["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate sample_id in train_directed.")

    pair_ids = set(
        train_pairs["pair_id"].astype(str)
    )

    directed_pair_ids = set(
        train_directed["pair_id"].astype(str)
    )

    if pair_ids != directed_pair_ids:
        raise RuntimeError(
            "train_pairs and train_directed pair_id sets differ."
        )

    direction_counts = (
        train_directed["direction"]
        .value_counts()
        .to_dict()
    )

    if set(direction_counts) != {"en_zh", "zh_en"}:
        raise RuntimeError(
            f"Unexpected directions: {direction_counts}"
        )

    if len(train_directed) != 2 * len(train_pairs):
        raise RuntimeError(
            "Expected exactly 2 directed rows per pair.\n"
            f"pairs={len(train_pairs)}\n"
            f"directed={len(train_directed)}"
        )

    pair_direction_count = (
        train_directed.groupby("pair_id")["direction"]
        .nunique()
    )

    if not (pair_direction_count == 2).all():
        raise RuntimeError(
            "Some train pairs do not contain both en_zh and zh_en."
        )

    # --------------------------------------------------------
    # Validation leakage sets
    # --------------------------------------------------------

    train_pair_ids = set(
        train_pairs["pair_id"]
        .astype(str)
        .tolist()
    )

    validation_pair_ids = set(
        validation_pairs["pair_id"]
        .astype(str)
        .tolist()
    )

    train_groups = set(
        train_pairs["split_group_id"]
        .astype(str)
        .tolist()
    )

    validation_groups = set(
        validation_pairs["split_group_id"]
        .astype(str)
        .tolist()
    )

    train_en = set(
        train_pairs["en"]
        .map(normalize_text)
        .tolist()
    )

    train_zh = set(
        train_pairs["zh"]
        .map(normalize_text)
        .tolist()
    )

    validation_en = set(
        validation_pairs["en"]
        .map(normalize_text)
        .tolist()
    )

    validation_zh = set(
        validation_pairs["zh"]
        .map(normalize_text)
        .tolist()
    )

    train_pair_keys = set(
        make_pair_key(en, zh)
        for en, zh
        in zip(
            train_pairs["en"],
            train_pairs["zh"],
        )
    )

    validation_pair_keys = set(
        make_pair_key(en, zh)
        for en, zh
        in zip(
            validation_pairs["en"],
            validation_pairs["zh"],
        )
    )

    # --------------------------------------------------------
    # Benchmark discovery
    # --------------------------------------------------------

    if args.benchmark_root:
        benchmark_roots = [
            resolve_path(project_root, p)
            for p in args.benchmark_root
        ]
    else:
        benchmark_roots = default_benchmark_roots(
            project_root
        )

    (
        benchmark_pair_keys,
        benchmark_en,
        benchmark_zh,
        benchmark_manifest,
    ) = discover_benchmark_sets(
        benchmark_roots
    )

    loaded_benchmark_files = [
        row
        for row in benchmark_manifest
        if row["status"] == "LOADED"
    ]

    print("\nBenchmark roots:")
    if benchmark_roots:
        for p in benchmark_roots:
            print("-", p)
    else:
        print("(none)")

    print("\nBenchmark files loaded:")
    if loaded_benchmark_files:
        for row in loaded_benchmark_files:
            print(
                f"- {row['path']} "
                f"| pairs={row['zh_en_pairs']}"
            )
    else:
        print("(none)")

    if (
        not loaded_benchmark_files
        and
        not args.allow_no_benchmark
    ):
        raise RuntimeError(
            "\nNo ZH-EN frozen benchmark file was discovered.\n"
            "Step18A refuses to continue without benchmark leakage audit.\n"
            "If your benchmark lives elsewhere, pass:\n"
            "--benchmark_root <path>"
        )

    # --------------------------------------------------------
    # Explicit 17C teacher-selection leakage sets
    # --------------------------------------------------------

    (
        teacher_pair_ids,
        teacher_pair_keys,
        teacher_en,
        teacher_zh,
    ) = teacher_selection_sets(
        teacher_selection
    )

    # --------------------------------------------------------
    # Leakage report
    # --------------------------------------------------------

    leakage = {
        "validation_pair_id_overlap": len(
            train_pair_ids & validation_pair_ids
        ),
        "validation_group_overlap": len(
            train_groups & validation_groups
        ),
        "validation_exact_pair_overlap": len(
            train_pair_keys & validation_pair_keys
        ),
        "validation_english_overlap": len(
            train_en & validation_en
        ),
        "validation_chinese_overlap": len(
            train_zh & validation_zh
        ),

        "benchmark_exact_pair_overlap": len(
            train_pair_keys & benchmark_pair_keys
        ),
        "benchmark_english_overlap": len(
            train_en & benchmark_en
        ),
        "benchmark_chinese_overlap": len(
            train_zh & benchmark_zh
        ),

        "teacher_selection_pair_id_overlap": len(
            train_pair_ids & teacher_pair_ids
        ),
        "teacher_selection_exact_pair_overlap": len(
            train_pair_keys & teacher_pair_keys
        ),
        "teacher_selection_english_overlap": len(
            train_en & teacher_en
        ),
        "teacher_selection_chinese_overlap": len(
            train_zh & teacher_zh
        ),
    }

    print("\nLeakage audit:")
    for key, value in leakage.items():
        print(f"{key}: {value}")

    nonzero_leakage = {
        key: int(value)
        for key, value in leakage.items()
        if value != 0
    }

    if nonzero_leakage:
        raise RuntimeError(
            "\nSTEP 18A ABORTED: frozen-evaluation leakage detected.\n"
            + json.dumps(
                nonzero_leakage,
                ensure_ascii=False,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # Pair-level output
    # --------------------------------------------------------

    pair_columns = [
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
        "normalized_pair_hash",
        "quality_score",
        "risk_flags",
        "quality_tier",
        "quality_tier_reason",
        "training_weight",
        "approved_for_training",
        "split_group_id",
    ]

    pair_columns = [
        c
        for c in pair_columns
        if c in train_pairs.columns
    ]

    pair_out = (
        train_pairs[pair_columns]
        .copy()
        .reset_index(drop=True)
    )

    pair_out["kd_pair_id"] = [
        f"zh_en_kd_pair_{i:07d}"
        for i in range(len(pair_out))
    ]

    pair_out["kd_pair_hash"] = [
        sha256_text(
            row["pair_id"],
            normalize_text(row["en"]),
            normalize_text(row["zh"]),
            STEP_VERSION,
        )
        for _, row in pair_out.iterrows()
    ]

    pair_out["teacher_policy_version"] = POLICY_VERSION
    pair_out["candidate_stage"] = "18A_TRAIN_ONLY"
    pair_out["candidate_status"] = "READY_FOR_TEACHER_GENERATION"
    pair_out["allowed_for_kd_training"] = True

    # --------------------------------------------------------
    # Directed output
    # --------------------------------------------------------

    directed_columns = [
        "pair_id",
        "sample_id",
        "source_dataset",
        "quality_tier",
        "training_weight",
        "split_group_id",
        "quality_score",
        "risk_flags",
        "direction",
        "source_lang",
        "target_lang",
        "source_text",
        "target_text",
    ]

    directed_columns = [
        c
        for c in directed_columns
        if c in train_directed.columns
    ]

    directed_out = (
        train_directed[directed_columns]
        .copy()
        .reset_index(drop=True)
    )

    directed_out["kd_candidate_id"] = [
        f"zh_en_kd_candidate_{i:07d}"
        for i in range(len(directed_out))
    ]

    directed_out["kd_candidate_hash"] = [
        sha256_text(
            row["pair_id"],
            row["direction"],
            normalize_text(row["source_text"]),
            normalize_text(row["target_text"]),
            STEP_VERSION,
        )
        for _, row in directed_out.iterrows()
    ]

    directed_out["human_reference"] = directed_out["target_text"]

    directed_out["human_training_weight"] = (
        directed_out["training_weight"]
        .astype(float)
    )

    directed_out["opus_specialist_key"] = (
        directed_out["direction"]
        .map(
            {
                "en_zh": "opus_mt_en_zh_exp1",
                "zh_en": "opus_mt_zh_en_exp1",
            }
        )
    )

    directed_out["madlad_teacher_key"] = (
        directed_out["direction"]
        .map(
            {
                "en_zh": "madlad400_3b_mt_en_zh",
                "zh_en": "madlad400_3b_mt_zh_en",
            }
        )
    )

    # Prior metadata only. Never hard-route Step18A samples.
    prior_map = {
        ("en_zh", "ALT"): "MADLAD",
        ("zh_en", "ALT"): "MADLAD",
        ("en_zh", "Tatoeba"): "NONE",
        ("zh_en", "Tatoeba"): "NONE",
    }

    directed_out["domain_teacher_prior"] = [
        prior_map.get(
            (
                str(direction),
                str(source_dataset),
            ),
            "NONE",
        )
        for direction, source_dataset
        in zip(
            directed_out["direction"],
            directed_out["source_dataset"],
        )
    ]

    directed_out[
        "domain_teacher_prior_is_hard_route"
    ] = False

    directed_out[
        "teacher_policy_version"
    ] = POLICY_VERSION

    directed_out[
        "candidate_stage"
    ] = "18A_TRAIN_ONLY"

    directed_out[
        "candidate_status"
    ] = "READY_FOR_OPUS_AND_MADLAD_GENERATION"

    directed_out[
        "allowed_for_kd_training"
    ] = True

    # --------------------------------------------------------
    # Final assertions
    # --------------------------------------------------------

    assertions = {
        "train_pairs_nonempty": len(pair_out) > 0,
        "directed_candidates_nonempty": len(directed_out) > 0,
        "pair_id_unique": pair_out["pair_id"].is_unique,
        "kd_pair_id_unique": pair_out["kd_pair_id"].is_unique,
        "sample_id_unique": directed_out["sample_id"].is_unique,
        "kd_candidate_id_unique": directed_out[
            "kd_candidate_id"
        ].is_unique,
        "exactly_two_directions": set(
            directed_out["direction"].unique()
        ) == {"en_zh", "zh_en"},
        "two_directed_rows_per_pair": (
            directed_out.groupby("pair_id")
            .size()
            .eq(2)
            .all()
        ),
        "no_empty_source": (
            directed_out["source_text"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_empty_target": (
            directed_out["target_text"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "all_frozen_leakage_zero": (
            len(nonzero_leakage) == 0
        ),
        "routing_policy_is_frozen_v1": (
            policy.get("policy_version")
            == POLICY_VERSION
        ),
        "domain_prior_is_metadata_only": (
            directed_out[
                "domain_teacher_prior_is_hard_route"
            ]
            .eq(False)
            .all()
        ),
    }

    failed = [
        key
        for key, ok in assertions.items()
        if not bool(ok)
    ]

    if failed:
        raise RuntimeError(
            "STEP18A assertion failure:\n"
            + "\n".join(failed)
        )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_output = (
        output_dir
        / "kd_candidate_pairs_train_only_v1.parquet"
    )

    directed_output = (
        output_dir
        / "kd_candidates_bidirectional_train_only_v1.parquet"
    )

    manifest_output = (
        output_dir
        / "kd_candidate_manifest_v1.csv"
    )

    benchmark_manifest_output = (
        output_dir
        / "benchmark_discovery_manifest_v1.csv"
    )

    report_output = (
        output_dir
        / "kd_candidate_report_v1.json"
    )

    output_files = [
        pair_output,
        directed_output,
        manifest_output,
        benchmark_manifest_output,
        report_output,
    ]

    if not args.overwrite:
        existing = [
            p
            for p in output_files
            if p.exists()
        ]

        if existing:
            raise RuntimeError(
                "Step18A output already exists:\n"
                + "\n".join(str(p) for p in existing)
                + "\nUse --overwrite to rebuild."
            )

    pair_out.to_parquet(
        pair_output,
        index=False,
    )

    directed_out.to_parquet(
        directed_output,
        index=False,
    )

    manifest = (
        directed_out.groupby(
            [
                "direction",
                "source_dataset",
                "quality_tier",
                "domain_teacher_prior",
            ],
            dropna=False,
        )
        .agg(
            directed_rows=(
                "kd_candidate_id",
                "size",
            ),
            unique_pairs=(
                "pair_id",
                "nunique",
            ),
            mean_human_weight=(
                "human_training_weight",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "direction",
                "source_dataset",
                "quality_tier",
            ]
        )
        .reset_index(drop=True)
    )

    manifest.to_csv(
        manifest_output,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        benchmark_manifest
    ).to_csv(
        benchmark_manifest_output,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "step": "18A",
        "step_version": STEP_VERSION,
        "policy_version": POLICY_VERSION,

        "counts": {
            "train_pairs": int(len(pair_out)),
            "directed_candidates": int(
                len(directed_out)
            ),
            "direction": {
                str(k): int(v)
                for k, v
                in directed_out[
                    "direction"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "source_dataset_pairs": {
                str(k): int(v)
                for k, v
                in pair_out[
                    "source_dataset"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "quality_tier_pairs": {
                str(k): int(v)
                for k, v
                in pair_out[
                    "quality_tier"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "domain_teacher_prior_directed": {
                str(k): int(v)
                for k, v
                in directed_out[
                    "domain_teacher_prior"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },

        "benchmark_discovery": {
            "roots": [
                str(p)
                for p in benchmark_roots
            ],
            "loaded_files": int(
                len(loaded_benchmark_files)
            ),
            "benchmark_pair_keys": int(
                len(benchmark_pair_keys)
            ),
            "benchmark_english_sentences": int(
                len(benchmark_en)
            ),
            "benchmark_chinese_sentences": int(
                len(benchmark_zh)
            ),
        },

        "leakage": {
            str(k): int(v)
            for k, v
            in leakage.items()
        },

        "assertions": {
            str(k): bool(v)
            for k, v
            in assertions.items()
        },

        "routing_policy": {
            "alt_en_zh_prior": "MADLAD",
            "alt_zh_en_prior": "MADLAD",
            "tatoeba_en_zh_prior": None,
            "tatoeba_zh_en_prior": None,
            "prior_is_hard_route": False,
            "fixed_disagreement_threshold": None,
        },

        "safety": {
            "validation_rows_allowed_in_kd": False,
            "teacher_selection_rows_allowed_in_kd": False,
            "benchmark_rows_allowed_in_kd": False,
            "step18a_runs_models": False,
            "step18a_selects_final_teacher": False,
        },

        "outputs": {
            "pair_candidates": str(
                pair_output
            ),
            "directed_candidates": str(
                directed_output
            ),
            "manifest": str(
                manifest_output
            ),
            "benchmark_manifest": str(
                benchmark_manifest_output
            ),
        },

        "created_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),

        "status": (
            "READY_FOR_STEP_18B_TEACHER_GENERATION"
        ),
    }

    save_json(
        report,
        report_output,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("STEP 18A RESULT")
    print("=" * 110)

    print("\nTrain pairs:")
    print(len(pair_out))

    print("\nDirected candidates:")
    print(len(directed_out))

    print("\nDirection:")
    print(
        directed_out["direction"]
        .value_counts()
        .to_string()
    )

    print("\nSource dataset (pair level):")
    print(
        pair_out["source_dataset"]
        .value_counts()
        .to_string()
    )

    print("\nQuality tier (pair level):")
    print(
        pair_out["quality_tier"]
        .value_counts()
        .to_string()
    )

    print("\nDomain teacher prior:")
    print(
        directed_out["domain_teacher_prior"]
        .value_counts()
        .to_string()
    )

    print(
        "\nIMPORTANT: domain prior is metadata only; "
        "Step18A does NOT hard-select a teacher."
    )

    print("\nLeakage:")
    for key, value in leakage.items():
        print(f"{key}: {value}")

    print("\nAssertions:")
    for key, value in assertions.items():
        print(f"{key}: {bool(value)}")

    print("\nPair candidates:")
    print(pair_output)

    print("\nDirected candidates:")
    print(directed_output)

    print("\nManifest:")
    print(manifest_output)

    print("\nBenchmark manifest:")
    print(benchmark_manifest_output)

    print("\nReport:")
    print(report_output)

    print("\nSTATUS:")
    print("READY_FOR_STEP_18B_TEACHER_GENERATION")


if __name__ == "__main__":
    main()
