from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STEP_VERSION = "18G_V1"

VALID_GATE_ACTIONS = {
    "AUTO_EQUAL",
    "AUTO_OPUS",
    "AUTO_MADLAD",
    "QWEN_REQUIRED",
}

VALID_QWEN_WINNERS = {
    "OPUS",
    "MADLAD",
    "TIE",
    "BOTH_BAD",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18G - merge Step18E3 auto-routing and Step18F selective-Qwen "
            "decisions into a final, audited ZH<->EN KD dataset."
        )
    )

    p.add_argument("--project_root", default=None)

    p.add_argument(
        "--projection",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration/18e3_gate_policy/"
            "teacher_gate_projection_20k_v1.parquet"
        ),
    )

    p.add_argument(
        "--qwen_results",
        default=(
            "data/distillation/zh_en/v1/"
            "18f_selective_qwen/"
            "selective_qwen_results_v1.parquet"
        ),
    )

    p.add_argument(
        "--split_root",
        default="data/splits/zh_en/v1",
    )

    p.add_argument(
        "--benchmark_root",
        action="append",
        default=None,
        help=(
            "Frozen benchmark directory. Can be passed multiple times. "
            "Default: data/benchmark/zh_en"
        ),
    )

    p.add_argument(
        "--canonical_tie_teacher",
        choices=["OPUS", "MADLAD"],
        default="OPUS",
        help=(
            "Used only for a valid Qwen TIE where both Teachers are acceptable "
            "and have no major error."
        ),
    )

    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18g_final_kd"
        ),
    )

    p.add_argument("--overwrite", action="store_true")

    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def norm_text(value) -> str:
    if value is None:
        return ""

    s = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    s = s.replace(
        "\u00a0",
        " ",
    )

    s = re.sub(
        r"\s+",
        " ",
        s,
    ).strip()

    return s.casefold()


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


def first_existing(columns, names):
    for name in names:
        if name in columns:
            return name
    return None


def extract_en_zh_pairs_from_frame(
    df: pd.DataFrame,
) -> list[tuple[str, str]]:
    """
    Supports:
      - pair-level frames with en / zh
      - pair-level frames with english / chinese
      - directed frames with direction + source_text + target_text
      - directed frames with direction + source_text + human_reference
    """
    cols = set(df.columns)

    en_col = first_existing(
        cols,
        [
            "en",
            "english",
            "en_text",
            "english_text",
            "benchmark_en",
        ],
    )

    zh_col = first_existing(
        cols,
        [
            "zh",
            "chinese",
            "zh_text",
            "chinese_text",
            "benchmark_zh",
        ],
    )

    pairs = []

    if en_col and zh_col:
        for en, zh in zip(
            df[en_col],
            df[zh_col],
        ):
            en_n = norm_text(en)
            zh_n = norm_text(zh)

            if en_n and zh_n:
                pairs.append(
                    (
                        en_n,
                        zh_n,
                    )
                )

        return pairs

    if "direction" in cols:
        source_col = first_existing(
            cols,
            [
                "source_text",
                "source",
                "src_text",
            ],
        )

        target_col = first_existing(
            cols,
            [
                "target_text",
                "human_reference",
                "reference",
                "target",
                "tgt_text",
            ],
        )

        if source_col and target_col:
            for _, row in df.iterrows():
                direction = str(
                    row[
                        "direction"
                    ]
                ).strip()

                source = norm_text(
                    row[
                        source_col
                    ]
                )

                target = norm_text(
                    row[
                        target_col
                    ]
                )

                if not source or not target:
                    continue

                if direction == "en_zh":
                    pairs.append(
                        (
                            source,
                            target,
                        )
                    )

                elif direction == "zh_en":
                    pairs.append(
                        (
                            target,
                            source,
                        )
                    )

        return pairs

    return pairs


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(
            path
        )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    raise ValueError(
        f"Unsupported file: {path}"
    )


def collect_validation_files(
    split_root: Path,
) -> list[Path]:
    if not split_root.exists():
        raise FileNotFoundError(
            split_root
        )

    files = []

    for p in split_root.rglob("*"):
        if (
            p.is_file()
            and
            p.suffix.lower()
            in {
                ".parquet",
                ".csv",
            }
            and
            "validation"
            in p.name.lower()
        ):
            files.append(
                p
            )

    if not files:
        raise RuntimeError(
            f"No validation files found under {split_root}"
        )

    return sorted(
        files
    )


def collect_benchmark_files(
    roots: list[Path],
) -> list[Path]:
    files = []

    for root in roots:
        if not root.exists():
            raise FileNotFoundError(
                root
            )

        for p in root.rglob("*"):
            if (
                p.is_file()
                and
                p.suffix.lower()
                in {
                    ".parquet",
                    ".csv",
                }
            ):
                files.append(
                    p
                )

    files = sorted(
        set(
            files
        )
    )

    if not files:
        raise RuntimeError(
            "No benchmark files found."
        )

    return files


def build_frozen_eval_sets(
    validation_files: list[Path],
    benchmark_files: list[Path],
):
    val_pairs = set()
    benchmark_pairs = set()

    val_loaded = []
    benchmark_loaded = []

    for path in validation_files:
        df = read_tabular(
            path
        )

        pairs = extract_en_zh_pairs_from_frame(
            df
        )

        val_pairs.update(
            pairs
        )

        val_loaded.append(
            {
                "file": str(
                    path
                ),
                "rows": int(
                    len(
                        df
                    )
                ),
                "pairs_extracted": int(
                    len(
                        pairs
                    )
                ),
            }
        )

    for path in benchmark_files:
        df = read_tabular(
            path
        )

        pairs = extract_en_zh_pairs_from_frame(
            df
        )

        benchmark_pairs.update(
            pairs
        )

        benchmark_loaded.append(
            {
                "file": str(
                    path
                ),
                "rows": int(
                    len(
                        df
                    )
                ),
                "pairs_extracted": int(
                    len(
                        pairs
                    )
                ),
            }
        )

    val_en = {
        en
        for en, _ in val_pairs
    }

    val_zh = {
        zh
        for _, zh in val_pairs
    }

    bench_en = {
        en
        for en, _ in benchmark_pairs
    }

    bench_zh = {
        zh
        for _, zh in benchmark_pairs
    }

    return {
        "validation_pairs": val_pairs,
        "validation_en": val_en,
        "validation_zh": val_zh,
        "benchmark_pairs": benchmark_pairs,
        "benchmark_en": bench_en,
        "benchmark_zh": bench_zh,
        "validation_files": val_loaded,
        "benchmark_files": benchmark_loaded,
    }


def choose_target_for_row(
    row: pd.Series,
    canonical_tie_teacher: str,
):
    gate_action = str(
        row[
            "gate_action"
        ]
    )

    opus = str(
        row[
            "opus_prediction"
        ]
    ).strip()

    madlad = str(
        row[
            "madlad_prediction"
        ]
    ).strip()

    # --------------------------------------------------------
    # Auto-routed cells
    # --------------------------------------------------------

    if gate_action == "AUTO_EQUAL":
        if norm_text(
            opus
        ) != norm_text(
            madlad
        ):
            return {
                "approved": False,
                "reason": (
                    "AUTO_EQUAL_TEACHERS_NOT_EQUAL"
                ),
            }

        selected_teacher = (
            "OPUS"
        )

        return {
            "approved": True,
            "selected_teacher": (
                selected_teacher
            ),
            "kd_target": opus,
            "target_origin": (
                "AUTO_EQUAL"
            ),
            "decision_source_final": (
                "STEP18E3_GATE"
            ),
            "qwen_winner": "",
        }

    if gate_action == "AUTO_OPUS":
        return {
            "approved": True,
            "selected_teacher": (
                "OPUS"
            ),
            "kd_target": opus,
            "target_origin": (
                "AUTO_OPUS"
            ),
            "decision_source_final": (
                "STEP18E3_GATE"
            ),
            "qwen_winner": "",
        }

    if gate_action == "AUTO_MADLAD":
        return {
            "approved": True,
            "selected_teacher": (
                "MADLAD"
            ),
            "kd_target": madlad,
            "target_origin": (
                "AUTO_MADLAD"
            ),
            "decision_source_final": (
                "STEP18E3_GATE"
            ),
            "qwen_winner": "",
        }

    # --------------------------------------------------------
    # Qwen-routed cells
    # --------------------------------------------------------

    if gate_action != "QWEN_REQUIRED":
        return {
            "approved": False,
            "reason": (
                "INVALID_GATE_ACTION"
            ),
        }

    winner = str(
        row.get(
            "winner_model",
            "",
        )
    ).strip()

    decision_source = str(
        row.get(
            "decision_source",
            "",
        )
    ).strip()

    if winner == "BOTH_BAD":
        return {
            "approved": False,
            "reason": (
                "QWEN_BOTH_BAD"
            ),
        }

    if winner == "OPUS":
        if (
            not as_bool(
                row.get(
                    "opus_acceptable"
                )
            )
            or
            as_bool(
                row.get(
                    "opus_major_error"
                )
            )
        ):
            return {
                "approved": False,
                "reason": (
                    "QWEN_OPUS_WINNER_NOT_SAFE"
                ),
            }

        return {
            "approved": True,
            "selected_teacher": (
                "OPUS"
            ),
            "kd_target": opus,
            "target_origin": (
                "QWEN_OPUS"
            ),
            "decision_source_final": (
                decision_source
            ),
            "qwen_winner": (
                "OPUS"
            ),
        }

    if winner == "MADLAD":
        if (
            not as_bool(
                row.get(
                    "madlad_acceptable"
                )
            )
            or
            as_bool(
                row.get(
                    "madlad_major_error"
                )
            )
        ):
            return {
                "approved": False,
                "reason": (
                    "QWEN_MADLAD_WINNER_NOT_SAFE"
                ),
            }

        return {
            "approved": True,
            "selected_teacher": (
                "MADLAD"
            ),
            "kd_target": madlad,
            "target_origin": (
                "QWEN_MADLAD"
            ),
            "decision_source_final": (
                decision_source
            ),
            "qwen_winner": (
                "MADLAD"
            ),
        }

    if winner == "TIE":
        opus_safe = (
            as_bool(
                row.get(
                    "opus_acceptable"
                )
            )
            and
            not as_bool(
                row.get(
                    "opus_major_error"
                )
            )
        )

        madlad_safe = (
            as_bool(
                row.get(
                    "madlad_acceptable"
                )
            )
            and
            not as_bool(
                row.get(
                    "madlad_major_error"
                )
            )
        )

        if not (
            opus_safe
            and
            madlad_safe
        ):
            return {
                "approved": False,
                "reason": (
                    "QWEN_TIE_NOT_BOTH_SAFE"
                ),
            }

        selected_teacher = (
            canonical_tie_teacher
        )

        target = (
            opus
            if selected_teacher
            ==
            "OPUS"
            else madlad
        )

        return {
            "approved": True,
            "selected_teacher": (
                selected_teacher
            ),
            "kd_target": target,
            "target_origin": (
                "QWEN_TIE_CANONICAL_"
                +
                selected_teacher
            ),
            "decision_source_final": (
                decision_source
            ),
            "qwen_winner": (
                "TIE"
            ),
        }

    return {
        "approved": False,
        "reason": (
            "QWEN_WINNER_MISSING_OR_INVALID"
        ),
    }


def main():
    args = parse_args()

    root = (
        Path(
            args.project_root
        ).resolve()
        if args.project_root
        else infer_project_root()
    )

    projection_path = resolve_path(
        root,
        args.projection,
    )

    qwen_path = resolve_path(
        root,
        args.qwen_results,
    )

    split_root = resolve_path(
        root,
        args.split_root,
    )

    benchmark_roots = [
        resolve_path(
            root,
            value,
        )
        for value in (
            args.benchmark_root
            if args.benchmark_root
            else [
                "data/benchmark/zh_en",
            ]
        )
    ]

    output_dir = resolve_path(
        root,
        args.output_dir,
    )

    if not projection_path.exists():
        raise FileNotFoundError(
            projection_path
        )

    if not qwen_path.exists():
        raise FileNotFoundError(
            qwen_path
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_path = (
        output_dir
        /
        "final_kd_dataset_v1.parquet"
    )

    rejected_path = (
        output_dir
        /
        "rejected_kd_candidates_v1.parquet"
    )

    summary_path = (
        output_dir
        /
        "final_kd_summary_v1.csv"
    )

    report_path = (
        output_dir
        /
        "final_kd_report_v1.json"
    )

    outputs = [
        final_path,
        rejected_path,
        summary_path,
        report_path,
    ]

    if (
        not args.overwrite
        and
        any(
            p.exists()
            for p in outputs
        )
    ):
        raise RuntimeError(
            "Step18G outputs already exist. Use --overwrite."
        )

    print(
        "=" * 115
    )

    print(
        "ZH-EN DISTILLATION PIPELINE"
    )

    print(
        "STEP 18G - BUILD FINAL AUDITED KD DATASET"
    )

    print(
        "=" * 115
    )

    print(
        "\nProjection:"
    )

    print(
        projection_path
    )

    print(
        "\nQwen results:"
    )

    print(
        qwen_path
    )

    projection = pd.read_parquet(
        projection_path
    ).copy()

    qwen = pd.read_parquet(
        qwen_path
    ).copy()

    required_projection = {
        "kd_candidate_id",
        "direction",
        "source_dataset",
        "calibration_band",
        "teacher_disagreement_score",
        "gate_action",
        "qwen_required",
        "source_text",
        "human_reference",
        "opus_prediction",
        "madlad_prediction",
    }

    missing = (
        required_projection
        -
        set(
            projection.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"Projection missing columns: {sorted(missing)}"
        )

    required_qwen = {
        "kd_candidate_id",
        "winner_model",
        "decision_source",
        "opus_acceptable",
        "opus_major_error",
        "madlad_acceptable",
        "madlad_major_error",
    }

    missing_qwen = (
        required_qwen
        -
        set(
            qwen.columns
        )
    )

    if missing_qwen:
        raise RuntimeError(
            f"Qwen results missing columns: {sorted(missing_qwen)}"
        )

    projection[
        "kd_candidate_id"
    ] = projection[
        "kd_candidate_id"
    ].astype(
        str
    )

    qwen[
        "kd_candidate_id"
    ] = qwen[
        "kd_candidate_id"
    ].astype(
        str
    )

    if not projection[
        "kd_candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Projection kd_candidate_id is not unique."
        )

    if not qwen[
        "kd_candidate_id"
    ].is_unique:
        raise RuntimeError(
            "Qwen kd_candidate_id is not unique."
        )

    invalid_gate = set(
        projection[
            "gate_action"
        ]
        .astype(str)
        .unique()
    ) - VALID_GATE_ACTIONS

    if invalid_gate:
        raise RuntimeError(
            f"Invalid gate_action: {sorted(invalid_gate)}"
        )

    qwen_required = projection.loc[
        projection[
            "qwen_required"
        ]
        .map(
            as_bool
        )
    ].copy()

    if (
        len(
            qwen_required
        )
        !=
        len(
            qwen
        )
    ):
        raise RuntimeError(
            "Qwen result count does not match QWEN_REQUIRED rows: "
            f"{len(qwen)} vs {len(qwen_required)}"
        )

    if (
        set(
            qwen_required[
                "kd_candidate_id"
            ]
        )
        !=
        set(
            qwen[
                "kd_candidate_id"
            ]
        )
    ):
        raise RuntimeError(
            "Qwen result IDs do not exactly match QWEN_REQUIRED IDs."
        )

    merge_cols = [
        "kd_candidate_id",
        "winner_model",
        "decision_source",
        "confidence",
        "opus_acceptable",
        "opus_major_error",
        "madlad_acceptable",
        "madlad_major_error",
        "reason",
    ]

    merge_cols = [
        c
        for c in merge_cols
        if c in qwen.columns
    ]

    work = projection.merge(
        qwen[
            merge_cols
        ],
        on="kd_candidate_id",
        how="left",
        validate="one_to_one",
    )

    print(
        "\nInput rows:",
        len(
            work
        ),
    )

    print(
        "QWEN_REQUIRED rows:",
        len(
            qwen_required
        ),
    )

    # --------------------------------------------------------
    # Frozen evaluation sets
    # --------------------------------------------------------

    validation_files = collect_validation_files(
        split_root
    )

    benchmark_files = collect_benchmark_files(
        benchmark_roots
    )

    frozen = build_frozen_eval_sets(
        validation_files,
        benchmark_files,
    )

    print(
        "\nValidation files loaded:"
    )

    for info in frozen[
        "validation_files"
    ]:
        print(
            "-",
            info[
                "file"
            ],
            "| rows=",
            info[
                "rows"
            ],
            "| pairs=",
            info[
                "pairs_extracted"
            ],
        )

    print(
        "\nBenchmark files loaded:"
    )

    for info in frozen[
        "benchmark_files"
    ]:
        print(
            "-",
            info[
                "file"
            ],
            "| rows=",
            info[
                "rows"
            ],
            "| pairs=",
            info[
                "pairs_extracted"
            ],
        )

    # --------------------------------------------------------
    # Route to final KD target
    # --------------------------------------------------------

    decision_rows = []

    for _, row in work.iterrows():
        decision_rows.append(
            choose_target_for_row(
                row=row,
                canonical_tie_teacher=(
                    args.canonical_tie_teacher
                ),
            )
        )

    decision_df = pd.DataFrame(
        decision_rows
    )

    work = pd.concat(
        [
            work.reset_index(
                drop=True
            ),
            decision_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # Normalize all text needed for audits.
    work[
        "_norm_source"
    ] = work[
        "source_text"
    ].map(
        norm_text
    )

    work[
        "_norm_human"
    ] = work[
        "human_reference"
    ].map(
        norm_text
    )

    work[
        "_norm_kd_target"
    ] = work[
        "kd_target"
    ].fillna(
        ""
    ).map(
        norm_text
    )

    # Reconstruct canonical EN/ZH pair for strict frozen-eval leakage audit.
    norm_en = []
    norm_zh = []

    for _, row in work.iterrows():
        direction = str(
            row[
                "direction"
            ]
        )

        source = row[
            "_norm_source"
        ]

        target = row[
            "_norm_kd_target"
        ]

        if direction == "en_zh":
            norm_en.append(
                source
            )

            norm_zh.append(
                target
            )

        elif direction == "zh_en":
            norm_en.append(
                target
            )

            norm_zh.append(
                source
            )

        else:
            norm_en.append(
                ""
            )

            norm_zh.append(
                ""
            )

    work[
        "_norm_en"
    ] = norm_en

    work[
        "_norm_zh"
    ] = norm_zh

    # --------------------------------------------------------
    # Final rejection reasons in deterministic priority order
    # --------------------------------------------------------

    rejection_reasons = []

    for _, row in work.iterrows():
        reasons = []

        if not as_bool(
            row.get(
                "approved"
            )
        ):
            reasons.append(
                str(
                    row.get(
                        "reason",
                        "ROUTING_REJECT",
                    )
                )
            )

        if (
            not row[
                "_norm_source"
            ]
        ):
            reasons.append(
                "EMPTY_SOURCE"
            )

        if (
            as_bool(
                row.get(
                    "approved"
                )
            )
            and
            not row[
                "_norm_kd_target"
            ]
        ):
            reasons.append(
                "EMPTY_KD_TARGET"
            )

        if (
            row[
                "_norm_source"
            ]
            and
            row[
                "_norm_kd_target"
            ]
            and
            row[
                "_norm_source"
            ]
            ==
            row[
                "_norm_kd_target"
            ]
        ):
            reasons.append(
                "KD_TARGET_EQUALS_SOURCE"
            )

        en = row[
            "_norm_en"
        ]

        zh = row[
            "_norm_zh"
        ]

        pair = (
            en,
            zh,
        )

        if (
            en
            and
            zh
        ):
            if (
                pair
                in frozen[
                    "validation_pairs"
                ]
            ):
                reasons.append(
                    "VALIDATION_EXACT_PAIR_LEAKAGE"
                )

            if (
                en
                in frozen[
                    "validation_en"
                ]
            ):
                reasons.append(
                    "VALIDATION_ENGLISH_LEAKAGE"
                )

            if (
                zh
                in frozen[
                    "validation_zh"
                ]
            ):
                reasons.append(
                    "VALIDATION_CHINESE_LEAKAGE"
                )

            if (
                pair
                in frozen[
                    "benchmark_pairs"
                ]
            ):
                reasons.append(
                    "BENCHMARK_EXACT_PAIR_LEAKAGE"
                )

            if (
                en
                in frozen[
                    "benchmark_en"
                ]
            ):
                reasons.append(
                    "BENCHMARK_ENGLISH_LEAKAGE"
                )

            if (
                zh
                in frozen[
                    "benchmark_zh"
                ]
            ):
                reasons.append(
                    "BENCHMARK_CHINESE_LEAKAGE"
                )

        rejection_reasons.append(
            reasons
        )

    work[
        "rejection_reasons"
    ] = [
        json.dumps(
            x,
            ensure_ascii=False,
        )
        for x in rejection_reasons
    ]

    work[
        "approved_for_kd"
    ] = [
        len(
            x
        )
        ==
        0
        for x in rejection_reasons
    ]

    work[
        "first_rejection_reason"
    ] = [
        (
            x[
                0
            ]
            if x
            else ""
        )
        for x in rejection_reasons
    ]

    # --------------------------------------------------------
    # Exact duplicate audit after target selection.
    # Keep first candidate deterministically, reject later duplicates.
    # --------------------------------------------------------

    approved_mask = work[
        "approved_for_kd"
    ].copy()

    duplicate_mask = (
        work.loc[
            approved_mask,
            [
                "direction",
                "_norm_source",
                "_norm_kd_target",
            ],
        ]
        .duplicated(
            keep="first"
        )
    )

    duplicate_indices = (
        duplicate_mask.loc[
            duplicate_mask
        ].index
    )

    if len(
        duplicate_indices
    ):
        for idx in duplicate_indices:
            existing = json.loads(
                work.at[
                    idx,
                    "rejection_reasons",
                ]
            )

            existing.append(
                "DUPLICATE_SOURCE_KD_TARGET"
            )

            work.at[
                idx,
                "rejection_reasons",
            ] = json.dumps(
                existing,
                ensure_ascii=False,
            )

            work.at[
                idx,
                "approved_for_kd",
            ] = False

            if not work.at[
                idx,
                "first_rejection_reason",
            ]:
                work.at[
                    idx,
                    "first_rejection_reason",
                ] = (
                    "DUPLICATE_SOURCE_KD_TARGET"
                )

    # Keep weight neutral here. Step18H controls Human Replay vs KD mixing.
    work[
        "training_weight"
    ] = work[
        "approved_for_kd"
    ].map(
        lambda x: (
            1.0
            if x
            else 0.0
        )
    )

    work[
        "teacher_equals_human_reference"
    ] = (
        work[
            "_norm_kd_target"
        ]
        ==
        work[
            "_norm_human"
        ]
    )

    final = work.loc[
        work[
            "approved_for_kd"
        ]
    ].copy()

    rejected = work.loc[
        ~work[
            "approved_for_kd"
        ]
    ].copy()

    # --------------------------------------------------------
    # Final strict assertions
    # --------------------------------------------------------

    leakage_cols = [
        "VALIDATION_EXACT_PAIR_LEAKAGE",
        "VALIDATION_ENGLISH_LEAKAGE",
        "VALIDATION_CHINESE_LEAKAGE",
        "BENCHMARK_EXACT_PAIR_LEAKAGE",
        "BENCHMARK_ENGLISH_LEAKAGE",
        "BENCHMARK_CHINESE_LEAKAGE",
    ]

    final_reason_text = (
        final[
            "rejection_reasons"
        ]
        .astype(str)
        .str.cat(
            sep="\n"
        )
        if len(
            final
        )
        else ""
    )

    assertions = {
        "input_rows_20000": (
            len(
                work
            )
            ==
            20000
        ),
        "candidate_id_unique": (
            work[
                "kd_candidate_id"
            ]
            .is_unique
        ),
        "qwen_result_count_4106": (
            len(
                qwen
            )
            ==
            4106
        ),
        "all_qwen_required_have_result": (
            work.loc[
                work[
                    "gate_action"
                ]
                ==
                "QWEN_REQUIRED",
                "winner_model",
            ]
            .notna()
            .all()
        ),
        "no_both_bad_in_final": (
            not (
                final[
                    "winner_model"
                ]
                .fillna(
                    ""
                )
                ==
                "BOTH_BAD"
            )
            .any()
        ),
        "no_empty_source_final": (
            (
                final[
                    "_norm_source"
                ]
                !=
                ""
            )
            .all()
        ),
        "no_empty_target_final": (
            (
                final[
                    "_norm_kd_target"
                ]
                !=
                ""
            )
            .all()
        ),
        "no_target_equals_source_final": (
            (
                final[
                    "_norm_source"
                ]
                !=
                final[
                    "_norm_kd_target"
                ]
            )
            .all()
        ),
        "no_duplicate_source_target_final": (
            not final[
                [
                    "direction",
                    "_norm_source",
                    "_norm_kd_target",
                ]
            ]
            .duplicated()
            .any()
        ),
        "all_frozen_leakage_zero_final": (
            all(
                token
                not in final_reason_text
                for token in leakage_cols
            )
        ),
        "training_weight_one_final": (
            (
                final[
                    "training_weight"
                ]
                ==
                1.0
            )
            .all()
        ),
    }

    failed = [
        key
        for key, value in assertions.items()
        if not bool(
            value
        )
    ]

    if failed:
        raise RuntimeError(
            "STEP18G assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    # --------------------------------------------------------
    # Clean helper columns from persisted user-facing datasets
    # --------------------------------------------------------

    helper_cols = [
        "_norm_source",
        "_norm_human",
        "_norm_kd_target",
        "_norm_en",
        "_norm_zh",
    ]

    final_save = final.drop(
        columns=[
            c
            for c in helper_cols
            if c in final.columns
        ]
    )

    rejected_save = rejected.drop(
        columns=[
            c
            for c in helper_cols
            if c in rejected.columns
        ]
    )

    final_save.to_parquet(
        final_path,
        index=False,
    )

    rejected_save.to_parquet(
        rejected_path,
        index=False,
    )

    summary = (
        final_save.groupby(
            [
                "direction",
                "source_dataset",
                "target_origin",
                "selected_teacher",
            ],
            dropna=False,
        )
        .agg(
            rows=(
                "kd_candidate_id",
                "size",
            ),
            mean_disagreement_score=(
                "teacher_disagreement_score",
                "mean",
            ),
            teacher_equals_human_rows=(
                "teacher_equals_human_reference",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "direction",
                "source_dataset",
                "target_origin",
                "selected_teacher",
            ]
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    rejected_counts = {
        str(k): int(v)
        for k, v in rejected_save[
            "first_rejection_reason"
        ]
        .value_counts()
        .to_dict()
        .items()
    }

    report = {
        "step": "18G",
        "step_version": STEP_VERSION,
        "policy": {
            "canonical_qwen_tie_teacher": (
                args.canonical_tie_teacher
            ),
            "qwen_both_bad": (
                "reject_from_kd"
            ),
            "qwen_tie": (
                "accept only when both Teachers are safe; "
                f"use canonical {args.canonical_tie_teacher}"
            ),
            "training_weight": (
                "1.0 for every approved KD row; "
                "Step18H controls Human/KD mixture"
            ),
        },
        "inputs": {
            "projection": str(
                projection_path
            ),
            "qwen_results": str(
                qwen_path
            ),
            "validation_files": frozen[
                "validation_files"
            ],
            "benchmark_files": frozen[
                "benchmark_files"
            ],
        },
        "counts": {
            "input_rows": int(
                len(
                    work
                )
            ),
            "approved_kd_rows": int(
                len(
                    final_save
                )
            ),
            "rejected_rows": int(
                len(
                    rejected_save
                )
            ),
            "approved_percent": (
                100.0
                *
                len(
                    final_save
                )
                /
                len(
                    work
                )
                if len(
                    work
                )
                else 0.0
            ),
            "direction": {
                str(k): int(v)
                for k, v in final_save[
                    "direction"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "selected_teacher": {
                str(k): int(v)
                for k, v in final_save[
                    "selected_teacher"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "target_origin": {
                str(k): int(v)
                for k, v in final_save[
                    "target_origin"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
            "teacher_equals_human_reference": int(
                final_save[
                    "teacher_equals_human_reference"
                ]
                .sum()
            ),
            "rejection_first_reason": (
                rejected_counts
            ),
        },
        "assertions": {
            key: bool(
                value
            )
            for key, value in assertions.items()
        },
        "outputs": {
            "final_kd_dataset": str(
                final_path
            ),
            "rejected": str(
                rejected_path
            ),
            "summary": str(
                summary_path
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18H_BUILD_EXP2_TRAINING"
        ),
    }

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Console result
    # --------------------------------------------------------

    print(
        "\n"
        +
        "=" * 115
    )

    print(
        "STEP 18G RESULT"
    )

    print(
        "=" * 115
    )

    print(
        "\nInput rows:",
        len(
            work
        ),
    )

    print(
        "Approved KD rows:",
        len(
            final_save
        ),
        f"({100.0 * len(final_save) / len(work):.2f}%)",
    )

    print(
        "Rejected rows:",
        len(
            rejected_save
        ),
    )

    print(
        "\nSelected Teacher:"
    )

    print(
        final_save[
            "selected_teacher"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTarget origin:"
    )

    print(
        final_save[
            "target_origin"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nDirection:"
    )

    print(
        final_save[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nTeacher == human reference:",
        int(
            final_save[
                "teacher_equals_human_reference"
            ]
            .sum()
        ),
    )

    print(
        "\nRejected first reason:"
    )

    if len(
        rejected_save
    ):
        print(
            rejected_save[
                "first_rejection_reason"
            ]
            .value_counts()
            .to_string()
        )

    else:
        print(
            "None"
        )

    print(
        "\nSummary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nAssertions:"
    )

    for key, value in assertions.items():
        print(
            f"{key}: {bool(value)}"
        )

    print(
        "\nFinal KD dataset:"
    )

    print(
        final_path
    )

    print(
        "\nRejected:"
    )

    print(
        rejected_path
    )

    print(
        "\nReport:"
    )

    print(
        report_path
    )

    print(
        "\nSTATUS:"
    )

    print(
        "READY_FOR_STEP_18H_BUILD_EXP2_TRAINING"
    )


if __name__ == "__main__":
    main()
