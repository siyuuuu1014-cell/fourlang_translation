from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


STEP_VERSION = "18E1_V1"
DEFAULT_SEED = 2026


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18E1 - build a balanced Qwen pairwise Teacher-routing "
            "calibration set from Step18D disagreement scores."
        )
    )
    p.add_argument("--project_root", default=None)
    p.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/"
            "18d_disagreement/"
            "teacher_disagreement_20k_v1.parquet"
        ),
    )
    p.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18e_qwen_calibration"
        ),
    )
    p.add_argument(
        "--per_direction_source",
        type=int,
        default=200,
        help=(
            "Rows per direction x source_dataset stratum. "
            "With 2 directions x 2 sources, 200 gives 800 total rows."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def stable_int(text: str) -> int:
    return int(
        hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        16,
    )


def assign_local_bands(part: pd.DataFrame) -> pd.DataFrame:
    out = part.copy()

    exact_mask = out["opus_madlad_exact_match"].fillna(False).astype(bool)

    out["calibration_band"] = ""
    out.loc[exact_mask, "calibration_band"] = "B0_EXACT"

    nonexact = out.loc[~exact_mask].copy()

    if len(nonexact) == 0:
        return out

    # Local quartiles inside each direction x source_dataset group.
    q25 = float(nonexact["teacher_disagreement_score"].quantile(0.25))
    q50 = float(nonexact["teacher_disagreement_score"].quantile(0.50))
    q75 = float(nonexact["teacher_disagreement_score"].quantile(0.75))

    s = nonexact["teacher_disagreement_score"]

    out.loc[
        nonexact.index[s <= q25],
        "calibration_band",
    ] = "B1_LOW"

    out.loc[
        nonexact.index[(s > q25) & (s <= q50)],
        "calibration_band",
    ] = "B2_MEDIUM"

    out.loc[
        nonexact.index[(s > q50) & (s <= q75)],
        "calibration_band",
    ] = "B3_HIGH"

    out.loc[
        nonexact.index[s > q75],
        "calibration_band",
    ] = "B4_VERY_HIGH"

    return out


def sample_group(
    part: pd.DataFrame,
    target: int,
    seed: int,
    group_key: str,
) -> pd.DataFrame:
    bands = [
        "B0_EXACT",
        "B1_LOW",
        "B2_MEDIUM",
        "B3_HIGH",
        "B4_VERY_HIGH",
    ]

    # Aim for equal coverage across five bands.
    base = target // len(bands)
    remainder = target % len(bands)

    quotas = {
        band: base + (1 if i < remainder else 0)
        for i, band in enumerate(bands)
    }

    selected = []
    selected_ids = set()

    for band in bands:
        pool = part.loc[
            part["calibration_band"] == band
        ].copy()

        n = min(
            quotas[band],
            len(pool),
        )

        if n <= 0:
            continue

        random_state = (
            seed
            + stable_int(f"{group_key}|{band}") % 1_000_000_000
        )

        pick = pool.sample(
            n=n,
            replace=False,
            random_state=random_state,
        )

        selected.append(pick)
        selected_ids.update(
            pick["kd_candidate_id"].astype(str).tolist()
        )

    if selected:
        out = pd.concat(selected, ignore_index=True)
    else:
        out = part.iloc[0:0].copy()

    # Fill any deficit from the remaining rows, preferring higher disagreement
    # because those are most informative for Teacher routing.
    deficit = target - len(out)

    if deficit > 0:
        remaining = part.loc[
            ~part["kd_candidate_id"]
            .astype(str)
            .isin(selected_ids)
        ].copy()

        remaining = remaining.sort_values(
            [
                "teacher_disagreement_score",
                "kd_candidate_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

        if len(remaining) < deficit:
            raise RuntimeError(
                f"Not enough rows to fill {group_key}: "
                f"need={deficit}, available={len(remaining)}"
            )

        out = pd.concat(
            [
                out,
                remaining.head(deficit),
            ],
            ignore_index=True,
        )

    if len(out) != target:
        raise RuntimeError(
            f"Calibration sample size mismatch for {group_key}: "
            f"expected={target}, observed={len(out)}"
        )

    return out


def main():
    args = parse_args()

    root = (
        Path(args.project_root).resolve()
        if args.project_root
        else infer_project_root()
    )

    input_path = resolve_path(root, args.input)
    outdir = resolve_path(root, args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    outdir.mkdir(parents=True, exist_ok=True)

    output_path = (
        outdir
        / "qwen_teacher_routing_calibration_800_v1.parquet"
    )
    manifest_path = (
        outdir
        / "qwen_teacher_routing_calibration_manifest_v1.csv"
    )
    report_path = (
        outdir
        / "qwen_teacher_routing_calibration_report_v1.json"
    )

    outputs = [
        output_path,
        manifest_path,
        report_path,
    ]

    if not args.overwrite and any(p.exists() for p in outputs):
        raise RuntimeError(
            "Step18E1 outputs already exist. Use --overwrite."
        )

    print("=" * 110)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18E1 - BUILD QWEN TEACHER-ROUTING CALIBRATION SET")
    print("=" * 110)

    print("\nInput:")
    print(input_path)

    df = pd.read_parquet(input_path).copy()

    required = {
        "kd_candidate_id",
        "pair_id",
        "direction",
        "source_dataset",
        "quality_tier",
        "source_lang",
        "target_lang",
        "source_text",
        "human_reference",
        "opus_prediction",
        "madlad_prediction",
        "opus_madlad_exact_match",
        "teacher_disagreement_score",
        "opus_reference_chrfpp",
        "madlad_reference_chrfpp",
        "reference_metric_winner",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["kd_candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate kd_candidate_id in Step18D input.")

    expected_groups = {
        ("en_zh", "ALT"),
        ("en_zh", "Tatoeba"),
        ("zh_en", "ALT"),
        ("zh_en", "Tatoeba"),
    }

    observed_groups = set(
        tuple(x)
        for x in (
            df[["direction", "source_dataset"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )

    if observed_groups != expected_groups:
        raise RuntimeError(
            f"Unexpected direction/source groups: {observed_groups}"
        )

    banded_parts = []

    for (direction, source), part in df.groupby(
        ["direction", "source_dataset"],
        sort=True,
    ):
        banded = assign_local_bands(part)

        banded["calibration_group"] = (
            str(direction)
            + "|"
            + str(source)
        )

        banded_parts.append(banded)

    work = pd.concat(
        banded_parts,
        ignore_index=True,
    )

    selected_parts = []

    for (direction, source), part in work.groupby(
        ["direction", "source_dataset"],
        sort=True,
    ):
        group_key = f"{direction}|{source}"

        selected = sample_group(
            part=part,
            target=args.per_direction_source,
            seed=args.seed,
            group_key=group_key,
        )

        selected_parts.append(selected)

    calibration = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    calibration["qwen_review_id"] = [
        f"zh_en_kd_cal_{i:05d}"
        for i in range(len(calibration))
    ]

    # Randomize final order deterministically so the judge does not see
    # long blocks of the same direction/source/band.
    calibration["_shuffle_key"] = calibration[
        "qwen_review_id"
    ].map(
        lambda x: stable_int(f"{args.seed}|{x}")
    )

    calibration = (
        calibration.sort_values("_shuffle_key")
        .drop(columns=["_shuffle_key"])
        .reset_index(drop=True)
    )

    total_expected = (
        args.per_direction_source
        * 4
    )

    group_counts = (
        calibration.groupby(
            ["direction", "source_dataset"]
        )
        .size()
    )

    assertions = {
        "total_rows_exact": len(calibration) == total_expected,
        "candidate_id_unique": calibration["kd_candidate_id"].is_unique,
        "review_id_unique": calibration["qwen_review_id"].is_unique,
        "all_four_direction_source_groups_present": (
            len(group_counts) == 4
        ),
        "each_direction_source_exact": (
            group_counts.eq(args.per_direction_source).all()
        ),
        "no_empty_source": (
            calibration["source_text"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_empty_opus": (
            calibration["opus_prediction"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_empty_madlad": (
            calibration["madlad_prediction"]
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
    }

    failed = [
        k
        for k, v in assertions.items()
        if not bool(v)
    ]

    if failed:
        raise RuntimeError(
            "STEP18E1 assertion failure:\n"
            + "\n".join(failed)
        )

    manifest = (
        calibration.groupby(
            [
                "direction",
                "source_dataset",
                "calibration_band",
            ],
            dropna=False,
        )
        .agg(
            rows=("kd_candidate_id", "size"),
            mean_disagreement_score=(
                "teacher_disagreement_score",
                "mean",
            ),
            min_disagreement_score=(
                "teacher_disagreement_score",
                "min",
            ),
            max_disagreement_score=(
                "teacher_disagreement_score",
                "max",
            ),
            mean_opus_reference_chrfpp=(
                "opus_reference_chrfpp",
                "mean",
            ),
            mean_madlad_reference_chrfpp=(
                "madlad_reference_chrfpp",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "direction",
                "source_dataset",
                "calibration_band",
            ]
        )
    )

    calibration.to_parquet(
        output_path,
        index=False,
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = {
        "step": "18E1",
        "step_version": STEP_VERSION,
        "seed": int(args.seed),
        "input": str(input_path),
        "sampling": {
            "per_direction_source": int(
                args.per_direction_source
            ),
            "expected_total": int(total_expected),
            "stratification": [
                "direction",
                "source_dataset",
                "local disagreement band",
            ],
            "bands": [
                "B0_EXACT",
                "B1_LOW",
                "B2_MEDIUM",
                "B3_HIGH",
                "B4_VERY_HIGH",
            ],
            "note": (
                "Bands are computed inside each direction x source_dataset "
                "stratum. Human-reference chrF++ is diagnostic only and is "
                "not used to select rows."
            ),
        },
        "counts": {
            "rows": int(len(calibration)),
            "direction_source": {
                f"{d}|{s}": int(n)
                for (d, s), n
                in group_counts.to_dict().items()
            },
            "band": {
                str(k): int(v)
                for k, v
                in calibration[
                    "calibration_band"
                ]
                .value_counts()
                .to_dict()
                .items()
            },
        },
        "assertions": {
            k: bool(v)
            for k, v
            in assertions.items()
        },
        "outputs": {
            "calibration": str(output_path),
            "manifest": str(manifest_path),
        },
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "status": "READY_FOR_STEP_18E2_QWEN_PAIRWISE_CALIBRATION",
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("STEP 18E1 RESULT")
    print("=" * 110)

    print("\nRows:", len(calibration))

    print("\nDirection × source:")
    print(
        calibration.groupby(
            ["direction", "source_dataset"]
        )
        .size()
        .to_string()
    )

    print("\nCalibration band:")
    print(
        calibration[
            "calibration_band"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nManifest:")
    print(
        manifest[
            [
                "direction",
                "source_dataset",
                "calibration_band",
                "rows",
                "mean_disagreement_score",
                "min_disagreement_score",
                "max_disagreement_score",
            ]
        ].to_string(index=False)
    )

    print("\nAssertions:")
    for k, v in assertions.items():
        print(f"{k}: {bool(v)}")

    print("\nCalibration set:")
    print(output_path)

    print("\nReport:")
    print(report_path)

    print("\nSTATUS:")
    print("READY_FOR_STEP_18E2_QWEN_PAIRWISE_CALIBRATION")


if __name__ == "__main__":
    main()
