from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF


STEP_VERSION = "18D_V1"
CHRFPP = CHRF(word_order=2)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 18D - analyze OPUS vs MADLAD disagreement on the "
            "20K train-only ZH<->EN Teacher subset. No teacher is selected here."
        )
    )
    p.add_argument("--project_root", default=None)
    p.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/"
            "18c1_madlad_generation_20k/"
            "opus_madlad_teacher_predictions_train_only_v1.parquet"
        ),
    )
    p.add_argument(
        "--output_dir",
        default="data/distillation/zh_en/v1/18d_disagreement",
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


def normalize_text(value) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    repl = {
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
    for a, b in repl.items():
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def sentence_chrfpp(hyp: str, ref: str) -> float:
    hyp = normalize_text(hyp)
    ref = normalize_text(ref)
    if hyp == "" and ref == "":
        return 100.0
    if hyp == "" or ref == "":
        return 0.0
    return float(CHRFPP.sentence_score(hyp, [ref]).score)


def safe_ratio(a: float, b: float) -> float:
    return float(a / b) if b > 0 else math.nan


def text_units(text: str, lang: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    if lang == "en":
        return len([x for x in text.split() if x])
    if lang == "zh":
        return sum(1 for ch in text if not ch.isspace())
    return len(text)


def quantile_dict(series: pd.Series) -> dict:
    qs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    return {
        f"q{int(q * 100):02d}": float(series.quantile(q))
        for q in qs
    }


def summarize(part: pd.DataFrame) -> dict:
    n = len(part)
    exact = int(part["opus_madlad_exact_match"].sum())
    margin = part["madlad_minus_opus_reference_chrfpp"]
    mad_better = int((margin > 0).sum())
    opus_better = int((margin < 0).sum())
    tied = int((margin == 0).sum())
    return {
        "rows": int(n),
        "exact_agreement_rows": exact,
        "exact_agreement_percent": 100.0 * exact / n if n else 0.0,
        "mean_opus_madlad_chrfpp": float(part["opus_madlad_chrfpp"].mean()),
        "median_opus_madlad_chrfpp": float(part["opus_madlad_chrfpp"].median()),
        "mean_disagreement_score": float(part["teacher_disagreement_score"].mean()),
        "median_disagreement_score": float(part["teacher_disagreement_score"].median()),
        "mean_opus_reference_chrfpp": float(part["opus_reference_chrfpp"].mean()),
        "mean_madlad_reference_chrfpp": float(part["madlad_reference_chrfpp"].mean()),
        "mean_madlad_minus_opus_reference_chrfpp": float(margin.mean()),
        "madlad_reference_better_rows": mad_better,
        "opus_reference_better_rows": opus_better,
        "reference_metric_tie_rows": tied,
        "madlad_reference_better_percent": 100.0 * mad_better / n if n else 0.0,
        "opus_reference_better_percent": 100.0 * opus_better / n if n else 0.0,
        "disagreement_quantiles": quantile_dict(part["teacher_disagreement_score"]),
    }


def main():
    args = parse_args()
    root = Path(args.project_root).resolve() if args.project_root else infer_project_root()
    input_path = resolve_path(root, args.input)
    outdir = resolve_path(root, args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    outdir.mkdir(parents=True, exist_ok=True)

    scored_output = outdir / "teacher_disagreement_20k_v1.parquet"
    group_output = outdir / "teacher_disagreement_group_summary_v1.csv"
    decile_output = outdir / "teacher_disagreement_decile_summary_v1.csv"
    report_output = outdir / "teacher_disagreement_report_v1.json"

    outputs = [scored_output, group_output, decile_output, report_output]
    if not args.overwrite and any(p.exists() for p in outputs):
        raise RuntimeError("Step18D outputs already exist. Use --overwrite to rebuild.")

    print("=" * 110)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18D - OPUS VS MADLAD DISAGREEMENT ANALYSIS")
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
        "allowed_for_kd_training",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")

    if df["kd_candidate_id"].duplicated().any():
        raise RuntimeError("Duplicate kd_candidate_id.")

    if df["opus_prediction"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Empty OPUS prediction exists.")

    if df["madlad_prediction"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Empty MADLAD prediction exists.")

    print("\nRows:", len(df))
    print("\nDirection:")
    print(df["direction"].value_counts().to_string())

    opus_madlad = []
    opus_ref = []
    madlad_ref = []
    exact = []
    src_units = []
    opus_units = []
    mad_units = []

    print("\nScoring sentence-level chrF++...")

    for i, row in df.iterrows():
        src = normalize_text(row["source_text"])
        human = normalize_text(row["human_reference"])
        opus = normalize_text(row["opus_prediction"])
        mad = normalize_text(row["madlad_prediction"])

        exact.append(opus == mad)
        opus_madlad.append(sentence_chrfpp(opus, mad))
        opus_ref.append(sentence_chrfpp(opus, human))
        madlad_ref.append(sentence_chrfpp(mad, human))

        su = text_units(src, str(row["source_lang"]))
        ou = text_units(opus, str(row["target_lang"]))
        mu = text_units(mad, str(row["target_lang"]))

        src_units.append(su)
        opus_units.append(ou)
        mad_units.append(mu)

        if (i + 1) % 2000 == 0 or (i + 1) == len(df):
            print(f"{i + 1}/{len(df)}")

    df["opus_madlad_exact_match"] = exact
    df["opus_madlad_chrfpp"] = opus_madlad
    df["teacher_disagreement_score"] = (
        100.0 - df["opus_madlad_chrfpp"]
    ).clip(0.0, 100.0)

    df["opus_reference_chrfpp"] = opus_ref
    df["madlad_reference_chrfpp"] = madlad_ref
    df["madlad_minus_opus_reference_chrfpp"] = (
        df["madlad_reference_chrfpp"] - df["opus_reference_chrfpp"]
    )

    df["source_units"] = src_units
    df["opus_target_units"] = opus_units
    df["madlad_target_units"] = mad_units
    df["opus_source_length_ratio"] = [
        safe_ratio(a, b) for a, b in zip(opus_units, src_units)
    ]
    df["madlad_source_length_ratio"] = [
        safe_ratio(a, b) for a, b in zip(mad_units, src_units)
    ]
    df["teacher_output_length_ratio"] = [
        safe_ratio(a, b) for a, b in zip(mad_units, opus_units)
    ]

    ranks = df["teacher_disagreement_score"].rank(method="first")
    df["disagreement_decile"] = pd.qcut(
        ranks,
        q=10,
        labels=[
            "D01_lowest", "D02", "D03", "D04", "D05",
            "D06", "D07", "D08", "D09", "D10_highest"
        ],
    ).astype(str)

    df["reference_metric_winner"] = "TIE"
    df.loc[
        df["madlad_minus_opus_reference_chrfpp"] > 0,
        "reference_metric_winner",
    ] = "MADLAD"
    df.loc[
        df["madlad_minus_opus_reference_chrfpp"] < 0,
        "reference_metric_winner",
    ] = "OPUS"

    # IMPORTANT: diagnostic only
    df["teacher_selected"] = False
    df["teacher_selection"] = "UNDECIDED"
    df["ready_for_qwen_calibration"] = True

    group_rows = []
    for (direction, source_dataset, quality_tier), part in df.groupby(
        ["direction", "source_dataset", "quality_tier"],
        dropna=False,
    ):
        s = summarize(part)
        row = {
            "direction": direction,
            "source_dataset": source_dataset,
            "quality_tier": quality_tier,
        }
        for k, v in s.items():
            if k == "disagreement_quantiles":
                for qk, qv in v.items():
                    row[f"disagreement_{qk}"] = qv
            else:
                row[k] = v
        group_rows.append(row)

    group_summary = pd.DataFrame(group_rows).sort_values(
        ["direction", "source_dataset", "quality_tier"]
    )

    decile_summary = (
        df.groupby(
            ["direction", "source_dataset", "disagreement_decile"],
            dropna=False,
        )
        .agg(
            rows=("kd_candidate_id", "size"),
            mean_disagreement_score=("teacher_disagreement_score", "mean"),
            min_disagreement_score=("teacher_disagreement_score", "min"),
            max_disagreement_score=("teacher_disagreement_score", "max"),
            mean_opus_reference_chrfpp=("opus_reference_chrfpp", "mean"),
            mean_madlad_reference_chrfpp=("madlad_reference_chrfpp", "mean"),
            madlad_reference_better_percent=(
                "reference_metric_winner",
                lambda s: 100.0 * (s == "MADLAD").mean(),
            ),
            opus_reference_better_percent=(
                "reference_metric_winner",
                lambda s: 100.0 * (s == "OPUS").mean(),
            ),
        )
        .reset_index()
    )

    overall = summarize(df)
    direction_summary = {
        d: summarize(part)
        for d, part in df.groupby("direction")
    }
    source_summary = {
        f"{d}|{s}": summarize(part)
        for (d, s), part in df.groupby(["direction", "source_dataset"])
    }

    assertions = {
        "rows_preserved": len(df) == 20000,
        "candidate_id_unique": df["kd_candidate_id"].is_unique,
        "en_zh_10000": int((df["direction"] == "en_zh").sum()) == 10000,
        "zh_en_10000": int((df["direction"] == "zh_en").sum()) == 10000,
        "no_empty_opus": df["opus_prediction"].astype(str).str.strip().ne("").all(),
        "no_empty_madlad": df["madlad_prediction"].astype(str).str.strip().ne("").all(),
        "disagreement_score_in_range": df["teacher_disagreement_score"].between(0, 100).all(),
        "no_teacher_selected_in_18d": (df["teacher_selected"] == False).all(),
    }

    failed = [k for k, v in assertions.items() if not bool(v)]
    if failed:
        raise RuntimeError("STEP18D assertion failure:\n" + "\n".join(failed))

    df.to_parquet(scored_output, index=False)
    group_summary.to_csv(group_output, index=False, encoding="utf-8-sig")
    decile_summary.to_csv(decile_output, index=False, encoding="utf-8-sig")

    report = {
        "step": "18D",
        "step_version": STEP_VERSION,
        "input": str(input_path),
        "metric": {
            "pairwise_similarity": "sentence-level chrF++",
            "chrf_word_order": 2,
            "teacher_disagreement_score": "100 - OPUS_vs_MADLAD_sentence_chrF++",
            "human_reference_metrics_are_diagnostic_only": True,
            "teacher_selected_in_this_step": False,
            "fixed_qwen_threshold_selected_in_this_step": False,
        },
        "overall": overall,
        "direction": direction_summary,
        "direction_source": source_summary,
        "assertions": {k: bool(v) for k, v in assertions.items()},
        "outputs": {
            "scored_rows": str(scored_output),
            "group_summary": str(group_output),
            "decile_summary": str(decile_output),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_STEP_18E_QWEN_GATE_CALIBRATION",
    }

    with report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 110)
    print("STEP 18D RESULT")
    print("=" * 110)

    print("\nRows:", len(df))
    print("\nOverall:")
    print(
        "Exact OPUS == MADLAD:",
        overall["exact_agreement_rows"],
        f"({overall['exact_agreement_percent']:.2f}%)",
    )
    print(
        "Mean OPUS↔MADLAD chrF++:",
        f"{overall['mean_opus_madlad_chrfpp']:.4f}",
    )
    print(
        "Median OPUS↔MADLAD chrF++:",
        f"{overall['median_opus_madlad_chrfpp']:.4f}",
    )
    print(
        "Mean disagreement score:",
        f"{overall['mean_disagreement_score']:.4f}",
    )
    print(
        "Median disagreement score:",
        f"{overall['median_disagreement_score']:.4f}",
    )

    print("\nDisagreement quantiles:")
    for k, v in overall["disagreement_quantiles"].items():
        print(f"{k}: {v:.4f}")

    print("\nReference diagnostic:")
    print(
        "Mean OPUS→Human chrF++:",
        f"{overall['mean_opus_reference_chrfpp']:.4f}",
    )
    print(
        "Mean MADLAD→Human chrF++:",
        f"{overall['mean_madlad_reference_chrfpp']:.4f}",
    )
    print(
        "MADLAD reference-better:",
        overall["madlad_reference_better_rows"],
        f"({overall['madlad_reference_better_percent']:.2f}%)",
    )
    print(
        "OPUS reference-better:",
        overall["opus_reference_better_rows"],
        f"({overall['opus_reference_better_percent']:.2f}%)",
    )

    print("\nDirection summary:")
    for direction in ["en_zh", "zh_en"]:
        s = direction_summary[direction]
        print(f"\n{direction}")
        print("rows:", s["rows"])
        print(
            "exact_agreement_percent:",
            f"{s['exact_agreement_percent']:.2f}",
        )
        print(
            "mean_disagreement_score:",
            f"{s['mean_disagreement_score']:.4f}",
        )
        print(
            "mean_opus_ref_chrfpp:",
            f"{s['mean_opus_reference_chrfpp']:.4f}",
        )
        print(
            "mean_madlad_ref_chrfpp:",
            f"{s['mean_madlad_reference_chrfpp']:.4f}",
        )
        print(
            "madlad_ref_better_percent:",
            f"{s['madlad_reference_better_percent']:.2f}",
        )

    print("\nIMPORTANT:")
    print("18D is diagnostic only.")
    print("No OPUS/MADLAD teacher is hard-selected here.")
    print("No fixed Qwen threshold is chosen here.")

    print("\nScored rows:")
    print(scored_output)

    print("\nGroup summary:")
    print(group_output)

    print("\nDecile summary:")
    print(decile_output)

    print("\nReport:")
    print(report_output)

    print("\nSTATUS:")
    print("READY_FOR_STEP_18E_QWEN_GATE_CALIBRATION")


if __name__ == "__main__":
    main()
