from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sacrebleu import corpus_bleu, corpus_chrf
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


STEP_VERSION = "20_EN_ZH_EXP1_VS_EXP2_V2_GENERATION_V1"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step20 - apples-to-apples generation evaluation for EN->ZH: "
            "Exp1 vs Exp2 V2 epoch1 vs epoch2."
        )
    )
    p.add_argument(
        "--project_root",
        default="/root/autodl-tmp/fourlang_translation",
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_source_length", type=int, default=256)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def clean_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def load_pairs(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path).copy()

    required = {"en", "zh"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{name} missing required columns: {sorted(missing)}"
        )

    out = pd.DataFrame({
        "source": df["en"].map(clean_text),
        "reference": df["zh"].map(clean_text),
    })

    if "pair_id" in df.columns:
        out["pair_id"] = df["pair_id"].astype(str)
    else:
        out["pair_id"] = [
            f"{name}_{i:06d}"
            for i in range(len(out))
        ]

    if (out["source"] == "").any():
        raise RuntimeError(f"{name}: empty source detected.")

    if (out["reference"] == "").any():
        raise RuntimeError(f"{name}: empty reference detected.")

    return out


def metric_bundle(predictions, references):
    # EN -> ZH: BLEU tokenizer MUST be `zh`.
    bleu = corpus_bleu(
        predictions,
        [references],
        tokenize="zh",
    ).score

    chrf = corpus_chrf(
        predictions,
        [references],
        char_order=6,
        word_order=0,
        beta=2,
    ).score

    chrfpp = corpus_chrf(
        predictions,
        [references],
        char_order=6,
        word_order=2,
        beta=2,
    ).score

    exact = (
        100.0
        * sum(
            p.strip() == r.strip()
            for p, r in zip(predictions, references)
        )
        / len(references)
    )

    empty = sum(
        not p.strip()
        for p in predictions
    )

    return {
        "bleu": float(bleu),
        "chrf": float(chrf),
        "chrfpp": float(chrfpp),
        "exact_match_percent": float(exact),
        "empty_predictions": int(empty),
    }


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )

    kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        kwargs["dtype"] = torch.float16

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        **kwargs,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


@torch.inference_mode()
def generate_dataset(
    *,
    model_name: str,
    model_path: Path,
    dataset_name: str,
    df: pd.DataFrame,
    batch_size: int,
    max_source_length: int,
):
    print("\n" + "=" * 112)
    print(f"{model_name} | {dataset_name}")
    print("=" * 112)
    print("Model:", model_path)
    print("Samples:", len(df))

    tokenizer, model, device = load_model(model_path)

    params = sum(p.numel() for p in model.parameters())
    print("Parameters:", f"{params:,}")

    # Reuse the generation configuration saved with Exp1/Exp2.
    # This keeps the comparison aligned with the specialist's own
    # frozen generation configuration (e.g. EN->ZH num_beams=4).
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False

    # Remove sampling-only fields if present, avoiding warnings.
    try:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    predictions = []
    truncated_rows = 0

    total_generation_seconds = 0.0

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]

        sources = batch["source"].tolist()

        # Check truncation against the same tokenizer inputs.
        raw_lengths = tokenizer(
            sources,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]

        truncated_rows += sum(
            len(ids) > max_source_length
            for ids in raw_lengths
        )

        encoded = tokenizer(
            sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_source_length,
        )

        encoded = {
            k: v.to(device)
            for k, v in encoded.items()
        }

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        generated = model.generate(
            **encoded,
            generation_config=generation_config,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_generation_seconds += (
            time.perf_counter() - t0
        )

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        predictions.extend(
            x.strip()
            for x in decoded
        )

        done = min(
            start + len(batch),
            len(df),
        )

        if (
            done == len(df)
            or done % 256 < batch_size
        ):
            print(f"{done}/{len(df)}")

    refs = df["reference"].tolist()

    metrics = metric_bundle(
        predictions,
        refs,
    )

    metrics.update({
        "model_name": model_name,
        "dataset": dataset_name,
        "samples": int(len(df)),
        "generation_seconds": float(total_generation_seconds),
        "avg_generation_latency_seconds": (
            float(total_generation_seconds / len(df))
        ),
        "peak_gpu_memory_gib": (
            float(
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )
            if torch.cuda.is_available()
            else 0.0
        ),
        "truncated_source_rows": int(truncated_rows),
        "parameters": int(params),
    })

    pred_df = df.copy()
    pred_df["model_name"] = model_name
    pred_df["dataset"] = dataset_name
    pred_df["prediction"] = predictions

    print()
    print(f"BLEU:   {metrics['bleu']:.4f}")
    print(f"chrF:   {metrics['chrf']:.4f}")
    print(f"chrF++: {metrics['chrfpp']:.4f}")
    print(
        "Exact: "
        f"{metrics['exact_match_percent']:.4f}%"
    )
    print(
        "Latency: "
        f"{metrics['avg_generation_latency_seconds']:.4f}s/sample"
    )
    print(
        "Peak GPU: "
        f"{metrics['peak_gpu_memory_gib']:.3f} GiB"
    )
    print(
        "Truncated sources:",
        metrics["truncated_source_rows"],
    )

    del model
    del tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, pred_df


def main():
    args = parse_args()

    root = Path(args.project_root).resolve()

    model_paths = {
        "EXP1_BASELINE": (
            root
            / "results/specialists/en_zh/"
              "opus_mt_en_zh/exp1_human/best_model"
        ),
        "EXP2_V2_EPOCH1": (
            root
            / "results/specialists/en_zh/"
              "opus_mt_en_zh/exp2_kd_v2/"
              "epoch_models/epoch_1"
        ),
        "EXP2_V2_EPOCH2": (
            root
            / "results/specialists/en_zh/"
              "opus_mt_en_zh/exp2_kd_v2/"
              "epoch_models/epoch_2"
        ),
    }

    datasets = {
        "frozen_validation": (
            root
            / "data/splits/zh_en/v1/"
              "validation_pairs_v1.parquet"
        ),
        "flores_devtest": (
            root
            / "data/benchmark/zh_en/"
              "flores_plus_zh_en_devtest_v1.parquet"
        ),
        "tatoeba": (
            root
            / "data/benchmark/zh_en/"
              "tatoeba_zh_en_test_v1.parquet"
        ),
    }

    output_dir = (
        root
        / "results/specialists/en_zh/"
          "opus_mt_en_zh/exp2_kd_v2/"
          "step20_generation_eval"
    )

    if output_dir.exists() and not args.overwrite:
        raise RuntimeError(
            f"Output exists: {output_dir}\n"
            "Use --overwrite to rebuild Step20."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 112)
    print("ZH-EN SPECIALIST DISTILLATION PIPELINE")
    print("STEP 20 - EN->ZH EXP1 VS EXP2 V2 GENERATION EVALUATION")
    print("=" * 112)

    print("\nGPU:")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    else:
        print("CPU")

    print("\nModels:")
    for name, path in model_paths.items():
        print(f"{name}: {path}")
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model: {name} -> {path}"
            )

    loaded_datasets = {
        name: load_pairs(path, name)
        for name, path in datasets.items()
    }

    print("\nDatasets:")
    for name, df in loaded_datasets.items():
        print(name, ":", len(df))

    metric_rows = []
    prediction_frames = []

    for model_name, model_path in model_paths.items():
        for dataset_name, df in loaded_datasets.items():
            metrics, pred_df = generate_dataset(
                model_name=model_name,
                model_path=model_path,
                dataset_name=dataset_name,
                df=df,
                batch_size=args.batch_size,
                max_source_length=args.max_source_length,
            )
            metric_rows.append(metrics)
            prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(metric_rows)

    # Exp1-relative deltas.
    baseline = (
        metrics_df[
            metrics_df["model_name"] == "EXP1_BASELINE"
        ][
            [
                "dataset",
                "bleu",
                "chrfpp",
                "exact_match_percent",
            ]
        ]
        .rename(
            columns={
                "bleu": "exp1_bleu",
                "chrfpp": "exp1_chrfpp",
                "exact_match_percent": "exp1_exact_match_percent",
            }
        )
    )

    comparison = metrics_df.merge(
        baseline,
        on="dataset",
        how="left",
        validate="many_to_one",
    )

    comparison["delta_bleu_vs_exp1"] = (
        comparison["bleu"]
        - comparison["exp1_bleu"]
    )

    comparison["delta_chrfpp_vs_exp1"] = (
        comparison["chrfpp"]
        - comparison["exp1_chrfpp"]
    )

    comparison["delta_exact_vs_exp1"] = (
        comparison["exact_match_percent"]
        - comparison["exp1_exact_match_percent"]
    )

    # Held-out summary only: FLORES + Tatoeba.
    heldout = comparison[
        comparison["dataset"].isin(
            ["flores_devtest", "tatoeba"]
        )
    ].copy()

    candidate_summary = (
        heldout[
            heldout["model_name"] != "EXP1_BASELINE"
        ]
        .groupby("model_name")
        .agg(
            mean_delta_bleu=(
                "delta_bleu_vs_exp1",
                "mean",
            ),
            mean_delta_chrfpp=(
                "delta_chrfpp_vs_exp1",
                "mean",
            ),
            min_delta_bleu=(
                "delta_bleu_vs_exp1",
                "min",
            ),
            min_delta_chrfpp=(
                "delta_chrfpp_vs_exp1",
                "min",
            ),
        )
        .reset_index()
    )

    # Diagnostic acceptance rule only.
    # Final freeze should still inspect per-benchmark numbers.
    candidate_summary["no_large_regression"] = (
        (candidate_summary["min_delta_bleu"] >= -0.5)
        &
        (candidate_summary["min_delta_chrfpp"] >= -0.5)
    )

    candidate_summary["mean_primary_improved"] = (
        (candidate_summary["mean_delta_bleu"] > 0)
        &
        (candidate_summary["mean_delta_chrfpp"] > 0)
    )

    candidate_summary["preliminary_pass"] = (
        candidate_summary["no_large_regression"]
        &
        candidate_summary["mean_primary_improved"]
    )

    metrics_path = output_dir / "metrics_all_v1.csv"
    comparison_path = output_dir / "comparison_vs_exp1_v1.csv"
    candidate_summary_path = output_dir / "candidate_summary_v1.csv"
    predictions_path = output_dir / "all_predictions_v1.parquet"
    report_path = output_dir / "step20_report_v1.json"

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_summary.to_csv(
        candidate_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    pd.concat(
        prediction_frames,
        ignore_index=True,
    ).to_parquet(
        predictions_path,
        index=False,
    )

    report = {
        "step": "20",
        "step_version": STEP_VERSION,
        "direction": "en_zh",
        "target_language": "zh",
        "bleu_tokenizer": "zh",
        "models": {
            k: str(v)
            for k, v in model_paths.items()
        },
        "datasets": {
            k: {
                "path": str(datasets[k]),
                "samples": int(len(v)),
            }
            for k, v in loaded_datasets.items()
        },
        "evaluation_policy": {
            "primary_heldout": [
                "flores_devtest",
                "tatoeba",
            ],
            "primary_metrics": [
                "BLEU",
                "chrF++",
            ],
            "preliminary_no_large_regression_threshold": -0.5,
            "note": (
                "Preliminary pass is diagnostic only. "
                "Final model freeze requires per-benchmark inspection."
            ),
        },
        "metrics": metrics_df.to_dict(orient="records"),
        "comparison_vs_exp1": comparison.to_dict(orient="records"),
        "candidate_summary": candidate_summary.to_dict(orient="records"),
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "READY_FOR_STEP20_MODEL_DECISION",
    }

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 112)
    print("STEP 20 RESULT")
    print("=" * 112)

    display_cols = [
        "model_name",
        "dataset",
        "samples",
        "bleu",
        "chrfpp",
        "exact_match_percent",
        "delta_bleu_vs_exp1",
        "delta_chrfpp_vs_exp1",
        "avg_generation_latency_seconds",
        "peak_gpu_memory_gib",
        "truncated_source_rows",
    ]

    print(
        comparison[
            display_cols
        ].to_string(index=False)
    )

    print("\nHeld-out candidate summary:")
    print(
        candidate_summary.to_string(
            index=False
        )
    )

    print("\nOutputs:")
    print(metrics_path)
    print(comparison_path)
    print(candidate_summary_path)
    print(predictions_path)
    print(report_path)

    print("\nSTATUS:")
    print("READY_FOR_STEP20_MODEL_DECISION")


if __name__ == "__main__":
    main()
