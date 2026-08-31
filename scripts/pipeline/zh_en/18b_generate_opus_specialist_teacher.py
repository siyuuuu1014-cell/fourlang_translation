from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


STEP_VERSION = "18B_V1"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 18B - generate OPUS specialist teacher predictions "
            "for the train-only ZH<->EN KD candidate pool."
        )
    )

    parser.add_argument("--project_root", default=None)

    parser.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/18a_candidates/"
            "kd_candidates_bidirectional_train_only_v1.parquet"
        ),
    )

    parser.add_argument(
        "--en_zh_model",
        default=(
            "results/specialists/en_zh/"
            "opus_mt_en_zh/exp1_human/best_model"
        ),
    )

    parser.add_argument(
        "--zh_en_model",
        default=(
            "results/specialists/zh_en/"
            "opus_mt_zh_en/exp1_human/best_model"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18b_opus_generation"
        ),
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def infer_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"
    return torch.cuda.get_device_name(0)


def empty_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_has_safetensors(model_path: Path) -> bool:
    if (model_path / "model.safetensors").exists():
        return True
    if list(model_path.glob("model-*.safetensors")):
        return True
    return False


def load_model_and_tokenizer(model_path: Path):
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
    )
    print("Tokenizer loaded.")

    print("Loading OPUS specialist...")
    kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16

    if model_has_safetensors(model_path):
        kwargs["use_safetensors"] = True

    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_path),
        **kwargs,
    )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    params = sum(p.numel() for p in model.parameters())

    print("Model loaded.")
    print("Parameters:", f"{params:,}")

    return tokenizer, model, params


def append_jsonl(rows: list[dict], path: Path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )


def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return pd.DataFrame(rows)


def generate_direction(
    *,
    df: pd.DataFrame,
    direction: str,
    model_path: Path,
    checkpoint_path: Path,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
):
    print("\n" + "=" * 110)
    print(f"STEP 18B | {direction}")
    print("=" * 110)

    part = (
        df.loc[df["direction"] == direction]
        .copy()
        .sort_values("_input_order")
        .reset_index(drop=True)
    )

    print("Rows:", len(part))
    print("Model:")
    print(model_path)

    checkpoint = load_checkpoint(checkpoint_path)

    if len(checkpoint):
        if checkpoint["kd_candidate_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate candidate IDs in checkpoint: {checkpoint_path}"
            )

        completed_ids = set(
            checkpoint["kd_candidate_id"]
            .astype(str)
            .tolist()
        )
    else:
        completed_ids = set()

    pending = (
        part.loc[
            ~part["kd_candidate_id"]
            .astype(str)
            .isin(completed_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    print("Existing checkpoint rows:", len(completed_ids))
    print("Pending rows:", len(pending))

    if len(pending) == 0:
        print("Nothing pending.")
        return checkpoint

    tokenizer, model, _ = load_model_and_tokenizer(
        model_path
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    new_rows = []
    total_pending = len(pending)

    for start in range(0, total_pending, batch_size):
        stop = min(start + batch_size, total_pending)

        batch = pending.iloc[start:stop].copy()

        source_texts = (
            batch["source_text"]
            .astype(str)
            .tolist()
        )

        tokenized = tokenizer(
            source_texts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            tokenized = {
                key: value.cuda(non_blocking=True)
                for key, value in tokenized.items()
            }

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        with torch.inference_mode():
            generated = model.generate(
                **tokenized,
                num_beams=num_beams,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0

        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if len(predictions) != len(batch):
            raise RuntimeError("Prediction count mismatch.")

        per_sample_latency = elapsed / max(len(batch), 1)

        batch_records = []

        for (_, row), prediction in zip(
            batch.iterrows(),
            predictions,
        ):
            prediction = str(prediction).strip()

            batch_records.append(
                {
                    "kd_candidate_id": str(row["kd_candidate_id"]),
                    "pair_id": str(row["pair_id"]),
                    "sample_id": str(row["sample_id"]),
                    "direction": direction,
                    "source_dataset": str(row["source_dataset"]),
                    "quality_tier": str(row["quality_tier"]),
                    "opus_prediction": prediction,
                    "opus_prediction_empty": prediction == "",
                    "opus_equals_source": (
                        prediction
                        == str(row["source_text"]).strip()
                    ),
                    "opus_equals_human_reference": (
                        prediction
                        == str(row["human_reference"]).strip()
                    ),
                    "opus_generation_latency_seconds": float(
                        per_sample_latency
                    ),
                    "opus_model_path": str(model_path),
                    "opus_num_beams": int(num_beams),
                    "opus_max_source_length": int(max_source_length),
                    "opus_max_new_tokens": int(max_new_tokens),
                    "step_version": STEP_VERSION,
                }
            )

        new_rows.extend(batch_records)
        append_jsonl(batch_records, checkpoint_path)

        print(
            f"{stop}/{total_pending}"
            f" | batch={len(batch)}"
            f" | {elapsed:.3f}s"
            f" | {per_sample_latency:.4f}s/sample"
        )

        del tokenized
        del generated

    new_df = pd.DataFrame(new_rows)

    if len(checkpoint):
        out = pd.concat(
            [checkpoint, new_df],
            ignore_index=True,
        )
    else:
        out = new_df

    if out["kd_candidate_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate candidate IDs after generation for {direction}."
        )

    if len(out) != len(part):
        raise RuntimeError(
            f"Direction output row count mismatch for {direction}: "
            f"expected={len(part)} observed={len(out)}"
        )

    print("\nDirection generation complete.")
    print("Rows:", len(out))
    print(
        "Empty predictions:",
        int(out["opus_prediction_empty"].fillna(True).sum()),
    )
    print(
        "Equals source:",
        int(out["opus_equals_source"].fillna(False).sum()),
    )
    print(
        "Equals human reference:",
        int(
            out["opus_equals_human_reference"]
            .fillna(False)
            .sum()
        ),
    )
    print(
        "Mean generation latency:",
        round(
            float(
                out["opus_generation_latency_seconds"].mean()
            ),
            6,
        ),
        "s/sample",
    )

    if torch.cuda.is_available():
        peak_gib = (
            torch.cuda.max_memory_allocated()
            / 1024
            / 1024
            / 1024
        )
        print(
            "Peak GPU memory:",
            round(peak_gib, 3),
            "GiB",
        )

    del model
    del tokenizer
    empty_cuda()

    return out


def main():
    args = parse_args()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else infer_project_root()
    )

    input_path = resolve_path(project_root, args.input)
    en_zh_model = resolve_path(project_root, args.en_zh_model)
    zh_en_model = resolve_path(project_root, args.zh_en_model)
    output_dir = resolve_path(project_root, args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    for model_path in [en_zh_model, zh_en_model]:
        if not model_path.exists():
            raise FileNotFoundError(
                f"OPUS specialist missing:\n{model_path}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    final_output = (
        output_dir
        / "opus_teacher_predictions_train_only_v1.parquet"
    )

    report_output = (
        output_dir
        / "opus_teacher_generation_report_v1.json"
    )

    en_zh_checkpoint = (
        output_dir
        / "checkpoint_en_zh_v1.jsonl"
    )

    zh_en_checkpoint = (
        output_dir
        / "checkpoint_zh_en_v1.jsonl"
    )

    if args.overwrite:
        for p in [
            final_output,
            report_output,
            en_zh_checkpoint,
            zh_en_checkpoint,
        ]:
            if p.exists():
                p.unlink()
    else:
        if final_output.exists():
            raise RuntimeError(
                f"Final output already exists:\n{final_output}\n"
                "Use --overwrite to regenerate."
            )

    print("=" * 110)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18B - OPUS SPECIALIST TEACHER GENERATION")
    print("=" * 110)

    print("\nProject root:")
    print(project_root)

    print("\nInput:")
    print(input_path)

    print("\nGPU:")
    print(gpu_name())

    print("\nConfiguration:")
    print("Batch size:", args.batch_size)
    print("Num beams:", args.num_beams)
    print("Max source length:", args.max_source_length)
    print("Max new tokens:", args.max_new_tokens)

    df = pd.read_parquet(input_path).copy()

    required = {
        "kd_candidate_id",
        "pair_id",
        "sample_id",
        "direction",
        "source_dataset",
        "quality_tier",
        "source_text",
        "human_reference",
        "allowed_for_kd_training",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "18A candidate pool missing fields: "
            f"{sorted(missing)}"
        )

    if df["kd_candidate_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate kd_candidate_id in input."
        )

    if not (
        df["allowed_for_kd_training"]
        .fillna(False)
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Input contains rows not allowed for KD training."
        )

    directions = set(
        df["direction"].astype(str).tolist()
    )

    if directions != {"en_zh", "zh_en"}:
        raise RuntimeError(
            f"Unexpected directions: {directions}"
        )

    df["_input_order"] = range(len(df))

    print("\nInput rows:", len(df))

    print("\nDirection:")
    print(
        df["direction"]
        .value_counts()
        .to_string()
    )

    expected_counts = (
        df["direction"]
        .value_counts()
        .to_dict()
    )

    en_zh_result = generate_direction(
        df=df,
        direction="en_zh",
        model_path=en_zh_model,
        checkpoint_path=en_zh_checkpoint,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )

    zh_en_result = generate_direction(
        df=df,
        direction="zh_en",
        model_path=zh_en_model,
        checkpoint_path=zh_en_checkpoint,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )

    pred = pd.concat(
        [en_zh_result, zh_en_result],
        ignore_index=True,
    )

    if pred["kd_candidate_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate candidate IDs in combined OPUS predictions."
        )

    if len(pred) != len(df):
        raise RuntimeError(
            "Combined OPUS prediction row count mismatch."
        )

    merge_cols = [
        "kd_candidate_id",
        "opus_prediction",
        "opus_prediction_empty",
        "opus_equals_source",
        "opus_equals_human_reference",
        "opus_generation_latency_seconds",
        "opus_model_path",
        "opus_num_beams",
        "opus_max_source_length",
        "opus_max_new_tokens",
        "step_version",
    ]

    final = df.merge(
        pred[merge_cols],
        on="kd_candidate_id",
        how="left",
        validate="one_to_one",
    )

    final = (
        final.sort_values("_input_order")
        .drop(columns=["_input_order"])
        .reset_index(drop=True)
    )

    empty_count = int(
        final["opus_prediction"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    missing_count = int(
        final["opus_prediction"]
        .isna()
        .sum()
    )

    assertions = {
        "row_count_preserved": len(final) == len(df),
        "candidate_id_unique": final["kd_candidate_id"].is_unique,
        "en_zh_count_preserved": (
            int((final["direction"] == "en_zh").sum())
            == int(expected_counts["en_zh"])
        ),
        "zh_en_count_preserved": (
            int((final["direction"] == "zh_en").sum())
            == int(expected_counts["zh_en"])
        ),
        "no_missing_prediction": missing_count == 0,
        "no_empty_prediction": empty_count == 0,
    }

    failed = [
        key
        for key, value in assertions.items()
        if not bool(value)
    ]

    if failed:
        raise RuntimeError(
            "STEP18B assertion failure:\n"
            + "\n".join(failed)
        )

    final.to_parquet(
        final_output,
        index=False,
    )

    direction_report = {}

    for direction in ["en_zh", "zh_en"]:
        part = final.loc[
            final["direction"] == direction
        ]

        direction_report[direction] = {
            "rows": int(len(part)),
            "empty_predictions": int(
                part["opus_prediction"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            ),
            "equals_source": int(
                part["opus_equals_source"]
                .fillna(False)
                .sum()
            ),
            "equals_human_reference": int(
                part["opus_equals_human_reference"]
                .fillna(False)
                .sum()
            ),
            "mean_generation_latency_seconds": float(
                part["opus_generation_latency_seconds"].mean()
            ),
        }

    report = {
        "step": "18B",
        "step_version": STEP_VERSION,
        "input": str(input_path),
        "models": {
            "en_zh": str(en_zh_model),
            "zh_en": str(zh_en_model),
        },
        "generation": {
            "batch_size": int(args.batch_size),
            "num_beams": int(args.num_beams),
            "max_source_length": int(args.max_source_length),
            "max_new_tokens": int(args.max_new_tokens),
            "do_sample": False,
        },
        "counts": {
            "input_rows": int(len(df)),
            "output_rows": int(len(final)),
            "missing_predictions": int(missing_count),
            "empty_predictions": int(empty_count),
        },
        "direction": direction_report,
        "assertions": {
            key: bool(value)
            for key, value in assertions.items()
        },
        "outputs": {
            "predictions": str(final_output),
            "en_zh_checkpoint": str(en_zh_checkpoint),
            "zh_en_checkpoint": str(zh_en_checkpoint),
        },
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18C_MADLAD_GENERATION"
        ),
    }

    save_json(report, report_output)

    print("\n" + "=" * 110)
    print("STEP 18B RESULT")
    print("=" * 110)

    print("\nRows:", len(final))
    print("Missing predictions:", missing_count)
    print("Empty predictions:", empty_count)

    for direction in ["en_zh", "zh_en"]:
        part = direction_report[direction]

        print("\n" + direction)
        print("Rows:", part["rows"])
        print("Equals source:", part["equals_source"])
        print(
            "Equals human reference:",
            part["equals_human_reference"],
        )
        print(
            "Mean generation latency:",
            round(
                part["mean_generation_latency_seconds"],
                6,
            ),
            "s/sample",
        )

    print("\nAssertions:")
    for key, value in assertions.items():
        print(f"{key}: {bool(value)}")

    print("\nPredictions:")
    print(final_output)

    print("\nReport:")
    print(report_output)

    print("\nSTATUS:")
    print("READY_FOR_STEP_18C_MADLAD_GENERATION")


if __name__ == "__main__":
    main()
