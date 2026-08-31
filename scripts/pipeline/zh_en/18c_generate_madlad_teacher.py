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


STEP_VERSION = "18C_V1"

DEFAULT_MADLAD_PATH = (
    "/root/autodl-tmp/huggingface/hub/"
    "models--google--madlad400-3b-mt/"
    "snapshots/fa184c675da0b5c9e1c8694fccd4e12e2d422094"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 18C - generate MADLAD-400-3B-MT teacher predictions "
            "for the exact Step18B train-only ZH<->EN candidate pool."
        )
    )

    parser.add_argument("--project_root", default=None)

    parser.add_argument(
        "--input",
        default=(
            "data/distillation/zh_en/v1/"
            "18b_opus_generation/"
            "opus_teacher_predictions_train_only_v1.parquet"
        ),
    )

    parser.add_argument(
        "--model_path",
        default=DEFAULT_MADLAD_PATH,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "data/distillation/zh_en/v1/"
            "18c_madlad_generation"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def infer_project_root() -> Path:
    # .../scripts/pipeline/zh_en/18c_xxx.py
    return Path(__file__).resolve().parents[3]


def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def append_jsonl(rows: list[dict], path: Path):
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_jsonl(path: Path) -> pd.DataFrame:
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


def model_has_safetensors(model_path: Path) -> bool:
    if (model_path / "model.safetensors").exists():
        return True

    if list(
        model_path.glob("model-*.safetensors")
    ):
        return True

    return False


def gpu_name() -> str:
    if not torch.cuda.is_available():
        return "CPU"

    return torch.cuda.get_device_name(0)


def empty_cuda():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_madlad(model_path: Path):
    print("\nLoading MADLAD tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        use_fast=False,
        legacy=True,
    )

    print("Tokenizer loaded.")

    print("Loading MADLAD model...")

    kwargs = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }

    if model_has_safetensors(model_path):
        kwargs["use_safetensors"] = True

    if torch.cuda.is_available():
        # Recent transformers prefers dtype; some older releases still
        # use torch_dtype. Try the current API first.
        kwargs["dtype"] = torch.float16

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            **kwargs,
        )

    except TypeError:
        kwargs.pop("dtype", None)

        if torch.cuda.is_available():
            kwargs["torch_dtype"] = torch.float16

        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_path),
            **kwargs,
        )

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    param_memory_gib = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    ) / 1024**3

    print("Model loaded.")
    print("Parameters:", f"{params:,}")
    print(
        "Parameter memory:",
        f"{param_memory_gib:.3f} GiB",
    )

    return tokenizer, model, params, param_memory_gib


def target_prefix(direction: str) -> str:
    if direction == "en_zh":
        return "<2zh>"

    if direction == "zh_en":
        return "<2en>"

    raise ValueError(
        f"Unsupported direction: {direction}"
    )


def generate_direction(
    *,
    df: pd.DataFrame,
    direction: str,
    tokenizer,
    model,
    checkpoint_path: Path,
    batch_size: int,
    num_beams: int,
    max_source_length: int,
    max_new_tokens: int,
):
    print("\n" + "=" * 110)
    print(f"MADLAD | {direction}")
    print("=" * 110)

    part = (
        df.loc[
            df["direction"] == direction
        ]
        .copy()
        .sort_values("_input_order")
        .reset_index(drop=True)
    )

    print("Rows:", len(part))

    checkpoint = load_jsonl(
        checkpoint_path
    )

    if len(checkpoint):
        if (
            checkpoint[
                "kd_candidate_id"
            ]
            .duplicated()
            .any()
        ):
            raise RuntimeError(
                "Duplicate kd_candidate_id "
                f"in checkpoint:\n{checkpoint_path}"
            )

        completed_ids = set(
            checkpoint[
                "kd_candidate_id"
            ]
            .astype(str)
            .tolist()
        )

    else:
        completed_ids = set()

    pending = (
        part.loc[
            ~part[
                "kd_candidate_id"
            ]
            .astype(str)
            .isin(
                completed_ids
            )
        ]
        .copy()
        .reset_index(drop=True)
    )

    print(
        "Existing checkpoint rows:",
        len(completed_ids),
    )

    print(
        "Pending rows:",
        len(pending),
    )

    if len(pending) == 0:
        print("Nothing pending.")
        return checkpoint

    prefix = target_prefix(
        direction
    )

    print(
        "MADLAD target prefix:",
        prefix,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    new_rows = []

    total_pending = len(
        pending
    )

    for start in range(
        0,
        total_pending,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            total_pending,
        )

        batch = (
            pending.iloc[
                start:stop
            ]
            .copy()
        )

        raw_sources = (
            batch[
                "source_text"
            ]
            .astype(str)
            .tolist()
        )

        # MADLAD translation instruction:
        # <2target_language> source sentence
        model_inputs = [
            f"{prefix} {text}"
            for text in raw_sources
        ]

        tokenized = tokenizer(
            model_inputs,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            tokenized = {
                key: value.cuda(
                    non_blocking=True
                )
                for key, value
                in tokenized.items()
            }

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

        elapsed = (
            time.perf_counter()
            -
            t0
        )

        predictions = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        if (
            len(predictions)
            !=
            len(batch)
        ):
            raise RuntimeError(
                "MADLAD prediction count mismatch."
            )

        latency_per_sample = (
            elapsed
            /
            max(
                len(batch),
                1,
            )
        )

        batch_records = []

        for (
            (_, row),
            prediction,
        ) in zip(
            batch.iterrows(),
            predictions,
        ):
            prediction = str(
                prediction
            ).strip()

            record = {
                "kd_candidate_id": str(
                    row[
                        "kd_candidate_id"
                    ]
                ),
                "pair_id": str(
                    row[
                        "pair_id"
                    ]
                ),
                "sample_id": str(
                    row[
                        "sample_id"
                    ]
                ),
                "direction": direction,
                "source_dataset": str(
                    row[
                        "source_dataset"
                    ]
                ),
                "quality_tier": str(
                    row[
                        "quality_tier"
                    ]
                ),
                "madlad_prediction": (
                    prediction
                ),
                "madlad_prediction_empty": (
                    prediction
                    ==
                    ""
                ),
                "madlad_equals_source": (
                    prediction
                    ==
                    str(
                        row[
                            "source_text"
                        ]
                    ).strip()
                ),
                "madlad_equals_human_reference": (
                    prediction
                    ==
                    str(
                        row[
                            "human_reference"
                        ]
                    ).strip()
                ),
                "madlad_equals_opus": (
                    prediction
                    ==
                    str(
                        row[
                            "opus_prediction"
                        ]
                    ).strip()
                ),
                "madlad_generation_latency_seconds": float(
                    latency_per_sample
                ),
                "madlad_target_prefix": (
                    prefix
                ),
                "madlad_num_beams": int(
                    num_beams
                ),
                "madlad_max_source_length": int(
                    max_source_length
                ),
                "madlad_max_new_tokens": int(
                    max_new_tokens
                ),
                "step_version": (
                    STEP_VERSION
                ),
            }

            batch_records.append(
                record
            )

        new_rows.extend(
            batch_records
        )

        append_jsonl(
            batch_records,
            checkpoint_path,
        )

        print(
            f"{stop}/{total_pending}"
            f" | batch={len(batch)}"
            f" | {elapsed:.3f}s"
            f" | {latency_per_sample:.4f}s/sample"
        )

        del tokenized
        del generated

    new_df = pd.DataFrame(
        new_rows
    )

    if len(checkpoint):
        out = pd.concat(
            [
                checkpoint,
                new_df,
            ],
            ignore_index=True,
        )

    else:
        out = new_df

    if (
        out[
            "kd_candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            f"Duplicate candidate IDs after {direction} MADLAD generation."
        )

    if len(out) != len(part):
        raise RuntimeError(
            f"{direction} MADLAD row count mismatch: "
            f"expected={len(part)} observed={len(out)}"
        )

    print("\nDirection generation complete.")
    print("Rows:", len(out))

    print(
        "Empty predictions:",
        int(
            out[
                "madlad_prediction_empty"
            ]
            .fillna(True)
            .sum()
        ),
    )

    print(
        "Equals source:",
        int(
            out[
                "madlad_equals_source"
            ]
            .fillna(False)
            .sum()
        ),
    )

    print(
        "Equals human reference:",
        int(
            out[
                "madlad_equals_human_reference"
            ]
            .fillna(False)
            .sum()
        ),
    )

    print(
        "Equals OPUS:",
        int(
            out[
                "madlad_equals_opus"
            ]
            .fillna(False)
            .sum()
        ),
    )

    print(
        "Mean generation latency:",
        round(
            float(
                out[
                    "madlad_generation_latency_seconds"
                ]
                .mean()
            ),
            6,
        ),
        "s/sample",
    )

    if torch.cuda.is_available():
        peak_gib = (
            torch.cuda.max_memory_allocated()
            /
            1024**3
        )

        print(
            "Peak GPU memory:",
            round(
                peak_gib,
                3,
            ),
            "GiB",
        )

    return out


def main():
    args = parse_args()

    project_root = (
        Path(
            args.project_root
        ).resolve()
        if args.project_root
        else infer_project_root()
    )

    input_path = resolve_path(
        project_root,
        args.input,
    )

    model_path = resolve_path(
        project_root,
        args.model_path,
    )

    output_dir = resolve_path(
        project_root,
        args.output_dir,
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Step18B input missing:\n{input_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"MADLAD model missing:\n{model_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_output = (
        output_dir
        /
        "opus_madlad_teacher_predictions_train_only_v1.parquet"
    )

    report_output = (
        output_dir
        /
        "madlad_teacher_generation_report_v1.json"
    )

    en_zh_checkpoint = (
        output_dir
        /
        "checkpoint_madlad_en_zh_v1.jsonl"
    )

    zh_en_checkpoint = (
        output_dir
        /
        "checkpoint_madlad_zh_en_v1.jsonl"
    )

    if args.overwrite:
        for path in [
            final_output,
            report_output,
            en_zh_checkpoint,
            zh_en_checkpoint,
        ]:
            if path.exists():
                path.unlink()

    else:
        if final_output.exists():
            raise RuntimeError(
                f"Final output already exists:\n{final_output}\n"
                "Use --overwrite to rebuild it."
            )

    print("=" * 110)
    print("ZH-EN DISTILLATION PIPELINE")
    print("STEP 18C - MADLAD TEACHER GENERATION")
    print("=" * 110)

    print("\nProject root:")
    print(project_root)

    print("\nInput:")
    print(input_path)

    print("\nMADLAD:")
    print(model_path)

    print("\nGPU:")
    print(gpu_name())

    print("\nConfiguration:")
    print("Batch size:", args.batch_size)
    print("Num beams:", args.num_beams)
    print(
        "Max source length:",
        args.max_source_length,
    )
    print(
        "Max new tokens:",
        args.max_new_tokens,
    )

    df = pd.read_parquet(
        input_path
    ).copy()

    required = {
        "kd_candidate_id",
        "pair_id",
        "sample_id",
        "direction",
        "source_dataset",
        "quality_tier",
        "source_text",
        "human_reference",
        "opus_prediction",
        "allowed_for_kd_training",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Step18B output missing required columns: "
            f"{sorted(missing)}"
        )

    if (
        df[
            "kd_candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate kd_candidate_id in Step18B input."
        )

    if not (
        df[
            "allowed_for_kd_training"
        ]
        .fillna(False)
        .astype(bool)
        .all()
    ):
        raise RuntimeError(
            "Step18B contains rows not allowed for KD training."
        )

    if (
        df[
            "opus_prediction"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise RuntimeError(
            "Step18B contains empty OPUS predictions."
        )

    direction_counts = (
        df[
            "direction"
        ]
        .value_counts()
        .to_dict()
    )

    if set(direction_counts) != {
        "en_zh",
        "zh_en",
    }:
        raise RuntimeError(
            f"Unexpected directions: {direction_counts}"
        )

    df[
        "_input_order"
    ] = range(
        len(df)
    )

    print("\nInput rows:", len(df))
    print("\nDirection:")
    print(
        df[
            "direction"
        ]
        .value_counts()
        .to_string()
    )

    tokenizer, model, params, param_memory_gib = (
        load_madlad(
            model_path
        )
    )

    en_zh_result = generate_direction(
        df=df,
        direction="en_zh",
        tokenizer=tokenizer,
        model=model,
        checkpoint_path=en_zh_checkpoint,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )

    zh_en_result = generate_direction(
        df=df,
        direction="zh_en",
        tokenizer=tokenizer,
        model=model,
        checkpoint_path=zh_en_checkpoint,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_source_length=args.max_source_length,
        max_new_tokens=args.max_new_tokens,
    )

    pred = pd.concat(
        [
            en_zh_result,
            zh_en_result,
        ],
        ignore_index=True,
    )

    if (
        pred[
            "kd_candidate_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate candidate IDs in combined MADLAD predictions."
        )

    if len(pred) != len(df):
        raise RuntimeError(
            "Combined MADLAD prediction row count mismatch."
        )

    merge_columns = [
        "kd_candidate_id",
        "madlad_prediction",
        "madlad_prediction_empty",
        "madlad_equals_source",
        "madlad_equals_human_reference",
        "madlad_equals_opus",
        "madlad_generation_latency_seconds",
        "madlad_target_prefix",
        "madlad_num_beams",
        "madlad_max_source_length",
        "madlad_max_new_tokens",
        "step_version",
    ]

    final = df.merge(
        pred[
            merge_columns
        ],
        on="kd_candidate_id",
        how="left",
        validate="one_to_one",
    )

    final = (
        final.sort_values(
            "_input_order"
        )
        .drop(
            columns=[
                "_input_order",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    missing_count = int(
        final[
            "madlad_prediction"
        ]
        .isna()
        .sum()
    )

    empty_count = int(
        final[
            "madlad_prediction"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    assertions = {
        "row_count_preserved": (
            len(final)
            ==
            len(df)
        ),
        "candidate_id_unique": (
            final[
                "kd_candidate_id"
            ]
            .is_unique
        ),
        "en_zh_count_preserved": (
            int(
                (
                    final[
                        "direction"
                    ]
                    ==
                    "en_zh"
                )
                .sum()
            )
            ==
            int(
                direction_counts[
                    "en_zh"
                ]
            )
        ),
        "zh_en_count_preserved": (
            int(
                (
                    final[
                        "direction"
                    ]
                    ==
                    "zh_en"
                )
                .sum()
            )
            ==
            int(
                direction_counts[
                    "zh_en"
                ]
            )
        ),
        "opus_predictions_preserved": (
            final[
                "opus_prediction"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .all()
        ),
        "no_missing_madlad_prediction": (
            missing_count
            ==
            0
        ),
        "no_empty_madlad_prediction": (
            empty_count
            ==
            0
        ),
    }

    failed = [
        key
        for key, value
        in assertions.items()
        if not bool(value)
    ]

    if failed:
        raise RuntimeError(
            "STEP18C assertion failure:\n"
            +
            "\n".join(
                failed
            )
        )

    final.to_parquet(
        final_output,
        index=False,
    )

    direction_report = {}

    for direction in [
        "en_zh",
        "zh_en",
    ]:
        part = final.loc[
            final[
                "direction"
            ]
            ==
            direction
        ]

        direction_report[
            direction
        ] = {
            "rows": int(
                len(part)
            ),
            "empty_predictions": int(
                part[
                    "madlad_prediction"
                ]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            ),
            "equals_source": int(
                part[
                    "madlad_equals_source"
                ]
                .fillna(False)
                .sum()
            ),
            "equals_human_reference": int(
                part[
                    "madlad_equals_human_reference"
                ]
                .fillna(False)
                .sum()
            ),
            "equals_opus": int(
                part[
                    "madlad_equals_opus"
                ]
                .fillna(False)
                .sum()
            ),
            "mean_generation_latency_seconds": float(
                part[
                    "madlad_generation_latency_seconds"
                ]
                .mean()
            ),
        }

    report = {
        "step": "18C",
        "step_version": STEP_VERSION,
        "input": str(
            input_path
        ),
        "model": {
            "path": str(
                model_path
            ),
            "parameters": int(
                params
            ),
            "parameter_memory_gib": float(
                param_memory_gib
            ),
        },
        "generation": {
            "batch_size": int(
                args.batch_size
            ),
            "num_beams": int(
                args.num_beams
            ),
            "max_source_length": int(
                args.max_source_length
            ),
            "max_new_tokens": int(
                args.max_new_tokens
            ),
            "en_zh_target_prefix": "<2zh>",
            "zh_en_target_prefix": "<2en>",
            "do_sample": False,
        },
        "counts": {
            "input_rows": int(
                len(df)
            ),
            "output_rows": int(
                len(final)
            ),
            "missing_madlad_predictions": int(
                missing_count
            ),
            "empty_madlad_predictions": int(
                empty_count
            ),
        },
        "direction": direction_report,
        "assertions": {
            key: bool(value)
            for key, value
            in assertions.items()
        },
        "outputs": {
            "predictions": str(
                final_output
            ),
            "en_zh_checkpoint": str(
                en_zh_checkpoint
            ),
            "zh_en_checkpoint": str(
                zh_en_checkpoint
            ),
        },
        "created_at_utc": (
            datetime.now(
                timezone.utc
            )
            .isoformat()
        ),
        "status": (
            "READY_FOR_STEP_18D_DISAGREEMENT_ANALYSIS"
        ),
    }

    save_json(
        report,
        report_output,
    )

    print("\n" + "=" * 110)
    print("STEP 18C RESULT")
    print("=" * 110)

    print("\nRows:", len(final))
    print(
        "Missing MADLAD predictions:",
        missing_count,
    )
    print(
        "Empty MADLAD predictions:",
        empty_count,
    )

    for direction in [
        "en_zh",
        "zh_en",
    ]:
        part = direction_report[
            direction
        ]

        print("\n" + direction)
        print(
            "Rows:",
            part[
                "rows"
            ],
        )
        print(
            "Equals source:",
            part[
                "equals_source"
            ],
        )
        print(
            "Equals human reference:",
            part[
                "equals_human_reference"
            ],
        )
        print(
            "Equals OPUS:",
            part[
                "equals_opus"
            ],
        )
        print(
            "Mean generation latency:",
            round(
                part[
                    "mean_generation_latency_seconds"
                ],
                6,
            ),
            "s/sample",
        )

    print("\nAssertions:")

    for key, value in assertions.items():
        print(
            f"{key}: {bool(value)}"
        )

    print("\nPredictions:")
    print(final_output)

    print("\nReport:")
    print(report_output)

    print("\nSTATUS:")
    print(
        "READY_FOR_STEP_18D_DISAGREEMENT_ANALYSIS"
    )

    del model
    del tokenizer
    empty_cuda()


if __name__ == "__main__":
    main()
