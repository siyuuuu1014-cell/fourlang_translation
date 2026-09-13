"""Generate one resumable teacher translation stream for targeted zh->uz data."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock  # noqa: E402
from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="data/targeted/zh_uz/v1/source_candidates.jsonl"
    )
    parser.add_argument("--teacher-id", required=True)
    parser.add_argument("--family", choices=("madlad", "m2m100", "nllb"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/multilingual/fourlang.toml")
    parser.add_argument("--checkpoint-rows", type=int, default=128)
    args = parser.parse_args()
    if args.checkpoint_rows < 8 or args.checkpoint_rows % 8:
        raise ValueError("--checkpoint-rows must be a positive multiple of 8")

    source_path = diag.project_path(args.source)
    source_manifest_path = source_path.with_name("source_manifest.json")
    model_path = diag.project_path(args.model)
    output = diag.checked_output(args.output)
    config = diag.load_config(args.config)
    rows = diag.read_rows(source_path)
    source_manifest = diag.read_json(source_manifest_path)
    if (
        source_manifest.get("status")
        != "TARGETED_SOURCES_READY_NOT_TRANSLATED_NOT_APPROVED"
        or source_manifest.get("rows") != 6000
        or source_manifest.get("output_sha256") != diag.file_sha256(source_path)
        or source_manifest.get("exact_protected_overlap") != 0
        or source_manifest.get("missing_protected_files")
    ):
        raise ValueError("Targeted source manifest is incomplete, stale, or unsafe")
    if len(rows) != 6000 or len({row["candidate_id"] for row in rows}) != 6000:
        raise ValueError("Expected 6000 unique targeted source candidates")
    if any(
        row.get("src_lang") != "zh"
        or row.get("tgt_lang") != "uz"
        or row.get("usage") != "TRAINING_CANDIDATE_NOT_APPROVED"
        for row in rows
    ):
        raise ValueError("Invalid targeted-source contract")

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        diag.log(f"Hashing teacher model for safe resume: {model_path}")
        model_hash = diag.model_signature(model_path)
        signature = diag.bind_manifest(
            output / "manifest.json",
            {
                "source": str(source_path),
                "source_sha256": diag.file_sha256(source_path),
                "source_manifest_sha256": diag.file_sha256(source_manifest_path),
                "teacher_id": args.teacher_id,
                "family": args.family,
                "model": str(model_path),
                "model_signature": model_hash,
                "config": str(args.config),
                "config_sha256": diag.file_sha256(diag.project_path(args.config)),
                "checkpoint_rows": args.checkpoint_rows,
                "code_sha256": diag.file_sha256(Path(__file__)),
                "versions": diag.versions(),
            },
        )
        inference_rows = [
            {
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": row["src_text"],
                "tgt_text": "",
            }
            for row in rows
        ]
        chunks = output / "chunks"
        chunks.mkdir(exist_ok=True)
        translations = diag.chunk_predictions(
            chunks, inference_rows, signature, args.checkpoint_rows, None
        )
        tokenizer = model = predict = None
        started = time.monotonic()
        try:
            if translations is None:
                tokenizer, model = diag.flow.load_model(
                    {
                        "id": args.teacher_id,
                        "family": args.family,
                        "path": str(model_path),
                        "require_local_artifact": True,
                    },
                    "zh",
                    "uz",
                )

                def predict(texts):
                    return diag.flow.translate(
                        tokenizer, model, args.family, "zh", "uz", texts, config
                    )

                translations = diag.chunk_predictions(
                    chunks,
                    inference_rows,
                    signature,
                    args.checkpoint_rows,
                    predict,
                )
        finally:
            del predict, model, tokenizer
            gc.collect()
            if diag.flow.torch.cuda.is_available():
                diag.flow.torch.cuda.empty_cache()

        assert translations is not None
        generated = [
            {
                **row,
                "teacher_id": args.teacher_id,
                "teacher_family": args.family,
                "teacher_text": text,
                "usage": "TEACHER_CANDIDATE_NOT_APPROVED",
            }
            for row, text in zip(rows, translations, strict=True)
        ]
        diag.save_jsonl(output / "teacher_candidates.jsonl", generated)
        diag.save_json(
            output / "summary.json",
            {
                "status": "TEACHER_CANDIDATES_READY_NOT_APPROVED",
                "teacher_id": args.teacher_id,
                "rows": len(generated),
                "empty_translations": sum(not text.strip() for text in translations),
                "elapsed_seconds": time.monotonic() - started,
                "training_started": False,
                "training_data_written": False,
            },
        )
        diag.log(
            f"TARGETED_TEACHER_READY: teacher={args.teacher_id} rows={len(generated)}"
        )


if __name__ == "__main__":
    main()
