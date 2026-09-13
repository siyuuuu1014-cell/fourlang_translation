"""Run the frozen 110-row Chinese-to-Uzbek direct semantic check."""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402


DEFAULT_INPUT = "acceptance/zh_uz_direct_v1/gold.jsonl"
DEFAULT_MODEL = "results/student/fourlang/exp2/best_model/shared"
DEFAULT_OUTPUT = (
    "reports/diagnostics/fourlang/zh_uz_direct_exp2_v1/predictions.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate the 110 frozen zh-uz semantic test sentences."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--config", default="configs/multilingual/fourlang.toml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = diag.project_path(args.input)
    model_path = diag.project_path(args.model)
    output_path = diag.project_path(args.output)
    config = diag.load_config(args.config)
    rows = diag.read_rows(input_path)

    if len(rows) != 110:
        raise ValueError(f"Expected 110 frozen rows, got {len(rows)}")
    required = {"id", "scenario", "source_zh", "reference_uz", "key_meaning_zh"}
    if any(required - set(row) for row in rows):
        raise ValueError("Input row is missing a required field")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test IDs")

    selected = diag.read_json(
        diag.PROJECT_ROOT / "results/model_selection/fourlang/selected_student.json"
    )
    candidate = {
        **selected["candidate"],
        "path": str(model_path),
        "require_local_artifact": True,
    }
    diag.model_signature(model_path)
    diag.flow.set_seed(2026)
    tokenizer, model = diag.flow.load_model(candidate, "zh", "uz")
    try:
        predictions = diag.flow.translate(
            tokenizer,
            model,
            candidate["family"],
            "zh",
            "uz",
            [row["source_zh"] for row in rows],
            config,
        )
    finally:
        del model, tokenizer
        gc.collect()
        if diag.flow.torch.cuda.is_available():
            diag.flow.torch.cuda.empty_cache()

    results = [
        {
            **row,
            "model_output_uz": prediction,
            "semantic_grade": "",
            "review_notes": "",
        }
        for row, prediction in zip(rows, predictions, strict=True)
    ]
    automatic_metrics = diag.flow.metrics(
        predictions,
        [row["reference_uz"] for row in rows],
        "uz",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diag.save_jsonl(output_path, results)
    diag.save_json(
        output_path.parent / "summary.json",
        {
            "status": "TRANSLATIONS_READY_PENDING_SEMANTIC_REVIEW",
            "direction": "zh-uz",
            "rows": len(results),
            "model": str(model_path),
            "input": str(input_path),
            "predictions": str(output_path),
            "automatic_metrics": automatic_metrics,
            "semantic_scores": "PENDING_HUMAN_REVIEW",
            "metric_warning": (
                "BLEU/chrF2 compare against one AI-authored reference and do not "
                "determine whether a semantically valid paraphrase passes."
            ),
            "training_started": False,
            "training_data_written": False,
        },
    )
    print(
        "ZH_UZ_DIRECT_TEST_READY: "
        f"rows={len(results)} bleu={automatic_metrics['bleu']:.4f} "
        f"chrf2={automatic_metrics['chrf2']:.4f} output={output_path}"
    )


if __name__ == "__main__":
    main()
