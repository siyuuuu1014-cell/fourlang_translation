"""Select zh->uz decoding on validation, then evaluate once on fixed devtest."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import (  # noqa: E402
    metrics,
    normalize_generated_texts,
    prepare_inputs,
)
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint, model_signature  # noqa: E402
from scripts.pipeline_v3.fourlang_m2m100_ru_uz_lora import (  # noqa: E402
    adapter_signature,
    configured_direction,
    load_metrics,
    load_routed_model,
    repair_rows,
)
from scripts.pipeline_v3.fourlang_m2m100_student import summarize_metrics  # noqa: E402

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_zh_uz_decode_calibration_v1.toml"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = load_config(config["model"]["config"])
    direction, _, _ = configured_direction(nested)
    if direction != config["experiment"]["direction"]:
        raise RuntimeError(f"Calibration/model direction mismatch: {direction}")
    return nested


def candidate_ids(config: dict[str, Any]) -> list[str]:
    values = [str(item["id"]) for item in config["decoding"]["candidates"]]
    if len(values) != len(set(values)):
        raise ValueError("Decoding candidate ids must be unique.")
    return values


def decode(
    tokenizer: Any,
    model: Any,
    source: str,
    target: str,
    texts: list[str],
    settings: dict[str, Any],
    common: dict[str, Any],
) -> list[str]:
    output = []
    batch_size = int(common["batch_size"])
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded, generation = prepare_inputs(
            tokenizer,
            "m2m100",
            source,
            target,
            batch,
            int(common["max_source_length"]),
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **encoded,
                **generation,
                do_sample=False,
                num_beams=int(settings["num_beams"]),
                length_penalty=float(settings["length_penalty"]),
                early_stopping=bool(settings["early_stopping"]),
                max_new_tokens=int(common["max_new_tokens"]),
            )
        decoded = [
            text.strip()
            for text in tokenizer.batch_decode(tokens, skip_special_tokens=True)
        ]
        output.extend(normalize_generated_texts(target, decoded))
    return output


def selection_signature(config: dict[str, Any], nested: dict[str, Any]) -> str:
    return fingerprint(
        {
            "adapter": adapter_signature(project_path(nested["outputs"]["adapter"])),
            "base": model_signature(project_path(nested["base_model"]["path"])),
            "validation": file_sha256(project_path(config["data"]["validation"])),
            "decoding": config["decoding"],
            "selection": {
                "primary_metric": config["selection"]["primary_metric"],
                "secondary_metric": config["selection"]["secondary_metric"],
            },
            "implementation": file_sha256(Path(__file__)),
        }
    )


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    if config["experiment"]["direction"] != "zh-uz":
        raise RuntimeError("This controlled calibration is locked to zh-uz.")
    nested = model_config(config)
    validation = project_path(config["data"]["validation"])
    benchmark = project_path(config["benchmark"]["path"])
    baseline = project_path(config["baseline"]["exp4_metrics"])
    for item in (validation, benchmark, baseline):
        if not item.is_file():
            raise FileNotFoundError(item)
    adapter_signature(project_path(nested["outputs"]["adapter"]))
    rows = repair_rows(validation, "zh-uz")
    frame = pd.read_parquet(benchmark, columns=["zh", "uz"])
    candidate_ids(config)
    report = {
        "status": "READY",
        "direction": "zh-uz",
        "validation_rows": len(rows),
        "benchmark_rows": len(frame),
        "candidate_count": len(config["decoding"]["candidates"]),
        "selection_data": str(validation.resolve()),
        "fixed_test_data": str(benchmark.resolve()),
        "test_used_for_parameter_selection": False,
    }
    print(f"M2M100_ZH_UZ_DECODE_PREFLIGHT_READY: {json.dumps(report)}", flush=True)
    return report


def calibrate(config: dict[str, Any]) -> dict[str, Any]:
    preflight(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for decoding calibration.")
    nested = model_config(config)
    destination = project_path(config["outputs"]["calibration"])
    signature = selection_signature(config, nested)
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("selection_signature") != signature:
            raise RuntimeError(f"Existing calibration differs; refusing overwrite: {destination}")
        print(f"M2M100_ZH_UZ_DECODE_CALIBRATION_REUSED: {destination}", flush=True)
        return payload
    rows = repair_rows(project_path(config["data"]["validation"]), "zh-uz")
    sources = [str(row["src_text"]) for row in rows]
    references = [str(row["tgt_text"]) for row in rows]
    tokenizer, model, active = load_routed_model(nested, "zh", "uz")
    if not active:
        raise RuntimeError("zh-uz adapter was not activated.")
    results = {}
    try:
        for settings in config["decoding"]["candidates"]:
            candidate_id = str(settings["id"])
            predictions = decode(
                tokenizer, model, "zh", "uz", sources, settings, config["decoding"]
            )
            values = metrics(predictions, references, "uz")
            results[candidate_id] = {"settings": settings, "metrics": values}
            print(f"Decode calibration {candidate_id}: {values}", flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    primary = str(config["selection"]["primary_metric"])
    secondary = str(config["selection"]["secondary_metric"])
    selected_id = max(
        results,
        key=lambda key: (
            results[key]["metrics"][primary],
            results[key]["metrics"][secondary],
            key,
        ),
    )
    payload = {
        "schema_version": 1,
        "status": "CALIBRATED_ON_VALIDATION_NOT_TESTED",
        "selection_signature": signature,
        "direction": "zh-uz",
        "selection_data": str(project_path(config["data"]["validation"]).resolve()),
        "test_used_for_parameter_selection": False,
        "primary_metric": primary,
        "secondary_metric": secondary,
        "candidates": results,
        "selected": {"id": selected_id, **results[selected_id]},
    }
    write_json(destination, payload)
    print(f"M2M100_ZH_UZ_DECODE_CALIBRATION_READY: {destination} selected={selected_id}", flush=True)
    return payload


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed devtest evaluation.")
    calibration = calibrate(config)
    nested = model_config(config)
    destination = project_path(config["outputs"]["metrics"])
    benchmark = project_path(config["benchmark"]["path"])
    selected = calibration["selected"]
    signature = fingerprint(
        {
            "selection_signature": calibration["selection_signature"],
            "selected": selected,
            "benchmark": file_sha256(benchmark),
        }
    )
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("evaluation_signature") != signature:
            raise RuntimeError(f"Existing evaluation differs; refusing overwrite: {destination}")
        print(f"M2M100_ZH_UZ_DECODE_EVALUATION_REUSED: {destination}", flush=True)
        return payload
    frame = pd.read_parquet(benchmark, columns=["zh", "uz"])
    tokenizer, model, active = load_routed_model(nested, "zh", "uz")
    if not active:
        raise RuntimeError("zh-uz adapter was not activated.")
    try:
        predictions = decode(
            tokenizer,
            model,
            "zh",
            "uz",
            frame["zh"].fillna("").astype(str).tolist(),
            selected["settings"],
            config["decoding"],
        )
        repaired = metrics(
            predictions, frame["uz"].fillna("").astype(str).tolist(), "uz"
        )
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    exp4 = load_metrics(project_path(config["baseline"]["exp4_metrics"]))
    values = {key: dict(value) for key, value in exp4["metrics"].items()}
    values["zh-uz"] = repaired
    payload = {
        "schema_version": 1,
        "status": "EVALUATED_FIXED_DEVTEST_NOT_PROMOTED",
        "evaluation_signature": signature,
        "selected_decoding": selected,
        "metrics": values,
        "summary": summarize_metrics(values),
        "routing": {"zh-uz": "exp4_plus_adapter_and_calibrated_decode", "all_other_directions": "exp4_base"},
    }
    write_json(destination, payload)
    print(f"M2M100_ZH_UZ_DECODE_EVALUATION_COMPLETE: {destination}", flush=True)
    return payload


def compare(config: dict[str, Any]) -> dict[str, Any]:
    candidate = evaluate(config)
    exp4 = load_metrics(project_path(config["baseline"]["exp4_metrics"]))
    delta = candidate["metrics"]["zh-uz"]["chrf2"] - exp4["metrics"]["zh-uz"]["chrf2"]
    settings = config["selection"]
    constraints = {
        "zh_uz_recovery_from_exp4_at_least_minimum": delta
        >= float(settings["minimum_repair_chrf2_from_exp4"]),
        "macro_chrf2_at_least_minimum": candidate["summary"]["macro_chrf2"]
        >= float(settings["minimum_macro_chrf2"]),
        "worst_chrf2_at_least_minimum": candidate["summary"]["worst_chrf2"]
        >= float(settings["minimum_worst_chrf2"]),
        "non_repair_directions_are_exact_exp4_routes": True,
        "test_not_used_for_parameter_selection": True,
    }
    report = {
        "schema_version": 1,
        "status": "CANDIDATE_PASSED" if all(constraints.values()) else "CANDIDATE_REJECTED",
        "candidate": candidate,
        "baseline": exp4,
        "deltas": {
            "zh_uz_chrf2_from_exp4": delta,
            "macro_chrf2_from_exp4": candidate["summary"]["macro_chrf2"]
            - exp4["summary"]["macro_chrf2"],
        },
        "constraints": constraints,
        "promotion_performed": False,
    }
    destination = project_path(config["outputs"]["comparison"])
    write_json(destination, report)
    print(f"M2M100_ZH_UZ_DECODE_COMPARISON_READY: {destination} status={report['status']}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "calibrate", "evaluate", "compare", "run-all"))
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "preflight":
        preflight(config)
    elif args.action == "calibrate":
        calibrate(config)
    elif args.action == "evaluate":
        evaluate(config)
    elif args.action == "compare":
        compare(config)
    else:
        calibrate(config)
        evaluate(config)
        compare(config)


if __name__ == "__main__":
    main()
