from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict

import sacrebleu
import torch
from peft import PeftModel

from config_utils import load_config, positive_limit, project_path
from data_utils import load_jsonl, validate_languages
from model_utils import load_base_model, load_tokenizer, prepare_generation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate M2M100 or a LoRA adapter")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", help="Adapter directory; omit to evaluate the base model")
    parser.add_argument("--output", help="Override result directory")
    return parser.parse_args()


def score_group(rows: list[dict[str, object]]) -> dict[str, float | int]:
    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    latencies = sorted(float(row["latency_seconds"]) for row in rows)
    p95_index = min(len(latencies) - 1, int(0.95 * len(latencies)))
    return {
        "samples": len(rows),
        "bleu": sacrebleu.corpus_bleu(predictions, [references]).score,
        "chrf2": sacrebleu.corpus_chrf(predictions, [references], word_order=2).score,
        "latency_mean_seconds": sum(latencies) / len(latencies),
        "latency_p95_seconds": latencies[p95_index],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["training"].get("seed", 42))
    test_dataset = load_jsonl(
        project_path(cfg["paths"]["test_file"]),
        limit=positive_limit(cfg["data"].get("max_test_samples")),
        seed=seed,
    )
    validate_languages(test_dataset, cfg["data"].get("supported_languages", ["zh", "en", "ru", "uz"]))

    model_name = cfg["model"]["base_model"]
    architecture = str(cfg["model"].get("architecture", "m2m100"))
    tokenizer = load_tokenizer(model_name, architecture)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = load_base_model(model_name, architecture, dtype=dtype)
    if args.adapter:
        model = PeftModel.from_pretrained(model, project_path(args.adapter))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.config.use_cache = True

    eval_cfg = cfg.get("evaluation", {})
    max_new_tokens = int(eval_cfg.get("max_new_tokens", 128))
    num_beams = int(eval_cfg.get("num_beams", 1))
    rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for example in test_dataset:
            src_lang = str(example["src_lang"])
            tgt_lang = str(example["tgt_lang"])
            generation_language = prepare_generation(
                tokenizer, architecture, src_lang, tgt_lang
            )
            inputs = tokenizer(
                str(example["src_text"]),
                return_tensors="pt",
                truncation=True,
                max_length=int(cfg["data"]["max_source_length"]),
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(
                **inputs,
                **generation_language,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - started

            rows.append(
                {
                    "direction": f"{src_lang}-{tgt_lang}",
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "source": str(example["src_text"]),
                    "reference": str(example["tgt_text"]),
                    "prediction": tokenizer.batch_decode(generated, skip_special_tokens=True)[0],
                    "latency_seconds": round(latency, 6),
                }
            )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["direction"])].append(row)

    metrics = {direction: score_group(group_rows) for direction, group_rows in sorted(grouped.items())}
    metrics["macro_average"] = {
        "bleu": sum(float(value["bleu"]) for value in metrics.values()) / len(grouped),
        "chrf2": sum(float(value["chrf2"]) for value in metrics.values()) / len(grouped),
    }

    result_dir = project_path(args.output or cfg["paths"]["results_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
