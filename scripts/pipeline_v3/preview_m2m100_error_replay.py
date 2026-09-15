"""Preview-only difficulty scoring for a proposed four-language Exp4 dataset.

This program deliberately has no training or formal-dataset writing action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config  # noqa: E402
from scripts.pipeline_v2.seq2seq_flow import load_model  # noqa: E402
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    atomic_json,
    file_sha256,
    fingerprint,
    model_signature,
)
from scripts.pipeline_v3.fourlang_flow import directions  # noqa: E402

CONFIG_DEFAULT = "configs/multilingual/fourlang_m2m100_exp4_preview_v1.toml"
REQUIRED_FIELDS = {"src_lang", "tgt_lang", "src_text", "tgt_text"}
PREVIEW_STATUS = "AWAITING_USER_APPROVAL_NOT_TRAINABLE"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_identity(source_name: str, line_number: int, row: dict[str, Any]) -> str:
    return sha256_text(f"{source_name}\n{line_number}\n{canonical_json(row)}")


def duplicate_identity(row: dict[str, Any]) -> str:
    source = str(row["src_lang"])
    target = str(row["tgt_lang"])
    # Conservative exact-duplicate grouping: normalize Unicode and whitespace only.
    # Script conversion or semantic similarity must not silently merge distinct rows.
    source_text = " ".join(unicodedata.normalize("NFKC", str(row["src_text"])).split()).casefold()
    target_text = " ".join(unicodedata.normalize("NFKC", str(row["tgt_text"])).split()).casefold()
    return sha256_text(canonical_json([source, target, source_text, target_text]))


def iter_source_rows(path: Path, source_name: str) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing {sorted(missing)}")
            direction = f"{row['src_lang']}-{row['tgt_lang']}"
            if direction not in directions():
                raise ValueError(f"{path}:{line_number} invalid direction {direction}")
            yield {
                "record_id": record_identity(source_name, line_number, row),
                "duplicate_group_id": duplicate_identity(row),
                "source_dataset": source_name,
                "source_line_number": line_number,
                "direction": direction,
                "row": row,
            }


def config_paths(config: dict[str, Any]) -> tuple[list[Path], Path, Path]:
    section = config["preview"]
    return (
        [project_path(value) for value in section["sources"]],
        project_path(section["model"]),
        project_path(section["output_root"]),
    )


def inventory(config: dict[str, Any]) -> dict[str, Any]:
    source_paths, model_path, output_root = config_paths(config)
    missing = [str(path.resolve()) for path in [*source_paths, model_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    dataset_counts: Counter[str] = Counter()
    direction_counts: Counter[str] = Counter()
    training_sources: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    missing_weights = 0
    invalid_weights = 0
    source_records = []
    for path in source_paths:
        source_name = path.relative_to(PROJECT_ROOT).as_posix()
        for item in iter_source_rows(path, source_name):
            row = item["row"]
            dataset_counts[source_name] += 1
            direction_counts[item["direction"]] += 1
            training_sources[str(row.get("training_source", "UNKNOWN"))] += 1
            duplicate_counts[item["duplicate_group_id"]] += 1
            if "weight" not in row:
                missing_weights += 1
            else:
                try:
                    if float(row["weight"]) <= 0:
                        invalid_weights += 1
                except (TypeError, ValueError):
                    invalid_weights += 1
        source_records.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "rows": dataset_counts[source_name],
            }
        )

    duplicate_groups = sum(count > 1 for count in duplicate_counts.values())
    duplicate_records = sum(count for count in duplicate_counts.values() if count > 1)
    report = {
        "schema_version": 1,
        "status": "INVENTORY_COMPLETE_PREVIEW_ONLY",
        "formal_training_data_written": False,
        "training_started": False,
        "sources": source_records,
        "model": {"path": str(model_path.resolve()), "signature": model_signature(model_path)},
        "total_rows": sum(dataset_counts.values()),
        "rows_by_dataset": dict(sorted(dataset_counts.items())),
        "rows_by_direction": dict(sorted(direction_counts.items())),
        "training_source_values": dict(sorted(training_sources.items())),
        "duplicate_groups": duplicate_groups,
        "records_in_duplicate_groups": duplicate_records,
        "missing_original_weights": missing_weights,
        "invalid_original_weights": invalid_weights,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "inventory.json", report)
    print(f"EXP4_INVENTORY_READY: {output_root / 'inventory.json'}", flush=True)
    return report


def score_fingerprint(config: dict[str, Any], report: dict[str, Any]) -> str:
    preview = config["preview"]
    return fingerprint(
        {
            "sources": report["sources"],
            "model": report["model"],
            "batch_size": preview["score_batch_size"],
            "shard_rows": preview["score_shard_rows"],
            "max_source_length": preview["max_source_length"],
            "max_target_length": preview["max_target_length"],
            "scoring": "teacher_forced_mean_token_nll_v1",
        }
    )


def batched(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def score_chunk(
    records: list[dict[str, Any]], tokenizer: Any, model: Any, config: dict[str, Any]
) -> list[dict[str, Any]]:
    scores: list[float | None] = [None] * len(records)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        grouped[item["direction"]].append(index)
    batch_size = int(config["preview"]["score_batch_size"])
    max_source = int(config["preview"]["max_source_length"])
    max_target = int(config["preview"]["max_target_length"])
    for direction, indices in grouped.items():
        source, target = direction.split("-", 1)
        tokenizer.src_lang = source
        tokenizer.tgt_lang = target
        for selection in batched(indices, batch_size):
            rows = [records[index]["row"] for index in selection]
            encoded = tokenizer(
                [str(row["src_text"]) for row in rows],
                padding=True,
                truncation=True,
                max_length=max_source,
                return_tensors="pt",
            )
            labels = tokenizer(
                text_target=[str(row["tgt_text"]) for row in rows],
                padding=True,
                truncation=True,
                max_length=max_target,
                return_tensors="pt",
            )["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            labels = labels.to(model.device)
            with torch.inference_mode():
                logits = model(**encoded, labels=labels).logits
                losses = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(labels.shape)
                mask = labels.ne(-100)
                means = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            for index, value in zip(selection, means.detach().cpu().tolist()):
                scores[index] = float(value)
    if any(value is None or not math.isfinite(value) for value in scores):
        raise RuntimeError("Non-finite or missing difficulty score")
    return [
        {**item, "difficulty_nll": scores[index]}
        for index, item in enumerate(records)
    ]


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
    temporary.replace(path)


def score(config: dict[str, Any], *, allow_cpu: bool = False) -> dict[str, Any]:
    source_paths, model_path, output_root = config_paths(config)
    report = inventory(config)
    signature = score_fingerprint(config, report)
    manifest_path = output_root / "score_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != signature:
            raise RuntimeError("Inputs/model/settings changed; refusing to mix scoring shards.")
    elif any((output_root / "scored_shards").glob("*.jsonl")):
        raise RuntimeError("Scoring shards exist without a matching manifest; preserve and use a new output.")
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "SCORING_IN_PROGRESS",
            "fingerprint": signature,
            "scoring_method": "teacher_forced_mean_token_nll_v1",
            "formal_training_data_written": False,
            "training_started": False,
        },
    )
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA is required for full scoring; use --allow-cpu only for a small test.")
    candidate = {
        **config["student"],
        "path": str(model_path),
    }
    tokenizer, model = load_model(candidate, "en", "zh")
    if not torch.cuda.is_available():
        model = model.float()
    model.eval()
    shard_size = int(config["preview"]["score_shard_rows"])
    shard_index = 0
    completed = 0
    for path in source_paths:
        source_name = path.relative_to(PROJECT_ROOT).as_posix()
        buffer: list[dict[str, Any]] = []
        for item in iter_source_rows(path, source_name):
            buffer.append(item)
            if len(buffer) == shard_size:
                shard_index += 1
                completed += score_one_shard(buffer, shard_index, output_root, tokenizer, model, config)
                buffer = []
        if buffer:
            shard_index += 1
            completed += score_one_shard(buffer, shard_index, output_root, tokenizer, model, config)
    result = {
        "schema_version": 1,
        "status": "SCORING_COMPLETE_PREVIEW_ONLY",
        "fingerprint": signature,
        "rows": completed,
        "shards": shard_index,
        "formal_training_data_written": False,
        "training_started": False,
    }
    atomic_json(manifest_path, result)
    print(f"EXP4_SCORING_READY: {manifest_path}", flush=True)
    return result


def score_one_shard(records, index, output_root, tokenizer, model, config) -> int:
    path = output_root / "scored_shards" / f"part-{index:06d}.jsonl"
    complete = path.with_suffix(".complete.json")
    expected_ids = [records[0]["record_id"], records[-1]["record_id"]]
    if path.exists() and complete.exists():
        meta = json.loads(complete.read_text(encoding="utf-8"))
        if meta.get("rows") != len(records) or meta.get("boundary_record_ids") != expected_ids:
            raise RuntimeError(f"Existing shard does not match current inputs: {path}")
        if meta.get("sha256") != file_sha256(path):
            raise RuntimeError(f"Existing shard checksum failed: {path}")
        print(f"EXP4_SCORE_SHARD_REUSED: {path}", flush=True)
        return len(records)
    if path.exists() or complete.exists():
        raise RuntimeError(f"Partial shard found; inspect it before continuing: {path}")
    scored = score_chunk(records, tokenizer, model, config)
    atomic_jsonl(path, scored)
    atomic_json(
        complete,
        {"rows": len(scored), "boundary_record_ids": expected_ids, "sha256": file_sha256(path)},
    )
    print(f"EXP4_SCORE_SHARD_WRITTEN: {path} rows={len(scored)}", flush=True)
    return len(scored)


def assign_bands(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    thresholds = config["difficulty_bands"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["direction"]].append(item)
    for direction, items in grouped.items():
        ordered = sorted(items, key=lambda item: (item["difficulty_nll"], item["record_id"]))
        count = len(ordered)
        for rank, item in enumerate(ordered, start=1):
            percentile = rank / count
            if percentile <= float(thresholds["easy_max_percentile"]):
                band = "easy"
            elif percentile <= float(thresholds["medium_max_percentile"]):
                band = "medium"
            elif percentile <= float(thresholds["hard_max_percentile"]):
                band = "hard"
            else:
                band = "extreme"
            item["difficulty_percentile_within_direction"] = percentile
            item["difficulty_band"] = band


def apply_provisional_weights(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    settings = config["provisional_weights"]
    duplicate_counts = Counter(item["duplicate_group_id"] for item in records)
    for item in records:
        raw = item["row"].get("weight")
        try:
            original = float(raw)
            if original <= 0:
                raise ValueError
            imputed = False
        except (TypeError, ValueError):
            original = float(settings["missing_original_weight"])
            imputed = True
        band_multiplier = float(settings[item["difficulty_band"]])
        direction_multiplier = float(
            settings["zh_uz_direction_multiplier"]
            if item["direction"] == "zh-uz"
            else settings["other_direction_multiplier"]
        )
        group_size = duplicate_counts[item["duplicate_group_id"]]
        duplicate_multiplier = 1.0 / group_size
        proposed = original * band_multiplier * direction_multiplier * duplicate_multiplier
        if proposed <= 0 or not math.isfinite(proposed):
            raise RuntimeError(f"Proposed weight is not positive: {item['record_id']}")
        item.update(
            {
                "original_weight_effective": original,
                "original_weight_imputed": imputed,
                "band_multiplier": band_multiplier,
                "direction_multiplier": direction_multiplier,
                "duplicate_group_size": group_size,
                "duplicate_multiplier": duplicate_multiplier,
                "proposed_weight": proposed,
            }
        )


def read_scored(output_root: Path) -> list[dict[str, Any]]:
    records = []
    paths = sorted((output_root / "scored_shards").glob("part-*.jsonl"))
    if not paths:
        raise FileNotFoundError("No scored shards found; run score first.")
    for path in paths:
        complete = path.with_suffix(".complete.json")
        if not complete.exists() or json.loads(complete.read_text(encoding="utf-8"))["sha256"] != file_sha256(path):
            raise RuntimeError(f"Unverified scored shard: {path}")
        with path.open(encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    return records


def preview(config: dict[str, Any]) -> dict[str, Any]:
    _, _, output_root = config_paths(config)
    manifest = json.loads((output_root / "score_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SCORING_COMPLETE_PREVIEW_ONLY":
        raise RuntimeError("Scoring is not complete; preview is unavailable.")
    records = read_scored(output_root)
    if len(records) != manifest["rows"]:
        raise RuntimeError("Scored row count does not match the scoring manifest.")
    assign_bands(records, config)
    apply_provisional_weights(records, config)

    band_counts: dict[str, Counter[str]] = defaultdict(Counter)
    direction_counts: Counter[str] = Counter()
    weight_sums: Counter[str] = Counter()
    for item in records:
        band_counts[item["direction"]][item["difficulty_band"]] += 1
        direction_counts[item["direction"]] += 1
        weight_sums[item["direction"]] += item["proposed_weight"]
    sample_count = int(config["planning"]["review_samples_per_band_per_direction"])
    samples = []
    for direction in directions():
        for band in ("easy", "medium", "hard", "extreme"):
            selected = [item for item in records if item["direction"] == direction and item["difficulty_band"] == band]
            selected.sort(key=lambda item: (item["difficulty_nll"], item["record_id"]), reverse=band in {"hard", "extreme"})
            samples.extend(selected[:sample_count])
    atomic_jsonl(output_root / "review_samples.jsonl", samples)

    total = len(records)
    effective_batch = int(config["planning"]["effective_batch_size"])
    steps = math.ceil(total / effective_batch)
    speed = float(config["planning"]["observed_optimizer_steps_per_second"])
    usage = shutil.disk_usage(output_root)
    result = {
        "schema_version": 1,
        "status": PREVIEW_STATUS,
        "hard_stop": "No formal train.jsonl is written and no training action exists in this program.",
        "formal_training_data_written": False,
        "training_started": False,
        "retention_contract": {
            "all_candidate_records_retained": True,
            "all_proposed_weights_positive": min(item["proposed_weight"] for item in records) > 0,
            "each_record_visited_per_full_epoch": True,
            "sampling_replacement": False,
        },
        "total_rows": total,
        "rows_by_direction": dict(sorted(direction_counts.items())),
        "rows_by_direction_and_band": {key: dict(value) for key, value in sorted(band_counts.items())},
        "proposed_weight_sum_by_direction": dict(sorted(weight_sums.items())),
        "proposed_weight_range": {
            "min": min(item["proposed_weight"] for item in records),
            "max": max(item["proposed_weight"] for item in records),
        },
        "provisional_parameters_requiring_user_approval": {
            "difficulty_band_percentiles": config["difficulty_bands"],
            "weight_multipliers": config["provisional_weights"],
            "missing_or_invalid_weight_is_temporarily_imputed": config["provisional_weights"]["missing_original_weight"],
            "zh_uz_priority_is_provisional": config["provisional_weights"]["zh_uz_direction_multiplier"],
        },
        "planning_only": {
            "optimizer_steps_per_epoch": steps,
            "estimated_training_hours_per_epoch_excluding_evaluation": steps / speed / 3600,
            "free_disk_gb_at_preview": usage.free / (1024 ** 3),
        },
        "review_samples": str((output_root / "review_samples.jsonl").resolve()),
        "next_action": "Review preview.json and review_samples.jsonl; do not train until explicit approval.",
    }
    atomic_json(output_root / "preview.json", result)
    print(f"EXP4_PREVIEW_READY_NOT_TRAINABLE: {output_root / 'preview.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "score", "preview", "run-preview"))
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.action == "inventory":
        inventory(config)
    elif args.action == "score":
        score(config, allow_cpu=args.allow_cpu)
    elif args.action == "preview":
        preview(config)
    else:
        score(config, allow_cpu=args.allow_cpu)
        preview(config)


if __name__ == "__main__":
    main()
