"""Read-only Exp2 data audit and resumable, aligned zh<->uz model comparison.

Only a new directory below reports/diagnostics is writable. No training, model
downloads, dataset regeneration, checkpoint selection or deployment takes place.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock, Timeout  # noqa: E402

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, read_json  # noqa: E402
from scripts.pipeline_v2.training_safety import (  # noqa: E402
    atomic_json,
    file_sha256,
    fingerprint,
    model_signature,
)
from scripts.pipeline_v3.fourlang_flow import _read_table, normalize_rows  # noqa: E402
from scripts.pipeline_v2 import seq2seq_flow as flow  # noqa: E402

DIRECTIONS = ("zh-uz", "uz-zh")
ROLES = ("teacher", "exp1", "exp2")
FIELDS = ("src_lang", "tgt_lang", "src_text", "tgt_text")
CHECKPOINT_ROOT = Path("results/student/fourlang/exp2/checkpoints/shared")
DEFAULT_OUTPUT = "reports/diagnostics/fourlang/exp2_zh_uz"
CODE_ROOT = Path(__file__).resolve().parents[2]


def log(message):
    print(message, flush=True)


def project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def identity(row):
    return tuple(row[field] for field in FIELDS)


def is_teacher(row):
    return "teacher" in str(row.get("training_source", "")).lower()


def versions():
    names = (
        "torch",
        "transformers",
        "accelerate",
        "datasets",
        "numpy",
        "sacrebleu",
        "tokenizers",
        "opencc-python-reimplemented",
    )
    return {name: importlib.metadata.version(name) for name in names}


def checked_output(value):
    path = project_path(value)
    parent = (PROJECT_ROOT / "reports/diagnostics").resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError("--output must be a NEW subdirectory of reports/diagnostics.")
    # Never follow a link inside the result tree into models or source data.
    if path.exists() and any(
        item.is_symlink() or item.resolve() != item.absolute()
        for item in path.rglob("*")
    ):
        raise ValueError("Diagnostic output must not contain links/junctions.")
    return path


def bind_manifest(path, manifest):
    record = {"fingerprint": fingerprint(manifest), "manifest": manifest}
    if path.exists():
        if read_json(path) != record:
            raise RuntimeError(
                f"Inputs/settings changed: {path}. Use a new --output; keep the old run."
            )
    else:
        atomic_json(path, record)
    return record["fingerprint"]


def preserve_text(path, content):
    """Generated reports are immutable too: do not overwrite human annotations."""
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing output differs (possibly edited): {path}. Use a new --output."
            )
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save_json(path, value):
    preserve_text(
        path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def save_jsonl(path, rows):
    preserve_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        ),
    )


def load_run_inputs(config):
    """Authenticate the actual training rows and saved subset, not a new sample."""
    checkpoint = PROJECT_ROOT / CHECKPOINT_ROOT
    paths = {
        name: checkpoint / f"{name}.json"
        for name in (
            "run_manifest",
            "validation_subset",
            "initial_validation",
            "finished",
        )
    }
    records = {name: read_json(path) for name, path in paths.items()}
    manifest = records["run_manifest"]["manifest"]
    signature = fingerprint(manifest)
    if manifest["experiment"] != "exp2" or not manifest["shared"]:
        raise RuntimeError("Expected the completed shared Exp2 run.")
    if any(record.get("fingerprint") != signature for record in records.values()):
        raise RuntimeError(
            "Training manifest/subset/baseline/export belong to different runs."
        )
    train_path = PROJECT_ROOT / "data/multilingual/fourlang/exp2/train.jsonl"
    validation_path = train_path.with_name("validation.jsonl")
    log(
        "Checking the actual Exp2 training/validation rows against the completed run..."
    )
    train, validation = read_rows(train_path), read_rows(validation_path)
    for name, rows in (("train", train), ("validation", validation)):
        if fingerprint(rows) != manifest[f"{name}_sha256"]:
            raise RuntimeError(
                f"{name} data changed since Exp2 training. Preserve the original run; do not re-aggregate."
            )
    saved_groups = records["validation_subset"]["groups"]
    # Recompute ONLY as an integrity check. Evaluation always consumes saved_groups.
    expected = flow.fixed_validation_groups(
        validation,
        int(manifest["settings"]["direction_validation_samples"]),
        manifest["seed"],
    )
    if saved_groups != expected:
        raise RuntimeError(
            "Saved validation subset does not match the recorded training data/seed."
        )
    groups = {direction: saved_groups[direction] for direction in DIRECTIONS}
    for direction, rows in groups.items():
        if not rows or any(
            f"{row['src_lang']}-{row['tgt_lang']}" != direction for row in rows
        ):
            raise RuntimeError(f"Invalid saved validation direction: {direction}")
    kd_entries = [entry for entry in config["pair_data"] if entry["pair"] == "zh_uz"]
    if len(kd_entries) != 1:
        raise ValueError("Expected exactly one zh_uz pair_data entry.")
    kd_path = project_path(kd_entries[0]["kd_train"])
    paths.update(train=train_path, validation=validation_path, kd_source=kd_path)
    baseline = records["initial_validation"]["metrics"]
    best = records["finished"]["report"].get("best_model_checkpoint")
    if best:
        best_name = str(best).replace("\\", "/").rsplit("/", 1)[-1]
        step = best_name.removeprefix("checkpoint-")
        if not step.isdigit():
            raise ValueError("Invalid recorded best-model checkpoint name.")
        metric_path = checkpoint / f"direction_metrics_step_{step}.json"
        if metric_path.exists():
            paths["best_direction_metrics"] = metric_path
    best_metrics = (
        read_json(paths["best_direction_metrics"])
        if "best_direction_metrics" in paths
        else {}
    )
    return train, groups, manifest, records["finished"], baseline, best_metrics, paths


def metadata_value(row, key):
    value = row.get(key)
    if value is None or flow.pd.isna(value):
        return "UNKNOWN"
    return str(value).strip() or "UNKNOWN"


def normalized_metadata(path):
    frame = _read_table(path).reset_index(drop=True)
    normalized = normalize_rows(frame, origin=str(path))
    output = []
    for index, row in normalized.iterrows():
        if f"{row['src_lang']}-{row['tgt_lang']}" not in DIRECTIONS:
            continue
        original = frame.loc[index].to_dict()
        output.append(
            {
                **row.to_dict(),
                **{
                    key: metadata_value(original, key)
                    for key in (
                        "judge_label",
                        "teacher_id",
                        "teacher_usefulness",
                        "source_corpus",
                        "pair_id",
                    )
                },
            }
        )
    return output


def resolve_metadata(row, index):
    matches = [
        item
        for item in index.get(identity(row), [])
        if item["training_source"] == row["training_source"]
        and math.isclose(
            float(item["weight"]), float(row["weight"]), rel_tol=1e-9, abs_tol=1e-9
        )
    ]
    status = "MATCHED" if matches else "UNMATCHED"
    values = {}
    for key in ("judge_label", "teacher_id", "teacher_usefulness", "source_corpus"):
        choices = sorted(
            {
                metadata_value(item, key).upper()
                if key == "judge_label"
                else metadata_value(item, key)
                for item in matches
            }
        )
        values[key] = (
            choices[0] if len(choices) == 1 else ("AMBIGUOUS" if choices else "UNKNOWN")
        )
        if len(choices) > 1:
            status = "AMBIGUOUS"
    return status, values


def audit_training(train, original):
    index = defaultdict(list)
    for row in original:
        index[identity(row)].append(row)
    samples, exemplars = Counter(), {}
    for row in train:
        if f"{row['src_lang']}-{row['tgt_lang']}" not in DIRECTIONS:
            continue
        weight = float(row["weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("Non-positive/non-finite training weight.")
        key = (*identity(row), row["training_source"], weight)
        samples[key] += 1
        exemplars[key] = row
    traces = []
    for key in sorted(samples):
        row = exemplars[key]
        status, meta = resolve_metadata(row, index)
        traces.append(
            {
                **{field: row[field] for field in FIELDS},
                "sample_id": fingerprint(identity(row)),
                "training_source": row["training_source"],
                "weight": float(row["weight"]),
                "occurrences": samples[key],
                "metadata_status": status,
                **meta,
            }
        )
    report = {
        "schema_version": 1,
        "scope": "actual Exp2 sampled rows, zh<->uz only",
        "notes": [
            "MINOR is read from matching source metadata, never inferred from a low weight.",
            "Matching uses normalized source/target + languages + training_source + weight.",
            "Source KD metadata is the current local file; historical provenance cannot be guaranteed where metadata is missing/ambiguous.",
            "Weighted shares describe dataset weight sums, not exact gradient contribution (loss normalizes within each batch).",
            "Occurrences count repetitions within one dataset, not training epochs; unique rows use the four text/language fields.",
        ],
        "directions": {},
    }
    for direction in DIRECTIONS:
        rows = [
            row for row in traces if f"{row['src_lang']}-{row['tgt_lang']}" == direction
        ]
        teacher = [row for row in rows if is_teacher(row)]
        if not rows or not teacher:
            raise ValueError(f"Missing training/teacher rows for {direction}.")

        def count(items):
            return sum(row["occurrences"] for row in items)

        def weight_sum(items):
            return sum(row["weight"] * row["occurrences"] for row in items)

        labels = {}
        for label in sorted({row["judge_label"] for row in teacher}):
            items = [row for row in teacher if row["judge_label"] == label]
            labels[label] = {
                "rows": count(items),
                "unique_text_rows": len({identity(row) for row in items}),
                "weight_sum": weight_sum(items),
            }
        minor = labels.get("MINOR", {"rows": 0, "weight_sum": 0})
        unknown_rows = sum(
            labels.get(label, {}).get("rows", 0) for label in ("UNKNOWN", "AMBIGUOUS")
        )
        teacher_ids = Counter()
        for row in teacher:
            teacher_ids[row["teacher_id"]] += row["occurrences"]
        repeats = Counter()
        for row in teacher:
            repeats[identity(row)] += row["occurrences"]
        report["directions"][direction] = {
            "training_rows": count(rows),
            "teacher_rows": count(teacher),
            "human_rows": count(rows) - count(teacher),
            "unique_teacher_text_rows": len(repeats),
            "max_teacher_occurrences": max(repeats.values()),
            "teacher_labels": labels,
            "teacher_ids": dict(teacher_ids),
            "teacher_metadata_status": dict(
                sum(
                    (
                        Counter({row["metadata_status"]: row["occurrences"]})
                        for row in teacher
                    ),
                    Counter(),
                )
            ),
            "minor_share_of_teacher_rows": minor["rows"] / count(teacher),
            "minor_share_bounds_with_unknown_labels": [
                minor["rows"] / count(teacher),
                (minor["rows"] + unknown_rows) / count(teacher),
            ],
            "minor_share_of_all_rows": minor["rows"] / count(rows),
            "minor_share_of_teacher_weight": minor["weight_sum"] / weight_sum(teacher),
            "teacher_weight_sum": weight_sum(teacher),
            "all_weight_sum": weight_sum(rows),
        }
    return report, traces


def model_plan(manifest, finished, audit, original, selection):
    candidates, signatures, coverage = {}, {}, {}
    for direction in DIRECTIONS:
        candidate = dict(selection["directions"][direction]["candidate"])
        selected_id = candidate["id"]
        v3_ids = {
            row["teacher_id"]
            for row in original
            if f"{row['src_lang']}-{row['tgt_lang']}" == direction
            and row["training_source"] == "teacher_kd_v3"
        }
        if v3_ids - {selected_id, "UNKNOWN"}:
            raise RuntimeError(
                f"Selected Teacher {selected_id} differs from v3 data teachers {sorted(v3_ids)}. Diagnose the selection first."
            )
        ids = audit["directions"][direction]["teacher_ids"]
        if not (v3_ids - {"UNKNOWN"}) and not ids.get(selected_id):
            raise RuntimeError(
                f"Cannot associate selected Teacher {selected_id} with actual KD rows in {direction}."
            )
        path_key = (
            direction.replace("-", "_") + "_path"
            if candidate["family"] == "marian_pair"
            else "path"
        )
        candidate[path_key] = str(project_path(candidate[path_key]))
        candidates[f"teacher/{direction}"] = {
            **candidate,
            "require_local_artifact": True,
        }
        coverage[direction] = {
            "evaluated_teacher_id": selected_id,
            "sampled_teacher_ids": ids,
            "known_matching_rows": ids.get(selected_id, 0),
            "note": "This is the selected v3 Teacher, not a reconstruction of mixed historical Teachers.",
        }
    for role in ("exp1", "exp2"):
        path = project_path(f"results/student/fourlang/{role}/best_model/shared")
        candidate = {
            "id": role,
            "family": manifest["family"],
            "path": str(path),
            "require_local_artifact": True,
        }
        for direction in DIRECTIONS:
            candidates[f"{role}/{direction}"] = candidate
    # Hash once per artifact directory; never fall back to Hub/base-model weights.
    for name, candidate in candidates.items():
        direction = name.split("/")[1]
        key = (
            direction.replace("-", "_") + "_path"
            if candidate["family"] == "marian_pair"
            else "path"
        )
        path = candidate[key]
        if path not in signatures:
            log(f"Checking local model files: {path}")
            signatures[path] = model_signature(Path(path))
        if (
            name.startswith("exp1/")
            and signatures[path] != manifest["source_artifacts"]
        ):
            raise RuntimeError(
                "Exp1 weights/tokenizer no longer match Exp2's recorded starting model."
            )
        if (
            name.startswith("exp2/")
            and signatures[path] != finished["export_signature"]
        ):
            raise RuntimeError(
                "Exp2 export differs from finished.json; refusing comparison."
            )
    return candidates, signatures, coverage


def chunk_predictions(directory, rows, signature, size, predict):
    """predict=None validates completed chunks without loading any model."""
    output = []
    for start in range(0, len(rows), size):
        batch = rows[start : start + size]
        ids = [fingerprint(identity(row)) for row in batch]
        path = directory / f"chunk_{start:06d}.json"
        if path.exists():
            record = read_json(path)
            expected_hash = record.pop("content_sha256", None)
            if (
                expected_hash != fingerprint(record)
                or record.get("fingerprint") != signature
                or record.get("sample_ids") != ids
            ):
                raise RuntimeError(f"Corrupted/mismatched inference chunk: {path}")
            predictions = record["predictions"]
        elif predict is None:
            return None
        else:
            log(
                f"{directory.parent.name}/{directory.name}: generating {start + 1}-{start + len(batch)}/{len(rows)}"
            )
            predictions = predict([row["src_text"] for row in batch])
            if len(predictions) != len(batch) or any(
                not isinstance(text, str) for text in predictions
            ):
                raise RuntimeError("Model returned misaligned/non-text predictions.")
            record = {
                "fingerprint": signature,
                "sample_ids": ids,
                "predictions": predictions,
            }
            atomic_json(path, {**record, "content_sha256": fingerprint(record)})
            log(
                f"{directory.parent.name}/{directory.name}: saved {start + len(batch)}/{len(rows)}"
            )
        if len(predictions) != len(batch) or any(
            not isinstance(text, str) for text in predictions
        ):
            raise RuntimeError(f"Invalid predictions in {path}")
        output.extend(predictions)
    return output


def run_inference(output, groups, candidates, config, signature, chunk_rows, seed):
    predictions, pending = {}, defaultdict(list)
    for name, candidate in candidates.items():
        _, direction = name.split("/")
        directory = output / "chunks" / name
        result = chunk_predictions(
            directory, groups[direction], signature, chunk_rows, None
        )
        if result is not None:
            log(f"Reusing completed {name}: {len(result)} rows")
            predictions[name] = result
        else:
            # One model resident at a time; share Teacher loading across directions.
            pending[fingerprint(candidate)].append((name, candidate))
    for tasks in pending.values():
        name, candidate = tasks[0]
        source, target = name.split("/")[1].split("-")
        log(f"Loading {name} (local weights only; no training)...")
        flow.set_seed(seed)
        # FP32 weights + CUDA FP16 autocast match the training-time direction eval.
        tokenizer, model = flow.load_model(candidate, source, target, training=True)
        predict = None
        try:
            model.eval()
            model.requires_grad_(False)
            runtime = {
                "tokenizer_class": type(tokenizer).__name__,
                "vocab_size": len(tokenizer),
                "model_class": type(model).__name__,
                "dtype": str(model.dtype),
                "generation_config": model.generation_config.to_dict(),
            }
            for name, candidate in tasks:
                _, direction = name.split("/")
                source, target = direction.split("-")
                info_path = output / "chunks" / name / "runtime.json"
                info_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(info_path, runtime)

                def predict(texts, tok=tokenizer, mdl=model):
                    with flow.torch.autocast(
                        device_type="cuda",
                        dtype=flow.torch.float16,
                        enabled=flow.torch.cuda.is_available(),
                    ):
                        return flow.translate(
                            tok, mdl, candidate["family"], source, target, texts, config
                        )

                predictions[name] = chunk_predictions(
                    output / "chunks" / name,
                    groups[direction],
                    signature,
                    chunk_rows,
                    predict,
                )
        finally:
            del predict
            del tokenizer, model
            gc.collect()
            if flow.torch.cuda.is_available():
                flow.torch.cuda.empty_cache()
    return predictions


def comparison_report(groups, predictions, baseline, best_metrics, seed):
    scores, comparisons = {}, []
    for direction, rows in groups.items():
        target = direction.split("-")[1]
        references = [row["tgt_text"] for row in rows]
        scored = {
            role: flow.metrics(predictions[f"{role}/{direction}"], references, target)
            for role in ROLES
        }
        scored["exp2_minus_exp1"] = {
            key: scored["exp2"][key] - scored["exp1"][key] for key in ("bleu", "chrf2")
        }
        scored["teacher_minus_exp1"] = {
            key: scored["teacher"][key] - scored["exp1"][key]
            for key in ("bleu", "chrf2")
        }
        scored["difference_from_training_log"] = {
            role: {
                key: scored[role][key] - recorded[f"eval_{direction}_{key}"]
                for key in ("bleu", "chrf2")
                if f"eval_{direction}_{key}" in recorded
            }
            for role, recorded in (("exp1", baseline), ("exp2", best_metrics))
        }
        scores[direction] = scored
        for index, row in enumerate(rows):
            outputs = {
                role: predictions[f"{role}/{direction}"][index] for role in ROLES
            }
            sentence_scores = {
                role: float(
                    flow.CHRF(word_order=2)
                    .sentence_score(text, [row["tgt_text"]])
                    .score
                )
                for role, text in outputs.items()
            }
            comparisons.append(
                {
                    "sample_id": fingerprint(identity(row)),
                    "direction": direction,
                    "source": row["src_text"],
                    "reference": row["tgt_text"],
                    **outputs,
                    "sentence_chrf2": sentence_scores,
                    "exp2_minus_exp1_sentence_chrf2": sentence_scores["exp2"]
                    - sentence_scores["exp1"],
                    "heuristics_not_judgments": {
                        role: {
                            "empty": not text.strip(),
                            "exact_source_copy": text == row["src_text"],
                        }
                        for role, text in outputs.items()
                    },
                }
            )
    review = []
    for direction in groups:
        rows = [row for row in comparisons if row["direction"] == direction]
        worst = sorted(
            [row for row in rows if row["exp2_minus_exp1_sentence_chrf2"] < 0],
            key=lambda row: (row["exp2_minus_exp1_sentence_chrf2"], row["sample_id"]),
        )[:20]
        random_rows = sorted(
            rows, key=lambda row: fingerprint([seed, row["sample_id"], "review"])
        )[:20]
        reasons = defaultdict(list)
        for reason, chosen in (
            ("largest_metric_decreases", worst),
            ("fixed_random", random_rows),
        ):
            for row in chosen:
                reasons[row["sample_id"]].append(reason)
        review.extend(
            {
                **row,
                "selection_reasons": reasons[row["sample_id"]],
                "human_review": {"preferred_output": "", "error_type": "", "notes": ""},
            }
            for row in rows
            if row["sample_id"] in reasons
        )
    return scores, comparisons, review


def summary_markdown(audit, scores, warnings):
    lines = [
        "# 中乌 Exp2 诊断（不训练、不晋级）",
        "",
        "## 固定验证集对比",
        "",
        "与原训练使用相同的已保存样本、解码设置及指标定义。chrF2 沿用 CHRF(word_order=2)。",
        "这里只比较中乌两个方向，不代表 12 方向整体效果，也不是 FLORES devtest。",
        "",
        "|方向|样本|Teacher chrF2|Exp1 chrF2|Exp2 chrF2|Exp2−Exp1|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction, item in scores.items():
        lines.append(
            f"|{direction}|{item['exp1']['samples']}|{item['teacher']['chrf2']:.4f}|{item['exp1']['chrf2']:.4f}|{item['exp2']['chrf2']:.4f}|{item['exp2_minus_exp1']['chrf2']:+.4f}|"
        )
    lines += ["", "## 实际采样训练数据", ""]
    for direction, item in audit["directions"].items():
        lines.append(
            f"- {direction}：Teacher {item['teacher_rows']} 条（{item['unique_teacher_text_rows']} 个不同文本对），MINOR 占 Teacher 条数 {item['minor_share_of_teacher_rows']:.2%}，占 Teacher 权重和 {item['minor_share_of_teacher_weight']:.2%}。"
        )
        labels = item["teacher_labels"]
        lines.append(
            f"  UNKNOWN={labels.get('UNKNOWN', {}).get('rows', 0)}，AMBIGUOUS={labels.get('AMBIGUOUS', {}).get('rows', 0)}；它们不是 PASS。有未知标签时，上述 MINOR 比例仅为已确认部分。"
        )
    lines += [
        "",
        "## 解释边界与下一步",
        "",
        "- 先看 metrics.json 中 difference_from_training_log；差异明显时先查环境、精度和 tokenizer，不先归因于 KD。",
        "- MINOR 占比仅说明相关性，不能证明它导致退步；需人工检查逐句译文，再决定是否做独立对照实验。",
        "- review_samples.jsonl 含每方向固定随机最多 20 条及指标实际下降最多 20 条（不足则取全部，重叠合并）。后者是有偏样本，不能用其错误率代表全量。",
        "- 如需填写人工意见，先复制 review_samples.jsonl 为另一个文件；生成产物不会覆盖已有修改。",
        "- 本诊断不更改训练数据、模型、选模结果和部署门槛。",
        "",
    ]
    if warnings:
        lines += ["## 注意", ""] + [f"- {warning}" for warning in warnings] + [""]
    return "\n".join(lines)


def diagnose(args):
    output = checked_output(args.output)
    if args.checkpoint_rows < 8 or args.checkpoint_rows % 8:
        raise ValueError(
            "--checkpoint-rows must be a positive multiple of 8 (the inference batch size)."
        )
    if output.exists() and not (output / "audit_manifest.json").exists():
        leftovers = {item.name for item in output.iterdir()} - {
            ".lock",
            "audit_manifest.json.tmp",
        }
        if leftovers:
            raise RuntimeError(
                "Output directory already contains unowned files. Choose a new --output."
            )
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        config = load_config(args.config)
        train, groups, manifest, finished, baseline, best_metrics, paths = (
            load_run_inputs(config)
        )
        log("Tracing sampled Teacher rows back to current source KD metadata...")
        original = normalized_metadata(paths["kd_source"])
        audit, traces = audit_training(train, original)
        implementation = {
            str(path.relative_to(CODE_ROOT)): file_sha256(path)
            for path in (
                Path(__file__),
                Path(flow.__file__),
                CODE_ROOT / "scripts/pipeline_v2/training_safety.py",
                CODE_ROOT / "scripts/pipeline_v3/fourlang_flow.py",
                CODE_ROOT / "scripts/pipeline_v3/language_normalization.py",
            )
        }
        data_manifest = {
            "schema_version": 1,
            "files": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in paths.items()
            },
            "implementation": implementation,
            "versions": versions(),
        }
        bind_manifest(output / "audit_manifest.json", data_manifest)
        preserve_text(output / ".gitignore", "*\n")
        save_json(output / "training_audit.json", audit)
        save_jsonl(output / "training_sample_trace.jsonl", traces)
        log(f"Data audit saved: {output / 'training_audit.json'}")
        if args.audit_only:
            log("Audit only: no model has been loaded, no inference/training started.")
            return
        selection_path = (
            PROJECT_ROOT / "results/model_selection/zh_uz/selected_teacher.json"
        )
        candidates, signatures, coverage = model_plan(
            manifest, finished, audit, original, read_json(selection_path)
        )
        config = {
            "training": manifest["settings"],
            "language_codes": manifest["language_codes"],
            "deployment": manifest["decoding"],
        }
        runtime_manifest = {
            "schema_version": 1,
            "audit_fingerprint": fingerprint(data_manifest),
            "models": candidates,
            "model_signatures": signatures,
            "teacher_selection_sha256": file_sha256(selection_path),
            "config": config,
            "seed": manifest["seed"],
            "checkpoint_rows": args.checkpoint_rows,
            "precision": "fp32_weights_cuda_fp16_autocast"
            if flow.torch.cuda.is_available()
            else "fp32_cpu",
            "device": flow.torch.cuda.get_device_name(0)
            if flow.torch.cuda.is_available()
            else "cpu",
        }
        signature = bind_manifest(output / "comparison_manifest.json", runtime_manifest)
        save_json(output / "teacher_coverage.json", coverage)
        warnings = [
            f"Dependency changed since training: {name}: {version} -> {versions().get(name)}"
            for name, version in manifest["versions"].items()
            if versions().get(name) != version
        ]
        for role, recorded in (("exp1", baseline), ("exp2", best_metrics)):
            if any(
                f"eval_{direction}_{metric}" not in recorded
                for direction in DIRECTIONS
                for metric in ("bleu", "chrf2")
            ):
                warnings.append(
                    f"Incomplete historical {role} direction metrics; missing log comparisons are left empty, not zero."
                )
        for name, digest in manifest.get("implementation", {}).items():
            current = [
                value
                for path, value in implementation.items()
                if Path(path).name == name
            ]
            if current != [digest]:
                warnings.append(
                    f"Training implementation changed: {name}; historical scores may not reproduce exactly."
                )
        if (manifest["precision"] == "fp16") != flow.torch.cuda.is_available():
            warnings.append(
                "Training and diagnostic precision/device differ; historical scores may differ."
            )
        for warning in warnings:
            log(f"WARNING: {warning}")
        started = time.monotonic()
        predictions = run_inference(
            output,
            groups,
            candidates,
            config,
            signature,
            args.checkpoint_rows,
            manifest["seed"],
        )
        scores, comparisons, review = comparison_report(
            groups, predictions, baseline, best_metrics, manifest["seed"]
        )
        save_json(output / "metrics.json", scores)
        save_jsonl(output / "comparisons.jsonl", comparisons)
        save_jsonl(output / "review_samples.jsonl", review)
        preserve_text(output / "summary.md", summary_markdown(audit, scores, warnings))
        artifacts = (
            "training_audit.json",
            "training_sample_trace.jsonl",
            "teacher_coverage.json",
            "metrics.json",
            "comparisons.jsonl",
            "review_samples.jsonl",
            "summary.md",
        )
        save_json(
            output / "done.json",
            {
                "status": "PASS",
                "meaning": "diagnostics completed, NOT model promotion",
                "fingerprint": signature,
                "output_sha256": {
                    name: file_sha256(output / name) for name in artifacts
                },
            },
        )
        log(
            f"Diagnostics complete ({time.monotonic() - started:.0f}s inference/report phase): {output / 'summary.md'}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multilingual/fourlang.toml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only audit sampled training rows; do not load models.",
    )
    parser.add_argument(
        "--checkpoint-rows",
        type=int,
        default=32,
        help="Persist every N predictions; multiple of 8, default 32.",
    )
    args = parser.parse_args()
    # Defense in depth; candidate_path also requires explicit local artifacts.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        diagnose(args)
    except Timeout:
        parser.exit(
            2,
            "Another diagnostic process owns this output directory. Do not launch a duplicate.\n",
        )
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Diagnostic stopped safely: {error}\n")


if __name__ == "__main__":
    main()
