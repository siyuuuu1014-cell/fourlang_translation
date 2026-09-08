"""CPU-only review packet for the actual Exp2 zh<->uz training rows.

Heuristics are review hints, never semantic judgments or automatic filters.
This entry point cannot train, infer, change datasets, or select a checkpoint.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock, Timeout  # noqa: E402

from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402

DEFAULT_OUTPUT = "reports/diagnostics/fourlang/zh_uz_training_quality_v1"
RULES = {
    "same_character_run": 12,
    "same_word_run": 4,
    "minimum_zh_han_share_of_letters": 0.60,
    "target_to_single_human_reference_length_range": [0.5, 2.0],
    "digits": "ASCII integer/decimal token sets; no semantic or unit conversion",
    "uz_script": "flag Cyrillic/Han/kana; Latin is not proof of Uzbek",
    "references": "same directed source, explicitly human-prefixed training_source only",
    "action": "review_only_no_filtering",
}
HAN = re.compile(r"[\u3400-\u9fff]")
NON_LATIN_UZ = re.compile(r"[\u0400-\u052f\u3040-\u30ff\u3400-\u9fff]")
DIGITS = re.compile(r"[0-9]+(?:[.,][0-9]+)*")


def source_key(row):
    return row["src_lang"], row["tgt_lang"], row["src_text"]


def human_references(original):
    """Never join on target text, opposite directions, or validation examples."""
    index = defaultdict(set)
    for row in original:
        if str(row["training_source"]).lower().startswith("human"):
            index[source_key(row)].add(row["tgt_text"])
    return index


def text_hints(text, language):
    hints = []
    if not text.strip():
        hints.append("empty")
    folded = text.casefold()
    if re.search(r"(\S)\1{11,}", folded):
        hints.append("long_character_run")
    if re.search(r"\b(\w+)(?:\s+\1){3,}\b", folded):
        hints.append("repeated_word")
    if re.search(r"https?://|<[^>]+>", text):
        hints.append("url_or_markup")
    letters = sum(char.isalpha() for char in text)
    if language == "zh" and letters and len(HAN.findall(text)) / letters < 0.60:
        hints.append("low_han_share")
    if language == "uz" and NON_LATIN_UZ.search(text):
        hints.append("non_latin_script")
    return hints


def review_hints(row, references):
    flags = [
        f"{side}_{hint}"
        for side, field, lang in (
            ("source", "src_text", "src_lang"),
            ("target", "tgt_text", "tgt_lang"),
        )
        for hint in text_hints(row[field], row[lang])
    ]
    if row["src_text"].strip() == row["tgt_text"].strip():
        flags.append("exact_source_copy")
    source_digits = set(DIGITS.findall(row["src_text"]))
    target_digits = set(DIGITS.findall(row["tgt_text"]))
    if source_digits != target_digits:
        flags.append("digit_token_set_difference")
    if len(references) == 1:
        reference_length = len(re.sub(r"\s", "", references[0]))
        target_length = len(re.sub(r"\s", "", row["tgt_text"]))
        ratio = target_length / max(1, reference_length)
        if ratio < 0.5 or ratio > 2.0:
            flags.append("length_vs_single_human_reference")
    return sorted(flags)


def known_pass_candidate(row):
    """Metadata-only count, not a declaration that the translation is correct."""
    return (
        diag.is_teacher(row)
        and row["metadata_status"] == "MATCHED"
        and row["judge_label"] == "PASS"
        and row["teacher_usefulness"].upper() in {"HIGH", "MEDIUM"}
        and row["teacher_id"].upper() not in {"UNKNOWN", "AMBIGUOUS"}
    )


def select_review(rows, seed, per_stratum):
    buckets = defaultdict(list)
    for row in rows:
        label = row["judge_label"] if diag.is_teacher(row) else "NON_TEACHER"
        buckets[(row["direction"], label)].append(row)
    chosen, reasons, strata = {}, defaultdict(list), []
    for (direction, label), group in sorted(buckets.items()):
        selected = sorted(
            group, key=lambda r: diag.fingerprint([seed, r["audit_id"], "stratum"])
        )[:per_stratum]
        strata.append(
            {
                "direction": direction,
                "label": label,
                "available_unique_records": len(group),
                "selected": len(selected),
            }
        )
        for row in selected:
            chosen[row["audit_id"]] = row
            reasons[row["audit_id"]].append("fixed_label_stratum")
    # Supplementary hint sample; not a random error-rate estimate.
    for direction in diag.DIRECTIONS:
        flagged = [r for r in rows if r["direction"] == direction and r["review_hints"]]
        selected = sorted(
            flagged, key=lambda r: diag.fingerprint([seed, r["audit_id"], "hint"])
        )[:per_stratum]
        for row in selected:
            chosen[row["audit_id"]] = row
            reasons[row["audit_id"]].append("fixed_hint_sample")
    review = []
    for key in sorted(
        chosen, key=lambda key: diag.fingerprint([seed, key, "blind_order"])
    ):
        row = chosen[key]
        # Hide the automatic label, source role, and flags from the initial reviewer.
        review.append(
            {
                "audit_id": key,
                "direction": row["direction"],
                "source": row["src_text"],
                "translation": row["tgt_text"],
                "human_training_reference_candidates": row[
                    "human_training_reference_candidates"
                ],
                "human_reference_count": row["human_reference_count"],
                "human_review": {
                    "meaning_preserved": "",
                    "omission": "",
                    "addition": "",
                    "terminology": "",
                    "numbers_names": "",
                    "script_register": "",
                    "reference_reliable": "",
                    "notes": "",
                },
            }
        )
    return review, dict(sorted(reasons.items())), strata


def build_report(train, original, seed=2026, per_stratum=20):
    if not 1 <= per_stratum <= 100:
        raise ValueError("--review-per-stratum must be between 1 and 100.")
    audit, traces = diag.audit_training(train, original)
    references = human_references(original)
    enriched = []
    for row in traces:
        refs = sorted(references.get(source_key(row), set()))
        enriched.append(
            {
                **row,
                "audit_id": diag.fingerprint(
                    [row["sample_id"], row["training_source"], row["weight"]]
                ),
                "direction": f"{row['src_lang']}-{row['tgt_lang']}",
                "review_hints": review_hints(row, refs),
                "human_reference_count": len(refs),
                "human_training_reference_candidates": refs[:5],
                "reference_preview_truncated": len(refs) > 5,
                "known_pass_candidate": known_pass_candidate(row),
            }
        )
    _, pool_traces = diag.audit_training(original, original)
    summary = {
        "schema_version": 1,
        "status": "REVIEW_PACKET_READY_NOT_QUALITY_APPROVED",
        "training_started": False,
        "rules": RULES,
        "seed": seed,
        "actual_sampling_audit": audit,
        "directions": {},
        "limitations": [
            "Hints are not judgments: numbers may be spelled out; names and citations may be valid.",
            "References come only from current human-tagged training rows; not verified gold.",
            "Monolingual Teacher rows may have no reference; no reference is not an error.",
            "Missing labels stay UNKNOWN, conflicting metadata stays AMBIGUOUS; weights do not infer labels.",
            "Known PASS candidate is metadata-only; parse history is not reconstructed from absent fields.",
            "Random sampling is within label strata over unique records, not occurrence-weighted overall sampling.",
            "Hint samples are biased; neither their error rate nor unweighted stratum averages estimates overall quality.",
            "Current source metadata is not proof of the historical provenance of missing labels.",
            "No semantic omission/terminology decision is made by this script.",
        ],
    }
    for direction in diag.DIRECTIONS:
        rows = [r for r in enriched if r["direction"] == direction]
        by_label = {}
        for label in sorted(
            {r["judge_label"] if diag.is_teacher(r) else "NON_TEACHER" for r in rows}
        ):
            group = [
                r
                for r in rows
                if (r["judge_label"] if diag.is_teacher(r) else "NON_TEACHER") == label
            ]
            hints_unique, hints_sampled = Counter(), Counter()
            for row in group:
                hints_unique.update(row["review_hints"])
                for hint in row["review_hints"]:
                    hints_sampled[hint] += row["occurrences"]
            by_label[label] = {
                "unique_records": len(group),
                "sampled_rows": sum(r["occurrences"] for r in group),
                "flagged_unique_records": sum(bool(r["review_hints"]) for r in group),
                "flagged_sampled_rows": sum(
                    r["occurrences"] for r in group if r["review_hints"]
                ),
                "hints_unique_records": dict(sorted(hints_unique.items())),
                "hints_sampled_rows": dict(sorted(hints_sampled.items())),
            }
        pool = [
            r for r in pool_traces if f"{r['src_lang']}-{r['tgt_lang']}" == direction
        ]
        summary["directions"][direction] = {
            "by_label": by_label,
            "teacher_without_human_reference_unique_records": sum(
                diag.is_teacher(r) and r["human_reference_count"] == 0 for r in rows
            ),
            "sampled_known_pass_unique_text_pairs": len(
                {diag.identity(r) for r in rows if known_pass_candidate(r)}
            ),
            "current_pool_known_pass_unique_text_pairs": len(
                {diag.identity(r) for r in pool if known_pass_candidate(r)}
            ),
        }
    review, selection, strata = select_review(enriched, seed, per_stratum)
    summary["review_sampling"] = {
        "per_stratum": per_stratum,
        "selected": len(review),
        "strata": strata,
    }
    return summary, enriched, review, selection


def summary_markdown(summary):
    lines = [
        "# 中乌训练数据：只读质量审查包",
        "",
        "状态：审查包已生成，尚未通过人工质量审核；未开始训练。",
        "",
        "|方向|训练条数|Teacher 条数|标签明确的 PASS 候选不同文本对（当前池）|",
        "|---|---:|---:|---:|",
    ]
    for direction in diag.DIRECTIONS:
        audit = summary["actual_sampling_audit"]["directions"][direction]
        quality = summary["directions"][direction]
        lines.append(
            f"|{direction}|{audit['training_rows']}|{audit['teacher_rows']}|{quality['current_pool_known_pass_unique_text_pairs']}|"
        )
    lines.extend(
        [
            "",
            f"抽取 {summary['review_sampling']['selected']} 条供复核。",
            "先看 review_blind.jsonl；确认后再查看 review_packet.json 中的标签和抽样来源。",
            "标记只是疑点，不自动删样本；无参考译文不等于有错误。",
            "不要原地填写生成文件；另存人工审阅副本。",
            "下载 review_packet.json 上传即可继续分析，不要用终端输出整份文件。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args):
    if not 1 <= args.review_per_stratum <= 100:
        raise ValueError("--review-per-stratum must be between 1 and 100.")
    output = diag.checked_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        if not (output / "quality_manifest.json").exists():
            leftovers = {p.name for p in output.iterdir()} - {
                ".lock",
                "quality_manifest.json.tmp",
            }
            if leftovers:
                raise RuntimeError(
                    "Output contains unowned files; choose a new --output."
                )
        config_path = diag.project_path(args.config)
        config_hash = diag.file_sha256(config_path)
        config = diag.load_config(config_path)
        checkpoint = diag.PROJECT_ROOT / diag.CHECKPOINT_ROOT
        data = diag.PROJECT_ROOT / "data/multilingual/fourlang/exp2"
        snapshot_paths = {config_path, data / "train.jsonl", data / "validation.jsonl"}
        snapshot_paths.update(
            checkpoint / f"{name}.json"
            for name in (
                "run_manifest",
                "validation_subset",
                "initial_validation",
                "finished",
            )
        )
        snapshot_paths.update(checkpoint.glob("direction_metrics_step_*.json"))
        snapshot_paths.update(
            diag.project_path(entry["kd_train"])
            for entry in config["pair_data"]
            if entry["pair"] == "zh_uz"
        )
        before = {path: diag.file_sha256(path) for path in snapshot_paths}
        if before[config_path] != config_hash:
            raise RuntimeError("Config changed during audit; retry with stable inputs.")
        train, _, manifest, _, _, _, paths = diag.load_run_inputs(config)
        paths = {**paths, "config": config_path}
        if not set(paths.values()).issubset(snapshot_paths):
            raise RuntimeError(
                "Input set changed during audit; retry with stable inputs."
            )
        hashes = {name: before[path] for name, path in paths.items()}
        diag.log("Reading current KD metadata and tracing the sampled training rows...")
        original = diag.normalized_metadata(paths["kd_source"])
        summary, enriched, review, selection = build_report(
            train, original, args.seed, args.review_per_stratum
        )
        if hashes != {name: diag.file_sha256(path) for name, path in paths.items()}:
            raise RuntimeError(
                "Input changed during audit; no report will be published."
            )
        implementation = [
            Path(__file__),
            Path(diag.__file__),
            Path(diag.flow.__file__),
            diag.CODE_ROOT / "scripts/pipeline_v3/fourlang_flow.py",
            diag.CODE_ROOT / "scripts/pipeline_v3/language_normalization.py",
            diag.CODE_ROOT / "scripts/pipeline_v2/training_safety.py",
            diag.CODE_ROOT / "scripts/pipeline_v2/common.py",
        ]
        signature = diag.bind_manifest(
            output / "quality_manifest.json",
            {
                "schema_version": 1,
                "exp2_run_fingerprint": diag.fingerprint(manifest),
                "inputs": {
                    name: {"path": str(path), "sha256": hashes[name]}
                    for name, path in paths.items()
                },
                "code": {
                    str(p.relative_to(diag.CODE_ROOT)): diag.file_sha256(p)
                    for p in implementation
                },
                "versions": diag.versions(),
                "rules": RULES,
                "seed": args.seed,
                "review_per_stratum": args.review_per_stratum,
            },
        )
        diag.preserve_text(output / ".gitignore", "*\n")
        diag.save_json(output / "quality_audit.json", summary)
        diag.save_jsonl(output / "training_flags.jsonl", enriched)
        diag.save_jsonl(output / "review_blind.jsonl", review)
        selected_ids = set(selection)
        diag.save_json(
            output / "review_packet.json",
            {
                "fingerprint": signature,
                "summary": summary,
                "review": review,
                "selection_reasons": selection,
                "metadata_for_after_blind_review": [
                    r for r in enriched if r["audit_id"] in selected_ids
                ],
            },
        )
        diag.preserve_text(output / "summary.md", summary_markdown(summary))
        files = (
            "quality_audit.json",
            "training_flags.jsonl",
            "review_blind.jsonl",
            "review_packet.json",
            "summary.md",
        )
        diag.save_json(
            output / "done.json",
            {
                "fingerprint": signature,
                "status": "REVIEW_PACKET_READY_NOT_QUALITY_APPROVED",
                "outputs": {name: diag.file_sha256(output / name) for name in files},
            },
        )
        diag.log(
            f"Review packet ready (no training/inference): {output / 'review_packet.json'}"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multilingual/fourlang.toml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--review-per-stratum", type=int, default=20)
    try:
        run(parser.parse_args())
    except Timeout:
        raise SystemExit(
            "Another audit owns this output directory. Do not start a duplicate."
        ) from None


if __name__ == "__main__":
    main()
