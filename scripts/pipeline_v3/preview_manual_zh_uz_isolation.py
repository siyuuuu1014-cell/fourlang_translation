"""Preview exact dataset impact of manual ZH-UZ isolation decisions; never mutate data."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock, Timeout  # noqa: E402

from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402

DEFAULT_ADJUDICATION = (
    "reports/diagnostics/fourlang/zh_uz_semantic_adjudication_v2"
)
DEFAULT_DECISIONS = (
    "configs/review/zh_uz_manual_review_decisions_v1.json"
)
DEFAULT_OUTPUT = (
    "reports/diagnostics/fourlang/zh_uz_manual_isolation_impact_v1"
)
DEFAULT_CONFIG = "configs/multilingual/fourlang.toml"
ADJUDICATION_OUTPUTS = {
    "offline_recovered_v1.jsonl",
    "supplement_input.jsonl",
    "supplement_results.jsonl",
    "corrected_conservative_preview.jsonl",
    "summary.json",
    "final_review_packet.json",
}
OUTPUTS = {
    "impact_preview.json",
    "isolate_matches.jsonl",
    "protected_matches.jsonl",
}


def validate_decisions(document):
    if (
        document.get("schema_version") != 1
        or document.get("status") != "MANUAL_REVIEW_DECISIONS_NOT_APPLIED"
        or document.get("application_policy")
        != "Only isolate_review_ids may be used by a future dry-run. Protected and pending rows must remain unchanged."
    ):
        raise ValueError("Expected the versioned, unapplied manual decision registry.")
    isolate = document.get("isolate_review_ids")
    protected_items = document.get("protect_from_automatic_isolation")
    if not isinstance(isolate, list) or not isinstance(protected_items, list):
        raise ValueError("Decision lists are missing.")
    protected = []
    for item in protected_items:
        if not isinstance(item, dict) or not str(item.get("reason", "")).strip():
            raise ValueError("Protected decisions require a nonempty reason.")
        protected.append(item.get("review_id"))
    for name, values in (("isolate", isolate), ("protected", protected)):
        if (
            len(values) != len(set(values))
            or any(not isinstance(v, str) or not re.fullmatch(r"[a-f0-9]{64}", v) for v in values)
        ):
            raise ValueError(f"Invalid or duplicate {name} review IDs.")
    if set(isolate) & set(protected):
        raise ValueError("Isolation and protection decisions overlap.")
    summary = document.get("summary", {})
    if (
        summary.get("isolate") != len(isolate)
        or summary.get("protect_from_automatic_isolation") != len(protected)
    ):
        raise ValueError("Decision summary does not match its lists.")
    return set(isolate), {item["review_id"]: item["reason"] for item in protected_items}


def load_adjudication(source):
    manifest_path = source / "manifest.json"
    done_path = source / "done.json"
    manifest = diag.read_json(manifest_path)
    done = diag.read_json(done_path)
    signature = diag.fingerprint(manifest["manifest"])
    if (
        manifest.get("fingerprint") != signature
        or done.get("fingerprint") != signature
        or done.get("status")
        != "CORRECTED_ADJUDICATION_PREVIEW_PENDING_HUMAN_CONFIRMATION"
        or set(done.get("outputs", {})) != ADJUDICATION_OUTPUTS
    ):
        raise ValueError("Incomplete or unexpected corrected adjudication.")
    snapshots = {manifest_path: diag.file_sha256(manifest_path), done_path: diag.file_sha256(done_path)}
    for name, digest in done["outputs"].items():
        path = source / name
        if path.resolve().parent != source or diag.file_sha256(path) != digest:
            raise ValueError(f"Corrected adjudication output changed: {name}")
        snapshots[path] = digest
    summary = diag.read_json(source / "summary.json")
    rows = diag.read_rows(source / "corrected_conservative_preview.jsonl")
    ids = [row.get("review_id") for row in rows]
    if len(rows) != summary.get("first_pass_pairs") or len(ids) != len(set(ids)):
        raise ValueError("Corrected adjudication accounting mismatch.")
    for row in rows:
        expected = diag.fingerprint(diag.identity(row))
        if row["review_id"] != expected:
            raise ValueError("Corrected adjudication review identity mismatch.")
    return rows, signature, snapshots


def row_counts(rows):
    counts = Counter()
    sources = {}
    for row in rows:
        review_id = diag.fingerprint(diag.identity(row))
        counts[review_id] += 1
        sources.setdefault(review_id, Counter())[str(row.get("training_source", "UNKNOWN"))] += 1
    return counts, sources


def build_impact(adjudicated, isolate, protected, datasets):
    by_id = {row["review_id"]: row for row in adjudicated}
    unknown = sorted((isolate | set(protected)) - set(by_id))
    if unknown:
        raise ValueError(f"Manual decisions contain unknown review IDs: {unknown[:5]}")
    indexed = {name: row_counts(rows) for name, rows in datasets.items()}

    def matches(review_ids, *, reasons=None):
        output = []
        for review_id in sorted(review_ids):
            row = by_id[review_id]
            counts = {name: indexed[name][0][review_id] for name in sorted(indexed)}
            output.append(
                {
                    "review_id": review_id,
                    "src_lang": row["src_lang"],
                    "tgt_lang": row["tgt_lang"],
                    "src_text": row["src_text"],
                    "tgt_text": row["tgt_text"],
                    "current_adjudication_category": row.get("adjudication_category"),
                    "dataset_rows": counts,
                    "exp2_training_sources": dict(
                        sorted(indexed.get("exp2_train", ({}, {}))[1].get(review_id, {}).items())
                    ),
                    **({"protection_reason": reasons[review_id]} if reasons else {}),
                    "action_applied": False,
                }
            )
        return output

    isolate_matches = matches(isolate)
    protected_matches = matches(set(protected), reasons=protected)
    summary = {
        "schema_version": 1,
        "status": "IMPACT_PREVIEW_ONLY_PENDING_EXPLICIT_APPLY",
        "training_started": False,
        "training_data_written": False,
        "decisions_applied": False,
        "manual_isolate_review_ids": len(isolate),
        "protected_review_ids": len(protected),
        "datasets": {},
        "directions": {},
        "limitations": [
            "Counts are exact identity matches against current files; no fuzzy text matching is used.",
            "No source, training, validation, model, checkpoint, or configuration file is changed.",
            "Protected and pending rows are never included in isolation impact.",
            "This assistant review is not native-speaker certification.",
        ],
    }
    for name, rows in datasets.items():
        counts = indexed[name][0]
        summary["datasets"][name] = {
            "rows": len(rows),
            "isolate_rows": sum(counts[key] for key in isolate),
            "isolate_unique_pairs": sum(bool(counts[key]) for key in isolate),
            "protected_rows": sum(counts[key] for key in protected),
        }
    for direction in diag.DIRECTIONS:
        selected = [
            row
            for row in isolate_matches
            if f"{row['src_lang']}-{row['tgt_lang']}" == direction
        ]
        summary["directions"][direction] = {
            "isolate_unique_pairs": len(selected),
            "exp2_train_rows": sum(row["dataset_rows"].get("exp2_train", 0) for row in selected),
            "kd_source_rows": sum(row["dataset_rows"].get("kd_source", 0) for row in selected),
            "validation_rows": sum(row["dataset_rows"].get("validation", 0) for row in selected),
        }
    if summary["datasets"].get("validation", {}).get("isolate_rows"):
        raise RuntimeError("An isolation decision overlaps validation data; manual investigation required.")
    return summary, isolate_matches, protected_matches


def load_completed_datasets(config_path):
    config = diag.load_config(config_path)
    train, _groups, _manifest, _finished, _baseline, _best, paths = diag.load_run_inputs(config)
    validation = diag.read_rows(paths["validation"])
    kd_source = diag.normalized_metadata(paths["kd_source"])
    return {
        "exp2_train": train,
        "validation": validation,
        "kd_source": kd_source,
    }, paths


def run(args):
    source = diag.project_path(args.adjudication).resolve()
    output = diag.checked_output(args.output).resolve()
    decisions_path = diag.project_path(args.decisions).resolve()
    config_path = diag.project_path(args.config).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Impact output must be separate from adjudication output.")
    decision_document = diag.read_json(decisions_path)
    isolate, protected = validate_decisions(decision_document)
    adjudicated, adjudication_signature, snapshots = load_adjudication(source)
    datasets, paths = load_completed_datasets(config_path)
    snapshots[decisions_path] = diag.file_sha256(decisions_path)
    snapshots[config_path] = diag.file_sha256(config_path)
    for name in ("train", "validation", "kd_source"):
        snapshots[paths[name]] = diag.file_sha256(paths[name])
    summary, isolate_matches, protected_matches = build_impact(
        adjudicated, isolate, protected, datasets
    )
    summary["adjudication_fingerprint"] = adjudication_signature
    summary["decision_sha256"] = snapshots[decisions_path]
    if any(diag.file_sha256(path) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Input changed during impact preview; no results published.")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        if not (output / "preview_manifest.json").exists() and set(
            path.name for path in output.iterdir()
        ) - {".lock"}:
            raise RuntimeError("Unowned output files; choose a new --output.")
        signature = diag.bind_manifest(
            output / "preview_manifest.json",
            {
                "schema_version": 1,
                "adjudication_fingerprint": adjudication_signature,
                "inputs": {str(path): digest for path, digest in sorted(snapshots.items())},
                "code": {
                    str(path): diag.file_sha256(path)
                    for path in (Path(__file__), Path(diag.__file__))
                },
            },
        )
        diag.preserve_text(output / ".gitignore", "*\n")
        diag.save_json(output / "impact_preview.json", summary)
        diag.save_jsonl(output / "isolate_matches.jsonl", isolate_matches)
        diag.save_jsonl(output / "protected_matches.jsonl", protected_matches)
        diag.save_json(
            output / "done.json",
            {
                "fingerprint": signature,
                "status": summary["status"],
                "outputs": {
                    name: diag.file_sha256(output / name) for name in sorted(OUTPUTS)
                },
            },
        )
    diag.log(f"Manual isolation impact ready (no data changed): {output / 'impact_preview.json'}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudication", default=DEFAULT_ADJUDICATION)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    try:
        run(parser.parse_args())
    except Timeout:
        raise SystemExit("Another impact preview owns the output directory.") from None


if __name__ == "__main__":
    main()
