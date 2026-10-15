"""Build an immutable ZH-UZ v4 KD dataset from an authenticated isolation preview."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock, Timeout  # noqa: E402

from scripts.pipeline_v3 import preview_manual_zh_uz_isolation as preview  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import normalize_rows  # noqa: E402

diag = preview.diag
DEFAULT_SOURCE = "data/distillation/zh_uz/v3/train.jsonl"
DEFAULT_DECISIONS = "configs/review/zh_uz_manual_review_decisions_v1.json"
DEFAULT_IMPACT = "reports/diagnostics/fourlang/zh_uz_manual_isolation_impact_v1"
DEFAULT_OUTPUT = "data/distillation/zh_uz/v4/train.jsonl"
DEFAULT_REPORT = "reports/diagnostics/fourlang/zh_uz_v4_build_v1"
IMPACT_OUTPUTS = {
    "impact_preview.json",
    "isolate_matches.jsonl",
    "protected_matches.jsonl",
}
REPORT_OUTPUTS = {"summary.json", "removed_rows.jsonl"}


def checked_dataset_path(value):
    path = diag.project_path(value).resolve()
    allowed = (diag.PROJECT_ROOT / "data/distillation/zh_uz/v4").resolve()
    if path != allowed / "train.jsonl":
        raise ValueError("Output must be data/distillation/zh_uz/v4/train.jsonl.")
    if allowed.exists() and any(
        item.is_symlink() or item.resolve() != item.absolute()
        for item in allowed.rglob("*")
    ):
        raise ValueError("V4 output directory must not contain links or junctions.")
    return path


def load_impact(source, decision_sha256):
    manifest_path = source / "preview_manifest.json"
    done_path = source / "done.json"
    manifest = diag.read_json(manifest_path)
    done = diag.read_json(done_path)
    signature = diag.fingerprint(manifest["manifest"])
    if (
        manifest.get("fingerprint") != signature
        or done.get("fingerprint") != signature
        or done.get("status") != "IMPACT_PREVIEW_ONLY_PENDING_EXPLICIT_APPLY"
        or set(done.get("outputs", {})) != IMPACT_OUTPUTS
    ):
        raise ValueError("Incomplete or unexpected manual isolation impact preview.")
    snapshots = {
        manifest_path: diag.file_sha256(manifest_path),
        done_path: diag.file_sha256(done_path),
    }
    for name, digest in done["outputs"].items():
        path = source / name
        if path.resolve().parent != source or diag.file_sha256(path) != digest:
            raise ValueError(f"Impact preview output changed: {name}")
        snapshots[path] = digest
    impact = diag.read_json(source / "impact_preview.json")
    if (
        impact.get("status") != "IMPACT_PREVIEW_ONLY_PENDING_EXPLICIT_APPLY"
        or impact.get("decisions_applied") is not False
        or impact.get("training_data_written") is not False
        or impact.get("decision_sha256") != decision_sha256
        or impact.get("datasets", {}).get("validation", {}).get("isolate_rows") != 0
    ):
        raise ValueError("Impact preview is unsafe, stale, or belongs to other decisions.")
    return impact, signature, snapshots


def filter_rows(raw_rows, source, isolate, protected):
    normalized = normalize_rows(pd.DataFrame(raw_rows), origin=str(source))
    if list(normalized.index) != list(range(len(raw_rows))):
        raise ValueError("Source normalization dropped or reordered rows.")
    kept, removed = [], []
    matched = Counter()
    protected_matches = Counter()
    for raw, (_, row) in zip(raw_rows, normalized.iterrows(), strict=True):
        review_id = diag.fingerprint(diag.identity(row))
        direction = f"{row['src_lang']}-{row['tgt_lang']}"
        if review_id in isolate:
            removed.append(
                {
                    "review_id": review_id,
                    "direction": direction,
                    "source_row": raw,
                }
            )
            matched[review_id] += 1
        else:
            kept.append(raw)
        if review_id in protected:
            protected_matches[review_id] += 1
    if set(matched) != isolate or any(count != 1 for count in matched.values()):
        raise ValueError("Every isolation decision must match exactly one KD source row.")
    if set(matched) & set(protected_matches):
        raise RuntimeError("A protected row was selected for isolation.")
    return kept, removed, protected_matches


def run(args):
    source = diag.project_path(args.source).resolve()
    decisions_path = diag.project_path(args.decisions).resolve()
    impact_dir = diag.project_path(args.impact).resolve()
    output = checked_dataset_path(args.output)
    report_dir = diag.checked_output(args.report).resolve()
    if source == output or output.is_relative_to(source):
        raise ValueError("V3 source and V4 output must be separate.")
    decisions_sha256 = diag.file_sha256(decisions_path)
    isolate, protected = preview.validate_decisions(diag.read_json(decisions_path))
    impact, impact_signature, snapshots = load_impact(
        impact_dir, decisions_sha256
    )
    source_sha256 = diag.file_sha256(source)
    snapshots[source] = source_sha256
    snapshots[decisions_path] = decisions_sha256
    raw_rows = diag.read_rows(source)
    expected = impact["datasets"]["kd_source"]
    if (
        len(raw_rows) != expected["rows"]
        or expected["isolate_rows"] != len(isolate)
        or expected["isolate_unique_pairs"] != len(isolate)
    ):
        raise ValueError("V3 source no longer matches the authenticated impact preview.")
    kept, removed, protected_matches = filter_rows(
        raw_rows, source, isolate, protected
    )
    if len(removed) != len(isolate) or len(kept) != len(raw_rows) - len(isolate):
        raise RuntimeError("V4 row accounting failed.")
    if any(diag.file_sha256(path) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Input changed during V4 build; no output published.")

    removed_by_direction = Counter(row["direction"] for row in removed)
    summary = {
        "schema_version": 1,
        "status": "CLEANED_KD_DATASET_V4_BUILT_NOT_TRAINED",
        "source": str(source),
        "source_sha256": source_sha256,
        "source_rows": len(raw_rows),
        "output": str(output),
        "output_rows": len(kept),
        "removed_rows": len(removed),
        "removed_unique_review_ids": len({row["review_id"] for row in removed}),
        "removed_by_direction": dict(sorted(removed_by_direction.items())),
        "protected_review_ids": len(protected),
        "protected_source_rows": sum(protected_matches.values()),
        "decision_sha256": decisions_sha256,
        "impact_preview_fingerprint": impact_signature,
        "training_started": False,
        "original_source_modified": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(report_dir / ".lock"), timeout=0):
        if not (report_dir / "build_manifest.json").exists() and set(
            path.name for path in report_dir.iterdir()
        ) - {".lock"}:
            raise RuntimeError("Unowned report output; choose a new --report.")
        manifest_signature = diag.bind_manifest(
            report_dir / "build_manifest.json",
            {
                "schema_version": 1,
                "inputs": {
                    str(path): digest
                    for path, digest in sorted(snapshots.items())
                },
                "impact_preview_fingerprint": impact_signature,
                "output": str(output),
                "code": {
                    str(path): diag.file_sha256(path)
                    for path in (Path(__file__), Path(preview.__file__), Path(diag.__file__))
                },
            },
        )
        diag.preserve_text(report_dir / ".gitignore", "*\n")
        diag.save_jsonl(output, kept)
        summary["output_sha256"] = diag.file_sha256(output)
        diag.save_jsonl(report_dir / "removed_rows.jsonl", removed)
        diag.save_json(report_dir / "summary.json", summary)
        diag.save_json(
            report_dir / "done.json",
            {
                "fingerprint": manifest_signature,
                "status": summary["status"],
                "dataset": {
                    "path": str(output),
                    "sha256": summary["output_sha256"],
                    "rows": len(kept),
                },
                "outputs": {
                    name: diag.file_sha256(report_dir / name)
                    for name in sorted(REPORT_OUTPUTS)
                },
            },
        )
    diag.log(
        f"Built immutable ZH-UZ v4 KD data: {output} "
        f"({len(raw_rows)} -> {len(kept)}; removed {len(removed)})."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--decisions", default=DEFAULT_DECISIONS)
    parser.add_argument("--impact", default=DEFAULT_IMPACT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    try:
        run(parser.parse_args())
    except Timeout:
        raise SystemExit("Another V4 build owns the report directory.") from None


if __name__ == "__main__":
    main()
