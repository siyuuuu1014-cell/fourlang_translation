"""CPU-only cleaning preview of a completed v2 audit; never export training data."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock, Timeout  # noqa: E402

from scripts.pipeline_v3 import audit_zh_uz_training_quality as quality  # noqa: E402

diag = quality.diag
DEFAULT_INPUT = "reports/diagnostics/fourlang/zh_uz_training_quality_v2"
DEFAULT_OUTPUT = "reports/diagnostics/fourlang/zh_uz_cleaning_preview_v1"
DEFAULT_CASES = "configs/review/zh_uz_isolation_candidates_v1.json"
CATEGORIES = ("ISOLATE_CANDIDATE", "REVIEW_REQUIRED", "KEEP_CANDIDATE")
OUTPUTS = {
    "ISOLATE_CANDIDATE": "isolate_candidates.jsonl",
    "REVIEW_REQUIRED": "review_required.jsonl",
    "KEEP_CANDIDATE": "keep_candidates.jsonl",
}
AUDIT_FILES = {
    "quality_audit.json",
    "metadata_recovery.json",
    "training_flags.jsonl",
    "review_blind.jsonl",
    "review_packet.json",
    "summary.md",
}
POLICY = {
    "action": "preview_only_no_training_dataset_export",
    "isolate": "exact audit IDs from assistant-reviewed case registry; pending confirmation",
    "review": "remaining hints, MINOR or unverifiable Teacher metadata",
    "keep": "no issue found by this bounded policy; not quality approval",
}


def validate_cases(document):
    if (
        document.get("schema_version") != 1
        or document.get("action") != "preview_only_no_deletion"
    ):
        raise ValueError("Expected a versioned preview-only case registry.")
    cases = {}
    for item in document["cases"]:
        key = item["audit_id"]
        if (
            not re.fullmatch(r"[a-f0-9]{64}", key)
            or key in cases
            or not item["reason"].strip()
        ):
            raise ValueError(
                "Case IDs must be unique full hashes with nonempty reasons."
            )
        cases[key] = item["reason"]
    return cases


def build_preview(records, cases):
    rows, seen = [], set()
    for original in records:
        key = original["audit_id"]
        weight = float(original["weight"])
        occurrences = original["occurrences"]
        if (
            not math.isfinite(weight)
            or weight <= 0
            or type(occurrences) is not int
            or occurrences <= 0
        ):
            raise ValueError("Invalid weight or occurrence count in audit.")
        expected = diag.fingerprint(
            [
                diag.fingerprint(diag.identity(original)),
                original["training_source"],
                weight,
            ]
        )
        if key != expected or key in seen:
            raise ValueError("Audit identity mismatch or duplicate audit ID.")
        seen.add(key)
        direction = f"{original['src_lang']}-{original['tgt_lang']}"
        if direction not in diag.DIRECTIONS:
            raise ValueError("Preview is restricted to zh<->uz.")
        numeric = quality.checks.numeric_review(
            original["src_text"], original["tgt_text"]
        )
        hints = quality.review_hints(
            original, original["human_training_reference_candidates"], numeric
        )
        reasons = list(hints)
        if diag.is_teacher(original):
            if original["judge_label"] != "PASS":
                reasons.append("teacher_label_requires_review")
            if not quality.known_pass_candidate(original):
                reasons.append("teacher_metadata_requires_review")
        elif not str(original["training_source"]).lower().startswith("human"):
            reasons.append("unknown_source_role")
        category = (
            "ISOLATE_CANDIDATE"
            if key in cases
            else "REVIEW_REQUIRED"
            if reasons
            else "KEEP_CANDIDATE"
        )
        rows.append(
            {
                **original,
                "numeric_evidence": numeric,
                "review_hints": sorted(hints),
                "preview_category": category,
                "preview_reasons": ([cases[key]] if key in cases else [])
                + sorted(set(reasons)),
                "removed_rule_hints": sorted(
                    set(original["review_hints"]) - set(hints)
                ),
                "quality_approved": False,
                "human_confirmation": "",
            }
        )
    rows.sort(key=lambda r: r["audit_id"])
    summary = {
        "schema_version": 1,
        "status": "PREVIEW_ONLY_PENDING_CONFIRMATION",
        "training_started": False,
        "training_data_written": False,
        "policy": POLICY,
        "directions": {},
        "matched_case_ids": sorted(set(cases) & seen),
        "unmatched_case_ids": sorted(set(cases) - seen),
        "limitations": [
            "Keep candidates are not certified translations; unflagged semantic errors can remain.",
            "Isolate candidates are assistant recommendations, not native-human certification.",
            "MINOR is routed to review, not automatically deleted.",
            "Counts describe records/occurrences in the frozen audit, not error rates or epochs.",
            "No rows are dropped, corrected, resampled, reweighted or exported as a training dataset.",
        ],
    }
    for direction in diag.DIRECTIONS:
        group = [r for r in rows if f"{r['src_lang']}-{r['tgt_lang']}" == direction]
        summary["directions"][direction] = {
            category: {
                "unique_records": len(
                    part := [r for r in group if r["preview_category"] == category]
                ),
                "sampled_rows": sum(r["occurrences"] for r in part),
                "teacher_sampled_rows": sum(
                    r["occurrences"] for r in part if diag.is_teacher(r)
                ),
                "non_teacher_sampled_rows": sum(
                    r["occurrences"] for r in part if not diag.is_teacher(r)
                ),
            }
            for category in CATEGORIES
        }
    summary["removed_hint_counts_unique_records"] = dict(
        sorted(
            Counter(hint for row in rows for hint in row["removed_rule_hints"]).items()
        )
    )
    return summary, rows


def load_audit(source):
    """Verify report ownership, completeness, hashes and original input snapshots."""
    manifest_path, done_path = source / "quality_manifest.json", source / "done.json"
    snapshots = {p: diag.file_sha256(p) for p in (manifest_path, done_path)}
    manifest, done = diag.read_json(manifest_path), diag.read_json(done_path)
    signature = diag.fingerprint(manifest["manifest"])
    if manifest["manifest"].get("schema_version") != 2:
        raise ValueError("Expected the completed v2 audit, not v1.")
    if manifest["fingerprint"] != signature or done["fingerprint"] != signature:
        raise ValueError("Audit fingerprint mismatch.")
    if (
        done.get("status") != "REVIEW_PACKET_READY_NOT_QUALITY_APPROVED"
        or set(done["outputs"]) != AUDIT_FILES
    ):
        raise ValueError("Incomplete or unexpected audit outputs.")
    for name, digest in done["outputs"].items():
        path = source / name
        if (
            path.resolve().parent != source.resolve()
            or diag.file_sha256(path) != digest
        ):
            raise ValueError(f"Audit output changed: {name}")
        snapshots[path] = digest
    for item in manifest["manifest"]["inputs"].values():
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(diag.PROJECT_ROOT.resolve()):
            raise ValueError(
                "Audit input is outside the project; run preview on the original server."
            )
        if diag.file_sha256(path) != item["sha256"]:
            raise ValueError(f"Original audit input changed: {path}")
        snapshots[path] = item["sha256"]
    audit = diag.read_json(source / "quality_audit.json")
    if audit.get("schema_version") != 2:
        raise ValueError("Expected v2 quality summary.")
    return diag.read_rows(source / "training_flags.jsonl"), audit, signature, snapshots


def run(args):
    source = diag.project_path(args.audit).resolve()
    output = diag.checked_output(args.output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Preview output must be separate from the source audit.")
    cases_path = diag.project_path(args.cases).resolve()
    case_hash = diag.file_sha256(cases_path)
    cases = validate_cases(diag.read_json(cases_path))
    diag.log("Verifying the completed v2 audit and unchanged original data...")
    records, audit, audit_signature, snapshots = load_audit(source)
    snapshots[cases_path] = case_hash
    summary, rows = build_preview(records, cases)
    summary["source_audit_fingerprint"] = audit_signature
    summary["input_audit_training_rows"] = {
        direction: audit["actual_sampling_audit"]["directions"][direction][
            "training_rows"
        ]
        for direction in diag.DIRECTIONS
    }
    for direction in diag.DIRECTIONS:
        total = sum(
            v["sampled_rows"] for v in summary["directions"][direction].values()
        )
        if (
            total
            != audit["actual_sampling_audit"]["directions"][direction]["training_rows"]
        ):
            raise ValueError(
                "Preview row accounting does not match the completed audit."
            )
    if any(diag.file_sha256(path) != digest for path, digest in snapshots.items()):
        raise RuntimeError("Input changed during preview; no results published.")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        if not (output / "preview_manifest.json").exists() and set(
            p.name for p in output.iterdir()
        ) - {".lock"}:
            raise RuntimeError("Unowned output files; choose a new --output.")
        signature = diag.bind_manifest(
            output / "preview_manifest.json",
            {
                "schema_version": 1,
                "source_audit_fingerprint": audit_signature,
                "inputs": {str(p): digest for p, digest in sorted(snapshots.items())},
                "code": {
                    str(p): diag.file_sha256(p)
                    for p in (
                        Path(__file__),
                        Path(quality.__file__),
                        Path(quality.checks.__file__),
                        Path(diag.__file__),
                    )
                },
                "policy": POLICY,
            },
        )
        diag.preserve_text(output / ".gitignore", "*\n")
        diag.save_json(output / "cleaning_preview.json", summary)
        for category, filename in OUTPUTS.items():
            diag.save_jsonl(
                output / filename,
                [r for r in rows if r["preview_category"] == category],
            )
        # All exact case recommendations plus fixed samples from the other queues.
        selected = [r for r in rows if r["preview_category"] == "ISOLATE_CANDIDATE"]
        for direction in diag.DIRECTIONS:
            for category in ("REVIEW_REQUIRED", "KEEP_CANDIDATE"):
                group = [
                    r
                    for r in rows
                    if f"{r['src_lang']}-{r['tgt_lang']}" == direction
                    and r["preview_category"] == category
                ]
                selected.extend(
                    sorted(
                        group,
                        key=lambda r: diag.fingerprint([2026, r["audit_id"], category]),
                    )[:20]
                )
        diag.save_json(
            output / "cleaning_preview_packet.json",
            {
                "fingerprint": signature,
                "summary": summary,
                "sample_is_not_an_error_rate_estimate": True,
                "review": selected,
            },
        )
        files = (
            "cleaning_preview.json",
            "cleaning_preview_packet.json",
            *OUTPUTS.values(),
        )
        diag.save_json(
            output / "done.json",
            {
                "fingerprint": signature,
                "status": summary["status"],
                "outputs": {name: diag.file_sha256(output / name) for name in files},
            },
        )
    diag.log(
        f"Cleaning preview ready (no data changed): {output / 'cleaning_preview_packet.json'}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", default=DEFAULT_CASES)
    try:
        run(parser.parse_args())
    except Timeout:
        raise SystemExit("Another preview owns the output directory.") from None


if __name__ == "__main__":
    main()
