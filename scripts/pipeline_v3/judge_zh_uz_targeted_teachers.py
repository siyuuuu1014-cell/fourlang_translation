"""Two-pass Qwen review and conservative selection of targeted teacher pairs."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock  # noqa: E402
from scripts.pipeline_v3 import auto_judge_acceptance_pilot as auto  # noqa: E402

diag = auto.diag


def load_teachers(paths: list[Path]):
    teachers = {}
    metadata = {}
    for path in paths:
        rows = diag.read_rows(path)
        if len(rows) != 6000:
            raise ValueError(f"Expected 6000 teacher rows: {path}")
        teacher_ids = {row.get("teacher_id") for row in rows}
        if len(teacher_ids) != 1 or None in teacher_ids:
            raise ValueError(f"Invalid teacher identity: {path}")
        teacher_id = str(next(iter(teacher_ids)))
        if teacher_id in teachers:
            raise ValueError(f"Duplicate teacher: {teacher_id}")
        by_id = {row["candidate_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate candidate IDs: {path}")
        teachers[teacher_id] = by_id
        metadata[teacher_id] = {
            "path": str(path),
            "sha256": diag.file_sha256(path),
        }
    if len(teachers) != 2:
        raise ValueError("Exactly two independent teacher files are required")
    ids = [set(rows) for rows in teachers.values()]
    if ids[0] != ids[1]:
        raise ValueError("Teacher candidate IDs do not align")
    return teachers, metadata, sorted(ids[0])


def blind_rows(teachers, candidate_ids):
    teacher_ids = sorted(teachers)
    rows, keys = [], {}
    for candidate_id in candidate_ids:
        first = teachers[teacher_ids[0]][candidate_id]
        second = teachers[teacher_ids[1]][candidate_id]
        if first["src_text"] != second["src_text"]:
            raise ValueError(f"Source mismatch: {candidate_id}")
        roles = list(teacher_ids)
        if int(hashlib.sha256(candidate_id.encode()).hexdigest()[:8], 16) % 2:
            roles.reverse()
        blind_id = "targeted-" + hashlib.sha256(candidate_id.encode()).hexdigest()[:20]
        rows.append(
            {
                "blind_id": blind_id,
                "candidate_id": candidate_id,
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": first["src_text"],
                "structured_facts": first.get("structured_facts", {}),
                "A": teachers[roles[0]][candidate_id]["teacher_text"],
                "B": teachers[roles[1]][candidate_id]["teacher_text"],
            }
        )
        keys[blind_id] = {"A": roles[0], "B": roles[1]}
    return rows, keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", required=True)
    parser.add_argument("--preferred-teacher", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/models/Qwen3-8B")
    parser.add_argument(
        "--output",
        default="reports/diagnostics/fourlang/zh_uz_targeted_v1_judge_v1",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.max_input_tokens, args.max_new_tokens) <= 0:
        raise ValueError("Generation settings must be positive")

    teacher_paths = [diag.project_path(path) for path in args.teacher]
    teachers, teacher_metadata, candidate_ids = load_teachers(teacher_paths)
    if args.preferred_teacher not in teachers:
        raise ValueError("--preferred-teacher is not one of the supplied teachers")
    rows, keys = blind_rows(teachers, candidate_ids)
    output = diag.checked_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        manifest = diag.bind_manifest(
            output / "manifest.json",
            {
                "teachers": teacher_metadata,
                "preferred_teacher": args.preferred_teacher,
                "judge_model": str(Path(args.model).resolve()),
                "passes": 2,
                "pass2_reverses_candidate_order": True,
                "acceptance_rule": "source usable and at least one teacher DIRECT with two-pass agreement",
                "batch_size": args.batch_size,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "prompt": auto.PROMPT,
                "code": {
                    str(path): diag.file_sha256(path)
                    for path in (Path(__file__), Path(auto.__file__), Path(auto.qwen.__file__))
                },
            },
        )
        diag.save_jsonl(output / "blind_teacher_candidates.jsonl", rows)
        if args.prepare_only:
            diag.log(f"Prepared {len(rows)} blind teacher comparisons; no model loaded.")
            return

        model_path = Path(args.model).resolve()
        if not (model_path / "config.json").is_file():
            raise ValueError("Local Qwen judge model is missing config.json")
        diag.log("Hashing local judge model for safe resume...")
        model_manifest = diag.bind_manifest(
            output / "model_manifest.json",
            {"path": str(model_path), "signature": diag.model_signature(model_path)},
        )
        signature = diag.fingerprint([manifest, model_manifest])
        first_inputs = auto.make_inputs(rows, reverse=False)
        second_inputs = auto.make_inputs(rows, reverse=True)
        first_dir = output / "chunks" / "pass1"
        second_dir = output / "chunks" / "pass2"
        first_dir.mkdir(parents=True, exist_ok=True)
        second_dir.mkdir(parents=True, exist_ok=True)
        first_raw = diag.chunk_predictions(
            first_dir, first_inputs, signature, args.batch_size, None
        )
        second_raw = diag.chunk_predictions(
            second_dir, second_inputs, signature, args.batch_size, None
        )
        predictor = None
        try:
            if first_raw is None or second_raw is None:
                predictor = auto.qwen.make_predict(
                    model_path, args.max_input_tokens, args.max_new_tokens
                )
            if first_raw is None:
                first_raw = diag.chunk_predictions(
                    first_dir, first_inputs, signature, args.batch_size, predictor
                )
            if second_raw is None:
                second_raw = diag.chunk_predictions(
                    second_dir, second_inputs, signature, args.batch_size, predictor
                )
        finally:
            del predictor
            gc.collect()

        judged = auto.reveal(rows, keys, first_raw, second_raw)
        selected = []
        rejection_reasons = Counter()
        for judgment in judged:
            candidate_id = judgment["candidate_id"]
            final = judgment["final"]
            if not final["agreement"] or final["source_usable"] is not True:
                rejection_reasons["unresolved_or_source_unusable"] += 1
                continue
            direct = [
                teacher_id
                for teacher_id, grade in judgment["model_grades"].items()
                if grade == "DIRECT"
            ]
            if not direct:
                rejection_reasons["no_direct_teacher"] += 1
                continue
            chosen = (
                args.preferred_teacher
                if args.preferred_teacher in direct
                else sorted(direct)[0]
            )
            row = teachers[chosen][candidate_id]
            selected.append(
                {
                    "pair_id": candidate_id,
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": row["src_text"],
                    "tgt_text": row["teacher_text"],
                    "category": row["category"],
                    "challenge_tags": row["challenge_tags"],
                    "structured_facts": row["structured_facts"],
                    "teacher_id": chosen,
                    "weight": 0.5,
                    "training_source": "targeted_zh_uz_v1_qwen_screened",
                    "usage": "TRAINING_CANDIDATE_PENDING_HUMAN_SPOTCHECK",
                }
            )

        summary = {
            "schema_version": 1,
            "status": "QWEN_SCREENED_CANDIDATES_PENDING_HUMAN_SPOTCHECK",
            "input_rows": len(rows),
            "selected_rows": len(selected),
            "rejected_rows": len(rows) - len(selected),
            "rejection_reasons": dict(rejection_reasons),
            "selected_by_teacher": dict(Counter(row["teacher_id"] for row in selected)),
            "selected_by_category": dict(Counter(row["category"] for row in selected)),
            "same_model_two_pass_review_not_independent_human_certification": True,
            "training_started": False,
            "training_data_written": False,
        }
        diag.save_jsonl(output / "auto_judgments.jsonl", judged)
        diag.save_jsonl(output / "selected_candidates.jsonl", selected)
        diag.save_json(output / "summary.json", summary)
        diag.log(
            f"TARGETED_JUDGE_READY: selected={len(selected)}/{len(rows)}; "
            "human spot-check still required."
        )


if __name__ == "__main__":
    main()
