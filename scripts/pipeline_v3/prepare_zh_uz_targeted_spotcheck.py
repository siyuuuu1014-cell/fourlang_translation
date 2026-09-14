"""Prepare a deterministic, stratified human spot-check for zh->uz candidates.

This command is deliberately read-only with respect to training data.  It writes
only a review packet below reports/diagnostics and never certifies translations.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock  # noqa: E402
from scripts.pipeline_v3 import diagnose_zh_uz as diag  # noqa: E402


EXPECTED_CATEGORIES = (
    "travel_hotel",
    "medical_emergency",
    "numbers_time",
    "negation_logic",
    "commerce",
    "complex_workflow",
)


def _rank(pair_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\0{pair_id}".encode("utf-8")).hexdigest()


def build_packet(
    rows: list[dict],
    *,
    per_category: int = 50,
    medical_extra: int = 50,
    seed: str = "zh-uz-targeted-spotcheck-v1",
) -> tuple[list[dict], list[dict]]:
    if per_category <= 0 or medical_extra < 0:
        raise ValueError("Sampling sizes must be positive")
    by_category: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        category = str(row.get("category", ""))
        if not pair_id or pair_id in seen:
            raise ValueError("Every candidate must have a unique pair_id")
        if category not in EXPECTED_CATEGORIES:
            raise ValueError(f"Unexpected category: {category!r}")
        if row.get("src_lang") != "zh" or row.get("tgt_lang") != "uz":
            raise ValueError(f"Unexpected direction: {pair_id}")
        seen.add(pair_id)
        by_category[category].append(row)

    missing = set(EXPECTED_CATEGORIES) - set(by_category)
    if missing:
        raise ValueError(f"Missing categories: {sorted(missing)}")

    selected = []
    for category in EXPECTED_CATEGORIES:
        count = per_category + (
            medical_extra if category == "medical_emergency" else 0
        )
        ranked = sorted(
            by_category[category], key=lambda row: _rank(row["pair_id"], seed)
        )
        if len(ranked) < count:
            raise ValueError(
                f"Not enough {category} rows: need {count}, got {len(ranked)}"
            )
        selected.extend(ranked[:count])

    selected.sort(
        key=lambda row: (
            EXPECTED_CATEGORIES.index(row["category"]),
            _rank(row["pair_id"], seed),
        )
    )
    packet, key = [], []
    for position, row in enumerate(selected, 1):
        review_id = "spot-" + hashlib.sha256(
            f"{seed}\0{row['pair_id']}".encode("utf-8")
        ).hexdigest()[:20]
        packet.append(
            {
                "review_id": review_id,
                "position": position,
                "category": row["category"],
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": row["src_text"],
                "tgt_text": row["tgt_text"],
                "challenge_tags": row.get("challenge_tags", []),
                "structured_facts": row.get("structured_facts", {}),
                "human_review": {
                    "decision": "",
                    "corrected_tgt_text": "",
                    "issue_tags": [],
                    "notes": "",
                },
            }
        )
        key.append(
            {
                "review_id": review_id,
                "pair_id": row["pair_id"],
                "teacher_id": row.get("teacher_id"),
                "training_source": row.get("training_source"),
            }
        )
    return packet, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="reports/diagnostics/fourlang/zh_uz_targeted_v1_judge_v1/selected_candidates.jsonl",
    )
    parser.add_argument(
        "--output",
        default="reports/diagnostics/fourlang/zh_uz_targeted_v1_spotcheck_v1",
    )
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--medical-extra", type=int, default=50)
    parser.add_argument("--seed", default="zh-uz-targeted-spotcheck-v1")
    args = parser.parse_args()

    source = diag.project_path(args.source)
    rows = diag.read_rows(source)
    packet, key = build_packet(
        rows,
        per_category=args.per_category,
        medical_extra=args.medical_extra,
        seed=args.seed,
    )
    output = diag.checked_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".lock"), timeout=0):
        manifest = {
            "schema_version": 1,
            "source": str(source),
            "source_sha256": diag.file_sha256(source),
            "source_rows": len(rows),
            "seed": args.seed,
            "per_category": args.per_category,
            "medical_extra": args.medical_extra,
        }
        diag.bind_manifest(output / "manifest.json", manifest)
        diag.save_jsonl(output / "blind_spotcheck.jsonl", packet)
        diag.save_jsonl(output / "review_key.jsonl", key)
        diag.save_json(
            output / "summary.json",
            {
                "schema_version": 1,
                "status": "HUMAN_SPOTCHECK_PACKET_READY_NOT_REVIEWED",
                "source_rows": len(rows),
                "sampled_rows": len(packet),
                "sampled_by_category": dict(Counter(row["category"] for row in packet)),
                "allowed_decisions": ["PASS", "MINOR_FIX", "FAIL"],
                "training_started": False,
                "training_data_written": False,
                "instructions": (
                    "Fill human_review in blind_spotcheck.jsonl. MINOR_FIX requires "
                    "corrected_tgt_text; do not inspect review_key.jsonl before review."
                ),
            },
        )
    diag.log(
        f"TARGETED_SPOTCHECK_READY: sampled={len(packet)}/{len(rows)}; "
        "no training data written."
    )


if __name__ == "__main__":
    main()
