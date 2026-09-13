"""Apply the frozen AI-assisted review to the Exp2 direct-test predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def preserve(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Existing output differs: {path}; use a new --output-dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="reports/diagnostics/fourlang/zh_uz_direct_exp2_v1/predictions.jsonl",
    )
    parser.add_argument(
        "--decisions",
        default="configs/review/zh_uz_direct_exp2_v1_semantic_review.json",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/diagnostics/fourlang/zh_uz_direct_exp2_v1_review_v1",
    )
    args = parser.parse_args()

    rows = read_jsonl(Path(args.predictions))
    decisions = read_json(Path(args.decisions))
    labels = {row_id: ("PASS", "") for row_id in decisions["pass_ids"]}
    for label in ("minor", "fail"):
        for row_id, note in decisions[label].items():
            if row_id in labels:
                raise ValueError(f"Duplicate decision: {row_id}")
            labels[row_id] = (label.upper(), note)
    row_ids = {row["id"] for row in rows}
    if len(rows) != 110 or len(row_ids) != 110 or row_ids != set(labels):
        raise ValueError("Predictions and frozen decisions do not contain the same 110 IDs")

    reviewed = []
    by_scenario = defaultdict(Counter)
    counts = Counter()
    for row in rows:
        label, note = labels[row["id"]]
        counts[label] += 1
        by_scenario[row["scenario"]][label] += 1
        reviewed.append({**row, "semantic_grade": label, "review_notes": note})

    total = len(reviewed)
    usable = counts["PASS"] + counts["MINOR"]
    automatic_path = Path(args.predictions).parent / "automatic_metrics.json"
    automatic = read_json(automatic_path) if automatic_path.exists() else None
    summary = {
        "schema_version": 1,
        "status": "AI_ASSISTED_REVIEW_COMPLETE_NOT_NATIVE_SPEAKER_CERTIFIED",
        "direction": "zh-uz",
        "rows": total,
        "labels": dict(counts),
        "direct_pass_rate": counts["PASS"] / total,
        "semantic_usable_rate_pass_plus_minor": usable / total,
        "fail_rate": counts["FAIL"] / total,
        "by_scenario": {
            name: dict(by_scenario[name]) for name in sorted(by_scenario)
        },
        "automatic_metrics": (
            automatic.get("automatic_metrics") if automatic else "NOT_FOUND"
        ),
        "promotion_recommendation": "DO_NOT_APPROVE_UNSUPERVISED_USE",
        "review_limitations": [
            "AI-assisted review, not Uzbek native-speaker certification.",
            "One AI-authored reference per sentence; valid paraphrases may score poorly.",
            "Medical, numerical, negation and entity failures require human safeguards.",
        ],
    }

    output = Path(args.output_dir)
    preserve(
        output / "reviewed_predictions.jsonl",
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed),
    )
    preserve(
        output / "semantic_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
