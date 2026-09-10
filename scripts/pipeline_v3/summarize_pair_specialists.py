"""Freeze a compact JSON/Markdown summary of the six-pair baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from scripts.pipeline_v2.common import PROJECT_ROOT, write_json  # noqa: E402

PAIR_IDS = ("en_zh", "en_uz", "en_ru", "zh_uz", "zh_ru", "uz_ru")
DEFAULT_OUTPUT = "reports/experiments/six_pair_baseline_v1"


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required baseline artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_summary(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    evaluation_root = root / "results/evaluation/pair_specialists"
    pairs: dict[str, Any] = {}
    sample_counts: set[int] = set()
    for pair_id in PAIR_IDS:
        pair_root = evaluation_root / pair_id
        exp1 = read_json(pair_root / "exp1.json")
        exp2 = read_json(pair_root / "exp2.json")
        gate = read_json(pair_root / "promotion_gate.json")
        if set(exp1) != set(exp2) or len(exp1) != 2:
            raise ValueError(f"{pair_id} must contain the same two directions.")
        directions = {}
        computed_pass = True
        for direction in exp1:
            before, after = exp1[direction], exp2[direction]
            samples = int(before["samples"])
            if samples != int(after["samples"]):
                raise ValueError(f"Sample-count mismatch for {direction}.")
            sample_counts.add(samples)
            passed = all(
                float(after[metric]) >= float(before[metric])
                for metric in ("bleu", "chrf2")
            )
            computed_pass = computed_pass and passed
            directions[direction] = {
                "exp1": before,
                "exp2": after,
                "delta": {
                    metric: float(after[metric]) - float(before[metric])
                    for metric in ("bleu", "chrf2")
                },
                "passed": passed,
            }
        expected_status = "PASS" if computed_pass else "FAIL"
        if gate.get("status") != expected_status:
            raise ValueError(
                f"Stored gate disagrees with metrics for {pair_id}: {gate.get('status')}"
            )
        pairs[pair_id] = {
            "gate": expected_status,
            "selected_baseline_stage": "exp2" if computed_pass else "exp1",
            "directions": directions,
        }
    if len(sample_counts) != 1:
        raise ValueError(f"All pair evaluations must share one sample count: {sample_counts}")
    return {
        "schema_version": 1,
        "experiment": "six_pair_specialists",
        "benchmark_role": "final_devtest_gate",
        "samples_per_direction": next(iter(sample_counts)),
        "pass_pairs": sum(item["gate"] == "PASS" for item in pairs.values()),
        "fail_pairs": sum(item["gate"] == "FAIL" for item in pairs.values()),
        "pairs": pairs,
        "decision": (
            "This closes the controlled SMaLL-100 baseline only; PASS does not "
            "select a final architecture. Architecture bake-off uses FLORES dev."
        ),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Six-pair SMaLL-100 baseline",
        "",
        f"Pairs: {summary['pass_pairs']} PASS, {summary['fail_pairs']} FAIL. ",
        "",
        "| Pair | Gate | Baseline stage | Direction | BLEU delta | chrF2 delta |",
        "|---|---|---|---|---:|---:|",
    ]
    for pair_id, pair in summary["pairs"].items():
        for direction, result in pair["directions"].items():
            lines.append(
                f"| {pair_id} | {pair['gate']} | {pair['selected_baseline_stage']} | "
                f"{direction} | {result['delta']['bleu']:+.4f} | "
                f"{result['delta']['chrf2']:+.4f} |"
            )
    lines.extend(["", summary["decision"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    summary = build_summary()
    write_json(output / "summary.json", summary)
    (output / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
