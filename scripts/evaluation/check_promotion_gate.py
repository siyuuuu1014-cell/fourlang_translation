from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block model freezing unless every requested direction meets its baseline."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--directions", nargs="+", required=True)
    parser.add_argument("--min-bleu-delta", type=float, default=0.0)
    parser.add_argument("--min-chrf-delta", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline).resolve()
    candidate_path = Path(args.candidate).resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    rows = []
    failures = []
    for direction in args.directions:
        if direction not in baseline or direction not in candidate:
            raise KeyError(
                f"Direction {direction!r} is missing from evaluation metrics."
            )
        scopes = {"aggregate": (baseline[direction], candidate[direction])}
        baseline_benchmarks = baseline[direction].get("benchmarks", {})
        candidate_benchmarks = candidate[direction].get("benchmarks", {})
        if (
            set(baseline_benchmarks) != set(candidate_benchmarks)
            or not baseline_benchmarks
        ):
            raise RuntimeError(
                f"Per-benchmark metrics are missing or mismatched for {direction}."
            )
        scopes.update(
            {
                name: (baseline_benchmarks[name], candidate_benchmarks[name])
                for name in baseline_benchmarks
            }
        )
        for scope, (base_metrics, candidate_metrics) in scopes.items():
            bleu_delta = float(candidate_metrics["bleu"]) - float(base_metrics["bleu"])
            chrf_delta = float(candidate_metrics["chrf2"]) - float(
                base_metrics["chrf2"]
            )
            passed = (
                bleu_delta >= args.min_bleu_delta and chrf_delta >= args.min_chrf_delta
            )
            rows.append(
                {
                    "direction": direction,
                    "scope": scope,
                    "baseline_bleu": float(base_metrics["bleu"]),
                    "candidate_bleu": float(candidate_metrics["bleu"]),
                    "bleu_delta": bleu_delta,
                    "baseline_chrf2": float(base_metrics["chrf2"]),
                    "candidate_chrf2": float(candidate_metrics["chrf2"]),
                    "chrf2_delta": chrf_delta,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(f"{direction}/{scope}")

    report = {
        "schema_version": 2,
        "status": "PASS" if not failures else "FAIL",
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "minimum_deltas": {
            "bleu": args.min_bleu_delta,
            "chrf2": args.min_chrf_delta,
        },
        "directions": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("Promotion gate failed for: " + ", ".join(failures))


if __name__ == "__main__":
    main()
