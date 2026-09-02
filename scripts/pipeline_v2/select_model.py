from __future__ import annotations

import argparse
from datetime import datetime, timezone

try:
    from .common import PROJECT_ROOT, candidate_by_id, commercial_candidates, load_config, pair_info, read_json, write_json
except ImportError:
    from common import PROJECT_ROOT, candidate_by_id, commercial_candidates, load_config, pair_info, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a model only from measured, commercially eligible candidates.")
    parser.add_argument("--role", choices=("student", "teacher"), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pair, source, target, _ = pair_info(config)
    root = PROJECT_ROOT / "results" / "model_selection" / pair
    scores = read_json(root / f"{args.role}_scores.json")
    eligible = {item["id"] for item in commercial_candidates(config, args.role)}
    directions = [f"{source}-{target}", f"{target}-{source}"]
    primary = config["selection"]["primary_metric"]
    secondary = config["selection"]["secondary_metric"]
    ranked = []
    for candidate_id, result in scores["candidates"].items():
        if candidate_id not in eligible or result.get("status") != "ok":
            continue
        if any(direction not in result["metrics"] for direction in directions):
            continue
        primary_values = [float(result["metrics"][direction][primary]) for direction in directions]
        secondary_values = [float(result["metrics"][direction][secondary]) for direction in directions]
        ranked.append((min(primary_values), sum(primary_values) / 2, min(secondary_values), candidate_id))
    if not ranked:
        raise RuntimeError(f"No measured, commercially eligible {args.role} candidate supports both directions.")
    ranked.sort(reverse=True)
    winner_id = ranked[0][-1]
    candidate = candidate_by_id(config, args.role, winner_id)
    payload = {
        "schema_version": 1,
        "role": args.role,
        "candidate_id": winner_id,
        "candidate": candidate,
        "selection_policy": "maximize worst-direction primary metric, then mean primary, then worst secondary",
        "ranking": [
            {"candidate_id": item[3], "worst_primary": item[0], "mean_primary": item[1], "worst_secondary": item[2]}
            for item in ranked
        ],
        "commercial_license_checked": True,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / f"selected_{args.role}.json", payload)


if __name__ == "__main__":
    main()
