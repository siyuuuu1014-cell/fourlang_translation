from __future__ import annotations

import argparse
from datetime import datetime, timezone

try:
    from .common import (
        PROJECT_ROOT,
        candidate_by_id,
        commercial_candidates,
        load_config,
        pair_info,
        read_json,
        write_json,
    )
except ImportError:
    from common import (
        PROJECT_ROOT,
        candidate_by_id,
        commercial_candidates,
        load_config,
        pair_info,
        read_json,
        write_json,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select each translation direction independently from complete measurements."
    )
    parser.add_argument("--role", choices=("student", "teacher"), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pair, source, target, _ = pair_info(config)
    root = PROJECT_ROOT / "results" / "model_selection" / pair
    scores = read_json(root / f"{args.role}_scores.json")
    candidates = commercial_candidates(config, args.role)
    eligible = {item["id"] for item in candidates}
    directions = [f"{source}-{target}", f"{target}-{source}"]
    failures = {
        candidate_id: result
        for candidate_id, result in scores["candidates"].items()
        if candidate_id in eligible and result.get("status") == "error"
    }
    if failures:
        details = "; ".join(
            f"{key}: {value.get('error_type')}" for key, value in failures.items()
        )
        raise RuntimeError(
            f"Model bake-off is incomplete; refusing partial selection: {details}"
        )
    primary, secondary = (
        config["selection"]["primary_metric"],
        config["selection"]["secondary_metric"],
    )
    selected = {}
    rankings = {}
    for direction in directions:
        ranked = []
        for candidate_id, result in scores["candidates"].items():
            if (
                candidate_id not in eligible
                or result.get("status") != "ok"
                or direction not in result.get("metrics", {})
            ):
                continue
            metrics = result["metrics"][direction]
            ranked.append(
                (float(metrics[primary]), float(metrics[secondary]), candidate_id)
            )
        if not ranked:
            raise RuntimeError(
                f"No eligible measured {args.role} candidate for {direction}."
            )
        ranked.sort(reverse=True)
        winner = ranked[0][2]
        selected[direction] = {
            "candidate_id": winner,
            "candidate": candidate_by_id(config, args.role, winner),
        }
        rankings[direction] = [
            {"candidate_id": item[2], primary: item[0], secondary: item[1]}
            for item in ranked
        ]
    ids = {item["candidate_id"] for item in selected.values()}
    shared = (
        args.role == "student"
        and len(ids) == 1
        and next(iter(selected.values()))["candidate"]["family"] != "marian_pair"
        and bool(
            config["selection"]["share_multilingual_student_when_same_candidate_wins"]
        )
    )
    payload = {
        "schema_version": 2,
        "role": args.role,
        "directions": selected,
        "selection_policy": "maximize metrics independently for each direction on selection-only benchmarks",
        "ranking": rankings,
        "training_layout": "shared_bidirectional" if shared else "separate_directional",
        "license_metadata_verified_during_prepare": True,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / f"selected_{args.role}.json", payload)


if __name__ == "__main__":
    main()
