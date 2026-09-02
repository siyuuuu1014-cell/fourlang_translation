from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from .common import PROJECT_ROOT, commercial_candidates, load_config, pair_info, project_path, write_json
except ImportError:
    from common import PROJECT_ROOT, commercial_candidates, load_config, pair_info, project_path, write_json


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def pair_hash(source: str, target: str) -> str:
    return hashlib.sha256(f"{source}\n{target}".encode("utf-8")).hexdigest()


def paths(config: dict) -> dict[str, Path]:
    pair, _, _, version = pair_info(config)
    pipeline = PROJECT_ROOT / "data" / "pipeline_v2" / pair
    reports = PROJECT_ROOT / "reports" / "pipeline" / pair
    return {
        "pipeline": pipeline,
        "reports": reports,
        "candidates": pipeline / "candidates.parquet",
        "rule_pass": pipeline / "rule_pass.parquet",
        "human_calibration": pipeline / "judge_calibration.parquet",
        "teacher_calibration": pipeline / "teacher_judge_calibration.parquet",
        "human_judged": pipeline / "human_judged.parquet",
        "teacher_judged": pipeline / "teacher_judged.parquet",
        "approved": PROJECT_ROOT / "data" / "approved" / pair / version / "approved_pairs.parquet",
        "splits": PROJECT_ROOT / "data" / "splits" / pair / version,
        "distillation": PROJECT_ROOT / "data" / "distillation" / pair / version,
    }


def validate(config: dict) -> None:
    pair, source, target, _ = pair_info(config)
    errors: list[str] = []
    if source == target or pair != f"{source}_{target}":
        errors.append("direction.pair must equal <source_lang>_<target_lang>.")
    if not config["direction"].get("commercial_use"):
        errors.append("This production pipeline requires commercial_use=true.")
    for role in ("student", "teacher"):
        candidates = commercial_candidates(config, role)
        ids = [item["id"] for item in candidates]
        if len(ids) != len(set(ids)):
            errors.append(f"Duplicate {role} candidate ids.")
        for item in candidates:
            if not item.get("license"):
                errors.append(f"{role} candidate {item['id']} has no declared license.")
    benchmark_paths = [project_path(config["benchmarks"][name]) for name in ("flores", "tatoeba")]
    data_paths = {
        project_path(config["data"]["source_file"]).resolve(),
        project_path(config["data"]["target_file"]).resolve(),
    }
    if any(path.resolve() in data_paths for path in benchmark_paths):
        errors.append("Protected benchmarks cannot also be raw training inputs.")
    report = {
        "schema_version": 2,
        "pair": pair,
        "commercial_use": True,
        "status": "PASS" if not errors else "FAIL",
        "student_candidates": [item["id"] for item in commercial_candidates(config, "student")],
        "teacher_candidates": [item["id"] for item in commercial_candidates(config, "teacher")],
        "protected_benchmarks": [str(path.relative_to(PROJECT_ROOT)) for path in benchmark_paths],
        "errors": errors,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths(config)["reports"] / "config_validation.json", report)
    if errors:
        raise SystemExit("Configuration validation failed: " + "; ".join(errors))


def candidates(config: dict) -> None:
    pair, source, target, _ = pair_info(config)
    settings = config["data"]
    source_path = project_path(settings["source_file"])
    target_path = project_path(settings["target_file"])
    rows = []
    with source_path.open("r", encoding="utf-8", errors="replace") as left:
        with target_path.open("r", encoding="utf-8", errors="replace") as right:
            for row_number, (source_text, target_text) in enumerate(zip(left, right, strict=True), 1):
                source_text, target_text = normalize(source_text), normalize(target_text)
                rows.append(
                    {
                        "pair_id": pair_hash(source_text, target_text),
                        source: source_text,
                        target: target_text,
                        "source_text": source_text,
                        "target_text": target_text,
                        "source_lang": source,
                        "target_lang": target,
                        "source_corpus": settings["corpus"],
                        "source_row": row_number,
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates(["source_text", "target_text"])
    source_counts = frame.groupby("source_text")["target_text"].nunique()
    target_counts = frame.groupby("target_text")["source_text"].nunique()
    frame = frame[
        ~frame["source_text"].isin(source_counts[source_counts > 1].index)
        & ~frame["target_text"].isin(target_counts[target_counts > 1].index)
    ].reset_index(drop=True)
    output = paths(config)["candidates"]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    write_json(paths(config)["reports"] / "candidates.json", {"pair": pair, "status": "CANDIDATE_ONLY", "rows": len(frame)})


def rule_reasons(source: str, target: str, settings: dict) -> list[str]:
    reasons = []
    if not source or not target:
        reasons.append("empty")
    if source == target:
        reasons.append("identical")
    if min(len(source), len(target)) < 2:
        reasons.append("too_short")
    if max(len(source), len(target)) > int(settings["max_characters"]):
        reasons.append("too_long")
    ratio = max(len(source), len(target)) / max(1, min(len(source), len(target)))
    if ratio > float(settings["max_length_ratio"]):
        reasons.append("length_ratio")
    if min(sum(char.isalpha() for char in source), sum(char.isalpha() for char in target)) < 2:
        reasons.append("insufficient_letters")
    if re.search(r"<[^>]+>|https?://\S+|\{\{.*?\}\}", source + " " + target):
        reasons.append("markup_or_url")
    return reasons


def rules(config: dict) -> None:
    frame = pd.read_parquet(paths(config)["candidates"])
    reasons = [rule_reasons(row.source_text, row.target_text, config["data"]) for row in frame.itertuples()]
    frame["rule_reasons"] = [json.dumps(item) for item in reasons]
    frame["rule_pass"] = [not item for item in reasons]
    eligible = frame[frame["rule_pass"]].copy()
    limit = int(config["data"]["candidate_limit"])
    if limit > 0 and len(eligible) > limit:
        passed = eligible.sort_values("pair_id").head(limit).copy()
    else:
        passed = eligible
    passed.to_parquet(paths(config)["rule_pass"], index=False)
    counts = Counter(reason for row in reasons for reason in row)
    write_json(paths(config)["reports"] / "rules.json", {
        "input_rows": len(frame),
        "rule_eligible_rows": len(eligible),
        "judge_candidate_rows": len(passed),
        "rule_reject_rows": len(frame) - len(eligible),
        "candidate_limit": limit,
        "reasons": counts,
    })


def calibrate(config: dict, mode: str) -> None:
    input_path = paths(config)[f"{mode}_calibration"]
    frame = pd.read_parquet(input_path)
    if "judge_score" not in frame:
        raise ValueError(f"{input_path} has no judge_score column.")
    judge = config["judge"]
    floor_key = "teacher_accept_score_floor" if mode == "teacher" else "accept_score_floor"
    quantile_key = "teacher_accept_quantile" if mode == "teacher" else "accept_quantile"
    valid = frame[frame["judge_parse_ok"]]["judge_score"].astype(float)
    if valid.empty:
        raise RuntimeError("Judge calibration produced no parseable scores.")
    threshold = max(float(judge[floor_key]), float(valid.quantile(float(judge[quantile_key]))))
    payload = {
        "schema_version": 1,
        "mode": mode,
        "model_path": judge["model_path"],
        "sample_rows": len(frame),
        "parseable_rows": len(valid),
        "floor": float(judge[floor_key]),
        "quantile": float(judge[quantile_key]),
        "accept_threshold": threshold,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths(config)["reports"] / f"{mode}_judge_policy.json", payload)


def approve(config: dict) -> None:
    frame = pd.read_parquet(paths(config)["human_judged"])
    policy = json.loads((paths(config)["reports"] / "human_judge_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["accept_threshold"])
    approved = frame[(frame["judge_parse_ok"]) & (frame["judge_score"] >= threshold)].copy()
    limit = int(config["data"]["candidate_limit"])
    if limit > 0:
        approved = approved.sort_values(["judge_score", "pair_id"], ascending=[False, True]).head(limit)
    output = paths(config)["approved"]
    output.parent.mkdir(parents=True, exist_ok=True)
    approved.to_parquet(output, index=False)
    write_json(paths(config)["reports"] / "approved.json", {"input_rows": len(frame), "approved_rows": len(approved), "threshold": threshold, "source": "qwen_judge"})


def write_directional(frame: pd.DataFrame, path: Path, source: str, target: str, *, weight: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in frame.itertuples(index=False):
            common = {"pair_id": row.pair_id, "source_corpus": getattr(row, "source_corpus", "unknown"), "weight": weight}
            records = [
                {**common, "src_lang": source, "tgt_lang": target, "src_text": row.source_text, "tgt_text": row.target_text},
                {**common, "src_lang": target, "tgt_lang": source, "src_text": row.target_text, "tgt_text": row.source_text},
            ]
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def split(config: dict) -> None:
    _, source, target, _ = pair_info(config)
    frame = pd.read_parquet(paths(config)["approved"]).sample(frac=1, random_state=int(config["direction"]["seed"]))
    validation_size = int(config["data"]["validation_pairs"])
    test_size = int(config["data"]["test_pairs"])
    if len(frame) <= validation_size + test_size:
        raise RuntimeError("Approved pairs are insufficient for requested validation and test sizes.")
    validation = frame.iloc[:validation_size]
    test = frame.iloc[validation_size:validation_size + test_size]
    train = frame.iloc[validation_size + test_size:]
    output = paths(config)["splits"]
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        part.to_parquet(output / f"{name}_pairs.parquet", index=False)
        write_directional(part, output / f"{name}.jsonl", source, target)
    ids = [set(part["pair_id"]) for part in (train, validation, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("Pair leakage detected between splits.")
    write_json(paths(config)["reports"] / "split.json", {"train_pairs": len(train), "validation_pairs": len(validation), "test_pairs": len(test), "pair_disjoint": True})


def protected_hashes(config: dict) -> set[str]:
    _, source, target, _ = pair_info(config)
    hashes: set[str] = set()
    for name in ("flores", "tatoeba"):
        frame = pd.read_parquet(project_path(config["benchmarks"][name]))
        left = source if source in frame.columns else "source_text"
        right = target if target in frame.columns else "target_text"
        for row in frame[[left, right]].itertuples(index=False, name=None):
            hashes.add(pair_hash(normalize(row[0]), normalize(row[1])))
    return hashes


def kd_candidates(config: dict) -> None:
    limit = int(config["distillation"]["candidate_pairs_per_direction"])
    blocked = protected_hashes(config)
    train_path = paths(config)["splits"] / "train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    selected = []
    direction_counts: Counter[str] = Counter()
    for row in rows:
        if row["pair_id"] in blocked:
            continue
        key = f"{row['src_lang']}-{row['tgt_lang']}"
        if direction_counts[key] >= limit:
            continue
        selected.append({"pair_id": row["pair_id"], "src_lang": row["src_lang"], "tgt_lang": row["tgt_lang"], "src_text": row["src_text"]})
        direction_counts[key] += 1
    output = paths(config)["pipeline"] / "kd_candidates.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    write_json(paths(config)["reports"] / "kd_candidates.json", {"rows": len(selected), "directions": direction_counts, "protected_pair_overlap": 0})


def kd_dataset(config: dict) -> None:
    frame = pd.read_parquet(paths(config)["teacher_judged"])
    policy = json.loads((paths(config)["reports"] / "teacher_judge_policy.json").read_text(encoding="utf-8"))
    threshold = float(policy["accept_threshold"])
    accepted = frame[(frame["judge_parse_ok"]) & (frame["judge_score"] >= threshold)].copy()
    output = paths(config)["distillation"]
    output.mkdir(parents=True, exist_ok=True)
    human_rows = [json.loads(line) for line in (paths(config)["splits"] / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in human_rows:
        row["weight"] = float(config["distillation"]["human_weight"])
        row["training_source"] = "human_replay"
    teacher_rows = []
    for row in accepted.itertuples(index=False):
        teacher_rows.append({
            "pair_id": row.pair_id, "src_lang": row.src_lang, "tgt_lang": row.tgt_lang,
            "src_text": row.src_text, "tgt_text": row.teacher_text,
            "weight": float(config["distillation"]["teacher_high_weight"]), "training_source": "teacher_kd",
        })
    all_rows = human_rows + teacher_rows
    (output / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    (output / "validation.jsonl").write_text((paths(config)["splits"] / "validation.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    write_json(paths(config)["reports"] / "kd_dataset.json", {"human_rows": len(human_rows), "teacher_rows": len(teacher_rows), "threshold": threshold})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "candidates", "rules", "calibrate", "approve", "split", "kd_candidates", "kd_dataset"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("human", "teacher"))
    args = parser.parse_args()
    config = load_config(args.config)
    actions = {"validate": validate, "candidates": candidates, "rules": rules, "approve": approve, "split": split, "kd_candidates": kd_candidates, "kd_dataset": kd_dataset}
    if args.action == "calibrate":
        if not args.mode:
            parser.error("calibrate requires --mode")
        calibrate(config, args.mode)
    else:
        actions[args.action](config)


if __name__ == "__main__":
    main()
