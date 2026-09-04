from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_v2.common import (  # noqa: E402
    load_config,
    pair_info,
    pipeline_namespace,
    project_path,
    write_json,
)
from scripts.pipeline_v2.data_flow import normalized_key, pair_hash  # noqa: E402
from scripts.pipeline_v3.language_normalization import (  # noqa: E402
    normalize_language_text,
)


REQUIRED_COLUMNS = {
    "pair_id",
    "source_text",
    "target_text",
    "source_lang",
    "target_lang",
}

# Deliberately conservative: these forms are strong signals that a Chinese sentence
# is Cantonese rather than standard written Mandarin. Qwen still reviews every row
# that survives this deterministic gate.
CANTONESE_PATTERN = re.compile(
    r"佢哋|我哋|你哋|点解|邊個|边个|幾多|几多|呢個|呢个|呢段|"
    r"[佢嘅喺咗哋冇唔嗰咁咩啲嚟攞睇嘢乜俾噉噃]"
)
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u052f]")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def language_contract_reason(zh_text: str, uz_text: str) -> str | None:
    if "\ufffd" in zh_text or "\ufffd" in uz_text:
        return "REPLACEMENT_CHARACTER"
    if not CJK_PATTERN.search(zh_text):
        return "ZH_SCRIPT_RISK"
    if CANTONESE_PATTERN.search(zh_text):
        return "NON_MANDARIN_CHINESE"
    if not re.search(r"[A-Za-z]", uz_text) or CYRILLIC_PATTERN.search(uz_text):
        return "UZ_NOT_LATIN"
    return None


def existing_training_sets(paths: list[Path]) -> tuple[dict[str, set[str]], list[str]]:
    sets = {"pairs": set(), "zh": set(), "uz": set()}
    loaded: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        loaded.append(str(path))
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            if not {"source_text", "target_text"}.issubset(frame.columns):
                raise ValueError(f"Existing parquet lacks canonical text columns: {path}")
            records = frame[["source_text", "target_text"]].itertuples(
                index=False, name=None
            )
        elif path.suffix == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            canonical: dict[str, tuple[str, str]] = {}
            for row in rows:
                if row.get("src_lang") == "zh" and row.get("tgt_lang") == "uz":
                    canonical[str(row["pair_id"])] = (row["src_text"], row["tgt_text"])
                elif row.get("src_lang") == "uz" and row.get("tgt_lang") == "zh":
                    canonical[str(row["pair_id"])] = (row["tgt_text"], row["src_text"])
            records = canonical.values()
        else:
            raise ValueError(f"Unsupported existing training file: {path}")
        for zh_text, uz_text in records:
            zh = normalize_language_text("zh", str(zh_text))
            uz = normalize_language_text("uz", str(uz_text))
            sets["pairs"].add(pair_hash(zh, uz))
            sets["zh"].add(normalized_key(zh))
            sets["uz"].add(normalized_key(uz))
    return sets, loaded


def import_candidates(
    config: dict[str, Any], *, allow_missing_existing_training: bool = False
) -> dict[str, Any]:
    pair, source, target, version = pair_info(config)
    if (pair, source, target, version) != ("zh_uz", "zh", "uz", "v2"):
        raise ValueError("This importer is restricted to the isolated zh_uz v2 flow.")
    settings = config["supplemental"]
    input_path = project_path(settings["input_candidates"])
    existing_paths = [project_path(value) for value in settings["existing_training"]]
    missing = [str(path) for path in existing_paths if not path.is_file()]
    if missing and not allow_missing_existing_training:
        raise FileNotFoundError(
            "Existing-training deduplication inputs are missing: " + ", ".join(missing)
        )
    existing, loaded_existing = existing_training_sets(existing_paths)
    frame = pd.read_parquet(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Supplemental candidates lack columns: {missing_columns}")

    rejection_counts: Counter[str] = Counter()
    rejected_samples: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    sample_size = int(settings.get("rejected_sample_size", 200))
    for row in frame.to_dict("records"):
        reason: str | None = None
        try:
            zh_text = normalize_language_text("zh", str(row["source_text"]))
            uz_text = normalize_language_text("uz", str(row["target_text"]))
        except ValueError:
            zh_text = str(row["source_text"]).strip()
            uz_text = str(row["target_text"]).strip()
            reason = "NORMALIZATION_ERROR"
        if reason is None:
            reason = language_contract_reason(zh_text, uz_text)
        current_id = pair_hash(zh_text, uz_text)
        if reason is None and (
            current_id in existing["pairs"]
            or normalized_key(zh_text) in existing["zh"]
            or normalized_key(uz_text) in existing["uz"]
        ):
            reason = "EXISTING_TRAINING_OVERLAP"
        if reason is not None:
            rejection_counts[reason] += 1
            if len(rejected_samples) < sample_size:
                rejected_samples.append(
                    {
                        "pair_id": str(row.get("pair_id", current_id)),
                        "reason": reason,
                        "source_text": zh_text,
                        "target_text": uz_text,
                    }
                )
            continue
        accepted.append(
            {
                **row,
                "pair_id": current_id,
                "pair": pair,
                "source_lang": source,
                "target_lang": target,
                "source_text": zh_text,
                "target_text": uz_text,
                "candidate_status": "CLEANED_UNREVIEWED_V2",
            }
        )

    output = pd.DataFrame(accepted)
    before_dedup = len(output)
    if len(output):
        output = output.drop_duplicates("pair_id", keep="first").sort_values("pair_id")
    rejection_counts["V2_EXACT_DUPLICATE"] += before_dedup - len(output)
    namespace = pipeline_namespace(config)
    pipeline_root = PROJECT_ROOT / "data" / "pipeline_v2" / namespace
    report_root = PROJECT_ROOT / "reports" / "pipeline" / namespace
    pipeline_root.mkdir(parents=True, exist_ok=True)
    destination = pipeline_root / "candidates.parquet"
    temporary = destination.with_suffix(".parquet.tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(destination)
    rejected_path = pipeline_root / "import_rejected_sample.jsonl"
    rejected_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected_samples),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "pair": pair,
        "version": version,
        "artifact_namespace": namespace,
        "status": "CLEANED_UNREVIEWED_V2",
        "input_path": str(input_path),
        "input_rows": len(frame),
        "candidate_rows": len(output),
        "rejections": dict(rejection_counts),
        "existing_training_configured": [str(path) for path in existing_paths],
        "existing_training_loaded": loaded_existing,
        "missing_existing_training": missing,
        "existing_training_overlap_checked": not missing,
        "force_full_qwen_review": True,
        "eligible_for_training": False,
        "outputs": {
            "candidates": str(destination),
            "rejected_sample": str(rejected_path),
        },
    }
    write_json(report_root / "supplemental_import.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import isolated supplemental ZH-UZ candidates into the v2 flow."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-missing-existing-training", action="store_true")
    args = parser.parse_args()
    report = import_candidates(
        load_config(args.config),
        allow_missing_existing_training=args.allow_missing_existing_training,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
