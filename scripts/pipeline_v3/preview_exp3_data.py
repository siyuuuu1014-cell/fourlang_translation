"""Preview the versioned Exp3 repair mix without writing data or training a model."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.pipeline_v2.common import PROJECT_ROOT, load_config, write_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256, fingerprint  # noqa: E402
from scripts.pipeline_v3.data_safety import protect_splits  # noqa: E402
from scripts.pipeline_v3.fourlang_flow import (  # noqa: E402
    LANGUAGES,
    _path,
    _read_table,
    balance_training_rows,
    directions,
    normalize_rows,
)

DEFAULT_CONFIG = "configs/multilingual/fourlang.toml"
DEFAULT_OUTPUT = "reports/pipeline/fourlang/exp3_data_preview.json"


def exp3_train_path(item: dict[str, Any]) -> Path:
    return _path(item.get("exp3_kd_train", item["kd_train"]))


def checked_output(value: str | Path) -> Path:
    path = _path(value).resolve()
    allowed = (PROJECT_ROOT / DEFAULT_OUTPUT).resolve()
    if path != allowed:
        raise ValueError(f"Preview output must be {DEFAULT_OUTPUT}.")
    return path


def prepare_data(
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the deterministic in-memory Exp3 mix and its preview report."""
    config = load_config(config_path)
    balancing = config.get("balancing", {}).get("exp3")
    if not balancing:
        raise ValueError("Missing balancing.exp3 configuration.")
    if int(balancing.get("max_teacher_repeats", 0)) != 1:
        raise ValueError("Exp3 preview requires max_teacher_repeats = 1.")

    train_parts: list[pd.DataFrame] = []
    validation_parts: list[pd.DataFrame] = []
    source_files: dict[str, str] = {}
    normalization = Counter()
    for item in config["pair_data"]:
        train_path = exp3_train_path(item)
        validation_path = _path(item["validation"])
        for path in (train_path, validation_path):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_files[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = (
                file_sha256(path)
            )
        train = normalize_rows(
            _read_table(train_path), origin=str(train_path.relative_to(PROJECT_ROOT))
        )
        validation = normalize_rows(
            _read_table(validation_path),
            origin=str(validation_path.relative_to(PROJECT_ROOT)),
        )
        expected = set(item["pair"].split("_"))
        for label, frame in (("train", train), ("validation", validation)):
            if any(
                {row.src_lang, row.tgt_lang} != expected
                for row in frame[["src_lang", "tgt_lang"]].itertuples(index=False)
            ):
                raise RuntimeError(f"{item['pair']} {label} contains another pair.")
        normalization.update(train.attrs["script_normalization"])
        normalization.update(validation.attrs["script_normalization"])
        train_parts.append(train)
        validation_parts.append(validation)

    train = pd.concat(train_parts, ignore_index=True).drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    )
    validation = pd.concat(validation_parts, ignore_index=True).drop_duplicates(
        ["src_lang", "tgt_lang", "src_text", "tgt_text"]
    )
    benchmark_paths = [
        _path(config["benchmarks"][name]) for name in ("flores_dev", "flores_devtest")
    ]
    previous_path = PROJECT_ROOT / "data/multilingual/fourlang/exp2/train.jsonl"
    for path in [*benchmark_paths, previous_path]:
        if not path.is_file():
            raise FileNotFoundError(path)
        source_files[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = (
            file_sha256(path)
        )
    previous = normalize_rows(_read_table(previous_path), origin="exp2_training")
    train, validation, leakage = protect_splits(
        train,
        validation,
        [pd.read_parquet(path) for path in benchmark_paths],
        LANGUAGES,
        previous,
    )
    if set(validation["src_lang"] + "-" + validation["tgt_lang"]) != set(
        directions()
    ):
        raise RuntimeError("Protected validation must retain all 12 directions.")

    balanced_train, sampling = balance_training_rows(
        train,
        seed=int(config["multilingual"]["seed"]),
        configured_rows=int(balancing.get("default_rows_per_direction", 0)),
        rows_by_direction={
            str(key): int(value)
            for key, value in balancing.get("rows_by_direction", {}).items()
        },
        teacher_ratio=float(balancing["teacher_ratio"]),
        max_teacher_repeats=int(balancing["max_teacher_repeats"]),
    )
    replacements = sampling.get("sampled_with_replacement", {})
    report = {
        "schema_version": 1,
        "status": (
            "EXP3_DATA_PREVIEW_READY_NOT_WRITTEN"
            if not replacements and leakage.get("protected_overlap_after") == 0
            else "EXP3_DATA_PREVIEW_BLOCKED"
        ),
        "experiment": "exp3",
        "source_model": "results/student/fourlang/exp2/best_model/shared",
        "training_started": False,
        "training_data_written": False,
        "original_sources_modified": False,
        "replacement_forbidden": True,
        "planned_rows": sampling["output_rows"],
        "validation_rows": len(validation),
        "sampling": sampling,
        "leakage_audit": leakage,
        "script_normalization": dict(sorted(normalization.items())),
        "source_sha256": dict(sorted(source_files.items())),
        "config_fingerprint": fingerprint(
            {
                "balancing": balancing,
                "seed": config["multilingual"]["seed"],
                "text_contract": config.get("text_contract", {}),
                "pair_data": config["pair_data"],
            }
        ),
        "next_step": (
            "Review and explicitly approve a separate versioned Exp3 dataset build."
            if not replacements
            else "Add unique source rows or reduce quotas before building Exp3 data."
        ),
    }
    return balanced_train, validation, report


def run(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    _, _, report = prepare_data(config_path)
    output = checked_output(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    print(
        f"{report['status']}: planned_rows={report['planned_rows']} "
        "replacement_directions="
        f"{len(report['sampling']['sampled_with_replacement'])}; "
        "no dataset or model written.",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.config, args.output)


if __name__ == "__main__":
    main()
