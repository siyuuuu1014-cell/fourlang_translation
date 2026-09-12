"""Build the immutable, versioned Exp3 repair dataset after an approved preview."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from filelock import FileLock  # noqa: E402

from scripts.pipeline_v2.common import PROJECT_ROOT, read_json  # noqa: E402
from scripts.pipeline_v2.training_safety import file_sha256  # noqa: E402
from scripts.pipeline_v3 import preview_exp3_data as preview  # noqa: E402

DEFAULT_CONFIG = "configs/multilingual/fourlang.toml"
DEFAULT_PREVIEW = "reports/pipeline/fourlang/exp3_data_preview.json"
DEFAULT_DATASET = "data/multilingual/fourlang/exp3"
DEFAULT_REPORT = "reports/pipeline/fourlang/exp3_data.json"
IDENTITY = ["src_lang", "tgt_lang", "src_text", "tgt_text"]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def dataset_output(version: str) -> str:
    if version not in preview.SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported Exp3 data version: {version}")
    return f"data/multilingual/fourlang/{version}"


def report_output(version: str) -> str:
    if version not in preview.SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported Exp3 data version: {version}")
    return f"reports/pipeline/fourlang/{version}_data.json"


def checked_dataset_path(value: str | Path, version: str = "exp3") -> Path:
    path = project_path(value)
    expected = (PROJECT_ROOT / dataset_output(version)).resolve()
    if path != expected:
        raise ValueError(f"Dataset output must be exactly {dataset_output(version)}")
    return path


def checked_report_path(value: str | Path, version: str = "exp3") -> Path:
    path = project_path(value)
    expected = (PROJECT_ROOT / report_output(version)).resolve()
    if path != expected:
        raise ValueError(f"Report output must be exactly {report_output(version)}")
    return path


def jsonl_text(frame) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in frame.to_dict("records")
    )


def preserve_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"Existing versioned output differs: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def preserve_json(path: Path, value) -> None:
    preserve_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def validate_frames(train, validation, preview_report, version: str = "exp3") -> dict:
    expected_status = f"{version.upper()}_DATA_PREVIEW_READY_NOT_WRITTEN"
    if preview_report.get("status") != expected_status:
        raise ValueError("Exp3 preview is not approved for a dataset build")
    if len(train) != preview_report.get("planned_rows"):
        raise ValueError("Built train size differs from the approved preview")
    if train.duplicated(IDENTITY).any() or validation.duplicated(IDENTITY).any():
        raise ValueError("Duplicate translation identities remain in Exp3")
    train_directions = Counter(train["src_lang"] + "-" + train["tgt_lang"])
    expected_directions = preview_report["sampling"]["target_rows_by_direction"]
    if dict(sorted(train_directions.items())) != dict(sorted(expected_directions.items())):
        raise ValueError("Exp3 direction counts differ from the approved preview")
    if len(validation) != preview_report.get("validation_rows"):
        raise ValueError("Built validation size differs from the approved preview")
    if preview_report["sampling"].get("sampled_with_replacement"):
        raise ValueError("Exp3 replacement sampling is forbidden")
    if preview_report["leakage_audit"].get("protected_overlap_after") != 0:
        raise ValueError("Protected benchmark/validation overlap remains")
    pseudo_validation = validation["training_source"].astype(str).str.contains(
        "teacher|pseudo|synthetic", case=False, regex=True
    )
    if pseudo_validation.any():
        raise ValueError("Teacher or synthetic rows are forbidden in validation")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_rows_by_direction": dict(sorted(train_directions.items())),
        "train_unique_identities": len(train.drop_duplicates(IDENTITY)),
        "validation_unique_identities": len(validation.drop_duplicates(IDENTITY)),
        "replacement_directions": 0,
        "protected_overlap_after": 0,
    }


def run(args) -> dict:
    version = args.version
    preview_path = project_path(args.preview or preview.preview_output(version))
    if not preview_path.is_file():
        raise FileNotFoundError(preview_path)
    approved_preview = read_json(preview_path)
    train, validation, recomputed_preview = preview.prepare_data(args.config, version)
    if approved_preview != recomputed_preview:
        raise RuntimeError(
            "Current inputs/config no longer match exp3_data_preview.json; rerun preview first"
        )
    validation_report = validate_frames(train, validation, approved_preview, version)
    output = checked_dataset_path(args.output or dataset_output(version), version)
    report_path = checked_report_path(args.report or report_output(version), version)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": version,
        "source_model": approved_preview["source_model"],
        "preview_sha256": file_sha256(preview_path),
        "source_sha256": approved_preview["source_sha256"],
        "config_fingerprint": approved_preview["config_fingerprint"],
        "planned_rows": approved_preview["planned_rows"],
        "code_sha256": {
            "scripts/pipeline_v3/build_exp3_data.py": file_sha256(Path(__file__)),
            "scripts/pipeline_v3/preview_exp3_data.py": file_sha256(
                Path(preview.__file__)
            ),
        },
    }
    with FileLock(str(output / ".build.lock"), timeout=0):
        if not (output / "manifest.json").exists() and set(
            path.name for path in output.iterdir()
        ) - {".build.lock"}:
            raise RuntimeError("Unowned Exp3 output directory; use a new version")
        preserve_json(output / "manifest.json", manifest)
        preserve_text(output / "train.jsonl", jsonl_text(train))
        preserve_text(output / "validation.jsonl", jsonl_text(validation))
        files = {
            "train": file_sha256(output / "train.jsonl"),
            "validation": file_sha256(output / "validation.jsonl"),
            "manifest": file_sha256(output / "manifest.json"),
        }
        report = {
            "schema_version": 1,
            "status": f"{version.upper()}_DATA_BUILT_NOT_TRAINED",
            "experiment": version,
            "source_model": approved_preview["source_model"],
            "training_started": False,
            "original_sources_modified": False,
            "teacher_generation_performed": False,
            "dataset": {
                "train": str((output / "train.jsonl").relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "validation": str(
                    (output / "validation.jsonl").relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "manifest": str(
                    (output / "manifest.json").relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "sha256": files,
            },
            "validation": validation_report,
            "sampling": approved_preview["sampling"],
            "leakage_audit": approved_preview["leakage_audit"],
            "source_sha256": approved_preview["source_sha256"],
            "preview_sha256": manifest["preview_sha256"],
            "config_fingerprint": approved_preview["config_fingerprint"],
            "next_step": "Review this report before starting Exp3 training from Exp2 best_model.",
        }
        preserve_json(report_path, report)
    print(
        f"{version.upper()}_DATA_BUILT_NOT_TRAINED: train={len(train)} "
        f"validation={len(validation)} replacement_directions=0",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--version", choices=preview.SUPPORTED_VERSIONS, default="exp3"
    )
    parser.add_argument("--preview")
    parser.add_argument("--output")
    parser.add_argument("--report")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
