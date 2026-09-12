import json
from pathlib import Path

import pytest

from scripts.pipeline_v2.training_safety import file_sha256, fingerprint
from scripts.pipeline_v3 import fourlang_flow as flow


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def config():
    return {
        "balancing": {"exp3_v2": {"default_rows_per_direction": 2000}},
        "multilingual": {"seed": 42},
        "text_contract": {"zh_script": "simplified"},
        "pair_data": [{"pair": "zh_uz"}],
    }


def make_valid_build(root: Path, settings) -> None:
    experiment = "exp3_v2"
    files = {
        "train": f"data/multilingual/fourlang/{experiment}/train.jsonl",
        "validation": f"data/multilingual/fourlang/{experiment}/validation.jsonl",
        "manifest": f"data/multilingual/fourlang/{experiment}/manifest.json",
    }
    for key, relative in files.items():
        write(root / relative, json.dumps({"file": key}) + "\n")
    preview = root / f"reports/pipeline/fourlang/{experiment}_data_preview.json"
    write(preview, "{}\n")
    report = {
        "schema_version": 1,
        "status": "EXP3_V2_DATA_BUILT_NOT_TRAINED",
        "experiment": experiment,
        "source_model": "results/student/fourlang/exp2/best_model/shared",
        "training_started": False,
        "config_fingerprint": fingerprint(
            {
                "balancing": settings["balancing"][experiment],
                "seed": settings["multilingual"]["seed"],
                "text_contract": settings["text_contract"],
                "pair_data": settings["pair_data"],
            }
        ),
        "validation": {
            "replacement_directions": 0,
            "protected_overlap_after": 0,
            "train_rows": 46000,
        },
        "dataset": {
            **files,
            "sha256": {
                key: file_sha256(root / relative) for key, relative in files.items()
            },
        },
        "preview_sha256": file_sha256(preview),
    }
    report_path = root / f"reports/pipeline/fourlang/{experiment}_data.json"
    write(report_path, json.dumps(report))


def test_exp3_v2_uses_exp2_best_model(monkeypatch, tmp_path):
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    candidate = {"family": "small100", "path": "unused"}
    assert flow.source_model_for_experiment(candidate, "exp3_v2") == str(
        tmp_path / "results/student/fourlang/exp2/best_model/shared"
    )


def test_versioned_exp3_data_verification_accepts_bound_files(monkeypatch, tmp_path):
    settings = config()
    make_valid_build(tmp_path, settings)
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    flow.verify_versioned_exp3_data(settings, "exp3_v2")


def test_versioned_exp3_data_verification_rejects_changed_train(monkeypatch, tmp_path):
    settings = config()
    make_valid_build(tmp_path, settings)
    monkeypatch.setattr(flow, "PROJECT_ROOT", tmp_path)
    write(
        tmp_path / "data/multilingual/fourlang/exp3_v2/train.jsonl",
        '{"changed": true}\n',
    )
    with pytest.raises(RuntimeError, match="Changed file"):
        flow.verify_versioned_exp3_data(settings, "exp3_v2")
