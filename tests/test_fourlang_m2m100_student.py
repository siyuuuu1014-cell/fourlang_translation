import json
from pathlib import Path

import pytest

from scripts.pipeline_v3 import fourlang_m2m100_student as runner


def test_candidate_is_locked_to_m2m100():
    settings = {
        "student": {
            "id": "m2m100_418m",
            "family": "m2m100",
            "repo_id": "facebook/m2m100_418M",
            "path": "unused",
        }
    }
    assert runner.candidate(settings)["family"] == "m2m100"
    settings["student"]["family"] = "nllb"
    with pytest.raises(RuntimeError, match="locked"):
        runner.candidate(settings)


def test_verify_m2m100_artifact_rejects_other_model_type(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "nllb"}), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "model_files", lambda path: [])
    with pytest.raises(RuntimeError, match="non-M2M100"):
        runner.verify_m2m100_artifact(tmp_path)


def test_validate_rows_requires_all_directions(tmp_path: Path):
    rows = [
        {
            "src_lang": source,
            "tgt_lang": target,
            "src_text": source,
            "tgt_text": target,
        }
        for source in ("en", "zh", "uz", "ru")
        for target in ("en", "zh", "uz", "ru")
        if source != target
    ]
    counts = runner.validate_rows(rows, tmp_path / "data.jsonl")
    assert len(counts) == 12
    with pytest.raises(RuntimeError, match="direction mismatch"):
        runner.validate_rows(rows[:-1], tmp_path / "data.jsonl")


def test_metric_summary_tracks_worst_direction():
    values = {
        "en-zh": {"bleu": 10.0, "chrf2": 20.0},
        "zh-en": {"bleu": 30.0, "chrf2": 40.0},
    }
    summary = runner.summarize_metrics(values)
    assert summary["macro_bleu"] == 20.0
    assert summary["macro_chrf2"] == 30.0
    assert summary["worst_direction"] == "en-zh"


def test_kd_v2_continues_from_kd_v1_and_reuses_kd_data(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    settings = {
        "student": {"path": "models/m2m100_418M"},
        "outputs": {"root": "results/student/fourlang_m2m100"},
    }
    experiment, source_model, data_stage = runner.stage_plan(settings, "kd_v2")
    assert experiment == "m2m100_kd_v2"
    assert source_model == (
        tmp_path
        / "results/student/fourlang_m2m100/m2m100_kd_v1/best_model/shared"
    )
    assert data_stage == "kd"


def test_human_route_uses_isolated_chained_models(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    settings = {
        "student": {"path": "models/m2m100_418M"},
        "outputs": {"root": "results/student/fourlang_m2m100"},
    }

    human = runner.stage_plan(settings, "human")
    kd = runner.stage_plan(settings, "kd_from_human")
    targeted = runner.stage_plan(settings, "targeted_from_human")

    assert human == (
        "m2m100_human_v1",
        tmp_path / "models/m2m100_418M",
        "human",
    )
    assert kd == (
        "m2m100_kd_from_human_v1",
        tmp_path
        / "results/student/fourlang_m2m100/m2m100_human_v1/best_model/shared",
        "kd",
    )
    assert targeted == (
        "m2m100_targeted_from_human_v1",
        tmp_path
        / "results/student/fourlang_m2m100/m2m100_kd_from_human_v1/best_model/shared",
        "targeted",
    )


def test_default_preflight_does_not_change_existing_route(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    observed = []
    monkeypatch.setattr(runner, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(runner, "candidate", lambda config: config["student"])
    monkeypatch.setattr(runner, "verify_m2m100_artifact", lambda path: {})

    def missing_paths(config, stage):
        observed.append(stage)
        return tmp_path / f"{stage}.train", tmp_path / f"{stage}.validation"

    monkeypatch.setattr(runner, "data_paths", missing_paths)
    settings = {
        "student": {"path": "base"},
        "outputs": {"root": "output"},
    }
    with pytest.raises(FileNotFoundError):
        runner.preflight(settings)
    assert observed == ["kd"]
