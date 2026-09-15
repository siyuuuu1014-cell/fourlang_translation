import json
import tomllib
from pathlib import Path

import pytest

from scripts.pipeline_v3 import fourlang_m2m100_student as runner


def test_human_route_uses_m2m100_specific_training_parameters():
    config_path = Path("configs/multilingual/fourlang_m2m100_v1.toml")
    with config_path.open("rb") as stream:
        settings = tomllib.load(stream)["training"]

    assert settings["m2m100_human_v1"]["epochs"] == 3
    kd = settings["m2m100_kd_from_human_v1"]
    assert kd["epochs"] == 3
    assert kd["learning_rate"] == pytest.approx(1e-5)
    assert kd["early_stopping_patience"] == 2
    assert settings["m2m100_targeted_from_human_v1"]["learning_rate"] == pytest.approx(
        1e-6
    )
    targeted_v2 = settings["m2m100_targeted_from_human_v2"]
    assert targeted_v2["epochs"] == 1
    assert targeted_v2["learning_rate"] == pytest.approx(3e-6)


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

    targeted_v2 = runner.stage_plan(settings, "targeted_from_human_v2")
    assert targeted_v2 == (
        "m2m100_targeted_from_human_v2",
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


def test_targeted_v2_comparison_applies_acceptance_gate_and_optional_nllb(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    evaluation_root = tmp_path / "evaluation"
    all_directions = runner.directions()

    def payload(experiment, bleu, chrf2):
        values = {
            direction: {"bleu": bleu, "chrf2": chrf2, "samples": 1012}
            for direction in all_directions
        }
        return {
            "experiment": experiment,
            "metrics": values,
            "summary": runner.summarize_metrics(values),
        }

    experiments = {
        runner.KD_FROM_HUMAN_EXPERIMENT: payload("kd", 20.0, 40.0),
        runner.TARGETED_FROM_HUMAN_EXPERIMENT: payload("v1", 20.1, 40.1),
        runner.TARGETED_FROM_HUMAN_V2_EXPERIMENT: payload("v2", 20.5, 40.5),
    }
    for name, value in experiments.items():
        path = evaluation_root / name / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    nllb_path = tmp_path / "nllb.json"
    nllb_path.write_text(
        json.dumps(payload("nllb", 22.0, 42.0)["metrics"]), encoding="utf-8"
    )
    config = {
        "outputs": {"evaluation_root": "evaluation"},
        "baselines": {"nllb_exp3_v2_metrics": "nllb.json"},
    }
    report = runner.compare_targeted_from_human_v2(config)
    assert report["acceptance"]["passed"] is True
    assert report["nllb_exp3_v2"]["exists"] is True
    assert report["candidate_minus_nllb_exp3_v2"]["summary"]["macro_chrf2"] == pytest.approx(-1.5)
