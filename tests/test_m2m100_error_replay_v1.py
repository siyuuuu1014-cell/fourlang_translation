import tomllib
from pathlib import Path

import pytest

from scripts.pipeline_v3 import build_m2m100_error_replay_v1 as builder
from scripts.pipeline_v3 import fourlang_m2m100_student as runner


CONFIG = Path("configs/multilingual/fourlang_m2m100_error_replay_v1.toml")


def load_settings():
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_approved_training_settings_are_locked_to_safe_values():
    config = load_settings()
    builder.validate_approval(config)
    training = config["training"]["m2m100_error_replay_v1"]
    assert training["epochs"] == 1
    assert training["learning_rate"] == pytest.approx(2e-6)
    assert training["save_total_limit"] == 1
    assert training["minimum_free_disk_gb"] == 16


def test_approval_must_match_training_settings():
    config = load_settings()
    config["training"]["m2m100_error_replay_v1"]["learning_rate"] = 3e-6
    with pytest.raises(RuntimeError, match="differ"):
        builder.validate_approval(config)


def test_error_replay_starts_from_kd_from_human_export(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    config = {
        "source_model": {
            "path": "results/student/fourlang_m2m100/m2m100_kd_from_human_v1/best_model/shared"
        }
    }
    experiment, source, data_stage = runner.stage_plan(config, "error_replay")
    assert experiment == "m2m100_error_replay_v1"
    assert source == tmp_path / config["source_model"]["path"]
    assert data_stage == "error_replay"


def test_validation_is_existing_exp2_holdout_not_newly_generated():
    config = load_settings()
    assert config["data"]["error_replay"]["validation"] == (
        "data/multilingual/fourlang/exp2/validation.jsonl"
    )
