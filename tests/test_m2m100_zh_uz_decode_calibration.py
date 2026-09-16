import tomllib
from pathlib import Path

from scripts.pipeline_v3 import calibrate_m2m100_zh_uz_decode as calibration


CONFIG = Path(
    "configs/multilingual/fourlang_m2m100_zh_uz_decode_calibration_v1.toml"
)


def load_config():
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_calibration_is_validation_only_and_direction_locked():
    config = load_config()
    assert config["experiment"]["direction"] == "zh-uz"
    assert config["data"]["validation"].endswith("exp3_v2/validation.jsonl")
    assert "benchmark" not in config["data"]


def test_grid_has_unique_stable_candidates_and_default_control():
    config = load_config()
    ids = calibration.candidate_ids(config)
    assert len(ids) == 7
    assert len(ids) == len(set(ids))
    control = next(
        item for item in config["decoding"]["candidates"] if item["id"] == "beam5_lp100"
    )
    assert control["num_beams"] == 5
    assert control["length_penalty"] == 1.0


def test_gate_is_not_lowered_after_first_lora_result():
    config = load_config()
    assert config["selection"]["minimum_repair_chrf2_from_exp4"] == 0.30
