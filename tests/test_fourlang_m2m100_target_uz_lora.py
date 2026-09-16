import tomllib
from pathlib import Path

import pytest

from scripts.pipeline_v3 import fourlang_m2m100_target_uz_lora as target_uz


CONFIG = Path("configs/multilingual/fourlang_m2m100_target_uz_lora_v1.toml")


def load_config():
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_route_is_limited_to_en_and_zh_into_uz():
    config = load_config()
    assert target_uz.validate_directions(config) == ("en-uz", "zh-uz")
    assert "ru-uz" not in config["experiment"]["repair_directions"]


def test_balancing_keeps_every_row_and_equalizes_weight_mass():
    rows = [
        {"src_lang": "en", "tgt_lang": "uz", "src_text": "a", "tgt_text": "b", "weight": 3.0},
        {"src_lang": "en", "tgt_lang": "uz", "src_text": "c", "tgt_text": "d", "weight": 1.0},
        {"src_lang": "zh", "tgt_lang": "uz", "src_text": "e", "tgt_text": "f", "weight": 1.0},
    ]
    balanced, report = target_uz.balance_direction_mass(rows, ("en-uz", "zh-uz"))
    assert len(balanced) == len(rows)
    assert report["balanced_weight_sum"]["en-uz"] == pytest.approx(
        report["balanced_weight_sum"]["zh-uz"]
    )


def test_selection_protects_en_uz_and_keeps_zh_uz_gate():
    config = load_config()
    assert config["selection"]["minimum_zh_uz_recovery_from_exp4"] == pytest.approx(0.30)
    assert config["selection"]["maximum_en_uz_regression_from_exp4"] == pytest.approx(0.10)
