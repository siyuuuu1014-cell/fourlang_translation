import tomllib
from pathlib import Path

import pytest

from scripts.pipeline_v3 import fourlang_m2m100_ru_uz_lora as route


CONFIG = Path("configs/multilingual/fourlang_m2m100_ru_uz_lora_v1.toml")
ZH_UZ_CONFIG = Path("configs/multilingual/fourlang_m2m100_zh_uz_lora_v1.toml")


def load_config():
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_route_is_locked_to_ru_uz_and_exp4():
    config = load_config()
    assert config["experiment"]["repair_direction"] == "ru-uz"
    assert config["base_model"]["path"].endswith(
        "m2m100_error_replay_v1/best_model/shared"
    )


def test_adapter_settings_are_small_and_directional():
    config = load_config()
    assert config["lora"]["r"] == 16
    assert config["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert config["training"]["epochs"] == 1
    assert config["training"]["learning_rate"] == pytest.approx(5e-5)


def test_repair_rows_keeps_only_ru_uz(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"src_lang": "ru", "tgt_lang": "uz", "src_text": "a", "tgt_text": "b"},
        {"src_lang": "en", "tgt_lang": "uz", "src_text": "c", "tgt_text": "d"},
    ]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")
    assert route.repair_rows(path) == [rows[0]]


def test_hard_gate_recovery_is_not_weakened():
    config = load_config()
    assert config["selection"]["minimum_ru_uz_recovery_from_exp4"] == pytest.approx(0.30)


def test_direction_configuration_supports_isolated_zh_uz_route():
    with ZH_UZ_CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    assert route.configured_direction(config) == ("zh-uz", "zh", "uz")
    assert config["outputs"]["adapter"].endswith("m2m100_zh_uz_lora_v1/adapter")
    assert config["selection"]["minimum_repair_chrf2_from_exp4"] == pytest.approx(0.30)


def test_repair_rows_can_select_zh_uz_without_changing_ru_uz_default(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [
        {"src_lang": "ru", "tgt_lang": "uz", "src_text": "a", "tgt_text": "b"},
        {"src_lang": "zh", "tgt_lang": "uz", "src_text": "c", "tgt_text": "d"},
    ]
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")
    assert route.repair_rows(path) == [rows[0]]
    assert route.repair_rows(path, "zh-uz") == [rows[1]]
