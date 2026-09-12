from pathlib import Path

import pytest

from scripts.pipeline_v3 import preview_exp3_data as preview


def test_exp3_uses_versioned_override(monkeypatch, tmp_path):
    monkeypatch.setattr(preview, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preview, "_path", lambda value: tmp_path / value)
    item = {
        "kd_train": "data/distillation/zh_uz/v3/train.jsonl",
        "exp3_kd_train": "data/distillation/zh_uz/v4/train.jsonl",
    }
    assert preview.exp3_train_path(item) == (
        tmp_path / "data/distillation/zh_uz/v4/train.jsonl"
    )


def test_preview_output_is_version_locked(monkeypatch, tmp_path):
    monkeypatch.setattr(preview, "PROJECT_ROOT", tmp_path)
    expected = tmp_path / preview.DEFAULT_OUTPUT
    assert preview.checked_output(expected) == expected.resolve()
    with pytest.raises(ValueError, match="exp3_data_preview.json"):
        preview.checked_output(Path("reports/pipeline/fourlang/other.json"))


def test_exp3_v2_preview_has_separate_version_locked_output(monkeypatch, tmp_path):
    monkeypatch.setattr(preview, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preview, "_path", lambda value: tmp_path / value)
    expected = tmp_path / "reports/pipeline/fourlang/exp3_v2_data_preview.json"
    assert preview.checked_output(expected, "exp3_v2") == expected.resolve()
    with pytest.raises(ValueError, match="exp3_v2_data_preview.json"):
        preview.checked_output(tmp_path / preview.DEFAULT_OUTPUT, "exp3_v2")
