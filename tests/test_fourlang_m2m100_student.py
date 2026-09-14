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

