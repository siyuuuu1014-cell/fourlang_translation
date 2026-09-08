import json
from pathlib import Path

import pytest

from scripts.pipeline_v3 import run_acceptance_pilot as pilot


def test_real_pilot_coverage():
    sources, tasks = pilot.load_tasks(Path("acceptance/fourlang_pilot_v1"))
    assert len(sources) == 80 and len(tasks) == 240
    assert len({t["task_id"] for t in tasks}) == 240


def test_overlap_both_sides_and_normalization(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "src_lang": "en",
                "tgt_lang": "zh",
                "src_text": "Hello, WORLD!",
                "tgt_text": "你好。",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = pilot.overlap(
        [
            {"src_lang": "en", "src_text": "hello world", "source_id": "one"},
            {"src_lang": "zh", "src_text": "你好", "source_id": "two"},
        ],
        [path],
    )
    assert {h["side"] for h in report["hits"]} == {"src", "tgt"}
    assert report["semantic_near_duplicate_check"] == "NOT_PERFORMED"
    with pytest.raises(FileNotFoundError):
        pilot.overlap([], [tmp_path / "missing.jsonl"])


def test_blind_mapping_and_escaping():
    _, tasks = pilot.load_tasks(Path("acceptance/fourlang_pilot_v1"))
    predictions = {
        role: {t["task_id"]: f"<script>{role}</script>" for t in tasks}
        for role in ("exp1", "exp2")
    }
    bundle = pilot.blind_packet(tasks, predictions)
    key = {r["blind_id"]: r for r in bundle["private_key"]}
    assert len(key) == 240
    for r in bundle["review"]:
        assert "task_id" not in r and "exp1_output" not in r
        k = key[r["blind_id"]]
        assert {k["A"], k["B"]} == {"exp1", "exp2"}
        assert r["A"] == predictions[k["A"]][k["task_id"]]
        assert not r["grade_A"]
    assert "<script>" not in pilot.render(bundle["review"])


def test_chunk_resume_without_model(tmp_path):
    rows = [
        {"src_lang": "en", "tgt_lang": "zh", "src_text": str(i), "tgt_text": ""}
        for i in range(5)
    ]
    calls = []

    def predict(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return ["translated:" + t for t in texts]

    with pytest.raises(RuntimeError, match="interrupted"):
        pilot.diag.chunk_predictions(tmp_path, rows, "sig", 2, predict)
    result = pilot.diag.chunk_predictions(tmp_path, rows, "sig", 2, predict)
    assert result == [f"translated:{i}" for i in range(5)]
    assert len(calls) == 4
    assert pilot.diag.chunk_predictions(tmp_path, rows, "sig", 2, None) == result
    with pytest.raises(RuntimeError, match="mismatched"):
        pilot.diag.chunk_predictions(tmp_path, rows, "other", 2, None)
