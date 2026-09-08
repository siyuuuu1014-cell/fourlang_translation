import copy
import json
from types import SimpleNamespace

import pytest

from scripts.pipeline_v3 import review_zh_uz_semantics as review


def row(i, category="REVIEW_REQUIRED", **extra):
    return {
        "src_lang": "zh",
        "tgt_lang": "uz",
        "src_text": f"句{i}",
        "tgt_text": f"Gap{i}",
        "audit_id": str(i),
        "direction": "zh-uz",
        "preview_category": category,
        "occurrences": 2,
        **extra,
    }


def test_dedup_and_blind_selection():
    rows = [
        row(1),
        row(1, audit_id="other", judge_label="FAIL"),
        row(2, "ISOLATE_CANDIDATE"),
        row(3, "KEEP_CANDIDATE"),
    ]
    original = copy.deepcopy(rows)
    pairs = review.select_pairs(rows, 1, 2026)
    assert len(pairs) == 3
    assert sum(len(p["members"]) for p in pairs) == 4
    assert pairs == review.select_pairs(list(reversed(rows)), 1, 2026)
    assert rows == original
    assert "judge_label" not in review.prompt(pairs[0])
    assert "audit_id" not in review.prompt(pairs[0])


@pytest.mark.parametrize(
    "raw",
    [
        "bad",
        "[]",
        '{"label":"PASS"}',
        '{"label":"PASS","reason":false}',
        '{"label":"PASS","reason":""}',
    ],
)
def test_fail_closed(raw):
    assert review.parse(raw)["parse_ok"] is False
    assert review.parse(raw)["label"] == "UNCERTAIN"


def test_resume_and_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(review.diag, "PROJECT_ROOT", tmp_path)
    rows = [row(i) for i in range(5)]
    monkeypatch.setattr(review, "load_preview", lambda p: (rows, "sig", {}))
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    args = SimpleNamespace(
        preview="preview",
        output="reports/diagnostics/review",
        model=str(model),
        keep_per_direction=1,
        seed=2026,
        batch_size=2,
        max_input_tokens=100,
        max_new_tokens=100,
        prepare_only=False,
    )
    calls = []

    def predict(texts):
        calls.append(len(texts))
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return [json.dumps({"label": "PASS", "reason": "test"}) for _ in texts]

    monkeypatch.setattr(review, "make_predict", lambda *a: predict)
    with pytest.raises(RuntimeError, match="interrupted"):
        review.run(args)
    review.run(args)
    assert calls == [2, 2, 2, 1]
    monkeypatch.setattr(
        review,
        "make_predict",
        lambda *a: pytest.fail("Completed run must not load model"),
    )
    review.run(args)
    summary = review.diag.read_json(tmp_path / args.output / "summary.json")
    assert summary["pairs"] == 5
    assert summary["training_data_written"] is False
    assert not list(tmp_path.rglob("train.jsonl"))
    args.seed += 1
    with pytest.raises(RuntimeError, match="changed"):
        review.run(args)
