import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.pipeline_v3 import run_acceptance_pilot as pilot


def test_infer_uses_real_local_resolver_and_reuses_chunks(tmp_path, monkeypatch):
    """Keep candidate_path real: previously mocking it hid the repo_id failure."""
    flow = pilot.diag.flow
    monkeypatch.setattr(flow.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        flow, "snapshot_download", lambda **k: pytest.fail("No Hub fallback")
    )
    models = {}
    for role in ("exp1", "exp2"):
        path = tmp_path / role
        path.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (path / name).write_text("{}")
        models[role] = path
    tok = SimpleNamespace(encode=lambda text: [1, 2])
    model = SimpleNamespace(eval=lambda: None, requires_grad_=lambda value: None)
    loaded = []

    def load(path, **kwargs):
        assert kwargs["local_files_only"] is True
        loaded.append(path)
        return model

    monkeypatch.setattr(flow.AutoTokenizer, "from_pretrained", lambda *a, **k: tok)
    monkeypatch.setattr(flow.AutoModelForSeq2SeqLM, "from_pretrained", load)
    monkeypatch.setattr(
        flow,
        "translate",
        lambda tok, mdl, family, src, tgt, texts, config: [
            f"{src}-{tgt}:{t}" for t in texts
        ],
    )
    tasks = [
        {
            "task_id": f"t-{lang}",
            "src_lang": lang,
            "tgt_lang": "zh",
            "src_text": "example",
        }
        for lang in ("en", "ru", "uz")
    ]
    config = {
        "language_codes": {"nllb": {lang: lang for lang in pilot.LANGS}},
        "training": {"max_source_length": 256},
    }
    result = pilot.infer(tasks, tmp_path / "out", "sig", config, models, 8)
    assert len(loaded) == 2
    assert result["exp1"]["t-ru"] == "ru-zh:example"
    assert pilot.infer(tasks, tmp_path / "out", "sig", config, models, 8) == result
    assert len(loaded) == 2
    with pytest.raises(FileNotFoundError):
        flow.candidate_path(
            {
                "family": "nllb",
                "path": str(tmp_path / "missing"),
                "require_local_artifact": True,
            },
            "en",
            "zh",
        )


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
