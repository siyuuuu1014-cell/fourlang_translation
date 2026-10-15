from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.pipeline_v3 import build_clean_zh_uz_v4 as build


def row(source, target, src="zh", tgt="uz", **extra):
    return {
        "src_lang": src,
        "tgt_lang": tgt,
        "src_text": source,
        "tgt_text": target,
        "weight": 0.5,
        "training_source": "teacher_kd_v3",
        **extra,
    }


def review_id(value):
    return build.diag.fingerprint(build.diag.identity(value))


def test_filter_is_exact_preserves_raw_rows_and_protected_rows():
    bad = row("坏。", "Yomon.", metadata="unchanged")
    safe = row("好。", "Yaxshi.", metadata="protected")
    other = row("中。", "O'rta.")
    original = [bad, safe, other]
    before = copy.deepcopy(original)
    kept, removed, protected = build.filter_rows(
        original,
        build.diag.PROJECT_ROOT / "v3.jsonl",
        {review_id(bad)},
        {review_id(safe): "false positive"},
    )
    assert original == before
    assert kept == [safe, other]
    assert removed == [
        {
            "review_id": review_id(bad),
            "direction": "zh-uz",
            "source_row": bad,
        }
    ]
    assert protected == {review_id(safe): 1}


def test_missing_or_duplicate_isolation_match_fails_closed():
    bad = row("坏。", "Yomon.")
    with pytest.raises(ValueError, match="exactly one"):
        build.filter_rows([bad], Path("v3.jsonl"), {"a" * 64}, {})
    with pytest.raises(ValueError, match="exactly one"):
        build.filter_rows([bad, bad], Path("v3.jsonl"), {review_id(bad)}, {})


def test_checked_output_is_version_locked(tmp_path, monkeypatch):
    monkeypatch.setattr(build.diag, "PROJECT_ROOT", tmp_path)
    expected = tmp_path / "data/distillation/zh_uz/v4/train.jsonl"
    assert build.checked_dataset_path(expected) == expected.resolve()
    with pytest.raises(ValueError, match="v4/train.jsonl"):
        build.checked_dataset_path(tmp_path / "data/distillation/zh_uz/v3/train.jsonl")
