import json
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from scripts.pipeline_v3 import build_exp3_data as build


def frame(rows):
    return pd.DataFrame(rows)


def rows(direction, count, source="human"):
    src, tgt = direction.split("-")
    return [
        {
            "src_lang": src,
            "tgt_lang": tgt,
            "src_text": f"source {direction} {index}",
            "tgt_text": f"target {direction} {index}",
            "training_source": source,
        }
        for index in range(count)
    ]


def approved(train, validation):
    counts = dict(sorted((train.src_lang + "-" + train.tgt_lang).value_counts().items()))
    return {
        "status": "EXP3_DATA_PREVIEW_READY_NOT_WRITTEN",
        "source_model": "results/student/fourlang/exp2/best_model/shared",
        "planned_rows": len(train),
        "validation_rows": len(validation),
        "sampling": {
            "target_rows_by_direction": counts,
            "sampled_with_replacement": {},
        },
        "leakage_audit": {"protected_overlap_after": 0},
        "source_sha256": {"source.jsonl": "abc"},
        "config_fingerprint": "fingerprint",
    }


def test_version_locked_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "PROJECT_ROOT", tmp_path)
    assert build.checked_dataset_path(tmp_path / build.DEFAULT_DATASET) == (
        tmp_path / build.DEFAULT_DATASET
    ).resolve()
    with pytest.raises(ValueError, match="exactly"):
        build.checked_dataset_path(tmp_path / "data/multilingual/fourlang/exp2")


def test_validation_rejects_duplicate_or_pseudo_validation():
    train = frame(rows("en-zh", 2))
    validation = frame(rows("en-zh", 1))
    report = approved(train, validation)
    duplicate = pd.concat([train, train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        build.validate_frames(duplicate, validation, {**report, "planned_rows": 3})
    pseudo = frame(rows("en-zh", 1, source="teacher_kd"))
    with pytest.raises(ValueError, match="forbidden"):
        build.validate_frames(train, pseudo, report)


def test_build_is_immutable_and_reproducible(monkeypatch, tmp_path):
    monkeypatch.setattr(build, "PROJECT_ROOT", tmp_path)
    train = frame(rows("en-zh", 2) + rows("zh-en", 1))
    validation = frame(rows("en-zh", 1) + rows("zh-en", 1))
    report = approved(train, validation)
    preview_path = tmp_path / build.DEFAULT_PREVIEW
    preview_path.parent.mkdir(parents=True)
    preview_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(build.preview, "prepare_data", lambda _: (train, validation, report))
    monkeypatch.setattr(build, "file_sha256", lambda path: f"hash:{Path(path).name}")
    args = Namespace(
        config="config.toml",
        preview=build.DEFAULT_PREVIEW,
        output=build.DEFAULT_DATASET,
        report=build.DEFAULT_REPORT,
    )
    first = build.run(args)
    second = build.run(args)
    assert first == second
    assert first["status"] == "EXP3_DATA_BUILT_NOT_TRAINED"
    assert first["teacher_generation_performed"] is False
    assert len((tmp_path / build.DEFAULT_DATASET / "train.jsonl").read_text().splitlines()) == 3
