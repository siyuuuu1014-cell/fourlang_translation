from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pipeline_v3 import manage_pair_model_artifacts as manager


def manifest() -> dict:
    return {
        "protected_prefixes": ["results/student/fourlang"],
        "pairs": [
            {
                "id": "en_zh",
                "directions": ["en-zh", "zh-en"],
                "status": "selected_validated",
                "model_path": "results/current/en_zh",
                "training_entrypoint": "train.py",
                "training_config": "config.toml",
                "training_data": ["data/train.jsonl"],
                "validation_data": "data/validation.jsonl",
                "evaluation_evidence": ["results/eval.json"],
            }
        ],
        "safe_delete": ["results/discarded/run1"],
    }


def test_cleanup_plan_only_lists_allowlisted_target(tmp_path: Path) -> None:
    discarded = tmp_path / "results/discarded/run1"
    discarded.mkdir(parents=True)
    (discarded / "weights.bin").write_bytes(b"1234")

    plan = manager.cleanup_plan(manifest(), tmp_path)

    assert plan[0]["path"] == "results/discarded/run1"
    assert plan[0]["bytes"] == 4
    assert discarded.exists()


def test_cleanup_rejects_protected_overlap(tmp_path: Path) -> None:
    payload = manifest()
    payload["safe_delete"] = ["results/student"]

    with pytest.raises(RuntimeError, match="protected"):
        manager.cleanup_plan(payload, tmp_path)


def test_resolve_inside_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        manager.resolve_inside(tmp_path, "../outside")


def test_checked_manifest_is_valid_json() -> None:
    payload = json.loads(manager.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    assert len(payload["pairs"]) == 6
    assert payload["protected_prefixes"]


def test_inventory_requires_config_and_weights(tmp_path: Path) -> None:
    payload = manifest()
    model = tmp_path / "results/current/en_zh"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")

    before = manager.model_inventory(payload, tmp_path)[0]
    (model / "model.safetensors").write_bytes(b"weights")
    after = manager.model_inventory(payload, tmp_path)[0]

    assert not before["model_complete"]
    assert after["model_complete"]
