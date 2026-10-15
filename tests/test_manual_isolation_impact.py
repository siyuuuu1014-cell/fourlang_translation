from __future__ import annotations

import copy

import pytest

from scripts.pipeline_v3 import preview_manual_zh_uz_isolation as preview


def row(source, target, src_lang="zh", tgt_lang="uz", **extra):
    value = {
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
        "src_text": source,
        "tgt_text": target,
        "weight": 0.5,
        "training_source": "teacher_kd_v3",
        **extra,
    }
    value["review_id"] = preview.diag.fingerprint(preview.diag.identity(value))
    return value


def decisions(isolate, protected):
    return {
        "schema_version": 1,
        "status": "MANUAL_REVIEW_DECISIONS_NOT_APPLIED",
        "summary": {
            "isolate": len(isolate),
            "protect_from_automatic_isolation": len(protected),
        },
        "isolate_review_ids": isolate,
        "protect_from_automatic_isolation": [
            {"review_id": key, "reason": "Reviewed false positive."}
            for key in protected
        ],
        "application_policy": "Only isolate_review_ids may be used by a future dry-run. Protected and pending rows must remain unchanged.",
    }


def test_exact_impact_counts_and_never_mutates_inputs():
    bad = row("坏。", "Yomon.", adjudication_category="REVIEW_REQUIRED")
    safe = row("好。", "Yaxshi.", adjudication_category="HIGH_CONFIDENCE_ISOLATE_CANDIDATE")
    other = row("中。", "O'rta.", adjudication_category="REVIEW_REQUIRED")
    adjudicated = [bad, safe, other]
    datasets = {
        "exp2_train": [bad, bad, safe, other],
        "kd_source": [bad, safe, other],
        "validation": [other],
    }
    before = copy.deepcopy((adjudicated, datasets))
    summary, isolate, protected = preview.build_impact(
        adjudicated, {bad["review_id"]}, {safe["review_id"]: "protect"}, datasets
    )
    assert (adjudicated, datasets) == before
    assert summary["datasets"]["exp2_train"]["isolate_rows"] == 2
    assert summary["datasets"]["exp2_train"]["protected_rows"] == 1
    assert summary["directions"]["zh-uz"]["isolate_unique_pairs"] == 1
    assert isolate[0]["action_applied"] is False
    assert protected[0]["protection_reason"] == "protect"
    assert not summary["training_data_written"]


def test_validation_overlap_fails_closed():
    bad = row("坏。", "Yomon.")
    with pytest.raises(RuntimeError, match="overlaps validation"):
        preview.build_impact(
            [bad], {bad["review_id"]}, {}, {"validation": [bad]}
        )


def test_unknown_and_overlapping_decisions_rejected():
    bad = row("坏。", "Yomon.")
    unknown = "a" * 64
    with pytest.raises(ValueError, match="unknown"):
        preview.build_impact([bad], {unknown}, {}, {"exp2_train": []})
    document = decisions([bad["review_id"]], [bad["review_id"]])
    with pytest.raises(ValueError, match="overlap"):
        preview.validate_decisions(document)


def test_registry_counts_and_hashes_are_strict():
    bad = row("坏。", "Yomon.")
    document = decisions([bad["review_id"]], [])
    isolate, protected = preview.validate_decisions(document)
    assert isolate == {bad["review_id"]}
    assert protected == {}
    document["summary"]["isolate"] = 2
    with pytest.raises(ValueError, match="summary"):
        preview.validate_decisions(document)
