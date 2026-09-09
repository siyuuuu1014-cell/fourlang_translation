import json

from scripts.pipeline_v3 import finalize_zh_uz_adjudication as finalize


def pair(source="原文。", target="Tarjima."):
    return {
        "src_lang": "zh",
        "tgt_lang": "uz",
        "src_text": source,
        "tgt_text": target,
    }


def raw(**updates):
    value = {
        "source_usable": True,
        "label": "PASS",
        "error_type": "NONE",
        "source_evidence": "",
        "target_evidence": "",
        "explanation": "Faithful.",
        "confidence": "HIGH",
    }
    value.update(updates)
    return json.dumps(value, ensure_ascii=False)


def test_recovers_pass_with_unnecessary_evidence():
    judgment, mode = finalize.parse_relaxed(
        raw(source_evidence="原文。", target_evidence="Tarjima."), pair()
    )
    assert judgment["parse_ok"]
    assert judgment["source_evidence"] == judgment["target_evidence"] == ""
    assert mode == "RECOVERED_PASS_EXTRA_EVIDENCE"


def test_recovers_only_punctuation_spacing_evidence_difference():
    judgment, mode = finalize.parse_relaxed(
        raw(
            label="MINOR",
            error_type="FLUENCY",
            source_evidence="原文",
            target_evidence="Tarjima",
        ),
        pair(source="原 文。", target="Tarjima."),
    )
    assert judgment["parse_ok"]
    assert mode == "RECOVERED_NORMALIZED_EVIDENCE"


def test_does_not_recover_hallucinated_evidence():
    judgment, mode = finalize.parse_relaxed(
        raw(
            label="MINOR",
            error_type="OMISSION",
            source_evidence="不存在",
            target_evidence="Tarjima",
        ),
        pair(),
    )
    assert not judgment["parse_ok"]
    assert mode == "UNRESOLVED_EVIDENCE_CONTRACT"


def test_exception_demotes_consensus_candidate():
    row = {
        **pair(),
        "review_id": "r",
        "members": [{"audit_id": "a"}],
        "judgment": {"label": "FAIL"},
    }
    final = {
        "r": {
            "judgment": {
                "label": "FAIL",
                "confidence": "HIGH",
                "parse_ok": True,
                "evidence_validation": "EXACT",
            },
            "provenance": "test",
        }
    }
    preview = finalize.build_preview([row], final, set(), {"r": "term"})
    assert (
        preview[0]["adjudication_category"]
        == "TERMINOLOGY_OR_FORMAT_REVIEW_REQUIRED"
    )
    assert not preview[0]["training_action_applied"]


def test_normalized_evidence_cannot_promote_isolation_candidate():
    row = {
        **pair(),
        "review_id": "r",
        "members": [{"audit_id": "a"}],
        "judgment": {"label": "FAIL"},
    }
    final = {
        "r": {
            "judgment": {
                "label": "FAIL",
                "confidence": "HIGH",
                "parse_ok": True,
                "evidence_validation": "NORMALIZED_PUNCTUATION_OR_SPACING",
            },
            "provenance": "test",
        }
    }
    preview = finalize.build_preview([row], final, set(), {})
    assert preview[0]["adjudication_category"] == "REVIEW_REQUIRED"
