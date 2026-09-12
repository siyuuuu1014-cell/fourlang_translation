import json

from scripts.pipeline_v3 import finalize_acceptance_auto_crosscheck as crosscheck


def response(winner, confidence="HIGH", source=True, extra=None):
    value = {
        "source_usable": source,
        "winner": winner,
        "confidence": confidence,
        "reason": "clear semantic comparison",
    }
    if extra is not None:
        value["extra"] = extra
    return json.dumps(value)


def row(a="相同内容。", b="相同内容!"):
    return {
        "blind_id": "case-1",
        "src_lang": "en",
        "tgt_lang": "zh",
        "direction": "en-zh",
        "src_text": "source",
        "A": a,
        "B": b,
        "private_mapping": {"A": "exp1", "B": "exp2"},
        "model_grades": {"exp1": "UNRESOLVED", "exp2": "UNRESOLVED"},
        "winner": "UNRESOLVED",
        "automated_diagnostic_only": True,
    }


def test_surface_equivalence_ignores_formatting_only():
    assert crosscheck.surface_equivalent(row("测试，文本。", " 测试, 文本 "))
    assert not crosscheck.surface_equivalent(row("测试甲", "测试乙"))
    assert not crosscheck.surface_equivalent(row("数值是1.2", "数值是12"))


def test_parser_accepts_one_embedded_strict_object():
    parsed = crosscheck.parse_judgment("answer: " + response("X") + " done")
    assert parsed["parse_ok"]
    assert parsed["winner"] == "X"


def test_parser_rejects_extra_fields_and_multiple_objects():
    assert not crosscheck.parse_judgment(response("X", extra=True))["parse_ok"]
    assert not crosscheck.parse_judgment(response("X") + response("X"))["parse_ok"]


def test_reconcile_maps_reversed_candidate_order():
    result = crosscheck.reconcile(
        crosscheck.parse_judgment(response("X")),
        crosscheck.parse_judgment(response("Y")),
    )
    assert result["agreement"]
    assert result["comparison"] == "A"


def test_reconcile_fails_closed_on_disagreement_or_low_confidence():
    disagreement = crosscheck.reconcile(
        crosscheck.parse_judgment(response("X")),
        crosscheck.parse_judgment(response("X")),
    )
    low = crosscheck.reconcile(
        crosscheck.parse_judgment(response("X", confidence="LOW")),
        crosscheck.parse_judgment(response("Y")),
    )
    assert disagreement["comparison"] == "UNRESOLVED"
    assert low["comparison"] == "UNRESOLVED"


def test_merge_reveals_model_only_after_two_order_agreement():
    case = row("候选甲", "候选乙")
    merged = crosscheck.merge(
        [case],
        [response("Y")],
        [response("X")],
    )
    assert merged[0]["winner"] == "exp2"


def test_merge_marks_only_deterministic_surface_tie_without_model():
    merged = crosscheck.merge([row()], [], [])
    assert merged[0]["winner"] == "TIE"
    assert merged[0]["crosscheck"]["status"] == "RESOLVED_SURFACE_EQUIVALENT"
