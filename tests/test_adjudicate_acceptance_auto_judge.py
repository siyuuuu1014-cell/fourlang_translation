import json

from scripts.pipeline_v3 import adjudicate_acceptance_auto_judge as adjudicate


def candidate(grade):
    return {
        "grade": grade,
        "errors": [] if grade in {"DIRECT", "UNJUDGEABLE"} else ["FLUENCY"],
        "reason": "reason",
        "confidence": "HIGH",
    }


def parsed_pair(x, y, source=True):
    return {
        "source_usable": source,
        "parse_ok": True,
        "X": candidate(x),
        "Y": candidate(y),
    }


def third_response(grade="DIRECT", source=True):
    return json.dumps(
        {
            "source_usable": source,
            "grade": grade,
            "errors": [] if grade in {"DIRECT", "UNJUDGEABLE"} else ["FLUENCY"],
            "reason": "independent reason",
            "confidence": "HIGH",
        }
    )


def unresolved_row():
    return {
        "blind_id": "case-1",
        "src_lang": "en",
        "tgt_lang": "zh",
        "scenario": "daily",
        "src_text": "source",
        "A": "translation a",
        "B": "translation b",
        "pass1": parsed_pair("DIRECT", "EDIT"),
        "pass2_reversed": parsed_pair("DIRECT", "EDIT"),
        "final": {"agreement": False},
        "private_mapping": {"A": "exp1", "B": "exp2"},
        "model_grades": {"exp1": "UNRESOLVED", "exp2": "UNRESOLVED"},
        "winner": "UNRESOLVED",
        "automated_diagnostic_only": True,
    }


def test_majority_requires_two_matching_votes():
    assert adjudicate.majority(["DIRECT", "EDIT", "DIRECT"]) == ("DIRECT", 2)
    assert adjudicate.majority(["DIRECT", "EDIT", "FAIL"])[0] is None


def test_candidate_isolated_prompt_does_not_include_other_candidate():
    row = unresolved_row()
    prompt = adjudicate.build_prompt(row, "A")
    assert "translation a" in prompt
    assert "translation b" not in prompt


def test_candidate_isolated_prompt_normalizes_chinese_punctuation():
    row = unresolved_row()
    row["tgt_lang"] = "zh"
    row["A"] = "翻译结果."
    prompt = adjudicate.build_prompt(row, "A")
    assert "翻译结果。" in prompt
    assert row["A"] == "翻译结果."


def test_third_vote_resolves_order_disagreement():
    row = unresolved_row()
    result = adjudicate.adjudicate_row(
        row,
        {
            "A": adjudicate.parse(third_response("DIRECT")),
            "B": adjudicate.parse(third_response("EDIT")),
        },
    )
    assert result["A"]["grade"] == "DIRECT"
    assert result["B"]["grade"] == "EDIT"
    assert result["comparison"] == "A"


def test_merge_reveals_actual_winner_and_preserves_resolved_rows():
    unresolved = unresolved_row()
    resolved = {**unresolved_row(), "blind_id": "case-2", "winner": "TIE"}
    merged = adjudicate.merge(
        [unresolved, resolved],
        [third_response("DIRECT"), third_response("EDIT")],
    )
    assert merged[0]["winner"] == "exp1"
    assert merged[1]["winner"] == "TIE"
    assert merged[1]["third_stage"]["status"] == "NOT_NEEDED_TWO_PASS_RESOLVED"


def test_invalid_third_response_fails_closed():
    result = adjudicate.parse("not json")
    assert not result["parse_ok"]
    assert result["grade"] == "UNRESOLVED"
