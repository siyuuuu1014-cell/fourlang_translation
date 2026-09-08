from scripts.pipeline_v3 import adjudicate_zh_uz_semantics as adjudicate


def pair(i, label="PASS"):
    return {
        "src_lang": "zh", "tgt_lang": "uz", "src_text": f"原文{i}",
        "tgt_text": f"tarjima{i}", "review_id": str(i), "members": [{"audit_id": str(i)}],
        "judgment": {"label": label, "reason": "x", "parse_ok": True},
    }


def test_second_prompt_is_blind():
    row = pair(1, "FAIL")
    row["judgment"]["reason"] = "secret old reason"
    text = adjudicate.second_prompt(row)
    assert "secret old reason" not in text
    assert "review_id" not in text


def test_second_parse_requires_exact_evidence():
    row = pair(1)
    good = '{"source_usable":true,"label":"FAIL","error_type":"MISTRANSLATION","source_evidence":"原文","target_evidence":"tarjima","explanation":"wrong","confidence":"HIGH"}'
    bad = good.replace('"tarjima"', '"invented"')
    assert adjudicate.parse_second(good, row)["parse_ok"]
    assert not adjudicate.parse_second(bad, row)["parse_ok"]


def test_selection_is_deterministic_and_keeps_all_non_pass():
    rows = [pair(i, "FAIL" if i == 0 else "PASS") for i in range(8)]
    chosen = adjudicate.select(rows, 2, 2026)
    assert len(chosen) == 3
    assert any(r["review_id"] == "0" for r in chosen)
    assert chosen == adjudicate.select(list(reversed(rows)), 2, 2026)


def test_selection_rechecks_recovered_pass():
    rows = [pair(1, "PASS"), pair(2, "PASS")]
    rows[0]["first_pass_recovered"] = True
    chosen = adjudicate.select(rows, 0, 2026)
    assert [r["review_id"] for r in chosen] == ["1"]


def test_conservative_policy_never_applies_training_action():
    rows = [pair(1, "FAIL"), pair(2, "PASS")]
    second = [{**rows[0], "second_judgment": {
        "label": "FAIL", "confidence": "HIGH", "parse_ok": True,
    }}]
    preview = adjudicate.conservative_preview(rows, second, set())
    assert preview[0]["adjudication_category"] == "HIGH_CONFIDENCE_ISOLATE_CANDIDATE"
    assert preview[1]["adjudication_category"] == "UNREVIEWED_FIRST_PASS_PASS"
    assert all(not row["training_action_applied"] for row in preview)
