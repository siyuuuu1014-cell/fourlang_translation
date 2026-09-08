from __future__ import annotations

import pytest

from scripts.pipeline_v3 import quality_checks as checks
from scripts.pipeline_v3.audit_zh_uz_training_quality import text_hints
from scripts.supplemental.import_pair_v2 import language_contract_reason


@pytest.mark.parametrize(
    "text",
    [
        "而家我只会用阿拉伯文创作",
        "而家只用阿拉伯文",
        "佢哋喺度睇书",
    ],
)
def test_cantonese_signal_shared_between_import_and_audit(text):
    assert language_contract_reason(text, "Bu tarjima.") == "NON_MANDARIN_CHINESE"
    assert "non_mandarin_review" in text_hints(text, "zh")


@pytest.mark.parametrize(
    "text",
    [
        "现在我只会用阿拉伯文创作",
        "然而家人并不知道。",
        "而家长需要参加会议。",
        "家庭和工作都很重要。",
        "没有规则提示不代表句子一定正确。",
    ],
)
def test_mandarin_negatives(text):
    assert language_contract_reason(text, "Bu tarjima.") is None
    assert "non_mandarin_review" not in text_hints(text, "zh")


@pytest.mark.parametrize(
    "source,target",
    [
        ("最低支付金额2万苏姆", "Eng kam to'lov 20000 so'm."),
        ("1万欧元", "10 000 evro"),
        ("5000万元", "50 million yuan"),
        ("3.5公里", "3,5 kilometr"),
        ("2公斤", "2000 gramm"),
        ("1200平方英尺", "111.48 kvadrat metr"),
        ("1200平方米", "1200 m²"),
        ("1200平方米", "1200 m2"),
        ("１２００平方米", "1200 kvadrat metr"),
    ],
)
def test_supported_correct_number_expressions_not_flagged(source, target):
    assert checks.numeric_review(source, target)["hints"] == []


@pytest.mark.parametrize(
    "source,target",
    [
        ("1,200平方英尺新型态加油站", "1200 kvadrat metrli yangi benzin stantsiyasi"),
        ("1200平方米", "1200 kvadrat fut"),
        ("20公里", "20 metr"),
        ("2小时", "2 daqiqa"),
        ("2公斤", "2 gramm"),
        ("2公斤", "2"),
    ],
)
def test_unit_change_requires_review(source, target):
    assert "number_unit_review" in checks.numeric_review(source, target)["hints"]


@pytest.mark.parametrize(
    "source,target",
    [
        ("2和2", "2"),
        ("20", "200"),
        ("2万", "2"),
        ("2", "-2"),
    ],
)
def test_value_and_occurrence_changes_reviewed(source, target):
    assert (
        "number_value_or_count_review" in checks.numeric_review(source, target)["hints"]
    )


def test_spelled_numbers_and_protocol_are_not_automatic_errors():
    evidence = checks.numeric_review("7-8 soat", "七到八小时")
    assert evidence["status"] == "REVIEW_NEEDED"
    assert "spelled numbers" in evidence["limitations"]
    assert "reject" not in evidence and "corrected_translation" not in evidence
    assert (
        checks.numeric_review("udp://tracker:port", "udp://tracker:port")["hints"] == []
    )


@pytest.mark.parametrize("apostrophe", ["'", "’", "‘", "ʻ", "ʼ", "`"])
def test_uzbek_word_prefix_is_not_grams(apostrophe):
    text = f"9 g{apostrophe}alaba"
    assert checks.number_evidence(text)[0]["unit"] is None
    assert checks.numeric_review(text, "9胜")["hints"] == []


@pytest.mark.parametrize("text", ["9g", "9 g", "9 g.", "9 g,", "9 gramm"])
def test_real_grams_still_recognized(text):
    assert checks.numeric_review(text, "9克")["hints"] == []
    assert checks.number_evidence(text)[0]["unit"]["dimension"] == "mass"


@pytest.mark.parametrize(
    "word", ["foiz", "foizdan", "foizi", "foizga", "foizning", "foizini", "foizli"]
)
def test_supported_percent_inflections(word):
    assert checks.numeric_review("3%", f"3 {word}")["hints"] == []
    assert "number_unit_review" in checks.numeric_review("30%", f"3 {word}")["hints"]


def test_percent_does_not_match_arbitrary_word_prefix():
    assert checks.number_evidence("3 foizunknown")[0]["unit"] is None
