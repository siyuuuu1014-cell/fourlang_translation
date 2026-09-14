from __future__ import annotations

import copy
from collections import Counter

import pytest

from scripts.pipeline_v3 import prepare_zh_uz_targeted_spotcheck as spot


def rows(per_category=4):
    return [
        {
            "pair_id": f"{category}-{index}",
            "src_lang": "zh",
            "tgt_lang": "uz",
            "src_text": f"中文 {category} {index}",
            "tgt_text": f"Uzbek {category} {index}",
            "category": category,
            "teacher_id": "hidden-teacher",
            "training_source": "candidate",
        }
        for category in spot.EXPECTED_CATEGORIES
        for index in range(per_category)
    ]


def test_stratified_packet_is_deterministic_blind_and_non_mutating():
    source = rows()
    before = copy.deepcopy(source)
    packet, key = spot.build_packet(source, per_category=2, medical_extra=1)
    packet_again, key_again = spot.build_packet(
        list(reversed(source)), per_category=2, medical_extra=1
    )

    assert (packet, key) == (packet_again, key_again)
    assert source == before
    assert len(packet) == len(key) == 13
    counts = Counter(row["category"] for row in packet)
    assert counts["medical_emergency"] == 3
    assert all(
        counts[category] == 2
        for category in spot.EXPECTED_CATEGORIES
        if category != "medical_emergency"
    )
    assert all("teacher_id" not in row and "pair_id" not in row for row in packet)
    assert all(row["human_review"]["decision"] == "" for row in packet)
    assert {row["review_id"] for row in packet} == {row["review_id"] for row in key}


def test_contract_rejects_bad_or_insufficient_input():
    source = rows()
    source[0]["pair_id"] = source[1]["pair_id"]
    with pytest.raises(ValueError, match="unique pair_id"):
        spot.build_packet(source, per_category=2)

    with pytest.raises(ValueError, match="Not enough medical_emergency"):
        spot.build_packet(rows(per_category=2), per_category=2, medical_extra=1)
