import copy

import pytest

from scripts.pipeline_v3 import preview_m2m100_error_replay as previewer


def settings():
    return {
        "difficulty_bands": {
            "easy_max_percentile": 0.30,
            "medium_max_percentile": 0.80,
            "hard_max_percentile": 0.95,
        },
        "provisional_weights": {
            "missing_original_weight": 1.0,
            "easy": 0.65,
            "medium": 1.0,
            "hard": 1.35,
            "extreme": 0.85,
            "zh_uz_direction_multiplier": 1.10,
            "other_direction_multiplier": 1.0,
            "duplicate_normalization": "inverse_group_size",
        },
    }


def record(index, direction="en-ru", duplicate="unique", weight=1.0):
    source, target = direction.split("-")
    row = {"src_lang": source, "tgt_lang": target, "src_text": str(index), "tgt_text": str(index)}
    if weight is not None:
        row["weight"] = weight
    return {
        "record_id": f"r{index}",
        "duplicate_group_id": duplicate if duplicate != "unique" else f"d{index}",
        "direction": direction,
        "difficulty_nll": float(index),
        "row": row,
    }


def test_record_identity_keeps_physical_duplicates_distinct():
    row = {"src_lang": "zh", "tgt_lang": "uz", "src_text": "你好", "tgt_text": "Salom"}
    assert previewer.record_identity("exp1", 1, row) != previewer.record_identity("exp2", 1, row)
    assert previewer.duplicate_identity(row) == previewer.duplicate_identity(copy.deepcopy(row))


def test_bands_keep_every_record_and_match_30_50_15_5_split():
    records = [record(index) for index in range(1, 101)]
    previewer.assign_bands(records, settings())
    counts = {}
    for item in records:
        counts[item["difficulty_band"]] = counts.get(item["difficulty_band"], 0) + 1
    assert counts == {"easy": 30, "medium": 50, "hard": 15, "extreme": 5}
    assert len(records) == 100


def test_duplicate_group_is_retained_and_total_weight_normalized():
    records = [record(1, duplicate="same"), record(2, duplicate="same")]
    for item in records:
        item["difficulty_band"] = "medium"
    previewer.apply_provisional_weights(records, settings())
    assert len(records) == 2
    assert sum(item["proposed_weight"] for item in records) == pytest.approx(1.0)
    assert all(item["proposed_weight"] > 0 for item in records)


def test_zh_uz_has_only_the_declared_slight_priority():
    records = [record(1, "zh-uz"), record(2, "en-ru")]
    for item in records:
        item["difficulty_band"] = "medium"
    previewer.apply_provisional_weights(records, settings())
    assert records[0]["proposed_weight"] == pytest.approx(1.1)
    assert records[1]["proposed_weight"] == pytest.approx(1.0)


def test_missing_weight_is_explicitly_marked_as_imputed():
    records = [record(1, weight=None)]
    records[0]["difficulty_band"] = "hard"
    previewer.apply_provisional_weights(records, settings())
    assert records[0]["original_weight_imputed"] is True
    assert records[0]["proposed_weight"] > 0
