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
        "quality_review": {
            "length_log_ratio_mad_threshold": 4.0,
            "length_log_ratio_min_deviation": 1.0,
            "minimum_script_characters": 8,
            "target_script_min_ratio": 0.5,
            "cross_script_max_ratio": 0.35,
            "token_overlap_threshold": 0.8,
            "token_overlap_min_tokens": 5,
            "one_flag_multiplier": 0.65,
            "multiple_flag_multiplier": 0.4,
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
        item["quality_flags"] = []
    previewer.apply_provisional_weights(records, settings())
    assert len(records) == 2
    assert sum(item["pre_direction_normalization_weight"] for item in records) == pytest.approx(1.0)
    assert all(item["pre_direction_normalization_weight"] > 0 for item in records)


def test_direction_mass_normalization_makes_zh_uz_exactly_slightly_higher():
    records = []
    index = 0
    for direction in previewer.directions():
        repeats = 3 if direction == "zh-uz" else 1
        for _ in range(repeats):
            index += 1
            records.append(record(index, direction))
    for item in records:
        item["difficulty_band"] = "medium"
        item["quality_flags"] = []
    previewer.apply_provisional_weights(records, settings())
    previewer.normalize_direction_mass(records, settings())
    totals = {}
    for item in records:
        totals[item["direction"]] = totals.get(item["direction"], 0) + item["proposed_weight"]
    ordinary = totals["en-ru"]
    assert all(value == pytest.approx(ordinary) for key, value in totals.items() if key != "zh-uz")
    assert totals["zh-uz"] == pytest.approx(ordinary * 1.1)


def test_missing_weight_is_explicitly_marked_as_imputed():
    records = [record(1, weight=None)]
    records[0]["difficulty_band"] = "hard"
    records[0]["quality_flags"] = []
    previewer.apply_provisional_weights(records, settings())
    assert records[0]["original_weight_imputed"] is True
    assert records[0]["pre_direction_normalization_weight"] > 0


def test_quality_flags_separate_obvious_mismatch_from_difficulty():
    records = [
        record(1, "en-zh"),
        record(2, "zh-en"),
    ]
    records[0]["row"].update(src_text="This remains English", tgt_text="This remains English")
    records[1]["row"].update(src_text="这是一个足够长的测试句子", tgt_text="这是一个足够长的测试句子")
    previewer.add_quality_flags(records, settings())
    assert "source_target_identical" in records[0]["quality_flags"]
    assert "target_script_mismatch" in records[0]["quality_flags"]
    assert "target_script_mismatch" in records[1]["quality_flags"]


def test_quality_flag_reduces_hard_example_weight():
    clean = record(1)
    flagged = record(2)
    for item in (clean, flagged):
        item["difficulty_band"] = "hard"
    clean["quality_flags"] = []
    flagged["quality_flags"] = ["length_ratio_outlier"]
    previewer.apply_provisional_weights([clean, flagged], settings())
    assert flagged["pre_direction_normalization_weight"] < clean["pre_direction_normalization_weight"]
