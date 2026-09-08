from __future__ import annotations

import copy

import pandas as pd
import pytest

from scripts.pipeline_v3 import judge_metadata_recovery as recovery
from scripts.pipeline_v3.audit_zh_uz_training_quality import build_report


def training(**extra):
    return {
        "src_lang": "zh",
        "tgt_lang": "uz",
        "src_text": "现在我写作。",
        "tgt_text": "Endi men yozaman.",
        "training_source": "teacher_kd",
        "weight": 0.5,
        "judge_label": "UNKNOWN",
        "teacher_id": "UNKNOWN",
        "teacher_usefulness": "HIGH",
        **extra,
    }


def judgment(**extra):
    return {
        **training(),
        "judge_label": "PASS",
        "teacher_id": "nllb-test",
        "parse_ok": True,
        "evidence_path": "saved.parquet",
        "evidence_row": 0,
        **extra,
    }


def both_directions(rows):
    return rows + [
        training(
            src_lang="uz", tgt_lang="zh", src_text="Boshqa gap.", tgt_text="另一句。"
        )
    ]


def test_recovery_only_changes_audit_copy_and_not_from_weight():
    original = [training()]
    before = copy.deepcopy(original)
    rows, report = recovery.recover_metadata(original, [judgment()])
    assert original == before
    assert rows[0]["judge_label"] == "PASS"  # Even though weight is 0.5.
    assert rows[0]["teacher_id"] == "nllb-test"
    assert rows[0]["weight"] == 0.5
    record = report["records"][0]
    assert record["status"] == "RECOVERED"
    assert record["before"]["judge_label"] == "UNKNOWN"
    assert record["evidence"][0]["path"] == "saved.parquet"
    summary, traces, _, _ = build_report(
        both_directions(original), both_directions(rows), recovery_report=report
    )
    trace = next(t for t in traces if t["src_lang"] == "zh")
    assert trace["metadata_recovery_status"] == "RECOVERED"
    assert trace["known_pass_candidate"]
    assert summary["metadata_recovery"]["counts_current_pool_teacher_rows"] == {
        "RECOVERED": 1
    }


@pytest.mark.parametrize(
    "other",
    [
        {"src_lang": "uz", "tgt_lang": "zh"},
        {"src_text": "现在我写作!"},
        {"tgt_text": "Endi u yozadi."},
    ],
)
def test_no_fuzzy_or_directionless_join(other):
    rows, report = recovery.recover_metadata([training()], [judgment(**other)])
    assert rows[0]["judge_label"] == "UNKNOWN"
    assert report["records"][0]["status"] == "NO_MATCH"


@pytest.mark.parametrize(
    "change",
    [
        {"judge_label": "MINOR"},
        {"teacher_id": "other-teacher"},
        {"teacher_usefulness": "MEDIUM"},
    ],
)
def test_disagreeing_evidence_blocks_all_recovery(change):
    rows, report = recovery.recover_metadata(
        [training()], [judgment(), judgment(**change)]
    )
    assert rows[0]["judge_label"] == "UNKNOWN"
    assert report["records"][0]["status"] == "CONFLICT"
    assert report["records"][0]["conflicting_fields"]


def test_existing_metadata_conflict_is_not_overwritten_or_known_pass():
    original = [training(judge_label="PASS", teacher_id="test")]
    rows, report = recovery.recover_metadata(original, [judgment(judge_label="FAIL")])
    assert rows == original
    _, traces, _, _ = build_report(
        both_directions(original), both_directions(rows), recovery_report=report
    )
    trace = next(t for t in traces if t["src_lang"] == "zh")
    assert not trace["known_pass_candidate"]
    assert "judge_metadata_review" in trace["review_hints"]


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"parse_ok": False}, "PARSE_UNVERIFIED"),
        ({"judge_label": "UNCERTAIN"}, "LABEL_UNVERIFIED"),
        ({"teacher_usefulness": "INVALID"}, "CONFLICT"),
    ],
)
def test_untrusted_evidence_does_not_fill(changes, status):
    rows, report = recovery.recover_metadata([training()], [judgment(**changes)])
    assert rows[0]["judge_label"] == "UNKNOWN"
    assert report["records"][0]["status"] == status


def test_missing_teacher_id_is_not_inferred():
    rows, report = recovery.recover_metadata(
        [training()], [judgment(teacher_id="UNKNOWN")]
    )
    assert rows[0]["judge_label"] == "PASS"
    assert rows[0]["teacher_id"] == "UNKNOWN"
    assert "teacher_id" not in report["records"][0]["filled_fields"]


def test_human_records_never_receive_teacher_labels():
    original = [training(training_source="human_replay")]
    rows, report = recovery.recover_metadata(original, [judgment()])
    assert rows == original
    assert report["records"] == []


@pytest.mark.parametrize("suffix", [".parquet", ".jsonl"])
def test_read_judged_artifacts_normalizes_like_training_and_requires_teacher_text(
    tmp_path, suffix
):
    path = tmp_path / ("judged" + suffix)
    record = {
        "source_lang": "zh",
        "target_lang": "uz",
        "source_text": "現在我寫作。",
        "teacher_text": "Endi men yozaman.",
        "judge_label": "PASS",
        "judge_parse_ok": True,
        "teacher_usefulness": "HIGH",
        "teacher_id": "test",
        "reference_text": "Do not join this.",
    }
    if suffix == ".parquet":
        pd.DataFrame([record]).to_parquet(path)
    else:
        recovery.diag.save_jsonl(path, [record])
    records = recovery.read_judgments(path)
    assert records[0]["src_text"] == "现在我写作。"
    assert records[0]["tgt_text"] == record["teacher_text"]
    assert records[0]["parse_ok"] is True
    assert records[0]["evidence_row"] == 0
    path = tmp_path / "no_teacher.jsonl"
    del record["teacher_text"]
    recovery.diag.save_jsonl(path, [record])
    with pytest.raises(ValueError, match="teacher_text"):
        recovery.read_judgments(path)


@pytest.mark.parametrize("parse_value", ["False", "True", False, None, 1, 0])
def test_parse_flag_must_be_real_boolean(tmp_path, parse_value):
    path = tmp_path / "judged.jsonl"
    recovery.diag.save_jsonl(
        path,
        [
            {
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": "现在我写作。",
                "teacher_text": "Endi men yozaman.",
                "judge_label": "PASS",
                "judge_parse_ok": parse_value,
            }
        ],
    )
    assert recovery.read_judgments(path)[0]["parse_ok"] is False
