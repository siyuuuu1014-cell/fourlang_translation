import json

import pytest

from scripts.pipeline_v3 import auto_judge_acceptance_pilot as judge


def response(x="DIRECT", y="EDIT", source=True):
    def item(grade):
        return {
            "grade": grade,
            "errors": [] if grade in {"DIRECT", "UNJUDGEABLE"} else ["FLUENCY"],
            "reason": "clear reason",
            "confidence": "HIGH",
        }

    return json.dumps({"source_usable": source, "X": item(x), "Y": item(y)})


def test_prompt_reverses_candidates():
    row = {
        "blind_id": "case-1",
        "src_lang": "en",
        "tgt_lang": "zh",
        "src_text": "source",
        "A": "first",
        "B": "second",
    }
    normal = judge.build_prompt(row, False)
    reverse = judge.build_prompt(row, True)
    assert '"X": "first"' in normal and '"Y": "second"' in normal
    assert '"X": "second"' in reverse and '"Y": "first"' in reverse


def test_chinese_punctuation_is_normalized_for_judge_only():
    original = "价格是3.14,可以吗?"
    assert judge.normalize_for_judge("zh", original) == "价格是3.14，可以吗？"
    assert judge.normalize_for_judge("en", original) == original
    row = {
        "src_lang": "uz",
        "tgt_lang": "zh",
        "src_text": "source",
        "A": "第一句.",
        "B": "第一句。",
    }
    prompt = judge.build_prompt(row)
    assert '"X": "第一句。"' in prompt
    assert '"Y": "第一句。"' in prompt
    assert row["A"] == "第一句."


def test_parse_rejects_direct_with_error():
    value = json.loads(response())
    value["X"]["errors"] = ["FLUENCY"]
    assert not judge.parse_judgment(json.dumps(value))["parse_ok"]


def test_reconcile_maps_reversed_second_pass():
    first = judge.parse_judgment(response("DIRECT", "EDIT"))
    second = judge.parse_judgment(response("EDIT", "DIRECT"))
    result = judge.reconcile(first, second)
    assert result["A"]["grade"] == "DIRECT"
    assert result["B"]["grade"] == "EDIT"
    assert result["comparison"] == "A"
    assert result["agreement"]


def test_reconcile_disagreement_is_unresolved():
    first = judge.parse_judgment(response("DIRECT", "EDIT"))
    second = judge.parse_judgment(response("FAIL", "EDIT"))
    result = judge.reconcile(first, second)
    assert result["A"]["grade"] == "UNRESOLVED"
    assert result["comparison"] == "UNRESOLVED"


def test_reveal_and_aggregate_by_actual_model():
    rows = [
        {
            "blind_id": "case-1",
            "src_lang": "en",
            "tgt_lang": "zh",
            "scenario": "daily",
            "src_text": "source",
            "A": "first",
            "B": "second",
            "grade_A": "",
            "grade_B": "",
            "errors_A": [],
            "errors_B": [],
            "reviewer": "",
            "notes": "",
        }
    ]
    keys = {"case-1": {"A": "exp2", "B": "exp1"}}
    judged = judge.reveal(
        rows,
        keys,
        [response("DIRECT", "FAIL")],
        [response("FAIL", "DIRECT")],
    )
    summary = judge.aggregate(judged)
    assert judged[0]["winner"] == "exp2"
    assert summary["overall"]["pairwise"]["exp2_wins"] == 1
    assert summary["overall"]["models"]["exp2"]["grades"]["DIRECT"] == 1
    assert summary["status"] == "AUTO_JUDGE_COMPLETE_NOT_HUMAN_ACCEPTANCE"


def make_acceptance_run(tmp_path):
    source = tmp_path / "acceptance"
    private = source / "organizer_only"
    private.mkdir(parents=True)
    review = {
        "blind_id": "case-1",
        "src_lang": "en",
        "tgt_lang": "zh",
        "scenario": "daily",
        "src_text": "source",
        "A": "first",
        "B": "second",
        "grade_A": "",
        "grade_B": "",
        "errors_A": [],
        "errors_B": [],
        "reviewer": "",
        "notes": "",
    }
    bundle = {
        "review": [review],
        "private_key": [
            {"blind_id": "case-1", "task_id": "task-1", "A": "exp1", "B": "exp2"}
        ],
    }
    acceptance_manifest = {"inputs": "fixed"}
    acceptance_fingerprint = judge.diag.fingerprint(acceptance_manifest)
    (source / "manifest.json").write_text(
        json.dumps(
            {"fingerprint": acceptance_fingerprint, "manifest": acceptance_manifest}
        ),
        encoding="utf-8",
    )
    assignment = private / "assignment.json"
    assignment.write_text(json.dumps(bundle), encoding="utf-8")
    assignment_manifest = {
        "sha256": judge.diag.file_sha256(assignment),
        "run": acceptance_fingerprint,
    }
    (private / "assignment_manifest.json").write_text(
        json.dumps(
            {
                "fingerprint": judge.diag.fingerprint(assignment_manifest),
                "manifest": assignment_manifest,
            }
        ),
        encoding="utf-8",
    )
    (source / "blind_review.jsonl").write_text(
        json.dumps(review) + "\n", encoding="utf-8"
    )
    return source, assignment


def test_load_acceptance_run_authenticates_assignment(tmp_path):
    source, _ = make_acceptance_run(tmp_path)
    rows, keys, signature = judge.load_acceptance_run(source)
    assert len(rows) == 1
    assert keys["case-1"] == {"A": "exp1", "B": "exp2"}
    assert signature


def test_load_acceptance_run_rejects_tampered_assignment(tmp_path):
    source, assignment = make_acceptance_run(tmp_path)
    assignment.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        judge.load_acceptance_run(source)
