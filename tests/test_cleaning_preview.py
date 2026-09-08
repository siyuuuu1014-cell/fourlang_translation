from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.pipeline_v3 import preview_zh_uz_cleaning as preview

diag = preview.diag


def record(source="第一句。", target="Birinchi gap.", **extras):
    row = {
        "src_lang": "zh",
        "tgt_lang": "uz",
        "src_text": source,
        "tgt_text": target,
        "weight": 1.0,
        "occurrences": 2,
        "training_source": "teacher_kd",
        "judge_label": "PASS",
        "teacher_usefulness": "HIGH",
        "teacher_id": "test",
        "metadata_status": "MATCHED",
        "metadata_recovery_status": "RECOVERED",
        "human_training_reference_candidates": [],
        "review_hints": [],
        **extras,
    }
    row["audit_id"] = diag.fingerprint(
        [
            diag.fingerprint(diag.identity(row)),
            row["training_source"],
            float(row["weight"]),
        ]
    )
    return row


def test_preview_accounting_precedence_and_no_mutation():
    rows = [
        record(),
        record("第二句。", judge_label="MINOR"),
        record("第三句。", training_source="human_replay"),
    ]
    before = copy.deepcopy(rows)
    cases = {rows[0]["audit_id"]: "A reviewed semantic issue."}
    summary, classified = preview.build_preview(rows, cases)
    assert rows == before
    assert {r["preview_category"] for r in classified} == set(preview.CATEGORIES)
    assert all(
        r["quality_approved"] is False and r["human_confirmation"] == ""
        for r in classified
    )
    assert sum(r["occurrences"] for r in classified) == 6
    assert sum(v["sampled_rows"] for v in summary["directions"]["zh-uz"].values()) == 6
    assert summary["matched_case_ids"] == list(cases)
    assert preview.build_preview(list(reversed(rows)), cases) == (summary, classified)


def test_minor_and_unknown_not_deleted():
    for label in ("MINOR", "UNKNOWN", "FAIL"):
        _, rows = preview.build_preview([record(judge_label=label)], {})
        assert rows[0]["preview_category"] == "REVIEW_REQUIRED"


def test_hints_alone_do_not_isolate_or_approve_semantics():
    _, rows = preview.build_preview(
        [record("7-8 soat", "七到八小时", src_lang="uz", tgt_lang="zh")], {}
    )
    assert rows[0]["preview_category"] == "REVIEW_REQUIRED"
    assert rows[0]["quality_approved"] is False


def test_removed_false_positives_not_retained_from_old_audit():
    _, rows = preview.build_preview(
        [
            record(
                "9 g'alaba",
                "9胜",
                src_lang="uz",
                tgt_lang="zh",
                review_hints=["number_unit_review"],
            )
        ],
        {},
    )
    assert rows[0]["preview_category"] == "KEEP_CANDIDATE"
    assert rows[0]["removed_rule_hints"] == ["number_unit_review"]
    assert not rows[0]["review_hints"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("audit_id", "bad"),
        ("occurrences", 0),
        ("occurrences", True),
        ("weight", float("nan")),
    ],
)
def test_invalid_identity_or_counts_rejected(field, value):
    item = record()
    item[field] = value
    with pytest.raises(ValueError):
        preview.build_preview([item], {})


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        preview.build_preview([record(), record()], {})


def test_case_registry_validation_and_missing_cases():
    document = {
        "schema_version": 1,
        "action": "preview_only_no_deletion",
        "cases": [{"audit_id": "a" * 64, "reason": "reviewed"}],
    }
    cases = preview.validate_cases(document)
    summary, _ = preview.build_preview([record()], cases)
    assert summary["unmatched_case_ids"] == ["a" * 64]
    document["cases"] *= 2
    with pytest.raises(ValueError, match="unique"):
        preview.validate_cases(document)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        diag.flow, "load_model", lambda *a, **k: pytest.fail("No models allowed")
    )
    source = tmp_path / preview.DEFAULT_INPUT
    source.mkdir(parents=True)
    rows = [record(), record("7-8 soat", "七到八小时", src_lang="uz", tgt_lang="zh")]
    train = tmp_path / "train.jsonl"
    diag.save_jsonl(train, rows)
    registry = tmp_path / "cases.json"
    diag.save_json(
        registry,
        {
            "schema_version": 1,
            "action": "preview_only_no_deletion",
            "cases": [{"audit_id": rows[0]["audit_id"], "reason": "reviewed example"}],
        },
    )
    signature = diag.bind_manifest(
        source / "quality_manifest.json",
        {
            "schema_version": 2,
            "inputs": {
                "train": {"path": str(train), "sha256": diag.file_sha256(train)}
            },
        },
    )
    diag.save_json(
        source / "quality_audit.json",
        {
            "schema_version": 2,
            "actual_sampling_audit": {
                "directions": {d: {"training_rows": 2} for d in diag.DIRECTIONS}
            },
        },
    )
    diag.save_jsonl(source / "training_flags.jsonl", rows)
    for name in preview.AUDIT_FILES - {"quality_audit.json", "training_flags.jsonl"}:
        diag.preserve_text(source / name, "{}\n")
    diag.save_json(
        source / "done.json",
        {
            "fingerprint": signature,
            "status": "REVIEW_PACKET_READY_NOT_QUALITY_APPROVED",
            "outputs": {
                name: diag.file_sha256(source / name) for name in preview.AUDIT_FILES
            },
        },
    )
    return tmp_path, SimpleNamespace(
        audit=preview.DEFAULT_INPUT, output=preview.DEFAULT_OUTPUT, cases="cases.json"
    )


def test_end_to_end_no_inputs_modified_and_idempotent(fixture):
    root, args = fixture
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    summary = preview.run(args)
    assert before == {p: p.read_bytes() for p in before}
    assert summary["training_data_written"] is False
    assert preview.run(args) == summary
    output = root / args.output
    assert not list(output.glob("train*"))
    done = diag.read_json(output / "done.json")
    for name, digest in done["outputs"].items():
        assert diag.file_sha256(output / name) == digest
    packet = diag.read_json(output / "cleaning_preview_packet.json")
    assert len(packet["review"]) == 2


@pytest.mark.parametrize("target", ["audit", "original"])
def test_changed_inputs_rejected(fixture, target):
    root, args = fixture
    path = (
        root / args.audit / "training_flags.jsonl"
        if target == "audit"
        else root / "train.jsonl"
    )
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        preview.run(args)
    assert not (root / args.output).exists()


def test_output_overlap_and_unsafe_path_rejected(fixture):
    root, args = fixture
    for path in (args.audit, args.audit + "/preview", "data/unsafe"):
        args.output = path
        with pytest.raises(ValueError):
            preview.run(args)


def test_existing_human_notes_not_overwritten(fixture):
    root, args = fixture
    preview.run(args)
    notes = root / args.output / "isolate_candidates.jsonl"
    notes.write_text("human notes", encoding="utf-8")
    with pytest.raises(RuntimeError, match="possibly edited"):
        preview.run(args)
    assert notes.read_text(encoding="utf-8") == "human notes"


def test_registry_changes_require_new_output(fixture):
    root, args = fixture
    preview.run(args)
    registry = root / args.cases
    document = diag.read_json(registry)
    document["cases"][0]["reason"] = "changed recommendation"
    registry.write_text(diag.json.dumps(document), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        preview.run(args)


def test_mid_preview_input_change_blocks_publication(fixture, monkeypatch):
    root, args = fixture
    build = preview.build_preview

    def change(records, cases):
        result = build(records, cases)
        (root / "train.jsonl").write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(preview, "build_preview", change)
    with pytest.raises(RuntimeError, match="Input changed"):
        preview.run(args)
    assert not (root / args.output).exists()
