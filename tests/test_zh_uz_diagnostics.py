from __future__ import annotations

import json
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

from scripts.pipeline_v3 import diagnose_zh_uz as diag
from scripts.pipeline_v2.training_safety import (
    atomic_json,
    file_sha256,
    fingerprint,
    model_signature,
)


def row(direction="zh-uz", text="a", target="b", **extra):
    source_lang, target_lang = direction.split("-")
    return {
        "src_lang": source_lang,
        "tgt_lang": target_lang,
        "src_text": text,
        "tgt_text": target,
        "training_source": "teacher_kd_v3",
        "weight": 1.0,
        "origin": "test",
        **extra,
    }


def original_rows():
    return [
        row(judge_label="MINOR", weight=0.5, teacher_id="test_teacher"),
        row(text="c", judge_label="PASS", teacher_id="test_teacher"),
        row(text="d", judge_label=None, weight=0.5, teacher_id="test_teacher"),
        row("uz-zh", judge_label="PASS", teacher_id="test_teacher"),
    ]


def stripped(items):
    return [
        {
            key: value
            for key, value in item.items()
            if key not in ("judge_label", "teacher_id")
        }
        for item in items
    ]


def test_actual_multiplicity_and_weights_not_source_pool_proportions():
    originals = original_rows()
    train = stripped(
        [originals[0]] * 3 + originals[1:] + [row(training_source="human_parallel")]
    )
    report, traces = diag.audit_training(train, originals)
    stats = report["directions"]["zh-uz"]
    assert stats["training_rows"] == 6
    assert stats["teacher_rows"] == 5
    assert stats["teacher_labels"]["MINOR"] == {
        "rows": 3,
        "unique_text_rows": 1,
        "weight_sum": 1.5,
    }
    assert stats["minor_share_of_teacher_rows"] == pytest.approx(3 / 5)
    assert stats["minor_share_bounds_with_unknown_labels"] == pytest.approx(
        [3 / 5, 4 / 5]
    )
    assert stats["minor_share_of_all_rows"] == pytest.approx(3 / 6)
    assert stats["minor_share_of_teacher_weight"] == pytest.approx(1.5 / 3)
    assert (
        stats["teacher_labels"]["UNKNOWN"]["rows"] == 1
    )  # Low weight is NOT proof of MINOR.
    assert stats["max_teacher_occurrences"] == 3
    assert len(traces) == 5


def test_ambiguous_and_unmatched_metadata_are_not_pass():
    original = original_rows()
    original.append({**original[0], "judge_label": "PASS"})
    train = stripped(original[:4]) + [row(text="missing")]
    report, _ = diag.audit_training(train, original)
    stats = report["directions"]["zh-uz"]
    assert stats["teacher_labels"]["AMBIGUOUS"]["rows"] == 1
    assert stats["teacher_metadata_status"]["UNMATCHED"] == 1
    assert stats["teacher_labels"]["UNKNOWN"]["rows"] == 2


def test_source_and_weight_disambiguate_same_text():
    a = row(judge_label="MINOR", weight=0.5)
    b = row(judge_label="PASS", weight=1.0)
    status, meta = diag.resolve_metadata(a, {diag.identity(a): [a, b]})
    assert status == "MATCHED"
    assert meta["judge_label"] == "MINOR"


def test_normalized_metadata_preserves_original_index_and_missing_label(tmp_path):
    path = tmp_path / "rows.jsonl"
    diag.save_jsonl(
        path,
        [
            row(text="", judge_label="FAIL"),
            row(text="  傳統  ", target="ўзбек", judge_label="MINOR"),
            row(text="甲", target="gap", judge_label=None),
        ],
    )
    normalized = diag.normalized_metadata(path)
    assert len(normalized) == 2
    assert normalized[0]["src_text"] == "传统"
    assert normalized[0]["tgt_text"] == "o'zbek"
    assert normalized[0]["judge_label"] == "MINOR"
    assert normalized[1]["judge_label"] == "UNKNOWN"
    assert diag.metadata_value({"label": pd.NA}, "label") == "UNKNOWN"


def test_chunk_interrupt_resume_only_missing_and_detect_corruption(tmp_path):
    rows = [row(text=str(i)) for i in range(19)]
    calls = []

    def interrupted(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("simulated interruption")
        return [text + " translated" for text in texts]

    with pytest.raises(RuntimeError, match="interruption"):
        diag.chunk_predictions(tmp_path, rows, "run", 8, interrupted)
    assert len(list(tmp_path.glob("chunk_*.json"))) == 1
    pending = mock.Mock(
        side_effect=lambda texts: [text + " translated" for text in texts]
    )
    predictions = diag.chunk_predictions(tmp_path, rows, "run", 8, pending)
    assert pending.call_count == 2
    assert pending.call_args_list[0].args[0] == [str(i) for i in range(8, 16)]
    assert len(predictions) == 19
    assert diag.chunk_predictions(tmp_path, rows, "run", 8, None) == predictions
    with pytest.raises(RuntimeError, match="mismatched"):
        diag.chunk_predictions(tmp_path, rows, "another-run", 8, None)
    first = tmp_path / "chunk_000000.json"
    record = diag.read_json(first)
    record["predictions"][0] = "corrupted"
    atomic_json(first, record)
    with pytest.raises(RuntimeError, match="Corrupted"):
        diag.chunk_predictions(tmp_path, rows, "run", 8, None)


def test_chunk_rejects_misaligned_predictions_before_saving(tmp_path):
    with pytest.raises(RuntimeError, match="misaligned"):
        diag.chunk_predictions(tmp_path, [row()], "run", 8, lambda texts: [])
    assert not list(tmp_path.glob("chunk_*.json"))


def test_manifest_and_human_edits_not_overwritten(tmp_path):
    path = tmp_path / "manifest.json"
    diag.bind_manifest(path, {"model": "original"})
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="changed"):
        diag.bind_manifest(path, {"model": "different"})
    assert path.read_bytes() == before
    review = tmp_path / "review.jsonl"
    diag.preserve_text(review, "human notes")
    with pytest.raises(RuntimeError, match="possibly edited"):
        diag.preserve_text(review, "new generation")
    assert review.read_text() == "human notes"


def fake_artifact(path):
    path.mkdir(parents=True)
    atomic_json(path / "config.json", {"model_type": "m2m_100"})
    atomic_json(path / "tokenizer.json", {"test": True})
    (path / "model.safetensors").write_bytes(
        b"fake weights: never loaded by a real model"
    )


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "PROJECT_ROOT", tmp_path)
    original = original_rows()
    train = stripped([original[0]] * 2 + original[1:])
    validation = [
        row(d, f"input {i}", f"reference {i}", training_source="human_parallel")
        for d in diag.DIRECTIONS
        for i in range(17)
    ]
    data = tmp_path / "data/multilingual/fourlang/exp2"
    data.mkdir(parents=True)
    diag.save_jsonl(data / "train.jsonl", train)
    diag.save_jsonl(data / "validation.jsonl", validation)
    kd = tmp_path / "kd.jsonl"
    diag.save_jsonl(kd, original)
    config = {"pair_data": [{"pair": "zh_uz", "kd_train": str(kd)}]}
    for role in ("exp1", "exp2"):
        fake_artifact(tmp_path / f"results/student/fourlang/{role}/best_model/shared")
    teacher = tmp_path / "teacher"
    fake_artifact(teacher)
    candidate = {"id": "test_teacher", "family": "nllb", "path": str(teacher)}
    atomic_json(
        tmp_path / "results/model_selection/zh_uz/selected_teacher.json",
        {"directions": {d: {"candidate": candidate} for d in diag.DIRECTIONS}},
    )
    exp1 = tmp_path / "results/student/fourlang/exp1/best_model/shared"
    exp2 = tmp_path / "results/student/fourlang/exp2/best_model/shared"
    manifest = {
        "experiment": "exp2",
        "shared": True,
        "family": "nllb",
        "seed": 2026,
        "settings": {"direction_validation_samples": 200, "max_source_length": 16},
        "train_sha256": fingerprint(train),
        "validation_sha256": fingerprint(validation),
        "source_artifacts": model_signature(exp1),
        "source_model": str(exp1),
        "language_codes": {"nllb": {"zh": "zho_Hans", "uz": "uzn_Latn"}},
        "decoding": {"num_beams": 1, "max_new_tokens": 3},
        "versions": diag.versions(),
        "precision": "fp32",
    }
    signature = fingerprint(manifest)
    checkpoints = tmp_path / diag.CHECKPOINT_ROOT
    atomic_json(
        checkpoints / "run_manifest.json",
        {"manifest": manifest, "fingerprint": signature},
    )
    groups = diag.flow.fixed_validation_groups(validation, 200, 2026)
    atomic_json(
        checkpoints / "validation_subset.json",
        {"fingerprint": signature, "groups": groups},
    )
    atomic_json(
        checkpoints / "initial_validation.json",
        {"fingerprint": signature, "metrics": {}},
    )
    atomic_json(
        checkpoints / "finished.json",
        {
            "fingerprint": signature,
            "report": {},
            "export_signature": model_signature(exp2),
        },
    )
    return SimpleNamespace(
        root=tmp_path,
        config=config,
        manifest=manifest,
        train=train,
        validation=validation,
        groups=groups,
        candidate=candidate,
    )


def test_saved_subset_authenticated_and_not_resampled(fixture_run):
    data = diag.load_run_inputs(fixture_run.config)
    assert data[1] == fixture_run.groups
    path = fixture_run.root / diag.CHECKPOINT_ROOT / "validation_subset.json"
    record = diag.read_json(path)
    record["groups"]["zh-uz"].reverse()
    atomic_json(path, record)
    with pytest.raises(RuntimeError, match="subset"):
        diag.load_run_inputs(fixture_run.config)


def test_changed_training_data_refused(fixture_run):
    path = fixture_run.root / "data/multilingual/fourlang/exp2/train.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row()) + "\n")
    with pytest.raises(RuntimeError, match="train data changed"):
        diag.load_run_inputs(fixture_run.config)


def test_wrong_run_baseline_refused(fixture_run):
    path = fixture_run.root / diag.CHECKPOINT_ROOT / "initial_validation.json"
    atomic_json(path, {"fingerprint": "different", "metrics": {}})
    with pytest.raises(RuntimeError, match="different runs"):
        diag.load_run_inputs(fixture_run.config)


def test_model_plan_rejects_teacher_mismatch_and_student_replacement(fixture_run):
    train, _, manifest, finished, _, _, paths = diag.load_run_inputs(fixture_run.config)
    original = diag.normalized_metadata(paths["kd_source"])
    audit, _ = diag.audit_training(train, original)
    selected = {
        "directions": {d: {"candidate": fixture_run.candidate} for d in diag.DIRECTIONS}
    }
    candidates, _, coverage = diag.model_plan(
        manifest, finished, audit, original, selected
    )
    assert len(candidates) == 6
    assert all(c["require_local_artifact"] for c in candidates.values())
    assert coverage["zh-uz"]["known_matching_rows"] == 4
    with pytest.raises(RuntimeError, match="differs from v3"):
        diag.model_plan(
            manifest,
            finished,
            audit,
            [{**r, "teacher_id": "wrong"} for r in original],
            selected,
        )
    exp2_weight = (
        fixture_run.root
        / "results/student/fourlang/exp2/best_model/shared/model.safetensors"
    )
    exp2_weight.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="export differs"):
        diag.model_plan(manifest, finished, audit, original, selected)


def test_report_is_aligned_corpus_scored_and_review_selection_explicit():
    groups = {
        d: [row(d, f"source {i}", f"reference words {i}") for i in range(30)]
        for d in diag.DIRECTIONS
    }
    predictions = {
        f"{role}/{d}": [
            r["tgt_text"] if role == "teacher" else f"{role} text {i}"
            for i, r in enumerate(rows)
        ]
        for role in diag.ROLES
        for d, rows in groups.items()
    }
    scores, comparisons, review = diag.comparison_report(
        groups, predictions, {}, {}, 2026
    )
    for d in groups:
        assert scores[d]["teacher"]["chrf2"] == pytest.approx(100)
        expected = diag.flow.metrics(
            predictions[f"exp1/{d}"],
            [r["tgt_text"] for r in groups[d]],
            d.split("-")[1],
        )
        assert scores[d]["exp1"] == expected
    assert comparisons[0]["teacher"] == comparisons[0]["reference"]
    assert comparisons[1]["exp1"] == "exp1 text 1"
    assert len({r["sample_id"] for r in review}) == len(review)
    for d in groups:
        assert (
            sum(
                "fixed_random" in r["selection_reasons"]
                for r in review
                if r["direction"] == d
            )
            == 20
        )
    assert all(r["human_review"]["notes"] == "" for r in review)


def test_review_does_not_call_unchanged_or_improved_rows_regressions():
    groups = {d: [row(d)] for d in diag.DIRECTIONS}
    predictions = {f"{role}/{d}": ["b"] for role in diag.ROLES for d in groups}
    _, _, review = diag.comparison_report(groups, predictions, {}, {}, 2026)
    assert all(r["selection_reasons"] == ["fixed_random"] for r in review)


class FakeTokenizer:
    def __len__(self):
        return 16


class FakeModel:
    dtype = "torch.float32"
    generation_config = SimpleNamespace(to_dict=lambda: {"num_beams": 1})

    def eval(self):
        return self

    def requires_grad_(self, enabled):
        assert enabled is False
        return self


def test_end_to_end_audit_full_resume_and_no_input_mutation(fixture_run, monkeypatch):
    monkeypatch.setattr(diag, "load_config", lambda _: fixture_run.config)
    monkeypatch.setattr(diag.flow.torch.cuda, "is_available", lambda: False)
    inputs = {p: file_sha256(p) for p in fixture_run.root.rglob("*") if p.is_file()}
    args = SimpleNamespace(
        config="unused", output=diag.DEFAULT_OUTPUT, checkpoint_rows=8, audit_only=True
    )
    loader = mock.Mock(return_value=(FakeTokenizer(), FakeModel()))
    translator = mock.Mock(
        side_effect=lambda tokenizer, model, family, source, target, texts, config: [
            f"{source}-{target} output {text}" for text in texts
        ]
    )
    monkeypatch.setattr(diag.flow, "load_model", loader)
    monkeypatch.setattr(diag.flow, "translate", translator)
    diag.diagnose(args)
    loader.assert_not_called()
    args.audit_only = False
    diag.diagnose(args)
    assert (
        loader.call_count == 3
    )  # Teacher once, Exp1 once, Exp2 once; not once per direction.
    assert translator.call_count == 18  # 3 models * 2 directions * ceil(17/8).
    output = fixture_run.root / diag.DEFAULT_OUTPUT
    assert len(diag.read_rows(output / "comparisons.jsonl")) == 34
    assert diag.read_json(output / "done.json")["status"] == "PASS"
    hashes = {
        p: file_sha256(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != ".lock"
    }
    diag.diagnose(args)
    assert loader.call_count == 3
    assert translator.call_count == 18
    assert all(file_sha256(p) == digest for p, digest in hashes.items())
    assert all(file_sha256(p) == digest for p, digest in inputs.items())
    args.checkpoint_rows = 16
    with pytest.raises(RuntimeError, match="changed"):
        diag.diagnose(args)


def test_output_scope_and_lock(fixture_run, monkeypatch):
    monkeypatch.setattr(diag, "load_config", lambda _: fixture_run.config)
    with pytest.raises(ValueError, match="subdirectory"):
        diag.checked_output("results/student/fourlang/exp2/best_model/shared")
    output = fixture_run.root / diag.DEFAULT_OUTPUT
    output.mkdir(parents=True)
    args = SimpleNamespace(
        config="unused", output=diag.DEFAULT_OUTPUT, checkpoint_rows=8, audit_only=True
    )
    with diag.FileLock(str(output / ".lock"), timeout=0):
        with pytest.raises(diag.Timeout):
            diag.diagnose(args)


def test_tiny_local_model_real_inference_no_training(tmp_path, monkeypatch):
    from test_training_optimizations import tiny_model

    path = tmp_path / "tiny"
    tiny_model(path)
    before = model_signature(path)
    monkeypatch.setattr(diag.flow.torch.cuda, "is_available", lambda: False)
    candidate = {
        "id": "tiny",
        "family": "nllb",
        "path": str(path),
        "require_local_artifact": True,
    }
    groups = {d: [row(d, "hello", "world")] for d in diag.DIRECTIONS}
    candidates = {
        f"{role}/{d}": {**candidate, "id": role} for role in diag.ROLES for d in groups
    }
    config = {
        "language_codes": {"nllb": {"zh": "zho_Hans", "uz": "uzn_Latn"}},
        "training": {"max_source_length": 8},
        "deployment": {"num_beams": 1, "max_new_tokens": 2},
    }
    predictions = diag.run_inference(
        tmp_path, groups, candidates, config, "tiny", 8, 2026
    )
    assert len(predictions) == 6
    assert all(
        len(texts) == 1 and isinstance(texts[0], str) for texts in predictions.values()
    )
    assert model_signature(path) == before
