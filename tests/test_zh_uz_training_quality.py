from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from scripts.pipeline_v3 import audit_zh_uz_training_quality as quality

diag = quality.diag


def row(direction="zh-uz", source="这是第一句。", target="Bu birinchi gap.", **extra):
    src, tgt = direction.split("-")
    return {
        "src_lang": src,
        "tgt_lang": tgt,
        "src_text": source,
        "tgt_text": target,
        "training_source": "teacher_kd_v3",
        "weight": 1.0,
        "judge_label": "PASS",
        "teacher_id": "test",
        "teacher_usefulness": "HIGH",
        **extra,
    }


def originals():
    return [
        row(),
        row(
            source="这是第二句。",
            target="Bu ikkinchi gap.",
            judge_label="MINOR",
            weight=0.5,
        ),
        row(
            source="这是第三句。",
            target="Bu uchinchi gap.",
            judge_label=None,
            weight=0.5,
        ),
        row(target="Bu inson tarjimasi.", training_source="human_parallel"),
        row(
            "uz-zh",
            "Bu inson gapi.",
            "这是人工句子。",
            training_source="human_parallel",
        ),
        row("uz-zh", "Bu boshqa gap.", "这是别的句子。"),
    ]


def test_hints_not_semantic_labels_and_no_mutation():
    item = row(source="2020年成立。", target="1999 " + "x" * 16)
    before = copy.deepcopy(item)
    hints = quality.review_hints(item, ["Bu juda uzun inson tarjimasi." * 8])
    assert "digit_token_set_difference" in hints
    assert "target_long_character_run" in hints
    assert "length_vs_single_human_reference" in hints
    assert item == before and item["judge_label"] == "PASS"
    assert not quality.text_hints("This is English, not Uzbek.", "uz")
    assert "non_latin_script" in quality.text_hints("Bu 日本語.", "uz")
    assert "low_han_share" in quality.text_hints("TCP里的ECN", "zh")
    assert "long_character_run" in quality.text_hints("WwwwwWwwwwWwwww", "uz")


def test_human_reference_join_requires_directed_source_and_explicit_human_role():
    items = [
        row(target="human reference", training_source="human_parallel"),
        row(target="teacher text"),
        row(target="unknown", training_source="legacy_unknown"),
        row(
            "uz-zh",
            source="这是第一句。",
            target="opposite",
            training_source="human_parallel",
        ),
    ]
    refs = quality.human_references(items)
    assert refs[quality.source_key(row())] == {"human reference"}
    assert "length_vs_single_human_reference" not in quality.review_hints(
        row(), ["a", "b"]
    )


def test_counts_sampling_provenance_and_blank_blind_review():
    original = originals()
    train = [original[0]] * 3 + [original[1]] * 2 + original[2:]
    before = copy.deepcopy((original, train))
    summary, records, review, selection = quality.build_report(
        train, original, per_stratum=2
    )
    groups = summary["directions"]["zh-uz"]["by_label"]
    assert groups["PASS"]["sampled_rows"] == 3
    assert groups["MINOR"]["sampled_rows"] == 2
    assert groups["UNKNOWN"]["sampled_rows"] == 1  # Low weight is not MINOR.
    assert (
        summary["directions"]["zh-uz"]["current_pool_known_pass_unique_text_pairs"] == 1
    )
    assert len({r["audit_id"] for r in records}) == len(records)
    assert len(review) == len({r["audit_id"] for r in review}) == len(selection)
    for item in review:
        assert (
            not {"judge_label", "teacher_id", "training_source", "review_hints"}
            & item.keys()
        )
        assert all(value == "" for value in item["human_review"].values())
    assert before == (original, train)
    again = quality.build_report(
        list(reversed(train)), list(reversed(original)), per_stratum=2
    )
    assert (summary, records, review, selection) == again


def test_conflicting_or_unmatched_metadata_not_known_pass():
    original = originals()
    original.append({**original[0], "judge_label": "MINOR"})
    train = original + [row(source="没有匹配来源。")]
    summary, records, _, _ = quality.build_report(train, original)
    assert (
        summary["directions"]["zh-uz"]["current_pool_known_pass_unique_text_pairs"] == 0
    )
    ambiguous = next(
        r for r in records if r["src_text"] == "这是第一句。" and diag.is_teacher(r)
    )
    assert ambiguous["metadata_status"] == "AMBIGUOUS"
    assert not ambiguous["known_pass_candidate"]
    unknown = next(r for r in records if r["src_text"] == "没有匹配来源。")
    assert unknown["metadata_status"] == "UNMATCHED"
    assert not unknown["known_pass_candidate"]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        diag.flow, "load_model", lambda *a, **k: pytest.fail("No model loading allowed")
    )
    original = originals()
    kd = tmp_path / "kd.jsonl"
    diag.save_jsonl(kd, original)
    train = diag.normalized_metadata(kd)
    train = [
        {k: r[k] for k in (*diag.FIELDS, "weight", "training_source", "origin")}
        for r in train
    ]
    validation = [
        row(source="验证中文句子。"),
        row("uz-zh", "Bu tekshiruv gapi.", "验证译文。"),
    ]
    data = tmp_path / "data/multilingual/fourlang/exp2"
    data.mkdir(parents=True)
    diag.save_jsonl(data / "train.jsonl", train)
    diag.save_jsonl(data / "validation.jsonl", validation)
    manifest = {
        "experiment": "exp2",
        "shared": True,
        "seed": 2026,
        "settings": {"direction_validation_samples": 200},
        "train_sha256": diag.fingerprint(train),
        "validation_sha256": diag.fingerprint(validation),
    }
    signature = diag.fingerprint(manifest)
    checkpoint = tmp_path / diag.CHECKPOINT_ROOT
    checkpoint.mkdir(parents=True)
    diag.save_json(
        checkpoint / "run_manifest.json",
        {"manifest": manifest, "fingerprint": signature},
    )
    diag.save_json(
        checkpoint / "validation_subset.json",
        {
            "groups": diag.flow.fixed_validation_groups(validation, 200, 2026),
            "fingerprint": signature,
        },
    )
    diag.save_json(
        checkpoint / "initial_validation.json",
        {"metrics": {}, "fingerprint": signature},
    )
    diag.save_json(
        checkpoint / "finished.json", {"report": {}, "fingerprint": signature}
    )
    config = tmp_path / "config.toml"
    config.write_text(
        '[[pair_data]]\npair = "zh_uz"\nkd_train = "kd.jsonl"\n', encoding="utf-8"
    )
    args = SimpleNamespace(
        config=str(config),
        output=quality.DEFAULT_OUTPUT,
        seed=2026,
        review_per_stratum=2,
    )
    return tmp_path, args


def test_end_to_end_cpu_only_idempotent_and_inputs_untouched(fixture):
    root, args = fixture
    inputs = list(root.rglob("*.json")) + list(root.rglob("*.jsonl"))
    before = {p: p.read_bytes() for p in inputs}
    summary = quality.run(args)
    output = root / args.output
    assert not summary["training_started"]
    done = diag.read_json(output / "done.json")
    assert done["status"] == "REVIEW_PACKET_READY_NOT_QUALITY_APPROVED"
    for name, digest in done["outputs"].items():
        assert diag.file_sha256(output / name) == digest
    packet_before = (output / "review_packet.json").read_bytes()
    assert quality.run(args) == summary
    assert (output / "review_packet.json").read_bytes() == packet_before
    assert before == {p: p.read_bytes() for p in inputs}


def test_refuse_modified_generated_review_or_parameters(fixture):
    root, args = fixture
    quality.run(args)
    output = root / args.output
    args.seed += 1
    with pytest.raises(RuntimeError, match="changed"):
        quality.run(args)
    args.seed -= 1
    review = output / "review_blind.jsonl"
    review.write_text("human notes", encoding="utf-8")
    with pytest.raises(RuntimeError, match="possibly edited"):
        quality.run(args)
    assert review.read_text() == "human notes"


def test_changed_run_data_and_bad_output_refused(fixture):
    root, args = fixture
    args.output = "data/distillation/unsafe"
    with pytest.raises(ValueError, match="subdirectory"):
        quality.run(args)
    assert not (root / args.output).exists()
    args.output = quality.DEFAULT_OUTPUT
    train = root / "data/multilingual/fourlang/exp2/train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="train data changed"):
        quality.run(args)
    assert not (root / args.output / "done.json").exists()


def test_mid_read_source_mutation_refused(fixture, monkeypatch):
    root, args = fixture
    load = diag.normalized_metadata

    def changed(path):
        rows = load(path)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        return rows

    monkeypatch.setattr(diag, "normalized_metadata", changed)
    with pytest.raises(RuntimeError, match="Input changed"):
        quality.run(args)
    assert not (root / args.output / "quality_manifest.json").exists()


@pytest.mark.parametrize("count", [0, -1, 101])
def test_invalid_review_budget(count):
    with pytest.raises(ValueError, match="between 1 and 100"):
        quality.build_report([], [], per_stratum=count)
