from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from filelock import FileLock, Timeout

from experiments.scratch_translation import common, data, engine
from experiments.scratch_translation.model import TranslationTransformer


@pytest.fixture(autouse=True)
def cpu_settings():
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    config = common.load_config()
    code = common.implementation()
    monkeypatch.setattr(common, "PROJECT", tmp_path)
    monkeypatch.setattr(common, "PACKAGE", tmp_path / "scratch")
    monkeypatch.setattr(engine, "implementation", lambda: code)
    config["experiment"]["suite"] = "test"
    config["tokenizer"].update(
        vocab_size=96, character_coverage=1.0, max_sentences_per_language=100
    )
    config["model"].update(
        d_model=16,
        nhead=2,
        encoder_layers=1,
        decoder_layers=1,
        dim_feedforward=32,
        dropout=0.15,
        max_length=24,
    )
    config["sampling"].update(
        rows_per_direction=10, kd_ratio=0.3, max_teacher_repeats_per_block=2
    )
    config["training"].update(
        max_steps=4,
        batch_size=2,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        checkpoint_steps=1,
        eval_steps=2,
        log_steps=1,
        keep_checkpoints=1,
        amp=False,
    )
    config["evaluation"].update(
        validation_per_direction=1, batch_size=2, max_new_tokens=4
    )
    for item in config["pair_data"]:
        a, b = item["pair"].split("_")
        human, teacher, val = [], [], []
        for source, target in ((a, b), (b, a)):
            for i in range(6):
                human.append(
                    dict(
                        src_lang=source,
                        tgt_lang=target,
                        src_text=f"{source} human {item['pair']} sentence {i}",
                        tgt_text=f"{target} human {item['pair']} sentence {i}",
                    )
                )
            for i in range(3):
                teacher.append(
                    dict(
                        src_lang=source,
                        tgt_lang=target,
                        src_text=f"{source} human {item['pair']} sentence {i}",
                        tgt_text=f"{target} KDONLY {item['pair']} translated {i}",
                        training_source="teacher_kd",
                    )
                )
            for i in range(2):
                val.append(
                    dict(
                        src_lang=source,
                        tgt_lang=target,
                        src_text=f"{source} VALIDONLY {item['pair']} source {i}",
                        tgt_text=f"{target} VALIDONLY {item['pair']} target {i}",
                    )
                )
        combined = [
            {**row, "training_source": "human_parallel"} for row in human
        ] + teacher
        for kind, rows in (("human", human), ("kd", combined), ("validation", val)):
            relative = f"data/{item['pair']}_{kind}.parquet"
            target_path = tmp_path / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(target_path, index=False)
            item[kind] = relative
    for split in ("dev", "test"):
        relative = f"data/flores_{split}.parquet"
        pd.DataFrame(
            [
                {
                    lang: f"{lang} BENCHONLY {split} original {i}"
                    for lang in common.LANGUAGES
                }
                for i in range(2)
            ]
        ).to_parquet(tmp_path / relative, index=False)
        config["benchmarks"][split] = relative
    return config


def test_output_fence_and_input_fence(tiny):
    assert common.suite_root(tiny).is_relative_to(common.PACKAGE / "artifacts")
    for path in (
        common.PROJECT / "models/bad.pt",
        common.PACKAGE / "artifacts",
        common.PACKAGE / "artifacts/test/../../outside",
    ):
        with pytest.raises(ValueError):
            common.writable(path)
    with pytest.raises(FileNotFoundError):
        common.source_path("../private.parquet")


def test_old_schema_and_strict_provenance():
    legacy = pd.DataFrame(
        [
            dict(
                direction="EN_ZH",
                source_text="hello",
                target_text="學習",
                sample_origin="TEACHER_KD",
            )
        ]
    )
    normalized = data.normalize_rows(legacy, "kd")
    assert normalized.iloc[0].direction == "en-zh"
    assert normalized.iloc[0].tgt_text == "学习"
    with pytest.raises(ValueError, match="HUMAN"):
        data.normalize_rows(legacy, "human")
    with pytest.raises(ValueError, match="explicitly"):
        data.normalize_rows(legacy.drop(columns="sample_origin"), "kd")


def test_scripts_and_wrong_script_prediction():
    assert data.normalize_text("uz", "Ўзбекистон  ғалаба") == "O'zbekiston g'alaba"
    assert data.normalize_text("zh", "學習　語言") == "学习 语言"
    with pytest.raises(ValueError):
        data.normalize_text("uz", "ї")
    assert engine.normalize_prediction("uz", "ї") == "ї"


def test_coverage_ring_and_sampling(tiny):
    first = data.cycle_indices(7, 0, 7, 5)
    assert sorted(first) == list(range(7))
    assert data.cycle_indices(7, 7, 7, 5) == first
    assert data.cycle_indices(7, 5, 4, 5) == first[5:] + first[:2]
    human = [{"direction": d} for d in common.DIRECTIONS for _ in range(6)]
    teacher = [{"direction": d} for d in common.DIRECTIONS for _ in range(3)]
    a, report_a = data.sampling_plan(tiny, "human_only", human, [], 0)
    b, report_b = data.sampling_plan(tiny, "human_kd", human, teacher, 0)
    assert len(a) == len(b) == 120
    assert set(kind for kind, _ in a) == {"human"}
    assert sum(kind == "teacher" for kind, _ in b) == 36
    assert b == data.sampling_plan(tiny, "human_kd", human, teacher, 0)[0]
    for direction in common.DIRECTIONS:
        assert report_a[direction]["human"]["draws"] == 10
        assert report_b[direction]["teacher"]["draws"] == 3
    tiny["sampling"]["kd_ratio"] = 0.9
    with pytest.raises(ValueError, match="cap"):
        data.sampling_plan(tiny, "human_kd", human, teacher, 0)


def test_prepare_protected_overlap_and_vocabulary(tiny):
    item = tiny["pair_data"][0]
    human_path = common.source_path(item["human"])
    human = pd.read_parquet(human_path)
    # Cross-pair, target-side overlap must also be removed, not merely pair overlap.
    protected_val = pd.read_parquet(
        common.source_path(tiny["pair_data"][1]["validation"])
    ).iloc[0]
    extra = {**human.iloc[0].to_dict(), "src_text": protected_val.src_text}
    pd.concat([human, pd.DataFrame([extra])], ignore_index=True).to_parquet(
        human_path, index=False
    )
    before = {
        path: common.file_hash(path) for path in (common.PROJECT / "data").glob("*")
    }
    manifest = data.prepare(tiny)
    assert manifest["audit"]["human_removed"] == 1
    assert manifest["audit"]["remaining_protected_overlap"] == 0
    assert before == {path: common.file_hash(path) for path in before}
    prepared, _ = data.load_prepared(tiny)
    corpus = (prepared / "tokenizer_corpus.txt").read_text(encoding="utf-8")
    assert (
        "KDONLY" not in corpus
        and "VALIDONLY" not in corpus
        and "BENCHONLY" not in corpus
    )
    assert set(pd.read_parquet(prepared / "teacher.parquet").kind) == {"teacher"}
    assert manifest == data.prepare(tiny)
    vocab = data.Vocabulary(prepared / "vocabulary.model", tiny["model"]["max_length"])
    encoded = vocab.encode(
        dict(src_lang="en", tgt_lang="zh", src_text="human", tgt_text="human")
    )
    assert encoded["src_ids"][:2] == [vocab.languages["en"], vocab.languages["zh"]]
    assert encoded["src_ids"][-1] == encoded["tgt_ids"][-1] == vocab.eos
    assert encoded["tgt_ids"][0] == vocab.bos
    changed = copy.deepcopy(tiny)
    changed["benchmarks"] = {
        "dev": tiny["benchmarks"]["test"],
        "test": tiny["benchmarks"]["dev"],
    }
    with pytest.raises(RuntimeError, match="mismatch"):
        data.load_prepared(changed)
    with (prepared / "human.parquet").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="artifact"):
        data.load_prepared(tiny)


def test_causal_padding_and_initialization(tiny):
    settings = {**tiny["model"], "encoder_layers": 2}
    model = TranslationTransformer(32, settings)
    model.eval()
    assert model.output.weight is model.embedding.weight
    layers = model.transformer.encoder.layers
    assert not torch.equal(layers[0].linear1.weight, layers[1].linear1.weight)
    source = torch.tensor([[4, 5, 12, 3]])
    decoder = torch.tensor([[2, 13, 14, 15]])
    changed = decoder.clone()
    changed[0, -1] = 16
    with torch.no_grad():
        left, right = model(source, decoder), model(source, changed)
        assert torch.allclose(left[:, :-1], right[:, :-1], atol=1e-6)
        padded = torch.cat([source, torch.zeros(1, 2, dtype=torch.long)], dim=1)
        assert torch.allclose(left, model(padded, decoder), atol=1e-5)
    assert model.generate(source, 2, 3, 3).shape[1] <= 3


def test_eos_generation_and_padding_loss(tiny, monkeypatch):
    model = TranslationTransformer(32, tiny["model"])

    def force_eos(target, memory, padding):
        logits = torch.zeros(len(target), target.shape[1], 32)
        logits[:, :, 3] = 10
        return logits

    monkeypatch.setattr(model, "decode", force_eos)
    assert model.generate(torch.tensor([[4, 5, 3]]), 2, 3, 10).tolist() == [[3]]
    logits = torch.randn(2, 3, 32)
    labels = torch.tensor([[8, 3, 0], [10, 11, 3]])
    loss = engine.translation_loss(logits, labels, 0, 0.1)
    logits[0, 2] = 1000
    assert torch.equal(loss, engine.translation_loss(logits, labels, 0, 0.1))


def test_resume_is_exact_and_arms_isolated(tiny, monkeypatch):
    data.prepare(tiny)
    root = common.suite_root(tiny)
    first = engine.train(tiny, "human_only", "cpu", stop_after_steps=1)
    assert first["status"] == "paused" and first["step"] == 1
    with pytest.raises(RuntimeError, match="Finish"):
        engine.evaluate(tiny, "human_only", "cpu")
    completed = engine.train(tiny, "human_only", "cpu")
    assert completed["status"] == "complete" and completed["examples_seen"] == 16
    saved_a = engine.read_checkpoint(root / "runs/human_only")
    assert not saved_a["rng"]["cuda"]
    # Same frozen prepared data, entirely separate output root, uninterrupted control.
    with monkeypatch.context() as patch:
        patch.setattr(engine, "suite_root", lambda config: root / "uninterrupted")
        engine.train(tiny, "human_only", "cpu")
        reference = engine.read_checkpoint(root / "uninterrupted/runs/human_only")
    for key in saved_a["model"]:
        assert torch.equal(saved_a["model"][key], reference["model"][key]), key
    assert saved_a["cursor"] == reference["cursor"]
    assert torch.equal(saved_a["rng"]["torch"], reference["rng"]["torch"])
    engine.train(tiny, "human_kd", "cpu")
    saved_b = engine.read_checkpoint(root / "runs/human_kd")
    assert saved_a["initial_hash"] == saved_b["initial_hash"]
    assert saved_a["fingerprint"] != saved_b["fingerprint"]
    assert any(
        not torch.equal(saved_a["model"][key], saved_b["model"][key])
        for key in saved_a["model"]
    )
    for arm in common.GROUPS:
        engine.evaluate(tiny, arm, "cpu")
    assert len(engine.compare(tiny)["directions"]) == 12
    assert engine.train(tiny, "human_only", "cpu")["step"] == 4
    changed = copy.deepcopy(tiny)
    changed["training"]["learning_rate"] *= 2
    with pytest.raises(RuntimeError, match="mismatch"):
        engine.train(changed, "human_only", "cpu")
    with FileLock(str(root / "runs/human_only/.train.lock")):
        with pytest.raises(Timeout):
            engine.train(tiny, "human_only", "cpu")


def test_corrupt_and_cross_arm_checkpoints(tiny):
    data.prepare(tiny)
    root = common.suite_root(tiny)
    engine.train(tiny, "human_only", "cpu", stop_after_steps=1)
    engine.train(tiny, "human_kd", "cpu", stop_after_steps=1)
    a = root / "runs/human_only"
    b = root / "runs/human_kd"
    record = common.read_json(a / "latest.json")
    # A syntactically valid checkpoint from the other arm must still be rejected.
    (b / "checkpoints" / record["file"]).write_bytes(
        (a / "checkpoints" / record["file"]).read_bytes()
    )
    common.atomic_json(b / "latest.json", record)
    with pytest.raises(RuntimeError, match="different arm"):
        engine.train(tiny, "human_kd", "cpu")
    (a / "checkpoints" / record["file"]).write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="corrupted"):
        engine.read_checkpoint(a)


def test_invalid_config_rejected(tmp_path):
    original = (Path(data.__file__).parent / "config.toml").read_text(encoding="utf-8")
    path = tmp_path / "bad.toml"
    for old, new in (
        ("nhead = 6", "nhead = 0"),
        ("max_steps = 30000", "max_steps = 0"),
        ('suite = "fourlang_scratch_v1"', 'suite = "../../escape"'),
    ):
        path.write_text(original.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError):
            common.load_config(path)


def test_rng_roundtrip_and_lr():
    engine.seed_all(123)
    state = engine.rng_state()
    expected_np, expected_torch = np.random.rand(5), torch.rand(5)
    engine.restore_rng(state)
    assert np.array_equal(expected_np, np.random.rand(5))
    assert torch.equal(expected_torch, torch.rand(5))
    config = {"warmup_steps": 100, "learning_rate": 0.001}
    assert engine.learning_rate(100, config) == 0.001
    assert engine.learning_rate(1, config) < engine.learning_rate(100, config)
    assert engine.learning_rate(400, config) == 0.0005


def test_checkpoint_publish_and_retention(tiny, monkeypatch):
    run = common.suite_root(tiny) / "retention"
    for step in (1, 2, 3):
        engine.save_checkpoint(
            run, {"step": step, "best_step": 1, "fingerprint": "local-test"}, keep=1
        )
    assert (run / "checkpoints/step_00000001.pt").is_file()
    assert not (run / "checkpoints/step_00000002.pt").exists()
    assert engine.read_checkpoint(run)["step"] == 3
    previous = common.read_json(run / "latest.json")

    def crash(path, payload):
        raise KeyboardInterrupt

    monkeypatch.setattr(engine, "atomic_torch", crash)
    with pytest.raises(KeyboardInterrupt):
        engine.save_checkpoint(
            run, {"step": 4, "best_step": 1, "fingerprint": "local-test"}, keep=1
        )
    assert common.read_json(run / "latest.json") == previous
    assert engine.read_checkpoint(run)["step"] == 3


def test_accumulation_uses_total_nonpadding_tokens():
    logits = torch.randn(2, 3, 12, requires_grad=True)
    labels = torch.tensor([[4, 3, 0], [8, 9, 3]])
    denominator = labels.ne(0).sum()
    full = engine.translation_loss(logits, labels, 0, 0.1) / denominator
    full.backward()
    expected = logits.grad.clone()
    logits.grad = None
    for i in range(2):
        (
            engine.translation_loss(logits[i : i + 1], labels[i : i + 1], 0, 0.1)
            / denominator
        ).backward()
    assert torch.allclose(logits.grad, expected, atol=1e-7)
