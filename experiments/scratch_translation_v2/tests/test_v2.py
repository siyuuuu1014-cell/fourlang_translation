from __future__ import annotations

import copy

import pandas as pd
import pytest
import sentencepiece as spm
import torch

from experiments.scratch_translation_v2 import common, data, engine
from experiments.scratch_translation_v2.model import TranslationTransformer


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
    torch.use_deterministic_algorithms(deterministic)


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    config = common.load_config()
    monkeypatch.setattr(common, "PROJECT", tmp_path)
    monkeypatch.setattr(common, "PACKAGE", tmp_path / "experiments/scratch_translation_v2")
    prepared = tmp_path / "experiments/scratch_translation/artifacts/fourlang_scratch_v1/prepared"
    prepared.mkdir(parents=True)
    corpus = prepared / "corpus.txt"
    corpus.write_text("\n".join(
        f"{lang} simple translation sentence number {i}"
        for lang in common.LANGUAGES for i in range(40)), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prepared / "vocabulary"), vocab_size=96,
        model_type="unigram", character_coverage=1.0, pad_id=0, unk_id=1,
        bos_id=2, eos_id=3,
        user_defined_symbols=[f"<lang:{lang}>" for lang in common.LANGUAGES],
        hard_vocab_limit=False,
    )
    processor = spm.SentencePieceProcessor(model_file=str(prepared / "vocabulary.model"))
    language_ids = {lang: processor.piece_to_id(f"<lang:{lang}>") for lang in common.LANGUAGES}
    rows = {name: [] for name in ("human", "teacher", "validation", "test")}
    for direction in common.DIRECTIONS:
        source, target = direction.split("-")
        for i in range(3):
            src_text = f"{source} simple translation sentence number {i}"
            tgt_text = f"{target} simple translation sentence number {i}"
            base = {"direction": direction, "src_text": src_text, "tgt_text": tgt_text,
                    "src_ids": [language_ids[source], language_ids[target],
                                *processor.encode(src_text), processor.eos_id()],
                    "tgt_ids": [processor.bos_id(), *processor.encode(tgt_text), processor.eos_id()]}
            for name in rows:
                rows[name].append({**base, "kind": "teacher" if name == "teacher" else "human"})
    files = {}
    for name, values in rows.items():
        path = prepared / f"{name}.parquet"
        pd.DataFrame(values).to_parquet(path, index=False)
        files[path.name] = common.file_hash(path)
    files["vocabulary.model"] = common.file_hash(prepared / "vocabulary.model")
    common.PACKAGE.mkdir(parents=True)
    manifest = {"schema": 1, "files": files}
    (prepared / "manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8")
    config["experiment"]["suite"] = "tiny_v2"
    config["model"].update(d_model=16, nhead=2, encoder_layers=1, decoder_layers=1,
                           dim_feedforward=32, dropout=0.1, max_length=32)
    config["sampling"].update(rows_per_direction=4, max_teacher_repeats_per_block=2)
    for direction in common.DIRECTIONS:
        config["sampling"]["kd_ratio"][direction] = 0.5
    config["training"].update(batch_size=2, gradient_accumulation_steps=1,
                              checkpoint_steps=1, eval_steps=1, log_steps=1,
                              keep_checkpoints=2, amp=False)
    config["training"]["phases"] = [
        {"name": "mixed_kd", "steps": 1, "learning_rate": 0.001,
         "warmup_steps": 1, "data": "mixed"},
        {"name": "human_polish", "steps": 1, "learning_rate": 0.0001,
         "warmup_steps": 1, "data": "human"},
    ]
    config["evaluation"].update(validation_batch_size=4, final_batch_size=4,
                                max_new_tokens=2, beam_size=2,
                                diagnostic_samples_per_direction=1)
    return config


def test_config_schedule_and_direction_ratios():
    config = common.load_config()
    assert engine.total_steps(config) == 60000
    assert config["sampling"]["kd_ratio"]["zh-uz"] == 0.50
    assert config["sampling"]["kd_ratio"]["en-zh"] == 0.30
    assert engine.phase_at(config, 49999)[1]["name"] == "mixed_kd"
    assert engine.phase_at(config, 50000)[1]["name"] == "human_polish"
    assert engine.phase_lr(0, config["training"]["phases"][0]) < 0.0003


def test_write_fence_and_frozen_v1_read_only(tiny):
    run = common.suite_root(tiny)
    assert run.is_relative_to(common.PACKAGE / "artifacts")
    for forbidden in (common.PROJECT / "models/model.pt",
                      common.PROJECT / "experiments/scratch_translation/artifacts/change.json",
                      common.PACKAGE / "artifacts"):
        with pytest.raises(ValueError):
            common.writable(forbidden)
    root, _ = common.frozen_prepared(tiny)
    before = {path: common.file_hash(path) for path in root.iterdir() if path.is_file()}
    data.load_data(tiny)
    assert before == {path: common.file_hash(path) for path in root.iterdir() if path.is_file()}


def test_sampling_ratios_repetition_and_human_polish(tiny):
    _, _, frames = data.load_data(tiny)
    human, teacher = frames["human"].to_dict("records"), frames["teacher"].to_dict("records")
    mixed, report = data.sampling_plan(tiny, "mixed", human, teacher, 0)
    polish, polish_report = data.sampling_plan(tiny, "human", human, teacher, 0)
    assert len(mixed) == len(polish) == 48
    assert sum(kind == "teacher" for kind, _ in mixed) == 24
    assert {kind for kind, _ in polish} == {"human"}
    assert all(report[d]["teacher"]["max_repeats"] <= 2 for d in common.DIRECTIONS)
    assert all("teacher" not in polish_report[d] for d in common.DIRECTIONS)
    changed = copy.deepcopy(tiny)
    changed["sampling"]["rows_per_direction"] = 20
    with pytest.raises(ValueError, match="cap"):
        data.sampling_plan(changed, "mixed", human, teacher, 0)


def test_model_causality_beam_and_parameter_scale():
    config = common.load_config()
    model = TranslationTransformer(16000, config["model"])
    assert 50_000_000 < sum(parameter.numel() for parameter in model.parameters()) < 60_000_000
    tiny_model = TranslationTransformer(24, {**config["model"], "d_model": 16, "nhead": 2,
                                              "encoder_layers": 1, "decoder_layers": 1,
                                              "dim_feedforward": 32, "max_length": 16})
    tiny_model.eval()
    source = torch.tensor([[4, 5, 3]])
    decoder = torch.tensor([[2, 8, 9]])
    changed = decoder.clone()
    changed[0, -1] = 10
    with torch.no_grad():
        left, right = tiny_model(source, decoder), tiny_model(source, changed)
    assert torch.allclose(left[:, :-1], right[:, :-1], atol=1e-6)
    assert torch.equal(tiny_model.generate_beam(source, 2, 3, 3, beam_size=1),
                       tiny_model.generate_greedy(source, 2, 3, 3))


def test_two_phase_resume_is_exact(tiny, monkeypatch):
    root = common.suite_root(tiny)
    first = engine.train(tiny, "cpu", stop_after_steps=1)
    assert first["status"] == "paused" and first["phase"] == "human_polish"
    final = engine.train(tiny, "cpu")
    assert final["status"] == "complete" and final["step"] == 2
    resumed = engine.read_checkpoint(root / "run")
    with monkeypatch.context() as patch:
        reference_root = common.PACKAGE / "artifacts/reference"
        patch.setattr(engine, "suite_root", lambda _config: reference_root)
        engine.train(tiny, "cpu")
        uninterrupted = engine.read_checkpoint(reference_root / "run")
    for key in resumed["model"]:
        assert torch.equal(resumed["model"][key], uninterrupted["model"][key]), key
    assert resumed["cursors"] == {"mixed": 2, "human": 2}
    assert resumed["cursors"] == uninterrupted["cursors"]


def test_checkpoint_corruption_is_rejected(tiny):
    engine.train(tiny, "cpu", stop_after_steps=1)
    run = common.suite_root(tiny) / "run"
    record = common.read_json(run / "latest.json")
    (run / "checkpoints" / record["file"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupted"):
        engine.read_checkpoint(run)
