from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import M2M100Config, M2M100ForConditionalGeneration, NllbTokenizerFast

from scripts.pipeline_v2 import seq2seq_flow as flow
from scripts.pipeline_v2.training_safety import (
    SafeCheckpointCallback,
    atomic_json,
    bind_run,
    checkpoint_complete,
    file_sha256,
    latest_complete_checkpoint,
    model_files,
)
from scripts.pipeline_v3.data_safety import protect_splits
from scripts.pipeline_v3 import fourlang_flow
from scripts.pipeline_v3.fourlang_flow import sample_coverage_first, directions


def row(source="en", target="zh", text="hello", translated="world"):
    return dict(
        src_lang=source,
        tgt_lang=target,
        src_text=text,
        tgt_text=translated,
        weight=1.0,
        training_source="human_parallel",
        origin="test",
    )


def tiny_model(path):
    vocab = {
        word: index
        for index, word in enumerate(
            [
                "<s>",
                "<pad>",
                "</s>",
                "<unk>",
                "eng_Latn",
                "zho_Hans",
                "uzn_Latn",
                "rus_Cyrl",
                "hello",
                "world",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = NllbTokenizerFast(
        tokenizer_object=backend, src_lang="eng_Latn", tgt_lang="zho_Hans"
    )
    model = M2M100ForConditionalGeneration(
        M2M100Config(
            vocab_size=len(tokenizer),
            d_model=16,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=32,
            decoder_start_token_id=2,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
        )
    )
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return tokenizer, model


class SamplingAndDataSafetyTests(unittest.TestCase):
    def test_coverage_before_repetition_with_real_pool_sizes(self):
        for available, quota in (
            (10126, 12000),
            (4277, 8000),
            (10917, 12000),
            (4876, 10200),
        ):
            sampled = sample_coverage_first(
                pd.DataFrame({"id": range(available)}), quota, 2026
            )
            counts = sampled.id.value_counts()
            self.assertEqual(len(sampled), quota)
            self.assertEqual(len(counts), min(available, quota))
            self.assertLessEqual(int(counts.max() - counts.min()), 1)
            pd.testing.assert_frame_equal(
                sampled,
                sample_coverage_first(
                    pd.DataFrame({"id": range(available)}), quota, 2026
                ),
            )

    def test_cross_pair_both_side_and_script_normalization_protection(self):
        train = pd.DataFrame(
            [
                row("en", "ru", "HELLO", "train"),
                row("uz", "zh", "safe", "繁體中文"),
                row("ru", "en", "clean", "unique"),
            ]
        )
        validation = pd.DataFrame(
            [row("en", "zh", "hello", "测试"), row("ru", "en", "seen", "was trained")]
        )
        benchmark = pd.DataFrame(
            [dict(en="bench", zh="繁体中文", uz="salom", ru="benchmark")]
        )
        previous = pd.DataFrame([row("zh", "en", "earlier", "was trained")])
        clean_train, clean_val, audit = protect_splits(
            train, validation, [benchmark], ("en", "zh", "uz", "ru"), previous
        )
        self.assertEqual(clean_train.src_text.tolist(), ["clean"])
        self.assertEqual(len(clean_val), 1)
        self.assertEqual(audit["protected_overlap_after"], 0)
        self.assertEqual(audit["train_rows_removed"], 2)
        self.assertEqual(audit["validation_previous_train_rows_removed"], 1)

    def test_fixed_validation_equal_counts_and_order_independence(self):
        rows = [
            row(*direction.split("-"), f"source{i}", f"target{i}")
            for direction in directions()
            for i in range(5 if direction == "en-zh" else 3)
        ]
        first = flow.fixed_validation_groups(rows, 200, 2026)
        second = flow.fixed_validation_groups(list(reversed(rows)), 200, 2026)
        self.assertEqual(first, second)
        self.assertEqual({len(values) for values in first.values()}, {3})

    def test_aggregate_and_reject_stale_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "multilingual": {"seed": 2026},
                "text_contract": {},
                "balancing": {
                    "exp2": {"default_rows_per_direction": 4, "teacher_ratio": 0.5}
                },
                "benchmarks": {
                    "flores_dev": "dev.parquet",
                    "flores_devtest": "test.parquet",
                },
                "pair_data": [],
            }
            for pair in fourlang_flow.UNORDERED_PAIRS:
                a, b = pair.split("_")
                train, val = [], []
                for source, target in ((a, b), (b, a)):
                    for i in range(4):
                        item = row(
                            source,
                            target,
                            f"{pair}train{source}{i}",
                            f"{pair}target{target}{i}",
                        )
                        item["training_source"] = (
                            "teacher_kd" if i < 2 else "human_replay"
                        )
                        train.append(item)
                    val.append(row(source, target, f"val{source}", f"val{target}"))
                pd.DataFrame(train).to_json(
                    root / f"{pair}train.jsonl", orient="records", lines=True
                )
                pd.DataFrame(val).to_json(
                    root / f"{pair}val.jsonl", orient="records", lines=True
                )
                config["pair_data"].append(
                    {
                        "pair": pair,
                        "kd_train": f"{pair}train.jsonl",
                        "validation": f"{pair}val.jsonl",
                    }
                )
            for name in ("dev.parquet", "test.parquet"):
                pd.DataFrame(
                    [{lang: f"benchmark{lang}" for lang in fourlang_flow.LANGUAGES}]
                ).to_parquet(root / name)
            previous = root / "data/multilingual/fourlang/exp1/train.jsonl"
            previous.parent.mkdir(parents=True)
            previous.write_text(
                json.dumps(row(text="previous", translated="previous target")) + "\n"
            )
            with mock.patch.object(fourlang_flow, "PROJECT_ROOT", root):
                fourlang_flow.aggregate(config, "exp2")
                fourlang_flow.verify_exp2_data(config)
                output = root / "data/multilingual/fourlang/exp2/train.jsonl"
                self.assertEqual(len(output.read_text().splitlines()), 48)
                original = output.read_text()
                output.write_text(original + "\n")
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    fourlang_flow.verify_exp2_data(config)


class TrainingSafetyTests(unittest.TestCase):
    def test_amp_scaler_roundtrip(self):
        trainer = object.__new__(flow.WeightedTrainer)
        scaler = mock.Mock()
        scaler.state_dict.return_value = {"scale": 16384.0, "_growth_tracker": 7}
        trainer.accelerator = SimpleNamespace(scaler=scaler)
        trainer.args = SimpleNamespace(should_save=True)
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                flow.Seq2SeqTrainer, "_save_optimizer_and_scheduler"
            ):
                trainer._save_optimizer_and_scheduler(temporary)
            with mock.patch.object(
                flow.Seq2SeqTrainer, "_load_optimizer_and_scheduler"
            ):
                trainer._load_optimizer_and_scheduler(temporary)
            scaler.load_state_dict.assert_called_once_with(
                scaler.state_dict.return_value
            )

    def test_all_language_prefixes_and_decoder_shifts(self):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer, model = tiny_model(Path(temporary))
            codes = dict(
                zip(
                    ("en", "zh", "uz", "ru"),
                    ("eng_Latn", "zho_Hans", "uzn_Latn", "rus_Cyrl"),
                )
            )
            config = {
                "training": {"max_source_length": 8, "max_target_length": 8},
                "language_codes": {"nllb": codes},
            }
            rows = [row(*direction.split("-")) for direction in directions()]
            tokens = flow.tokenize_rows(rows, tokenizer, "nllb", config)
            for item, encoded in zip(rows, tokens):
                self.assertEqual(
                    encoded["input_ids"][0],
                    tokenizer.convert_tokens_to_ids(codes[item["src_lang"]]),
                )
                self.assertEqual(
                    encoded["labels"][0],
                    tokenizer.convert_tokens_to_ids(codes[item["tgt_lang"]]),
                )
            batch = flow.DataCollatorForSeq2Seq(tokenizer, model=model)(
                [dict(item) for item in tokens]
            )
            loss = flow.WeightedTrainer.compute_loss(None, model, batch)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()

    def test_exp2_missing_local_model_never_falls_back(self):
        with mock.patch.object(flow, "snapshot_download") as download:
            with self.assertRaises(FileNotFoundError):
                flow.candidate_path(
                    {
                        "family": "nllb",
                        "path": "missing-exp1-weights",
                        "require_local_artifact": True,
                    },
                    "en",
                    "zh",
                )
            download.assert_not_called()

    def test_missing_shard_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            atomic_json(path / "config.json", {})
            atomic_json(
                path / "model.safetensors.index.json",
                {"weight_map": {"weight": "missing.safetensors"}},
            )
            with self.assertRaises(FileNotFoundError):
                model_files(path, tokenizer=False)

    def test_manifest_rejects_changed_data_settings_and_legacy(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            bind_run(directory, {"rows": "original", "lr": 0.1})
            for manifest in (
                {"rows": "changed", "lr": 0.1},
                {"rows": "original", "lr": 0.2},
            ):
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    bind_run(directory, manifest)
            legacy = directory / "legacy"
            (legacy / "checkpoint-2").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Legacy"):
                bind_run(legacy, {})

    def test_weighted_loss_matches_reference_and_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, model = tiny_model(Path(temporary))
            model.eval()
            labels = torch.tensor([[4, 8, 2], [5, 9, -100]])
            batch = {
                "input_ids": torch.tensor([[4, 8, 2], [5, 9, 2]]),
                "attention_mask": torch.ones(2, 3, dtype=torch.long),
                "labels": labels,
                "weight": torch.tensor([0.8, 1.0]),
            }
            logits = model(
                **{key: value for key, value in batch.items() if key != "weight"}
            ).logits
            losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            expected = (
                (losses.sum(1) / labels.ne(-100).sum(1)) * batch["weight"]
            ).sum() / batch["weight"].sum()
            expected.backward()
            reference_gradient = model.model.shared.weight.grad.clone()
            model.zero_grad()
            with mock.patch.object(model, "forward", wraps=model.forward) as forward:
                actual = flow.WeightedTrainer.compute_loss(None, model, batch)
            actual.backward()
            self.assertNotIn("labels", forward.call_args.kwargs)
            self.assertIn("weight", batch)  # caller's batch is not mutated
            torch.testing.assert_close(expected, actual)
            torch.testing.assert_close(
                reference_gradient, model.model.shared.weight.grad
            )

    def test_tiny_training_interrupt_resume_selection_and_finished_reuse(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            torch.set_num_threads(1)
            root = Path(temporary)
            source = root / "initial"
            torch.manual_seed(1)
            tiny_model(source)
            initial_hash = file_sha256(source / "model.safetensors")
            candidate = {"family": "nllb", "path": str(source)}
            codes = dict(
                zip(
                    ("en", "zh", "uz", "ru"),
                    ("eng_Latn", "zho_Hans", "uzn_Latn", "rus_Cyrl"),
                )
            )
            config = {
                "multilingual": {"seed": 2026, "languages": list(codes)},
                "language_codes": {"nllb": codes},
                "training": {
                    "batch_size": 6,
                    "gradient_accumulation_steps": 1,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.05,
                    "max_source_length": 8,
                    "max_target_length": 8,
                    "early_stopping_patience": 1,
                    "exp2": {
                        "epochs": 2,
                        "learning_rate": 0.001,
                        "direction_validation_samples": 1,
                        "checkpoint_interval_steps": 2,
                    },
                },
                "deployment": {"num_beams": 1, "max_new_tokens": 4},
            }
            rows = [row(*direction.split("-")) for direction in directions()] * 2
            destination = root / "run/best_model/shared"
            save = SafeCheckpointCallback.on_save

            def interrupt(callback, args, state, control, **kwargs):
                save(callback, args, state, control, **kwargs)
                raise KeyboardInterrupt("simulated interruption after a complete save")

            with mock.patch.object(SafeCheckpointCallback, "on_save", interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    flow.train_model(
                        candidate,
                        str(source),
                        "en",
                        "zh",
                        rows,
                        rows,
                        destination,
                        config,
                        experiment="exp2",
                        shared=True,
                    )
            directory = root / "run/checkpoints/shared"
            signature = json.loads((directory / "run_manifest.json").read_text())[
                "fingerprint"
            ]
            self.assertTrue(checkpoint_complete(directory / "checkpoint-2", signature))
            (directory / "checkpoint-3").mkdir()  # partial newer checkpoint is skipped
            self.assertTrue(
                latest_complete_checkpoint(directory, signature).endswith(
                    "checkpoint-2"
                )
            )
            report = flow.train_model(
                candidate,
                str(source),
                "en",
                "zh",
                rows,
                rows,
                destination,
                config,
                experiment="exp2",
                shared=True,
            )
            self.assertTrue(report["resumed_from_checkpoint"].endswith("checkpoint-2"))
            self.assertEqual(report["metric_for_best_model"], "eval_macro_chrf2")
            self.assertIn("eval_macro_chrf2", report["initial_validation"])
            self.assertTrue(report["best_model_checkpoint"].endswith("checkpoint-4"))
            self.assertEqual(file_sha256(source / "model.safetensors"), initial_hash)
            with mock.patch.object(flow, "load_model") as load:
                self.assertEqual(
                    flow.train_model(
                        candidate,
                        str(source),
                        "en",
                        "zh",
                        rows,
                        rows,
                        destination,
                        config,
                        experiment="exp2",
                        shared=True,
                    ),
                    report,
                )
                load.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                flow.train_model(
                    candidate,
                    str(source),
                    "en",
                    "zh",
                    rows + rows[:1],
                    rows,
                    destination,
                    config,
                    experiment="exp2",
                    shared=True,
                )
            # An uninterrupted reference run must export the same selected weights.
            full_destination = root / "reference/best_model/shared"
            save_model = flow.DirectionAwareTrainer.save_model

            def interrupted_export(trainer, output_dir, *args, **kwargs):
                if Path(output_dir) == full_destination:
                    raise KeyboardInterrupt("simulated interrupted final export")
                return save_model(trainer, output_dir, *args, **kwargs)

            with mock.patch.object(
                flow.DirectionAwareTrainer, "save_model", interrupted_export
            ):
                with self.assertRaises(KeyboardInterrupt):
                    flow.train_model(
                        candidate,
                        str(source),
                        "en",
                        "zh",
                        rows,
                        rows,
                        full_destination,
                        config,
                        experiment="exp2",
                        shared=True,
                    )
            recovered = flow.train_model(
                candidate,
                str(source),
                "en",
                "zh",
                rows,
                rows,
                full_destination,
                config,
                experiment="exp2",
                shared=True,
            )
            self.assertTrue(recovered["recovered_final_export"])
            self.assertEqual(
                file_sha256(full_destination / "model.safetensors"),
                file_sha256(destination / "model.safetensors"),
            )


if __name__ == "__main__":
    unittest.main()
