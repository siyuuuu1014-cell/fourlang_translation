from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.pipeline_v3 import weak_pair_ablation as ablation


def row(source: str, target: str, index: int, origin: str = "teacher_kd"):
    return {
        "src_lang": source,
        "tgt_lang": target,
        "src_text": f"{source} source {index}",
        "tgt_text": f"{target} target {index}",
        "weight": 1.0,
        "training_source": origin,
    }


def config() -> dict:
    return {
        "experiment": {"id": "test", "seed": 2026, "backbone": "small100"},
        "direction": {"seed": 2026},
        "model": {"family": "small100", "path": "base"},
        "benchmarks": {
            "flores_dev": "dev.parquet",
            "protected_devtest": "devtest.parquet",
        },
        "training": {"exp2": {"epochs": 2, "learning_rate": 5e-6}},
        "deployment": {"num_beams": 5, "max_new_tokens": 256},
        "comparison": {
            "noise_floor_chrf2": 0.3,
            "meaningful_gain_chrf2": 1.0,
        },
        "variants": {
            "bidir_full": {"learning_rate": 5e-6},
            "directional_full": {"learning_rate": 5e-6},
            "full_weighted_60_40": {
                "learning_rate": 5e-6,
                "teacher_effective_ratio": 0.6,
            },
            "full_native_lr2e6": {"learning_rate": 2e-6},
            "flores_relaxed_8k": {
                "learning_rate": 5e-6,
                "kd_train": "relaxed.jsonl",
            },
        },
        "pairs": [
            {
                "id": "zh_uz",
                "languages": ["zh", "uz"],
                "kd_quality_status": "approved",
                "kd_train": "kd.jsonl",
                "human_train": "human.jsonl",
                "validation": "validation.jsonl",
                "exp1_model": "exp1",
                "exp2_model": "exp2",
                "selected_baseline_stage": "exp1",
            },
            {
                "id": "uz_ru",
                "languages": ["uz", "ru"],
                "kd_quality_status": "approved",
                "kd_train": "missing-kd.jsonl",
                "human_train": "missing-human.jsonl",
                "validation": "missing-validation.jsonl",
                "exp1_model": "missing-exp1",
                "exp2_model": "missing-exp2",
                "selected_baseline_stage": "exp2",
            },
        ],
    }


class WeakPairAblationTests(unittest.TestCase):
    def test_run_ids_keep_directional_artifacts_separate(self):
        self.assertEqual(ablation.run_id("bidir_full"), "bidir_full")
        self.assertEqual(
            ablation.run_id("directional_full", "zh-uz"),
            "directional_full__zh_uz",
        )

    def test_prepare_full_keeps_every_unique_eligible_row(self):
        cfg = config()
        train = pd.DataFrame(
            [
                row("zh", "uz", 1),
                row("zh", "uz", 2, "human_replay"),
                row("uz", "zh", 3),
                row("uz", "zh", 4, "human_replay"),
                row("zh", "uz", 1),
            ]
        )
        validation = pd.DataFrame([row("zh", "uz", 10), row("uz", "zh", 11)])
        human = pd.DataFrame([row("zh", "uz", 20), row("uz", "zh", 21)])
        benchmark = pd.DataFrame({"zh": ["benchmark zh"], "uz": ["benchmark uz"]})
        frames = {
            "kd.jsonl": train,
            "validation.jsonl": validation,
            "human.jsonl": human,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("kd.jsonl", "validation.jsonl", "human.jsonl"):
                (root / name).write_text(name)
            benchmark.to_parquet(root / "dev.parquet")
            benchmark.to_parquet(root / "devtest.parquet")

            def read_table(path):
                return frames[Path(path).name].copy()

            with mock.patch.object(ablation, "PROJECT_ROOT", root), mock.patch.object(
                ablation, "_read_table", side_effect=read_table
            ):
                report = ablation.prepare(cfg, "zh_uz", "bidir_full")
                prepared = ablation._read_jsonl(
                    root
                    / "data/experiments/weak_pair_ablation/zh_uz/bidir_full/train.jsonl"
                )
        self.assertEqual(len(prepared), 4)
        self.assertEqual(report["composition"]["by_direction"], {"uz-zh": 2, "zh-uz": 2})
        self.assertEqual(report["selection"], "all unique eligible rows; no resampling or repetition")

    def test_directional_prepare_filters_train_and_validation(self):
        cfg = config()
        train = pd.DataFrame([row("zh", "uz", 1), row("uz", "zh", 2)])
        validation = pd.DataFrame([row("zh", "uz", 10), row("uz", "zh", 11)])
        human = pd.DataFrame([row("zh", "uz", 20), row("uz", "zh", 21)])
        benchmark = pd.DataFrame({"zh": ["benchmark zh"], "uz": ["benchmark uz"]})
        frames = {
            "kd.jsonl": train,
            "validation.jsonl": validation,
            "human.jsonl": human,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("kd.jsonl", "validation.jsonl", "human.jsonl"):
                (root / name).write_text(name)
            benchmark.to_parquet(root / "dev.parquet")
            benchmark.to_parquet(root / "devtest.parquet")

            def read_table(path):
                return frames[Path(path).name].copy()

            with mock.patch.object(ablation, "PROJECT_ROOT", root), mock.patch.object(
                ablation, "_read_table", side_effect=read_table
            ):
                report = ablation.prepare(
                    cfg, "zh_uz", "directional_full", "zh-uz"
                )
                output = root / (
                    "data/experiments/weak_pair_ablation/zh_uz/"
                    "directional_full__zh_uz"
                )
                prepared_train = ablation._read_jsonl(output / "train.jsonl")
                prepared_validation = ablation._read_jsonl(output / "validation.jsonl")
        self.assertTrue(all(item["src_lang"] == "zh" for item in prepared_train))
        self.assertTrue(all(item["src_lang"] == "zh" for item in prepared_validation))
        self.assertEqual(report["direction"], "zh-uz")

    def test_relaxed_variant_uses_its_isolated_dataset(self):
        cfg = config()
        relaxed = pd.DataFrame(
            [row("zh", "uz", 30), row("uz", "zh", 31)]
        )
        validation = pd.DataFrame([row("zh", "uz", 10), row("uz", "zh", 11)])
        human = pd.DataFrame([row("zh", "uz", 20), row("uz", "zh", 21)])
        benchmark = pd.DataFrame({"zh": ["benchmark zh"], "uz": ["benchmark uz"]})
        frames = {
            "relaxed.jsonl": relaxed,
            "validation.jsonl": validation,
            "human.jsonl": human,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in frames:
                (root / name).write_text(name)
            benchmark.to_parquet(root / "dev.parquet")
            benchmark.to_parquet(root / "devtest.parquet")

            def read_table(path):
                return frames[Path(path).name].copy()

            with mock.patch.object(ablation, "PROJECT_ROOT", root), mock.patch.object(
                ablation, "_read_table", side_effect=read_table
            ):
                report = ablation.prepare(cfg, "zh_uz", "flores_relaxed_8k")
        self.assertEqual(report["composition"]["rows"], 2)
        self.assertIn("relaxed.jsonl", report["file_sha256"])

    def test_weighted_full_uses_all_rows_and_rebalances_loss_mass(self):
        frame = pd.DataFrame(
            [
                row("zh", "uz", 1, "teacher_kd"),
                row("zh", "uz", 2, "teacher_kd_v3"),
                row("zh", "uz", 3, "teacher_kd_v3"),
                row("zh", "uz", 4, "human_replay"),
                row("uz", "zh", 5, "teacher_kd"),
                row("uz", "zh", 6, "teacher_kd_v3"),
                row("uz", "zh", 7, "human_replay"),
            ]
        )
        frame.loc[frame["training_source"] == "teacher_kd_v3", "weight"] = 0.8
        balanced, report = ablation._balance_teacher_mass(frame, 0.6)
        self.assertEqual(len(balanced), len(frame))
        for ratio in report["achieved_teacher_effective_ratio"].values():
            self.assertAlmostEqual(ratio, 0.6)

    def test_low_lr_variant_overrides_only_exp2_learning_rate(self):
        cfg = config()
        pair = cfg["pairs"][0]
        prepared = [row("zh", "uz", 1), row("uz", "zh", 2)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / pair["exp1_model"]
            source.mkdir(parents=True)
            with mock.patch.object(ablation, "PROJECT_ROOT", root), mock.patch.object(
                ablation, "verify_prepared"
            ), mock.patch.object(ablation, "model_files"), mock.patch.object(
                ablation, "_read_jsonl", return_value=prepared
            ), mock.patch.object(ablation, "train_model", return_value={}) as trainer:
                ablation.train(cfg, "zh_uz", "full_native_lr2e6")
        runtime_config = trainer.call_args.args[7]
        self.assertEqual(runtime_config["training"]["exp2"]["learning_rate"], 2e-6)
        self.assertEqual(cfg["training"]["exp2"]["learning_rate"], 5e-6)

    def test_compare_uses_selected_baseline_and_chrf_thresholds(self):
        cfg = config()
        baseline = {
            "scores": {
                "zh-uz": {"bleu": 4.0, "chrf2": 30.0, "samples": 10},
                "uz-zh": {"bleu": 18.0, "chrf2": 14.0, "samples": 10},
            }
        }
        bidir = {
            "scores": {
                "zh-uz": {"bleu": 4.5, "chrf2": 30.2, "samples": 10},
                "uz-zh": {"bleu": 18.2, "chrf2": 13.5, "samples": 10},
            }
        }
        directional = {
            "scores": {
                "zh-uz": {"bleu": 5.2, "chrf2": 31.2, "samples": 10}
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "results/evaluation/weak_pair_ablation/zh_uz"
            target.mkdir(parents=True)
            (target / "baseline_exp1.json").write_text(json.dumps(baseline))
            (target / "bidir_full.json").write_text(json.dumps(bidir))
            (target / "directional_full__zh_uz.json").write_text(
                json.dumps(directional)
            )
            with mock.patch.object(ablation, "PROJECT_ROOT", root):
                report = ablation.compare(cfg, "zh_uz")
        self.assertEqual(report["baseline_variant"], "baseline_exp1")
        self.assertEqual(
            report["winners"]["zh-uz"]["variant"], "directional_full__zh_uz"
        )
        self.assertEqual(report["winners"]["zh-uz"]["signal"], "meaningful_gain")
        regressed = [
            item
            for item in report["comparisons"]
            if item["variant"] == "bidir_full" and item["direction"] == "uz-zh"
        ][0]
        self.assertEqual(regressed["signal"], "regression")


if __name__ == "__main__":
    unittest.main()
