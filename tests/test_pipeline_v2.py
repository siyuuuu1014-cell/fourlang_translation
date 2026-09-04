from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.pipeline_v2.common import commercial_candidates, load_config
from scripts.pipeline_v2.data_flow import (
    pair_hash,
    rule_assessment,
    rule_reasons,
    select_split_pool,
    stratified_auto_accept_audit,
)
from scripts.pipeline_v2.qwen_judge import judge_id, parse_result, second_review_mask
from scripts.pipeline_v2 import seq2seq_flow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("FOURLANG_MODEL_ROOT", str(PROJECT_ROOT / "models"))
        os.environ.setdefault(
            "FOURLANG_QWEN_MODEL_PATH", str(PROJECT_ROOT / "models/Qwen3-8B")
        )
        cls.config = load_config(PROJECT_ROOT / "configs/directions/en_ru.toml")

    def test_only_commercial_candidates_are_eligible(self) -> None:
        for role in ("student", "teacher"):
            candidates = commercial_candidates(self.config, role)
            self.assertTrue(candidates)
            self.assertTrue(
                all(item["commercial_allowed"] is True for item in candidates)
            )
            self.assertFalse(any("nllb" in item["id"].lower() for item in candidates))

    def test_rule_filter_rejects_markup_and_length_mismatch(self) -> None:
        settings = self.config["data"]
        self.assertIn(
            "markup_or_url",
            rule_reasons("hello https://example.com", "привет", settings),
        )
        self.assertIn("critical_length_ratio", rule_reasons("a" * 100, "два", settings))
        self.assertEqual(
            rule_reasons("A useful sentence.", "Полезное предложение.", settings), []
        )

    def test_rules_use_three_explicit_routes(self) -> None:
        settings = self.config["data"]
        self.assertEqual(rule_assessment("", "текст", settings)[2], "HARD_REJECT")
        self.assertEqual(
            rule_assessment("The value is 2.", "Значение равно 3.", settings)[2],
            "NEEDS_QWEN",
        )
        self.assertEqual(
            rule_assessment("A useful sentence.", "Полезное предложение.", settings)[2],
            "AUTO_ACCEPT",
        )

    def test_judge_parser_is_strict_and_directional(self) -> None:
        parsed = parse_result(
            '{"label":"PASS","semantic_consistent":true,"reason":"equivalent"}'
        )
        self.assertTrue(parsed["judge_parse_ok"])
        self.assertEqual(parsed["judge_label"], "PASS")
        self.assertEqual(parsed["judge_reason"], "equivalent")
        self.assertFalse(parse_result("not json")["judge_parse_ok"])
        forward = judge_id({"pair_id": "p", "src_lang": "en", "tgt_lang": "ru"})
        reverse = judge_id({"pair_id": "p", "src_lang": "ru", "tgt_lang": "en"})
        self.assertNotEqual(forward, reverse)

    def test_pair_hash_is_stable(self) -> None:
        self.assertEqual(pair_hash("a", "б"), pair_hash("a", "б"))
        self.assertNotEqual(pair_hash("a", "б"), pair_hash("б", "a"))

    def test_auto_accept_audit_is_exact_and_reproducible(self) -> None:
        frame = pd.DataFrame(
            {
                "pair_id": [f"pair-{index:04d}" for index in range(800)],
                "source_text": ["x" * (index % 97 + 2) for index in range(800)],
                "target_text": ["я" * (index % 89 + 2) for index in range(800)],
            }
        )
        first = stratified_auto_accept_audit(frame, size=500, seed=2026)
        second = stratified_auto_accept_audit(frame, size=500, seed=2026)
        self.assertEqual(len(first), 500)
        self.assertEqual(first["pair_id"].tolist(), second["pair_id"].tolist())

    def test_teacher_judge_requires_explicit_usefulness(self) -> None:
        missing = parse_result(
            '{"label":"PASS","semantic_consistent":true}', teacher=True
        )
        accepted = parse_result(
            '{"label":"PASS","semantic_consistent":true,'
            '"teacher_usefulness":"HIGH"}',
            teacher=True,
        )
        self.assertFalse(missing["judge_parse_ok"])
        self.assertTrue(accepted["judge_parse_ok"])
        self.assertEqual(accepted["teacher_usefulness"], "HIGH")

    def test_qwen_judge_restores_original_batch_size(self) -> None:
        self.assertEqual(self.config["judge"]["batch_size"], 32)
        self.assertEqual(self.config["judge"]["max_input_tokens"], 1536)

    def test_parse_failures_are_sent_to_independent_second_review(self) -> None:
        frame = pd.DataFrame(
            {
                "judge_parse_ok": [False, True, True, True],
                "judge_label": ["UNCERTAIN", "FAIL", "UNCERTAIN", "PASS"],
            }
        )
        self.assertEqual(second_review_mask(frame).tolist(), [True, True, True, False])

    def test_exp1_pool_has_explicit_size_and_quality_tiers(self) -> None:
        settings = self.config["data"]
        required = (
            settings["train_pairs"]
            + settings["validation_pairs"]
            + settings["test_pairs"]
        )
        frame = pd.DataFrame(
            {
                "pair_id": [f"pair-{index}" for index in range(required + 20)],
                "quality_tier": ["GOLD"] * 10
                + ["SILVER"] * (required + 10),
            }
        )
        selected, report = select_split_pool(frame, settings, seed=2026)
        self.assertEqual(len(selected), required)
        self.assertEqual(set(selected["quality_tier"]), {"GOLD", "SILVER"})
        self.assertEqual(report["configured_train_pairs"], 61216)

    def test_training_preserves_effective_batch_and_experiment_schedule(self) -> None:
        training = self.config["training"]
        self.assertEqual(
            training["batch_size"] * training["gradient_accumulation_steps"], 32
        )
        self.assertEqual(training["exp1"]["epochs"], 3)
        self.assertEqual(training["exp1"]["learning_rate"], 3e-5)
        self.assertEqual(training["exp2"]["epochs"], 2)
        self.assertEqual(training["exp2"]["learning_rate"], 5e-6)
        self.assertEqual(training["batch_size"], 16)
        self.assertEqual(training["gradient_accumulation_steps"], 2)
        self.assertFalse(training["gradient_checkpointing"])

    def test_small100_is_the_explicit_measured_student_choice(self) -> None:
        self.assertEqual(self.config["selection"]["student_override"], "small100")

    def test_teacher_generation_resumes_from_atomic_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "data/pipeline_v2/en_ru/kd_candidates.jsonl"
            candidates.parent.mkdir(parents=True)
            rows = [
                {
                    "pair_id": f"pair-{index}",
                    "src_lang": source,
                    "tgt_lang": target,
                    "src_text": f"source-{index}",
                    "reference_text": f"reference-{index}",
                }
                for index, (source, target) in enumerate(
                    [("en", "ru")] * 3 + [("ru", "en")] * 2
                )
            ]
            candidates.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            selection_path = root / "results/model_selection/en_ru/selected_teacher.json"
            selection_path.parent.mkdir(parents=True)
            selection_path.write_text(
                json.dumps(
                    {
                        "directions": {
                            "en-ru": {"candidate": {"id": "teacher", "family": "fake"}},
                            "ru-en": {"candidate": {"id": "teacher", "family": "fake"}},
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "direction": {
                    "pair": "en_ru",
                    "source_lang": "en",
                    "target_lang": "ru",
                    "version": "v1",
                },
                "distillation": {"teacher_checkpoint_rows": 2},
                "training": {"max_source_length": 32},
                "deployment": {"num_beams": 1, "max_new_tokens": 16},
            }

            def fake_translate(*args, **kwargs):
                texts = args[5]
                return [f"translated:{text}" for text in texts]

            with (
                patch.object(seq2seq_flow, "PROJECT_ROOT", root),
                patch.object(seq2seq_flow, "load_model", return_value=(object(), object())),
                patch.object(seq2seq_flow, "translate", side_effect=fake_translate),
            ):
                seq2seq_flow.generate_teacher(config)

            output = root / "data/pipeline_v2/en_ru/teacher_generated.parquet"
            self.assertEqual(len(pd.read_parquet(output)), len(rows))
            shards = sorted(
                (root / "data/pipeline_v2/en_ru/teacher_generation_checkpoints").glob(
                    "*.parquet"
                )
            )
            shards[0].unlink()
            output.unlink()
            with (
                patch.object(seq2seq_flow, "PROJECT_ROOT", root),
                patch.object(seq2seq_flow, "load_model", return_value=(object(), object())),
                patch.object(
                    seq2seq_flow, "translate", side_effect=fake_translate
                ) as resumed_translate,
            ):
                seq2seq_flow.generate_teacher(config)
            self.assertEqual(resumed_translate.call_count, 1)
            resumed = pd.read_parquet(output)
            self.assertEqual(len(resumed), len(rows))
            self.assertEqual(
                resumed["teacher_text"].tolist(),
                [f"translated:source-{index}" for index in range(len(rows))],
            )
            output.unlink()
            with (
                patch.object(seq2seq_flow, "PROJECT_ROOT", root),
                patch.object(
                    seq2seq_flow,
                    "load_model",
                    side_effect=AssertionError("completed shards must skip model loading"),
                ),
            ):
                seq2seq_flow.generate_teacher(config)
            self.assertEqual(len(pd.read_parquet(output)), len(rows))


if __name__ == "__main__":
    unittest.main()
