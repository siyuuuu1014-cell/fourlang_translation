from __future__ import annotations

import os
import unittest
from pathlib import Path

import pandas as pd

from scripts.pipeline_v2.common import commercial_candidates, load_config
from scripts.pipeline_v2.data_flow import (
    pair_hash,
    rule_assessment,
    rule_reasons,
    stratified_auto_accept_audit,
)
from scripts.pipeline_v2.qwen_judge import judge_id, parse_result


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


if __name__ == "__main__":
    unittest.main()
