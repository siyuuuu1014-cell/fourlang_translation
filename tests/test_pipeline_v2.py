from __future__ import annotations

import os
import unittest
from pathlib import Path

from scripts.pipeline_v2.common import commercial_candidates, load_config
from scripts.pipeline_v2.data_flow import pair_hash, rule_reasons
from scripts.pipeline_v2.qwen_judge import judge_id, parse_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("FOURLANG_MODEL_ROOT", str(PROJECT_ROOT / "models"))
        os.environ.setdefault("FOURLANG_QWEN_MODEL_PATH", str(PROJECT_ROOT / "models/Qwen3-8B"))
        cls.config = load_config(PROJECT_ROOT / "configs/directions/en_ru.toml")

    def test_only_commercial_candidates_are_eligible(self) -> None:
        for role in ("student", "teacher"):
            candidates = commercial_candidates(self.config, role)
            self.assertTrue(candidates)
            self.assertTrue(all(item["commercial_allowed"] is True for item in candidates))
            self.assertFalse(any("nllb" in item["id"].lower() for item in candidates))

    def test_rule_filter_rejects_markup_and_length_mismatch(self) -> None:
        settings = self.config["data"]
        self.assertIn("markup_or_url", rule_reasons("hello https://example.com", "привет", settings))
        self.assertIn("length_ratio", rule_reasons("a" * 100, "два", settings))
        self.assertEqual(rule_reasons("A useful sentence.", "Полезное предложение.", settings), [])

    def test_judge_parser_is_strict_and_directional(self) -> None:
        ok, score, reason = parse_result('{"score": 0.91, "reason": "equivalent"}')
        self.assertTrue(ok)
        self.assertEqual(score, 0.91)
        self.assertEqual(reason, "equivalent")
        self.assertFalse(parse_result("not json")[0])
        forward = judge_id({"pair_id": "p", "src_lang": "en", "tgt_lang": "ru"})
        reverse = judge_id({"pair_id": "p", "src_lang": "ru", "tgt_lang": "en"})
        self.assertNotEqual(forward, reverse)

    def test_pair_hash_is_stable(self) -> None:
        self.assertEqual(pair_hash("a", "б"), pair_hash("a", "б"))
        self.assertNotEqual(pair_hash("a", "б"), pair_hash("б", "a"))


if __name__ == "__main__":
    unittest.main()
