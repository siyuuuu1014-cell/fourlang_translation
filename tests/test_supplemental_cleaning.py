from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.supplemental.clean_pair_subsets import (
    PAIR_SPECS,
    _canonical_pair,
    clean_file,
    deterministic_rejection_reason,
)


class SupplementalCleaningTest(unittest.TestCase):
    def test_canonical_pair_reverses_observed_reverse_direction(self) -> None:
        spec = PAIR_SPECS["uz-zh.jsonl"]
        self.assertEqual(
            _canonical_pair(spec, "uz", "zh", "salom", "你好"),
            ("你好", "salom"),
        )

    def test_deterministic_filter_rejects_number_mismatch(self) -> None:
        self.assertEqual(
            deterministic_rejection_reason(
                "en", "ru", "Version 2026 is ready.", "Версия 2025 готова."
            ),
            "NUMBER_MISMATCH",
        )

    def test_clean_file_normalizes_and_keeps_outputs_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "uz-zh.jsonl"
            rows = [
                {
                    "instruction": "请将下面的乌兹别克文翻译成简体中文：",
                    "input": "Бу синов гапидир.",
                    "output": "這是一個測試句子。",
                },
                {
                    "instruction": "Quyidagi xitoycha matnni o'zbek tiliga (lotin yozuvida) tarjima qiling:",
                    "input": "這是一個測試句子。",
                    "output": "Бу синов гапидир.",
                },
                {
                    "instruction": "请将下面的乌兹别克文翻译成简体中文：",
                    "input": "2026-yilda chiqarildi.",
                    "output": "于2025年正式发布。",
                },
            ]
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "isolated"
            report = clean_file(
                source,
                output,
                root / "missing-benchmarks",
                rejected_sample_size=10,
            )
            cleaned = pd.read_parquet(
                output / "zh_uz" / "cleaned_candidates.parquet"
            )
            self.assertEqual(len(cleaned), 1)
            self.assertEqual(cleaned.iloc[0]["source_text"], "这是一个测试句子。")
            self.assertEqual(cleaned.iloc[0]["target_text"], "Bu sinov gapidir.")
            self.assertEqual(cleaned.iloc[0]["candidate_status"], "CLEANED_UNREVIEWED")
            self.assertEqual(report["rejections"]["EXACT_DUPLICATE"], 1)
            self.assertEqual(report["rejections"]["NUMBER_MISMATCH"], 1)
            self.assertFalse(report["eligible_for_training"])
            self.assertTrue(report["requires_protected_recheck"])


if __name__ == "__main__":
    unittest.main()
