from __future__ import annotations

import unittest

import pandas as pd

from scripts.pipeline_v3 import build_deepseek_partial_zh_uz as builder


class BuildDeepSeekPartialZhUzTests(unittest.TestCase):
    def test_build_training_rows_filters_direction_failures_and_duplicates(self):
        baseline = pd.DataFrame(
            [
                {
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "已有句子。",
                    "tgt_text": "Mavjud gap.",
                    "weight": 1.0,
                    "training_source": "human_replay",
                },
                {
                    "src_lang": "uz",
                    "tgt_lang": "zh",
                    "src_text": "Teskari gap.",
                    "tgt_text": "反向句子。",
                    "weight": 1.0,
                    "training_source": "human_replay",
                },
            ]
        )
        audited = pd.DataFrame(
            [
                {
                    "pair_id": "new",
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "新句子。",
                    "teacher_text": "Yangi gap.",
                    "combined_usable": True,
                },
                {
                    "pair_id": "duplicate",
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "已有句子。",
                    "teacher_text": "Takror gap.",
                    "combined_usable": True,
                },
                {
                    "pair_id": "failed",
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "失败句子。",
                    "teacher_text": "Noto'g'ri gap.",
                    "combined_usable": False,
                },
                {
                    "pair_id": "reverse",
                    "src_lang": "uz",
                    "tgt_lang": "zh",
                    "src_text": "Yangi teskari gap.",
                    "teacher_text": "新的反向句子。",
                    "combined_usable": True,
                },
            ]
        )

        combined, accepted, rejected = builder.build_training_rows(
            baseline, audited, deepseek_weight=1.5
        )

        self.assertEqual(len(combined), 3)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source_id"], "new")
        self.assertEqual(accepted[0]["weight"], 1.5)
        self.assertEqual(rejected, {"DUPLICATE_SOURCE": 1})


if __name__ == "__main__":
    unittest.main()
