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
                {
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "旧教师句子。",
                    "tgt_text": "Eski o'qituvchi gapi.",
                    "weight": 0.8,
                    "training_source": "teacher_kd_v3",
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
                    "pair_id": "replacement",
                    "src_lang": "zh",
                    "tgt_lang": "uz",
                    "src_text": "旧教师句子。",
                    "teacher_text": "Yaxshilangan o'qituvchi gapi.",
                    "combined_usable": True,
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

        self.assertEqual(len(combined), 4)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(
            {row["source_id"] for row in accepted}, {"new", "replacement"}
        )
        self.assertTrue(all(row["weight"] == 1.5 for row in accepted))
        self.assertEqual(
            rejected,
            {
                "BASELINE_TEACHER_ROWS_REMOVED": 1,
                "BASELINE_TEACHER_SOURCE_REPLACED": 1,
                "DUPLICATE_HUMAN_SOURCE_REJECTED": 1,
                "NEW_SOURCE_ADDED": 1,
            },
        )
        self.assertNotIn(
            "Eski o'qituvchi gapi.", {row["tgt_text"] for row in combined}
        )


if __name__ == "__main__":
    unittest.main()
