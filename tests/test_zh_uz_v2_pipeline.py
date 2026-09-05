from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.pipeline_v2 import data_flow, qwen_judge
from scripts.pipeline_v2.common import pipeline_namespace
from scripts.supplemental import import_pair_v2


class ZhUzV2PipelineTests(unittest.TestCase):
    def test_namespace_is_separate_from_semantic_pair(self) -> None:
        config = {
            "direction": {
                "pair": "zh_uz",
                "source_lang": "zh",
                "target_lang": "uz",
                "version": "v2",
            },
            "artifacts": {"pipeline_namespace": "zh_uz_v2"},
        }
        self.assertEqual(pipeline_namespace(config), "zh_uz_v2")

    def test_language_contract_rejects_cantonese_and_cyrillic_uzbek(self) -> None:
        self.assertEqual(
            import_pair_v2.language_contract_reason(
                "我哋今日去学校。", "Bugun maktabga boramiz."
            ),
            "NON_MANDARIN_CHINESE",
        )
        self.assertEqual(
            import_pair_v2.language_contract_reason("我们今天去学校。", "Бугун"),
            "UZ_NOT_LATIN",
        )
        self.assertIsNone(
            import_pair_v2.language_contract_reason(
                "我们今天去学校。", "Bugun maktabga boramiz."
            )
        )

    def test_import_isolated_v2_filters_existing_training_and_cantonese(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incoming = root / "incoming.parquet"
            existing = root / "approved_v1.parquet"
            pd.DataFrame(
                [
                    {
                        "pair_id": "old",
                        "source_text": "这是已经存在的句子。",
                        "target_text": "Bu allaqachon mavjud gap.",
                    }
                ]
            ).to_parquet(existing, index=False)
            pd.DataFrame(
                [
                    {
                        "pair_id": "a",
                        "pair": "zh_uz",
                        "source_lang": "zh",
                        "target_lang": "uz",
                        "source_text": "我哋今日去学校。",
                        "target_text": "Bugun maktabga boramiz.",
                    },
                    {
                        "pair_id": "b",
                        "pair": "zh_uz",
                        "source_lang": "zh",
                        "target_lang": "uz",
                        "source_text": "这是已经存在的句子。",
                        "target_text": "Bu yangi tarjima.",
                    },
                    {
                        "pair_id": "c",
                        "pair": "zh_uz",
                        "source_lang": "zh",
                        "target_lang": "uz",
                        "source_text": "这是新的标准普通话句子。",
                        "target_text": "Bu yangi standart gap.",
                    },
                ]
            ).to_parquet(incoming, index=False)
            config = {
                "direction": {
                    "pair": "zh_uz",
                    "source_lang": "zh",
                    "target_lang": "uz",
                    "version": "v2",
                },
                "artifacts": {"pipeline_namespace": "zh_uz_v2"},
                "supplemental": {
                    "input_candidates": str(incoming),
                    "existing_training": [str(existing)],
                    "rejected_sample_size": 10,
                },
            }
            with patch.object(import_pair_v2, "PROJECT_ROOT", root):
                report = import_pair_v2.import_candidates(config)
            output = pd.read_parquet(
                root / "data/pipeline_v2/zh_uz_v2/candidates.parquet"
            )
            self.assertEqual(len(output), 1)
            self.assertEqual(output.iloc[0]["source_text"], "这是新的标准普通话句子。")
            self.assertEqual(report["rejections"]["NON_MANDARIN_CHINESE"], 1)
            self.assertEqual(report["rejections"]["EXISTING_TRAINING_OVERLAP"], 1)
            self.assertTrue(report["existing_training_overlap_checked"])

    def test_force_full_review_routes_every_non_rejected_row_to_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.parquet"
            routed = root / "rule_routed.parquet"
            review = root / "human_review_input.parquet"
            reports = root / "reports"
            pd.DataFrame(
                [
                    {
                        "pair_id": "good",
                        "source_text": "这是测试句子。",
                        "target_text": "Bu sinov gapidir.",
                    },
                    {
                        "pair_id": "ambiguous",
                        "source_text": "版本是2026。",
                        "target_text": "Versiya 2025.",
                    },
                    {"pair_id": "bad", "source_text": "", "target_text": "Matn"},
                ]
            ).to_parquet(candidates, index=False)
            config = {
                "direction": {
                    "pair": "zh_uz",
                    "source_lang": "zh",
                    "target_lang": "uz",
                    "version": "v2",
                    "seed": 2026,
                },
                "data": {"max_characters": 1000, "max_length_ratio": 6.0},
                "judge": {
                    "force_full_human_review": True,
                    "auto_accept_audit_pairs": 0,
                },
            }
            path_map = {
                "candidates": candidates,
                "routed": routed,
                "human_review": review,
                "reports": reports,
            }
            with (
                patch.object(data_flow, "paths", return_value=path_map),
                patch.object(
                    data_flow,
                    "benchmark_sets",
                    return_value={"pairs": set(), "source": set(), "target": set()},
                ),
            ):
                data_flow.rules(config)
            judged = pd.read_parquet(review)
            self.assertEqual(set(judged["pair_id"]), {"good", "ambiguous"})
            self.assertEqual(set(judged["qwen_review_type"]), {"FULL_REVIEW"})
            report = json.loads((reports / "rules.json").read_text(encoding="utf-8"))
            self.assertTrue(report["force_full_human_review"])

    def test_qwen_prompt_enforces_script_and_register_contract(self) -> None:
        rendered = qwen_judge.prompt("human", "zh", "uz", "你好", "Salom")
        self.assertIn("standard Simplified Mandarin", rendered)
        self.assertIn("Uzbek must use the Latin script", rendered)

    def test_qwen_source_prompt_rejects_fragments_and_wrong_language(self) -> None:
        rendered = qwen_judge.prompt("source", "uz", "zh", "English text", "")
        self.assertIn("Latin-script Uzbek", rendered)
        self.assertIn("keyword or entity lists", rendered)
        self.assertIn("another language", rendered)


if __name__ == "__main__":
    unittest.main()
