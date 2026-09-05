from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.supplemental import monolingual_v3


def base_config(root: Path) -> dict:
    return {
        "direction": {
            "pair": "zh_uz",
            "source_lang": "zh",
            "target_lang": "uz",
            "version": "v3",
            "seed": 2026,
        },
        "artifacts": {"pipeline_namespace": "zh_uz_v3"},
        "monolingual": {
            "target_zh_sources": 1,
            "target_uz_sources": 1,
            "collection_target_zh_sources": 1,
            "collection_target_uz_sources": 1,
            "minimum_accepted_per_direction": 1,
            "require_full_targets": True,
            "require_minimum_accepted": True,
            "checkpoint_rows": 1,
            "audit_sample_rows_per_language": 1,
            "min_zh_characters": 4,
            "max_zh_characters": 256,
            "min_uz_characters": 8,
            "max_uz_characters": 400,
            "min_zh_cjk_ratio": 0.6,
            "min_uz_latin_ratio": 0.85,
            "max_digit_ratio": 0.25,
            "source_minimum_group_pass_rate": 0.0,
            "source_minimum_calibration_rows": 1,
            "base_train": str(root / "base_train.jsonl"),
            "base_validation": str(root / "base_validation.jsonl"),
            "deduplicate_against": [str(root / "base_train.jsonl")],
        },
        "monolingual_sources": [],
        "distillation": {"teacher_high_weight": 1.0, "teacher_medium_weight": 0.8},
    }


class ZhUzV3PipelineTests(unittest.TestCase):
    def test_quality_contract_normalizes_chinese_and_uzbek(self) -> None:
        settings = {
            "min_zh_characters": 4,
            "max_zh_characters": 256,
            "min_uz_characters": 8,
            "max_uz_characters": 400,
            "min_zh_cjk_ratio": 0.6,
            "min_uz_latin_ratio": 0.85,
            "max_digit_ratio": 0.25,
        }
        reason, text = monolingual_v3.quality_reason("zh", "這是一個標準中文句子。", settings)
        self.assertIsNone(reason)
        self.assertEqual(text, "这是一个标准中文句子。")
        reason, _ = monolingual_v3.quality_reason("zh", "我哋今日去学校。", settings)
        self.assertEqual(reason, "NON_MANDARIN_CHINESE")
        reason, text = monolingual_v3.quality_reason("uz", "Ўзбекистон жуда гўзал мамлакат.", settings)
        self.assertIsNone(reason)
        self.assertNotRegex(text, r"[\u0400-\u052f]")

    def test_collect_isolated_sources_and_excludes_existing_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "pair_id": "base",
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": "这是已经存在的训练句子。",
                "tgt_text": "Bu mavjud tarjima.",
            }
            (root / "base_train.jsonl").write_text(json.dumps(base, ensure_ascii=False) + "\n", encoding="utf-8")
            (root / "base_validation.jsonl").write_text("", encoding="utf-8")
            incoming = root / "incoming.jsonl"
            incoming.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in [
                        {"id": "1", "text": "这是已经存在的训练句子。"},
                        {"id": "2", "text": "这是全新的标准中文句子。"},
                    ]
                ),
                encoding="utf-8",
            )
            uz = root / "uz.jsonl"
            uz.write_text(json.dumps({"id": "u1", "text": "Bu mutlaqo yangi o'zbekcha gapdir."}) + "\n", encoding="utf-8")
            config = base_config(root)
            config["monolingual_sources"] = [
                {"id": "local_zh", "kind": "local_jsonl", "language": "zh", "path": str(incoming), "text_field": "text", "split_documents": False, "max_rows": 1},
                {"id": "local_uz", "kind": "local_jsonl", "language": "uz", "path": str(uz), "text_field": "text", "split_documents": False, "max_rows": 1},
            ]
            with (
                patch.object(monolingual_v3, "PROJECT_ROOT", root),
                patch.object(monolingual_v3, "benchmark_sets", return_value={"pairs": set(), "source": set(), "target": set()}),
            ):
                report = monolingual_v3.collect(config)
            rows = monolingual_v3._read_jsonl(
                root / "data/pipeline_v2/zh_uz_v3/monolingual_candidates.jsonl"
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["src_lang"] for row in rows}, {"zh", "uz"})
            self.assertNotIn("这是已经存在的训练句子。", {row["src_text"] for row in rows})
            self.assertTrue(report["eligible_for_source_review"])

    def test_finalize_keeps_only_audited_teacher_rows_and_merges_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {
                "pair_id": "base",
                "src_lang": "zh",
                "tgt_lang": "uz",
                "src_text": "基础句子。",
                "tgt_text": "Asosiy gap.",
                "training_source": "human_replay",
            }
            (root / "base_train.jsonl").write_text(json.dumps(base, ensure_ascii=False) + "\n", encoding="utf-8")
            (root / "base_validation.jsonl").write_text(json.dumps(base, ensure_ascii=False) + "\n", encoding="utf-8")
            pipeline = root / "data/pipeline_v2/zh_uz_v3"
            pipeline.mkdir(parents=True)
            pd.DataFrame(
                [
                    {"pair_id": "z", "src_lang": "zh", "tgt_lang": "uz", "src_text": "这是新的中文来源。", "teacher_text": "Bu yangi xitoycha manba.", "judge_parse_ok": True, "judge_label": "PASS", "teacher_usefulness": "HIGH", "source_corpus": "test"},
                    {"pair_id": "u", "src_lang": "uz", "tgt_lang": "zh", "src_text": "Bu yangi o'zbekcha manba.", "teacher_text": "这是新的乌兹别克语来源。", "judge_parse_ok": True, "judge_label": "PASS", "teacher_usefulness": "MEDIUM", "source_corpus": "test"},
                    {"pair_id": "bad", "src_lang": "zh", "tgt_lang": "uz", "src_text": "这是失败的来源。", "teacher_text": "Noto'g'ri.", "judge_parse_ok": True, "judge_label": "FAIL", "teacher_usefulness": "REJECT", "source_corpus": "test"},
                ]
            ).to_parquet(pipeline / "teacher_judged.parquet", index=False)
            config = base_config(root)
            config["monolingual_sources"] = [{"id": "test", "kind": "local_jsonl", "language": "zh", "max_rows": 1}]
            with (
                patch.object(monolingual_v3, "PROJECT_ROOT", root),
                patch.object(monolingual_v3, "benchmark_sets", return_value={"pairs": set(), "source": set(), "target": set()}),
            ):
                report = monolingual_v3.finalize(config)
            output = monolingual_v3._read_jsonl(root / "data/distillation/zh_uz/v3/train.jsonl")
            self.assertEqual(len(output), 3)
            self.assertEqual(report["teacher_rows_by_direction"], {"zh-uz": 1, "uz-zh": 1})
            self.assertEqual({row["training_source"] for row in output}, {"human_replay", "teacher_kd_v3"})

    def test_select_sources_requires_qwen_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config(root)
            config["monolingual_sources"] = [
                {"id": "test", "kind": "local_jsonl", "language": "zh", "max_rows": 1}
            ]
            pipeline = root / "data/pipeline_v2/zh_uz_v3"
            pipeline.mkdir(parents=True)
            common = {
                "reference_text": "",
                "source_corpus": "test",
                "source_license": "test",
                "source_record": "1",
                "judge_parse_ok": True,
            }
            records = [
                {**common, "pair_id": "zh", "src_lang": "zh", "tgt_lang": "uz", "src_text": "这是自然完整的中文句子。", "judge_label": "PASS"},
                {**common, "pair_id": "uz", "src_lang": "uz", "tgt_lang": "zh", "src_text": "Bu tabiiy va to'liq o'zbekcha gap.", "judge_label": "PASS"},
                {**common, "pair_id": "bad", "src_lang": "zh", "tgt_lang": "uz", "src_text": "关键词 列表 碎片", "judge_label": "FAIL"},
            ]
            pd.DataFrame(records).to_parquet(
                pipeline / "source_judge_calibration.parquet", index=False
            )
            monolingual_v3._write_jsonl(
                [
                    {
                        key: row[key]
                        for key in (
                            "pair_id",
                            "src_lang",
                            "tgt_lang",
                            "src_text",
                            "reference_text",
                            "source_corpus",
                            "source_license",
                            "source_record",
                        )
                    }
                    for row in records
                ],
                pipeline / "monolingual_candidates.jsonl",
            )
            with patch.object(monolingual_v3, "PROJECT_ROOT", root):
                report = monolingual_v3.select_sources(config)
            rows = monolingual_v3._read_jsonl(pipeline / "kd_candidates.jsonl")
            self.assertEqual(len(rows), 2)
            self.assertTrue(report["eligible_for_teacher_generation"])
            self.assertEqual(report["review_mode"], "stratified_calibration")

    def test_select_sources_prefers_complete_full_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = base_config(root)
            config["monolingual_sources"] = [
                {"id": "test", "kind": "local_jsonl", "language": "zh", "max_rows": 1}
            ]
            pipeline = root / "data/pipeline_v2/zh_uz_v3"
            pipeline.mkdir(parents=True)
            records = [
                {"pair_id": "zh", "src_lang": "zh", "tgt_lang": "uz", "src_text": "这是自然完整的中文句子。", "reference_text": "", "source_corpus": "test", "source_license": "test", "source_record": "1"},
                {"pair_id": "uz", "src_lang": "uz", "tgt_lang": "zh", "src_text": "Bu tabiiy va to'liq o'zbekcha gap.", "reference_text": "", "source_corpus": "test", "source_license": "test", "source_record": "2"},
                {"pair_id": "bad", "src_lang": "zh", "tgt_lang": "uz", "src_text": "关键词 列表 碎片", "reference_text": "", "source_corpus": "test", "source_license": "test", "source_record": "3"},
            ]
            monolingual_v3._write_jsonl(records, pipeline / "monolingual_candidates.jsonl")
            judged = pd.DataFrame(records)
            judged["judge_parse_ok"] = True
            judged["judge_label"] = ["PASS", "PASS", "FAIL"]
            judged.to_parquet(pipeline / "source_judged.parquet", index=False)
            with patch.object(monolingual_v3, "PROJECT_ROOT", root):
                report = monolingual_v3.select_sources(config)
            rows = monolingual_v3._read_jsonl(pipeline / "kd_candidates.jsonl")
            self.assertEqual({row["pair_id"] for row in rows}, {"zh", "uz"})
            self.assertEqual(report["review_mode"], "full")
            self.assertTrue(report["full_review_coverage"])


if __name__ == "__main__":
    unittest.main()
