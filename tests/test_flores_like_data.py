from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.pipeline_v3 import flores_like_data as flow


class FloresLikeDataTests(unittest.TestCase):
    def test_quotas_sum_to_requested_total(self):
        quotas = flow._quotas(["a", "a", "b", "c"], 11)
        self.assertEqual(sum(quotas.values()), 11)
        self.assertGreater(quotas["a"], quotas["b"])

    def test_near_duplicate_index_detects_close_text(self):
        index = flow.NearDuplicateIndex(
            ["This is a protected benchmark sentence."], size=5, threshold=0.8
        )
        self.assertTrue(index.matches("This is a protected benchmark sentence!"))
        self.assertFalse(index.matches("Completely unrelated words live here."))

    def test_wikipedia_extension_collects_both_languages_and_checkpoints(self):
        config = {
            "inputs": {
                "candidate_pool": "base.parquet",
                "candidate_extensions": ["extension.parquet"],
            },
            "outputs": {"report_root": "reports"},
            "extension": {
                "filter_config": "filter.toml",
                "checkpoint_rows": 1,
                "require_full_targets": True,
                "sources": [
                    {
                        "id": "wikipedia_flores_like_zh",
                        "kind": "hf_dataset",
                        "language": "zh",
                        "text_field": "text",
                        "split_documents": True,
                        "target_rows": 1,
                        "max_documents": 10,
                        "license": "cc-by-sa",
                    },
                    {
                        "id": "wikipedia_flores_like_uz",
                        "kind": "hf_dataset",
                        "language": "uz",
                        "text_field": "text",
                        "split_documents": True,
                        "target_rows": 1,
                        "max_documents": 10,
                        "license": "cc-by-sa",
                    },
                ],
            },
        }
        base = pd.DataFrame(
            [
                {"pair_id": "old-zh", "src_lang": "zh", "src_text": "旧句子"},
                {"pair_id": "old-uz", "src_lang": "uz", "src_text": "Eski gap."},
            ]
        )

        def records(source):
            yield {
                "id": source["id"],
                "text": "新的百科句子。"
                if source["language"] == "zh"
                else "Yangi ensiklopediya jumlasi.",
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base.to_parquet(root / "base.parquet")
            with mock.patch.object(flow, "PROJECT_ROOT", root), mock.patch.object(
                flow, "validate"
            ), mock.patch.object(
                flow, "load_config", return_value={"monolingual": {}}
            ), mock.patch.object(
                flow, "_excluded_source_keys", return_value={"zh": set(), "uz": set()}
            ), mock.patch.object(
                flow, "_iter_records", side_effect=records
            ), mock.patch.object(
                flow, "quality_reason", side_effect=lambda language, text, settings: (None, text)
            ):
                report = flow.collect_extension(config)
            collected = pd.read_parquet(root / "extension.parquet")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["by_language"], {"uz": 1, "zh": 1})
        self.assertEqual(set(collected["src_lang"]), {"zh", "uz"})

    def test_assemble_hits_six_effective_mass_cells(self):
        def row(src, tgt, index, origin, weight=1.0):
            return {
                "src_lang": src,
                "tgt_lang": tgt,
                "src_text": f"{src} source sentence {index}",
                "tgt_text": f"{tgt} target sentence {index}",
                "training_source": origin,
                "weight": weight,
            }

        existing = pd.DataFrame(
            [
                row("zh", "uz", 1, "human_replay"),
                row("zh", "uz", 2, "teacher_kd"),
                row("uz", "zh", 3, "human_replay"),
                row("uz", "zh", 4, "teacher_kd"),
            ]
        )
        new = pd.DataFrame(
            [
                {**row("zh", "uz", 5, "teacher"), "teacher_text": "uz target sentence 5", "judge_parse_ok": True, "judge_label": "PASS", "teacher_usefulness": "HIGH"},
                {**row("uz", "zh", 6, "teacher"), "teacher_text": "zh target sentence 6", "judge_parse_ok": True, "judge_label": "PASS", "teacher_usefulness": "MEDIUM"},
            ]
        )
        validation = pd.DataFrame([row("zh", "uz", 20, "human_parallel"), row("uz", "zh", 21, "human_parallel")])
        benchmark = pd.DataFrame({"zh": ["基准句子内容"], "uz": ["himoyalangan mezon jumlasi"]})
        config = {
            "direction": {"pair": "zh_uz", "seed": 2026},
            "inputs": {"existing_train": "existing.jsonl", "validation": "validation.jsonl", "flores_dev": "dev.parquet", "flores_devtest": "devtest.parquet"},
            "outputs": {"root": "out", "report_root": "reports", "teacher_pipeline_root": "teacher"},
            "selection": {
                "minimum_teacher_rows_per_direction": 1,
                "shingle_size": 5,
                "near_duplicate_jaccard": 0.8,
                "source_caps": {"other": 1.0},
            },
            "extension": {
                "sources": [
                    {"language": "zh"},
                    {"language": "uz"},
                ]
            },
            "mixture": {"human": 0.4, "existing_kd": 0.3, "flores_like_kd": 0.3, "direction_share": 0.5},
            "distillation": {"teacher_high_weight": 1.0, "teacher_medium_weight": 0.8},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("existing.jsonl", "validation.jsonl"):
                (root / name).write_text("placeholder", encoding="utf-8")
            benchmark.to_parquet(root / "dev.parquet")
            benchmark.to_parquet(root / "devtest.parquet")
            (root / "teacher").mkdir()
            new.to_parquet(root / "teacher/teacher_judged.parquet")

            def read_table(path):
                known = {"existing.jsonl": existing, "validation.jsonl": validation}
                if Path(path).name in known:
                    return known[Path(path).name].copy()
                return pd.read_parquet(path)

            with mock.patch.object(flow, "PROJECT_ROOT", root), mock.patch.object(flow, "_read_table", side_effect=read_table):
                report = flow.assemble(config)
        total = report["rows"]
        for direction in flow.DIRECTIONS:
            self.assertAlmostEqual(report["effective_mass"][f"{direction}|human"], total * 0.20)
            self.assertAlmostEqual(report["effective_mass"][f"{direction}|existing_kd"], total * 0.15)
            self.assertAlmostEqual(report["effective_mass"][f"{direction}|flores_like_kd"], total * 0.15)


if __name__ == "__main__":
    unittest.main()
