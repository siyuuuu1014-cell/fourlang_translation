from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.pipeline.run_direction import DirectionPipeline
from scripts.pipeline_v2.common import load_config
from scripts.pipeline_v2.data_flow import rule_assessment
from scripts.pipeline_v2.opus_parallel import read_parallel_archive
from scripts.pipeline_v3.language_normalization import normalize_language_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAIRS = ("zh_uz", "zh_ru", "uz_ru")


class PairDataPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("FOURLANG_MODEL_ROOT", str(PROJECT_ROOT / "models"))
        os.environ.setdefault(
            "FOURLANG_QWEN_MODEL_PATH", str(PROJECT_ROOT / "models/Qwen3-8B")
        )
        cls.configs = {
            pair: load_config(PROJECT_ROOT / f"configs/directions/{pair}.toml")
            for pair in PAIRS
        }

    def test_every_remaining_pair_has_a_data_only_pipeline(self) -> None:
        for pair in PAIRS:
            with self.subTest(pair=pair):
                pipeline = DirectionPipeline(
                    PROJECT_ROOT / f"configs/pipelines/{pair}.toml",
                    profile_name="local",
                )
                stage_ids = [stage.stage_id for stage in pipeline.stages]
                self.assertEqual(pipeline.direction, pair)
                self.assertIn("download_parallel_data", stage_ids)
                self.assertIn("build_frozen_split", stage_ids)
                self.assertIn("build_kd_dataset", stage_ids)
                self.assertNotIn("train_exp1", stage_ids)
                self.assertNotIn("train_exp2", stage_ids)

    def test_pinned_corpora_match_each_language_pair(self) -> None:
        expected_counts = {"zh_uz": 5, "zh_ru": 3, "uz_ru": 4}
        for pair, config in self.configs.items():
            source = config["direction"]["source_lang"]
            target = config["direction"]["target_lang"]
            corpora = config["data"]["corpora"]
            with self.subTest(pair=pair):
                self.assertEqual(len(corpora), expected_counts[pair])
                self.assertEqual(config["data"]["train_pairs"], 0)
                self.assertTrue(
                    all(
                        {
                            corpus["archive_source_lang"],
                            corpus["archive_target_lang"],
                        }
                        == {source, target}
                        for corpus in corpora
                    )
                )
                self.assertEqual(
                    len({(item["name"], item["version"]) for item in corpora}),
                    len(corpora),
                )

    def test_script_contract_is_explicit_for_every_relevant_pair(self) -> None:
        for pair, config in self.configs.items():
            contract = config["text_contract"]
            languages = {
                config["direction"]["source_lang"],
                config["direction"]["target_lang"],
            }
            if "zh" in languages:
                self.assertEqual(contract["zh_script"], "simplified")
                self.assertTrue(contract["convert_traditional_to_simplified"])
            if "uz" in languages:
                self.assertEqual(contract["uz_script"], "latin")
                self.assertTrue(contract["transliterate_cyrillic_to_latin"])

    def test_generic_rules_understand_zh_uz_and_ru_scripts(self) -> None:
        settings = {"max_characters": 1000, "max_length_ratio": 6.0}
        clean_zh = normalize_language_text("zh", "這是一個測試句子。")
        clean_uz = normalize_language_text("uz", "Бу синов гапидир.")
        self.assertEqual(
            rule_assessment(clean_zh, clean_uz, settings, "zh", "uz")[2],
            "AUTO_ACCEPT",
        )
        flags, _, _ = rule_assessment(clean_zh, "Бу синов", settings, "zh", "uz")
        self.assertIn("TARGET_SCRIPT_RISK", flags)
        self.assertEqual(
            rule_assessment(clean_zh, "Это тестовое предложение.", settings, "zh", "ru")[2],
            "AUTO_ACCEPT",
        )

    def test_opus_zip_reader_preserves_alignment_and_orientation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "ru-uz.txt.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("corpus/ru-uz.ru", "Один\nДва\n")
                archive.writestr("corpus/ru-uz.uz", "Бир\nИкки\n")
            russian, uzbek = read_parallel_archive(archive_path, "ru", "uz")
            self.assertEqual(russian, ["Один", "Два"])
            self.assertEqual(uzbek, ["Бир", "Икки"])
            self.assertEqual(
                [normalize_language_text("uz", text) for text in uzbek],
                ["Bir", "Ikki"],
            )


if __name__ == "__main__":
    unittest.main()
