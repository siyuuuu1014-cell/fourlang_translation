from __future__ import annotations

import os
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from scripts.pipeline.run_direction import DirectionPipeline
from scripts.pipeline_v2.common import load_config
from scripts.pipeline_v3.fourlang_flow import (
    LANGUAGES,
    UNORDERED_PAIRS,
    balance_training_rows,
    directions,
    normalize_rows,
)
from scripts.pipeline_v3.language_normalization import (
    normalize_language_text,
    to_simplified_chinese,
    to_uzbek_latin,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FourLanguagePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("FOURLANG_MODEL_ROOT", str(PROJECT_ROOT / "models"))
        cls.config = load_config("configs/multilingual/fourlang.toml")

    def test_contract_has_four_languages_six_pairs_and_twelve_directions(self) -> None:
        self.assertEqual(tuple(self.config["multilingual"]["languages"]), LANGUAGES)
        self.assertEqual(
            {item["pair"] for item in self.config["pair_data"]},
            set(UNORDERED_PAIRS),
        )
        self.assertEqual(len(directions()), 12)
        self.assertEqual(len(set(directions())), 12)
        self.assertEqual(self.config["language_codes"]["nllb"]["zh"], "zho_Hans")
        self.assertEqual(self.config["language_codes"]["nllb"]["uz"], "uzn_Latn")

    def test_chinese_is_converted_to_simplified(self) -> None:
        self.assertEqual(to_simplified_chinese("繁體中文與軟體"), "繁体中文与软体")

    def test_uzbek_cyrillic_is_transliterated_to_latin(self) -> None:
        self.assertEqual(to_uzbek_latin("Ўзбекистон Республикаси"), "O'zbekiston Respublikasi")
        self.assertNotRegex(to_uzbek_latin("Салом, дунё!"), r"[\u0400-\u052f]")

    def test_language_rules_apply_only_to_the_declared_language(self) -> None:
        russian = "Россия"
        self.assertEqual(normalize_language_text("ru", russian), russian)

    def test_normalize_rows_applies_script_contract_to_both_sides(self) -> None:
        normalized = normalize_rows(
            pd.DataFrame(
                {
                    "src_lang": ["zh", "uz"],
                    "tgt_lang": ["uz", "zh"],
                    "src_text": ["這是測試", "Ўзбекистон"],
                    "tgt_text": ["Бу синов", "這是測試"],
                }
            ),
            origin="script-contract",
        )
        self.assertEqual(normalized.loc[0, "src_text"], "这是测试")
        self.assertEqual(normalized.loc[0, "tgt_text"], "Bu sinov")
        self.assertEqual(normalized.loc[1, "src_text"], "O'zbekiston")
        self.assertEqual(normalized.loc[1, "tgt_text"], "这是测试")
        self.assertEqual(
            normalized.attrs["script_normalization"]["zh_converted"], 2
        )
        self.assertEqual(
            normalized.attrs["script_normalization"]["uz_converted"], 2
        )

    def test_global_candidates_are_exactly_the_confirmed_shortlist(self) -> None:
        self.assertEqual(
            {item["id"] for item in self.config["student_candidates"]},
            {"small100", "m2m100_418m", "nllb_600m"},
        )
        nllb = next(
            item
            for item in self.config["student_candidates"]
            if item["id"] == "nllb_600m"
        )
        self.assertEqual(
            nllb["revision"], "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
        )

    def test_balancing_uses_every_direction_equally(self) -> None:
        rows = []
        for direction in directions():
            source, target = direction.split("-")
            for index in range(3):
                rows.append(
                    {
                        "src_lang": source,
                        "tgt_lang": target,
                        "src_text": f"{direction}-source-{index}",
                        "tgt_text": f"{direction}-target-{index}",
                        "weight": 1.0,
                        "training_source": "human_parallel",
                        "origin": "test",
                    }
                )
        balanced, report = balance_training_rows(
            pd.DataFrame(rows), seed=2026, configured_rows=2
        )
        counts = Counter(
            balanced["src_lang"] + "-" + balanced["tgt_lang"]
        )
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(report["output_rows"], 24)

    def test_existing_pair_kd_outputs_are_reused(self) -> None:
        pairs = {item["pair"]: item for item in self.config["pair_data"]}
        self.assertTrue(
            pairs["en_zh"]["kd_train"].endswith(
                "zh_en/v1/18h_exp2_training/exp2_train_combined_v1.parquet"
            )
        )
        self.assertTrue(
            pairs["en_uz"]["kd_train"].endswith(
                "en_uz/v1/11a_exp2_training/exp2_train_combined_v1.parquet"
            )
        )
        self.assertEqual(
            pairs["en_ru"]["kd_train"],
            "data/distillation/en_ru/v1/train.jsonl",
        )

    def test_legacy_directed_column_names_are_normalized(self) -> None:
        normalized = normalize_rows(
            pd.DataFrame(
                {
                    "source_lang": ["en"],
                    "target_lang": ["uz"],
                    "source_text": ["hello"],
                    "target_text": ["salom"],
                    "training_weight": [0.8],
                }
            ),
            origin="legacy",
        )
        self.assertEqual(normalized.loc[0, "src_lang"], "en")
        self.assertEqual(normalized.loc[0, "tgt_lang"], "uz")
        self.assertEqual(normalized.loc[0, "weight"], 0.8)

    def test_manifest_trains_and_freezes_only_one_model(self) -> None:
        pipeline = DirectionPipeline(
            PROJECT_ROOT / "configs/pipelines/fourlang.toml", profile_name="local"
        )
        ids = [stage.stage_id for stage in pipeline.stages]
        self.assertEqual(ids.count("train_exp1"), 1)
        self.assertEqual(ids.count("train_exp2"), 1)
        self.assertLess(ids.index("student_bakeoff"), ids.index("select_student"))
        self.assertLess(ids.index("promotion_gate"), ids.index("freeze"))
        aggregate_exp2 = next(
            stage for stage in pipeline.stages if stage.stage_id == "aggregate_exp2"
        )
        required = {path.replace("\\", "/") for path in aggregate_exp2.requires}
        configured = {
            item["kd_train"] for item in self.config["pair_data"]
        } | {item["validation"] for item in self.config["pair_data"]}
        self.assertEqual(required, configured)


if __name__ == "__main__":
    unittest.main()
