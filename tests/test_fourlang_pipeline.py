from __future__ import annotations

import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.pipeline_v3 import fourlang_flow
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

    def test_balancing_oversamples_low_resource_directions_deterministically(self) -> None:
        rows = []
        for direction in directions():
            source, target = direction.split("-")
            rows.append(
                {
                    "src_lang": source,
                    "tgt_lang": target,
                    "src_text": f"{direction}-source",
                    "tgt_text": f"{direction}-target",
                    "weight": 1.0,
                    "training_source": "human_parallel",
                    "origin": "test",
                }
            )
        frame = pd.DataFrame(rows)

        first, report = balance_training_rows(frame, seed=2026, configured_rows=3)
        second, _ = balance_training_rows(frame, seed=2026, configured_rows=3)

        pd.testing.assert_frame_equal(first, second)
        counts = Counter(first["src_lang"] + "-" + first["tgt_lang"])
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(report["output_rows"], 36)
        self.assertEqual(set(report["sampled_with_replacement"]), set(directions()))

    def test_exp2_sampling_obeys_direction_and_source_quotas(self) -> None:
        rows = []
        for direction in directions():
            source, target = direction.split("-")
            for index in range(2):
                rows.append(
                    {
                        "src_lang": source,
                        "tgt_lang": target,
                        "src_text": f"{direction}-teacher-source-{index}",
                        "tgt_text": f"{direction}-teacher-target-{index}",
                        "weight": 1.0,
                        "training_source": "teacher_kd",
                        "origin": "test",
                    }
                )
            for index in range(5):
                rows.append(
                    {
                        "src_lang": source,
                        "tgt_lang": target,
                        "src_text": f"{direction}-human-source-{index}",
                        "tgt_text": f"{direction}-human-target-{index}",
                        "weight": 1.0,
                        "training_source": "human_replay",
                        "origin": "test",
                    }
                )

        sampled, report = balance_training_rows(
            pd.DataFrame(rows),
            seed=2026,
            configured_rows=5,
            rows_by_direction={"zh-uz": 10},
            teacher_ratio=0.6,
            max_teacher_repeats=3,
        )

        counts = Counter(sampled["src_lang"] + "-" + sampled["tgt_lang"])
        self.assertEqual(counts["zh-uz"], 10)
        self.assertEqual({counts[item] for item in directions() if item != "zh-uz"}, {5})
        self.assertEqual(report["source_mix_by_direction"]["zh-uz"]["teacher_kd"], 6)
        self.assertEqual(report["source_mix_by_direction"]["zh-uz"]["human_replay"], 4)
        teacher = sampled[
            (sampled["src_lang"] == "zh")
            & (sampled["tgt_lang"] == "uz")
            & (sampled["training_source"] == "teacher_kd")
        ]
        self.assertLessEqual(teacher["src_text"].value_counts().max(), 3)

    def test_multilingual_curriculum_sizes_are_locked(self) -> None:
        exp1 = self.config["balancing"]["exp1"]
        exp2 = self.config["balancing"]["exp2"]
        self.assertEqual(exp1["default_rows_per_direction"] * 12, 120000)
        exp2_targets = {
            direction: exp2["rows_by_direction"].get(
                direction, exp2["default_rows_per_direction"]
            )
            for direction in directions()
        }
        self.assertEqual(
            exp2_targets,
            {
                "en-zh": 10000,
                "en-uz": 10000,
                "en-ru": 10000,
                "zh-en": 10000,
                "zh-uz": 20000,
                "zh-ru": 12000,
                "uz-en": 10000,
                "uz-zh": 20000,
                "uz-ru": 17000,
                "ru-en": 10000,
                "ru-zh": 12000,
                "ru-uz": 17000,
            },
        )
        self.assertEqual(sum(exp2_targets.values()), 158000)
        self.assertEqual(exp2["teacher_ratio"], 0.6)
        self.assertEqual(exp2["max_teacher_repeats"], 3)

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

    def test_legacy_zh_en_exp2_schema_is_normalized(self) -> None:
        normalized = normalize_rows(
            pd.DataFrame(
                {
                    "direction": ["en_zh", "zh_en"],
                    "source_text": ["hello", "你好"],
                    "target_text": ["你好", "hello"],
                    "training_weight": [1.0, 1.0],
                    "training_origin": ["TEACHER_KD", "HUMAN_REPLAY"],
                }
            ),
            origin="legacy-zh-en-exp2",
        )
        self.assertEqual(normalized["src_lang"].tolist(), ["en", "zh"])
        self.assertEqual(normalized["tgt_lang"].tolist(), ["zh", "en"])
        self.assertEqual(
            normalized["training_source"].tolist(),
            ["TEACHER_KD", "HUMAN_REPLAY"],
        )

    def test_legacy_en_uz_exp2_schema_is_normalized(self) -> None:
        normalized = normalize_rows(
            pd.DataFrame(
                {
                    "direction": ["en_uz", "uz_en"],
                    "source_text": ["hello", "salom"],
                    "target_text": ["salom", "hello"],
                    "sample_weight": [0.9, 1.0],
                    "sample_origin": ["TEACHER_KD", "HUMAN_REPLAY"],
                }
            ),
            origin="legacy-en-uz-exp2",
        )
        self.assertEqual(normalized["src_lang"].tolist(), ["en", "uz"])
        self.assertEqual(normalized["tgt_lang"].tolist(), ["uz", "en"])
        self.assertEqual(normalized["weight"].tolist(), [0.9, 1.0])
        self.assertEqual(
            normalized["training_source"].tolist(),
            ["TEACHER_KD", "HUMAN_REPLAY"],
        )

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

    def test_bakeoff_resumes_after_last_completed_candidate(self) -> None:
        config = {
            "student_candidates": [
                {"id": "small100"},
                {"id": "m2m100_418m"},
                {"id": "nllb_600m"},
            ]
        }
        completed = {"status": "ok", "macro_chrf2": 1.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(fourlang_flow, "PROJECT_ROOT", root):
                with mock.patch.object(
                    fourlang_flow,
                    "evaluate_candidate",
                    side_effect=[completed, KeyboardInterrupt()],
                ) as first_evaluation:
                    with self.assertRaises(KeyboardInterrupt):
                        fourlang_flow.bakeoff(config)
                self.assertEqual(first_evaluation.call_count, 2)

                checkpoint = fourlang_flow.read_json(
                    root / "results/model_selection/fourlang/student_scores.json"
                )
                self.assertEqual(checkpoint["candidates"], {"small100": completed})

                with mock.patch.object(
                    fourlang_flow,
                    "evaluate_candidate",
                    return_value=completed,
                ) as resumed_evaluation:
                    fourlang_flow.bakeoff(config)

                resumed_ids = [
                    call.args[1]["id"] for call in resumed_evaluation.call_args_list
                ]
                self.assertEqual(resumed_ids, ["m2m100_418m", "nllb_600m"])


if __name__ == "__main__":
    unittest.main()
