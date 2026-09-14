from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.pipeline_v3 import deepseek_teacher as teacher


def row(pair_id: str, source: str, target: str, text: str) -> dict:
    return {
        "pair_id": pair_id,
        "src_lang": source,
        "tgt_lang": target,
        "src_text": text,
    }


class DeepSeekTeacherTests(unittest.TestCase):
    def test_numeric_equivalence_accepts_localized_forms(self):
        cases = [
            ("上海降水pH为5.38，频率40.2%。", "pH 5,38, chastota 40,2%.", "zh", "uz"),
            ("领土超过140万平方公里。", "1,4 million kvadrat kilometr.", "zh", "uz"),
            ("会议在7月5日举行。", "Yig'ilish 5-iyulda bo'ladi.", "zh", "uz"),
            ("下辖6县。", "Oltita tumanni boshqargan.", "zh", "uz"),
            (
                "Qonun 1989-yil 21-oktabrda qabul qilingan.",
                "法律于1989年10月21日通过。",
                "uz",
                "zh",
            ),
            ("40-12-mingyilliklarga oid.", "属于公元前4万至1.2万年。", "uz", "zh"),
        ]
        for source, target, source_lang, target_lang in cases:
            with self.subTest(source=source):
                self.assertTrue(
                    teacher.numbers_preserved(source, target, source_lang, target_lang)
                )

    def test_numeric_equivalence_rejects_changed_magnitude(self):
        self.assertFalse(
            teacher.numbers_preserved(
                "共有437万失信被执行人。",
                "Jami 437 ming qarzdor bor.",
                "zh",
                "uz",
            )
        )

    def test_source_filter_rejects_high_confidence_spam(self):
        self.assertEqual(
            teacher.source_filter_reasons(
                "Mostbet Casino yangi slot bonuslarini taklif qiladi."
            ),
            ["GAMBLING_OR_BETTING"],
        )
        self.assertIn(
            "SEO_OR_PROMO_PREFIX",
            teacher.source_filter_reasons("top9、业内人士分析称露营产业将发展"),
        )
        self.assertEqual(
            teacher.source_filter_reasons("撒马尔罕拥有许多历史古迹。"), []
        )

    def test_source_filter_changes_full_selection_only(self):
        rows = [
            row("zh-clean", "zh", "uz", "正常的中文句子。"),
            row("zh-spam", "zh", "uz", "赌场提供老虎机奖金。"),
            row("uz-clean", "uz", "zh", "Bu oddiy jumla."),
            row("uz-spam", "uz", "zh", "Mostbet kazino bonusi."),
        ]
        config = {
            "pipeline": {"seed": 2026},
            "pilot": {"rows_per_direction": 2},
            "source_filter": {"enabled": True},
        }
        pilot, pilot_rejections = teacher.select_for_mode(rows, config, False)
        full, full_rejections = teacher.select_for_mode(rows, config, True)
        self.assertEqual(len(pilot), 4)
        self.assertFalse(pilot_rejections)
        self.assertEqual({item["pair_id"] for item in full}, {"zh-clean", "uz-clean"})
        self.assertEqual(full_rejections["GAMBLING_OR_BETTING"], 2)

    def test_full_selection_is_capped_per_direction(self):
        rows = [
            row(f"zh-{index}", "zh", "uz", f"中文句子{index}。") for index in range(3)
        ] + [
            row(f"uz-{index}", "uz", "zh", f"Oddiy jumla {index}.")
            for index in range(3)
        ]
        config = {
            "pipeline": {"seed": 2026},
            "pilot": {"rows_per_direction": 1},
            "full": {"rows_per_direction": 2},
            "source_filter": {"enabled": True},
        }

        selected, rejections = teacher.select_for_mode(rows, config, True)

        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {
                direction: sum(
                    teacher.direction(item) == direction for item in selected
                )
                for direction in teacher.DIRECTIONS
            },
            {"zh-uz": 2, "uz-zh": 2},
        )
        self.assertFalse(rejections)

    def test_reaudit_is_offline_and_preserves_original_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "teacher"
            pilot = root / "pilot"
            pilot.mkdir(parents=True)
            original = pilot / "teacher_generated.parquet"
            pd.DataFrame(
                [
                    {
                        **row("clean", "zh", "uz", "价格为5.38元。"),
                        "teacher_text": "Narxi 5,38 yuan.",
                        "hard_check_pass": False,
                    },
                    {
                        **row("spam", "uz", "zh", "Mostbet kazino bonusi."),
                        "teacher_text": "Mostbet赌场奖金。",
                        "hard_check_pass": True,
                    },
                ]
            ).to_parquet(original, index=False)
            config = {
                "output": {"root": str(root)},
                "quality": {
                    "reject_source_copy": True,
                    "require_arabic_numbers": True,
                    "reject_excessive_repetition": True,
                    "require_target_script_signal": True,
                },
                "source_filter": {"enabled": True},
            }
            before = original.read_bytes()
            with mock.patch.object(
                teacher.httpx, "post", side_effect=AssertionError("API called")
            ):
                report = teacher.reaudit(config, False)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(
                report["directions"]["zh-uz"]["revised_translation_pass_rows"],
                1,
            )
            self.assertEqual(report["directions"]["uz-zh"]["combined_usable_rows"], 0)
            self.assertEqual(original.read_bytes(), before)
            self.assertTrue((pilot / "reaudit.jsonl").is_file())

    def test_runtime_transport_override_does_not_mutate_config(self):
        config = {
            "api": {"batch_size": 10, "concurrency": 4},
            "quality": {"require_arabic_numbers": True},
        }
        runtime_config = teacher.with_runtime_overrides(
            config, batch_size_override=1, concurrency_override=1
        )

        self.assertEqual(config["api"], {"batch_size": 10, "concurrency": 4})
        self.assertNotEqual(runtime_config["api"], config["api"])

    def test_runtime_transport_override_rejects_zero(self):
        config = {"api": {"batch_size": 10, "concurrency": 4}}
        with self.assertRaisesRegex(ValueError, "batch_size_override"):
            teacher.with_runtime_overrides(config, batch_size_override=0)

    def test_pilot_selection_is_balanced_and_deterministic(self):
        rows = [
            row(f"zh-{index}", "zh", "uz", f"中文 {index}") for index in range(5)
        ] + [row(f"uz-{index}", "uz", "zh", f"uzbek {index}") for index in range(5)]
        first = teacher.select_rows(rows, 2026, 3)
        second = teacher.select_rows(list(reversed(rows)), 2026, 3)
        self.assertEqual(
            [teacher.row_key(item) for item in first],
            [teacher.row_key(item) for item in second],
        )
        self.assertEqual([teacher.direction(item) for item in first].count("zh-uz"), 3)
        self.assertEqual([teacher.direction(item) for item in first].count("uz-zh"), 3)

    def test_response_requires_exact_ids(self):
        content = json.dumps({"items": [{"id": "a", "translation": "Tarjima"}]})
        self.assertEqual(
            teacher.parse_translation_response(content, ["a"]), {"a": "Tarjima"}
        )
        with self.assertRaisesRegex(ValueError, "exactly match"):
            teacher.parse_translation_response(content, ["a", "b"])

    def test_hard_checks_detect_number_loss_and_wrong_script(self):
        config = {
            "quality": {
                "reject_source_copy": True,
                "require_arabic_numbers": True,
                "reject_excessive_repetition": True,
                "require_target_script_signal": True,
            }
        }
        failures = teacher.hard_check(
            row("a", "zh", "uz", "价格是 120 元"), "Нархи юз сўм", config
        )
        self.assertIn("ARABIC_NUMBER_MISMATCH", failures)
        self.assertIn("UZ_LATIN_SIGNAL_MISSING", failures)

    def test_hard_checks_accept_clean_translation(self):
        config = {
            "quality": {
                "reject_source_copy": True,
                "require_arabic_numbers": True,
                "reject_excessive_repetition": True,
                "require_target_script_signal": True,
            }
        }
        failures = teacher.hard_check(
            row("a", "zh", "uz", "价格是 120 元"),
            "Narxi 120 yuan.",
            config,
        )
        self.assertEqual(failures, [])

    def test_failed_normalization_keeps_raw_text_for_audit(self):
        text = "Unsupported Cyrillic: \u0462"
        self.assertEqual(teacher.normalized_teacher_text("uz", text), text)

    def test_client_error_is_fatal_without_retry(self):
        config = {
            "api": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-v4-pro",
                "thinking": False,
                "max_tokens": 100,
                "temperature": 0.1,
                "max_retries": 4,
                "timeout_seconds": 10,
            }
        }
        response = mock.Mock(status_code=401)
        with mock.patch.object(teacher.httpx, "post", return_value=response) as post:
            with self.assertRaisesRegex(teacher.FatalDeepSeekError, "HTTP 401"):
                teacher.request_batch(
                    [row("a", "zh", "uz", "测试")], config, "not-a-real-key"
                )
        self.assertEqual(post.call_count, 1)


if __name__ == "__main__":
    unittest.main()
