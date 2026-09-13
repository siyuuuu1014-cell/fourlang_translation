from __future__ import annotations

import json
import unittest
from unittest import mock

from scripts.pipeline_v3 import deepseek_teacher as teacher


def row(pair_id: str, source: str, target: str, text: str) -> dict:
    return {
        "pair_id": pair_id,
        "src_lang": source,
        "tgt_lang": target,
        "src_text": text,
    }


class DeepSeekTeacherTests(unittest.TestCase):
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
        ] + [
            row(f"uz-{index}", "uz", "zh", f"uzbek {index}") for index in range(5)
        ]
        first = teacher.select_rows(rows, 2026, 3)
        second = teacher.select_rows(list(reversed(rows)), 2026, 3)
        self.assertEqual([teacher.row_key(item) for item in first], [teacher.row_key(item) for item in second])
        self.assertEqual([teacher.direction(item) for item in first].count("zh-uz"), 3)
        self.assertEqual([teacher.direction(item) for item in first].count("uz-zh"), 3)

    def test_response_requires_exact_ids(self):
        content = json.dumps(
            {"items": [{"id": "a", "translation": "Tarjima"}]}
        )
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
