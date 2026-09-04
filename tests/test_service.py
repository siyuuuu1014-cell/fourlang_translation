from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from service.app import app
from service.dependencies import get_translator_engine


class FakeTranslatorEngine:
    def available_directions(self, *, ready_only: bool = True) -> list[str]:
        return ["en_ru"]

    def warmup(self, _direction: str) -> dict[str, float]:
        return {"generation_latency_seconds": 0.0}

    def translate(self, direction: str, text: str) -> dict[str, object]:
        return {
            "direction": direction,
            "source_lang": "en",
            "target_lang": "ru",
            "model": "en_ru_small100_v1",
            "architecture": "small100",
            "text": text,
            "translation": "перевод",
            "generation_latency_seconds": 0.01,
            "device": "cpu",
        }


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_translator_engine] = FakeTranslatorEngine
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_and_registry_are_available_without_weights(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        models = self.client.get("/models")
        self.assertEqual(models.status_code, 200)
        directions = {row["direction"] for row in models.json()["models"]}
        self.assertIn("en_ru", directions)
        self.assertIn("en_zh", directions)

    def test_translation_contract(self) -> None:
        response = self.client.post(
            "/translate",
            json={"source_lang": "en", "target_lang": "ru", "text": "hello"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["translation"], "перевод")
        self.assertEqual(response.json()["model"], "en_ru_small100_v1")


if __name__ == "__main__":
    unittest.main()

