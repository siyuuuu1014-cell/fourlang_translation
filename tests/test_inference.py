from __future__ import annotations

import json
import subprocess
import sys
import unittest

import torch

from inference.engine import (
    SUPPORTED_DIRECTIONS,
    TranslationEngine,
    parse_direction,
)
from inference.loader import LoadedModel


class DirectionTests(unittest.TestCase):
    def test_all_twelve_directions_are_available(self) -> None:
        self.assertEqual(len(SUPPORTED_DIRECTIONS), 12)
        self.assertIn("zh-en", SUPPORTED_DIRECTIONS)
        self.assertIn("uz-ru", SUPPORTED_DIRECTIONS)

    def test_direction_aliases(self) -> None:
        self.assertEqual(parse_direction("ZH_en"), ("zh", "en"))
        self.assertEqual(parse_direction("ru->uz"), ("ru", "uz"))

    def test_same_language_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_direction("en-en")

    def test_engine_switches_direction_without_reloading(self) -> None:
        loaded = LoadedModel(
            tokenizer=None,
            model=None,
            device=torch.device("cpu"),
            model_path="models/small100",
            adapter_path=None,
        )
        engine = TranslationEngine(loaded, direction="zh-en")
        same_loaded_model = engine.loaded
        self.assertEqual(engine.set_direction("ru_uz"), "ru-uz")
        self.assertIs(engine.loaded, same_loaded_model)

    def test_list_directions_cli_is_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "inference", "--list-directions"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["directions"]), 12)


if __name__ == "__main__":
    unittest.main()
