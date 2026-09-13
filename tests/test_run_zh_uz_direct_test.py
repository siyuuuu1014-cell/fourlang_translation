from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.pipeline_v3.run_zh_uz_direct_test import resolve_model_family


class RunZhUzDirectTestTests(unittest.TestCase):
    def test_detects_exported_small100_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary)
            (model / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "SMALL100Tokenizer"}),
                encoding="utf-8",
            )
            family = resolve_model_family(model, "transformers")
        self.assertEqual(family, "small100")

    def test_explicit_family_takes_precedence(self):
        with tempfile.TemporaryDirectory() as temporary:
            family = resolve_model_family(Path(temporary), "nllb", "small100")
        self.assertEqual(family, "small100")

    def test_falls_back_to_selected_family(self):
        with tempfile.TemporaryDirectory() as temporary:
            family = resolve_model_family(Path(temporary), "nllb")
        self.assertEqual(family, "nllb")


if __name__ == "__main__":
    unittest.main()
