from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from inference.engine import (
    SUPPORTED_DIRECTIONS,
    TranslationEngine,
    parse_direction,
)
from inference.loader import LoadedModel
from inference.router import ModelRouter, RoutedTranslationEngine


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.src_lang = None
        self.tgt_lang = None

    def __call__(self, *_args, **_kwargs):
        return FakeBatch(input_ids=torch.tensor([[1]]))

    def get_lang_id(self, language: str) -> int:
        return {"zh": 10, "en": 11, "ru": 12, "uz": 13}[language]

    def batch_decode(self, *_args, **_kwargs):
        return ["translated"]


class FakeModel:
    def __init__(self) -> None:
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return torch.tensor([[1]])


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

    def test_small100_uses_target_prefix_without_forced_bos(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        loaded = LoadedModel(
            tokenizer=tokenizer,
            model=model,
            device=torch.device("cpu"),
            model_path="models/small100",
            adapter_path=None,
            tokenizer_kind="small100",
        )
        result = TranslationEngine(loaded, direction="zh-en").translate("你好")
        self.assertEqual(tokenizer.tgt_lang, "en")
        self.assertNotIn("forced_bos_token_id", model.generate_kwargs)
        self.assertEqual(result["translation"], "translated")

    def test_m2m100_uses_source_language_and_forced_bos(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        loaded = LoadedModel(
            tokenizer=tokenizer,
            model=model,
            device=torch.device("cpu"),
            model_path="models/m2m100",
            adapter_path=None,
            tokenizer_kind="m2m100",
        )
        TranslationEngine(loaded, direction="zh-en").translate("你好")
        self.assertEqual(tokenizer.src_lang, "zh")
        self.assertEqual(model.generate_kwargs["forced_bos_token_id"], 11)

    def test_marian_uses_fixed_direction_without_language_id(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        loaded = LoadedModel(
            tokenizer=tokenizer,
            model=model,
            device=torch.device("cpu"),
            model_path="models/en-zh",
            adapter_path=None,
            tokenizer_kind="marian",
        )
        TranslationEngine(loaded, direction="en-zh").translate("hello")
        self.assertIsNone(tokenizer.src_lang)
        self.assertIsNone(tokenizer.tgt_lang)
        self.assertNotIn("forced_bos_token_id", model.generate_kwargs)

    def test_auto_router_prefers_available_specialists(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            project_root = Path(raw_dir) / "fourlang_translation"
            (project_root / "models" / "small100").mkdir(parents=True)
            specialists = project_root / "models" / "final_specialists"
            (specialists / "en_uz_small100_v1").mkdir(parents=True)
            (specialists / "en_zh_v1").mkdir(parents=True)
            zh_en = (
                project_root
                / "results"
                / "specialists"
                / "zh_en"
                / "opus_mt_zh_en"
                / "exp2_kd_v1"
                / "best_model"
            )
            zh_en.mkdir(parents=True)

            router = ModelRouter(project_root=project_root)
            self.assertEqual(
                router.route_for("en-uz").model_name,
                "en_uz_small100_v1",
            )
            self.assertEqual(router.route_for("en-zh").model_name, "en_zh_v1")
            self.assertEqual(router.route_for("zh-en").model_name, "zh_en_exp2_kd_v1")
            self.assertEqual(router.route_for("ru-uz").model_name, "small100_base")

    def test_single_model_router_caches_one_load_across_directions(self) -> None:
        calls = []

        def fake_load(model_path, **_kwargs):
            calls.append(str(model_path))
            return LoadedModel(
                tokenizer=FakeTokenizer(),
                model=FakeModel(),
                device=torch.device("cpu"),
                model_path=str(model_path),
                adapter_path=None,
            )

        router = ModelRouter(model_path="one-model", load_fn=fake_load)
        first, _ = router.load_for("zh-en")
        second, _ = router.load_for("en-ru")
        self.assertIs(first, second)
        self.assertEqual(calls, ["one-model"])

    def test_interactive_engine_switches_architectures_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            project_root = Path(raw_dir) / "fourlang_translation"
            (project_root / "models" / "small100").mkdir(parents=True)
            specialists = project_root / "models" / "final_specialists"
            (specialists / "en_uz_small100_v1").mkdir(parents=True)
            (specialists / "en_zh_v1").mkdir(parents=True)
            calls = []

            def fake_load(model_path, **_kwargs):
                path = str(model_path)
                calls.append(path)
                return LoadedModel(
                    tokenizer=FakeTokenizer(),
                    model=FakeModel(),
                    device=torch.device("cpu"),
                    model_path=path,
                    adapter_path=None,
                    tokenizer_kind=(
                        "marian" if path.endswith("en_zh_v1") else "small100"
                    ),
                )

            router = ModelRouter(project_root=project_root, load_fn=fake_load)
            engine = RoutedTranslationEngine(router, direction="en-uz")
            self.assertEqual(
                engine.translate("hello")["model_name"],
                "en_uz_small100_v1",
            )
            engine.set_direction("en-zh")
            self.assertEqual(engine.translate("hello")["model_name"], "en_zh_v1")
            engine.set_direction("uz-en")
            result = engine.translate("salom")
            self.assertEqual(result["model_name"], "en_uz_small100_v1")
            self.assertEqual(result["loaded_model_count"], 2)
            self.assertEqual(len(calls), 2)

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

    def test_list_routes_cli_does_not_load_weights(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "inference", "--list-routes"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["routing_mode"], "auto")
        self.assertEqual(len(payload["routes"]), 12)


if __name__ == "__main__":
    unittest.main()
