from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.pipeline_v3 import translate_current_models as current


class TranslateCurrentModelsTests(unittest.TestCase):
    def test_catalog_has_all_twelve_directions_and_six_models(self) -> None:
        catalog = current.direction_catalog()

        self.assertEqual(len(catalog), 12)
        self.assertEqual(len({item["relative_path"] for item in catalog.values()}), 6)
        self.assertEqual(
            catalog["zh-uz"]["model_name"],
            "zh_uz_flores_relaxed_8k_ep3",
        )
        self.assertEqual(
            catalog["uz-zh"]["relative_path"],
            catalog["zh-uz"]["relative_path"],
        )

    def test_direction_alias_and_project_relative_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            route, path = current.resolve_model("ru_en", project_root=root)

        self.assertEqual(route["pair"], "en_ru")
        self.assertEqual(
            path,
            (root / "models/final_pair_specialists/en_ru_v1").resolve(),
        )

    def test_require_model_rejects_project_root_instead_of_loading_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                current.require_model(root, "uz-zh")

    def test_manifest_checks_paths_without_loading_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / current.PAIR_MODELS["en_uz"]["path"]
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")

            manifest = current.model_manifest(root)

        en_uz = next(item for item in manifest if item["direction"] == "en-uz")
        uz_en = next(item for item in manifest if item["direction"] == "uz-en")
        self.assertTrue(en_uz["available"])
        self.assertTrue(en_uz["config_available"])
        self.assertEqual(en_uz["model_path"], uz_en["model_path"])

    def test_multidirection_input_switches_or_uses_inline_direction(self) -> None:
        self.assertEqual(
            current.parse_multidirection_input("/direction uz-ru", "zh-en"),
            ("switch", "uz-ru", None),
        )
        self.assertEqual(
            current.parse_multidirection_input("ru-uz: Добрый день", "zh-en"),
            ("translate", "ru-uz", "Добрый день"),
        )
        self.assertEqual(
            current.parse_multidirection_input("你好", "zh-en"),
            ("translate", "zh-en", "你好"),
        )

    def test_bare_direction_switches_without_translating(self) -> None:
        self.assertEqual(
            current.parse_multidirection_input("zh-uz", "en-ru"),
            ("switch", "zh-uz", None),
        )
        self.assertEqual(
            current.parse_multidirection_input("uz_zh", "en-ru"),
            ("switch", "uz-zh", None),
        )

    def test_bare_direction_command_requires_argument(self) -> None:
        with self.assertRaises(ValueError):
            current.parse_multidirection_input("/direction", "zh-en")
        with self.assertRaises(ValueError):
            current.parse_multidirection_input("/direction ", "zh-en")

    def test_fourlang_stages_resolve_to_shared_exports(self) -> None:
        self.assertEqual(set(current.FOURLANG_STAGES), {"exp1", "exp2", "exp3_v2"})
        for path in current.FOURLANG_STAGES.values():
            self.assertTrue(path.endswith("best_model/shared"))

    def test_multidirection_control_commands(self) -> None:
        self.assertEqual(
            current.parse_multidirection_input("/directions", "en-uz"),
            ("directions", "en-uz", None),
        )
        self.assertEqual(
            current.parse_multidirection_input("/quit", "en-uz"),
            ("quit", "en-uz", None),
        )


if __name__ == "__main__":
    unittest.main()
