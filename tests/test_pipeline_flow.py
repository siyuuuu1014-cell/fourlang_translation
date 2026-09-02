from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.pipeline.run_direction import DirectionPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineManifestTests(unittest.TestCase):
    def test_en_ru_pipeline_has_complete_order(self) -> None:
        pipeline = DirectionPipeline(
            PROJECT_ROOT / "configs/pipelines/en_ru.toml",
            profile_name="local",
        )
        stage_ids = [stage.stage_id for stage in pipeline.stages]
        self.assertEqual(stage_ids[0], "validate_config")
        self.assertLess(stage_ids.index("judge_human_data"), stage_ids.index("build_approved"))
        self.assertLess(stage_ids.index("select_student"), stage_ids.index("train_exp1"))
        self.assertLess(stage_ids.index("select_teacher"), stage_ids.index("generate_teacher"))
        self.assertLess(stage_ids.index("judge_teacher"), stage_ids.index("build_kd_dataset"))
        self.assertLess(stage_ids.index("train_exp2"), stage_ids.index("promotion_gate"))
        self.assertLess(stage_ids.index("promotion_gate"), stage_ids.index("freeze"))
        rendered = " ".join(part for stage in pipeline.stages for part in stage.command).lower()
        self.assertNotIn("lora", rendered)
        self.assertEqual(
            [stage.stage_id for stage in pipeline.select("train_exp2", "freeze", None)],
            ["train_exp2", "evaluate_exp2", "promotion_gate", "freeze"],
        )

    def test_en_ru_is_not_ready_before_promotion(self) -> None:
        registry = json.loads((PROJECT_ROOT / "models/model_registry.json").read_text(encoding="utf-8"))
        self.assertEqual(registry["models"]["en_ru"]["status"], "staged")
        self.assertEqual(registry["models"]["ru_en"]["status"], "staged")

if __name__ == "__main__":
    unittest.main()
