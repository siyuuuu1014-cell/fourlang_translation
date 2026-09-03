from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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
        self.assertLess(
            stage_ids.index("build_protected_benchmarks"),
            stage_ids.index("rule_route"),
        )
        self.assertLess(
            stage_ids.index("rule_route"), stage_ids.index("judge_human_data")
        )
        self.assertLess(
            stage_ids.index("judge_human_data"),
            stage_ids.index("second_review_human"),
        )
        self.assertLess(
            stage_ids.index("judge_human_data"), stage_ids.index("build_approved")
        )
        self.assertLess(
            stage_ids.index("select_student"), stage_ids.index("train_exp1")
        )
        self.assertLess(
            stage_ids.index("select_teacher"), stage_ids.index("generate_teacher")
        )
        self.assertLess(
            stage_ids.index("judge_teacher"), stage_ids.index("build_kd_dataset")
        )
        self.assertLess(
            stage_ids.index("train_exp2"), stage_ids.index("promotion_gate")
        )
        self.assertLess(stage_ids.index("promotion_gate"), stage_ids.index("freeze"))
        rendered = " ".join(
            part for stage in pipeline.stages for part in stage.command
        ).lower()
        self.assertNotIn("lora", rendered)
        self.assertEqual(
            [stage.stage_id for stage in pipeline.select("train_exp2", "freeze", None)],
            ["train_exp2", "evaluate_exp2", "promotion_gate", "freeze"],
        )

    def test_promotion_gate_rejects_one_regressed_benchmark(self) -> None:
        baseline = {
            direction: {
                "bleu": 20,
                "chrf2": 40,
                "benchmarks": {
                    "flores_devtest": {"bleu": 20, "chrf2": 40},
                    "tatoeba": {"bleu": 20, "chrf2": 40},
                },
            }
            for direction in ("en-ru", "ru-en")
        }
        candidate = json.loads(json.dumps(baseline))
        candidate["en-ru"]["bleu"] = 21
        candidate["en-ru"]["chrf2"] = 41
        candidate["en-ru"]["benchmarks"]["tatoeba"]["chrf2"] = 39
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.json"
            candidate_path = root / "candidate.json"
            output_path = root / "gate.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(
                        PROJECT_ROOT
                        / "scripts"
                        / "evaluation"
                        / "check_promotion_gate.py"
                    ),
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                    "--directions",
                    "en-ru",
                    "ru-en",
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(report["status"], "FAIL")
        failed = {
            (row["direction"], row["scope"])
            for row in report["directions"]
            if not row["passed"]
        }
        self.assertEqual(failed, {("en-ru", "tatoeba")})

    def test_en_ru_is_not_ready_before_promotion(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "models/model_registry.json").read_text(encoding="utf-8")
        )
        self.assertEqual(registry["models"]["en_ru"]["status"], "staged")
        self.assertEqual(registry["models"]["ru_en"]["status"], "staged")


if __name__ == "__main__":
    unittest.main()
