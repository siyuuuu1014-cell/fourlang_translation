from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.pipeline_v3 import pair_student_bakeoff as bakeoff
from scripts.pipeline_v3 import summarize_pair_specialists as summary


def config() -> dict:
    return {
        "experiment": {
            "id": "test",
            "commercial_use": True,
        },
        "selection": {
            "max_student_parameters": 1_300_000_000,
            "require_all_candidates": True,
        },
        "training": {"max_source_length": 256},
        "deployment": {"num_beams": 5, "max_new_tokens": 256},
        "student_candidates": [
            {
                "id": "a",
                "family": "small100",
                "license": "mit",
                "commercial_allowed": True,
                "path": "a",
                "repo_id": "a",
                "revision": "1",
            },
            {
                "id": "b",
                "family": "m2m100",
                "license": "mit",
                "commercial_allowed": True,
                "path": "b",
                "repo_id": "b",
                "revision": "1",
            },
        ],
        "pairs": [
            {
                "id": pair_id,
                "languages": pair_id.split("_"),
                "benchmark": "benchmark.parquet",
            }
            for pair_id in bakeoff.PAIR_IDS
        ],
    }


class PairStudentBakeoffTests(unittest.TestCase):
    def test_completed_candidate_is_reused_without_force(self):
        cfg = config()
        existing = {"a": {"status": "ok"}}
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(bakeoff, "PROJECT_ROOT", Path(temporary)):
                with mock.patch.object(
                    bakeoff,
                    "load_scores",
                    return_value=(existing, {"test": True}, "fingerprint"),
                ):
                    with mock.patch.object(
                        bakeoff, "evaluate_pair_candidate"
                    ) as evaluate_candidate:
                        report = bakeoff.evaluate(cfg, "zh_uz", "a")
        evaluate_candidate.assert_not_called()
        self.assertEqual(report["candidates"], existing)

    def test_ranking_selects_one_candidate_for_the_pair(self):
        cfg = config()
        results = {
            "a": {
                "status": "ok",
                "parameters": 300,
                "model_disk_bytes": 100,
                "selection_metrics": {
                    "minimum_direction_chrf2": 20.0,
                    "macro_chrf2": 30.0,
                    "macro_bleu": 10.0,
                    "mean_seconds_per_sample": 0.02,
                },
            },
            "b": {
                "status": "ok",
                "parameters": 400,
                "model_disk_bytes": 200,
                "selection_metrics": {
                    "minimum_direction_chrf2": 21.0,
                    "macro_chrf2": 29.0,
                    "macro_bleu": 9.0,
                    "mean_seconds_per_sample": 0.03,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(bakeoff, "PROJECT_ROOT", Path(temporary)):
                with mock.patch.object(
                    bakeoff,
                    "load_scores",
                    return_value=(results, {"test": True}, "fingerprint"),
                ):
                    report = bakeoff.rank(cfg, "zh_uz")
        self.assertEqual(report["winner"], "b")
        self.assertEqual([row["candidate_id"] for row in report["ranking"]], ["b", "a"])

    def test_ranking_refuses_partial_measurements(self):
        cfg = config()
        with mock.patch.object(
            bakeoff,
            "load_scores",
            return_value=(
                {
                    "a": {
                        "status": "ok",
                        "selection_metrics": {},
                    }
                },
                {},
                "fingerprint",
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, r"missing=\['b'\]"):
                bakeoff.rank(cfg, "zh_uz")

    def test_baseline_failure_selects_exp1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for pair_id in summary.PAIR_IDS:
                destination = root / "results/evaluation/pair_specialists" / pair_id
                destination.mkdir(parents=True)
                left, right = pair_id.split("_")
                directions = (f"{left}-{right}", f"{right}-{left}")
                exp1 = {
                    direction: {"bleu": 10.0, "chrf2": 20.0, "samples": 5}
                    for direction in directions
                }
                exp2 = {
                    direction: {"bleu": 11.0, "chrf2": 21.0, "samples": 5}
                    for direction in directions
                }
                gate = "PASS"
                if pair_id == "zh_uz":
                    exp2[directions[1]]["chrf2"] = 19.0
                    gate = "FAIL"
                (destination / "exp1.json").write_text(json.dumps(exp1))
                (destination / "exp2.json").write_text(json.dumps(exp2))
                (destination / "promotion_gate.json").write_text(
                    json.dumps({"status": gate})
                )
            report = summary.build_summary(root)
        self.assertEqual(report["pass_pairs"], 5)
        self.assertEqual(report["fail_pairs"], 1)
        self.assertEqual(report["pairs"]["zh_uz"]["selected_baseline_stage"], "exp1")


if __name__ == "__main__":
    unittest.main()
