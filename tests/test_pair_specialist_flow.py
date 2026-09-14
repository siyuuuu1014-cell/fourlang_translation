from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.pipeline_v3 import pair_specialist_flow as flow


def row(source: str, target: str, index: int, origin: str = "human_replay"):
    return {
        "src_lang": source,
        "tgt_lang": target,
        "src_text": f"{source} source {index}",
        "tgt_text": f"{target} target {index}",
        "weight": 1.0,
        "training_source": origin,
    }


class PairSpecialistFlowTests(unittest.TestCase):
    def test_pair_modes_and_order_are_explicit(self):
        config = {
            "pairs": [
                {
                    "id": pair_id,
                    "languages": pair_id.split("_"),
                    "mode": "train_full",
                    "kd_quality_status": "approved",
                    "human_train": "missing",
                    "validation": "missing",
                    "kd_train": "missing",
                }
                for pair_id in flow.PAIR_IDS
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(flow, "PROJECT_ROOT", Path(temporary)):
                report = flow.validate(config)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(list(report["pairs"]), list(flow.PAIR_IDS))
        self.assertFalse(any(item["exp2_ready"] for item in report["pairs"].values()))

    def test_exp2_sampling_keeps_both_sources_and_directions(self):
        pair = {"id": "en_ru", "languages": ["en", "ru"]}
        rows = []
        for source, target in (("en", "ru"), ("ru", "en")):
            rows.extend(row(source, target, i, "teacher_kd") for i in range(3))
            rows.extend(row(source, target, i + 10, "human_replay") for i in range(4))
        selected, report = flow._sample_training_rows(
            pd.DataFrame(rows),
            pair,
            {"rows_per_direction": 10, "teacher_ratio": 0.6, "max_teacher_repeats": 2},
            2026,
            "exp2",
        )
        counts = (
            selected.groupby(["direction", "training_source"], dropna=False)
            .size()
            .to_dict()
        )
        self.assertEqual(counts[("en-ru", "teacher_kd")], 6)
        self.assertEqual(counts[("en-ru", "human_replay")], 4)
        self.assertEqual(counts[("ru-en", "teacher_kd")], 6)
        self.assertEqual(counts[("ru-en", "human_replay")], 4)
        self.assertEqual(report["en-ru"]["teacher_max_repeats"], 2)

    def test_quality_gate_blocks_preparation(self):
        config = {
            "pairs": [
                {
                    "id": "zh_uz",
                    "languages": ["zh", "uz"],
                    "kd_quality_status": "review_required",
                    "human_train": "human.jsonl",
                    "validation": "validation.jsonl",
                    "kd_train": "kd.jsonl",
                }
            ]
        }
        with self.assertRaisesRegex(RuntimeError, "review_required"):
            flow.prepare(config, "zh_uz", "exp2")

    def test_reuse_and_resume_modes_refuse_wrong_training_stage(self):
        base = {
            "kd_quality_status": "approved",
            "languages": ["en", "uz"],
        }
        with mock.patch.object(flow, "pair_config") as find:
            find.return_value = {"id": "en_uz", "mode": "reuse_exp2", **base}
            with self.assertRaisesRegex(RuntimeError, "must not be retrained"):
                flow.train({}, "en_uz", "exp2")
            find.return_value = {
                "id": "en_ru",
                "mode": "resume_exp2",
                **{**base, "languages": ["en", "ru"]},
            }
            with self.assertRaisesRegex(RuntimeError, "already has Exp1"):
                flow.train({}, "en_ru", "exp1")

    def test_gate_requires_both_metrics_in_both_directions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "results/evaluation/pair_specialists/en_ru"
            target.mkdir(parents=True)
            baseline = {
                "en-ru": {"bleu": 10.0, "chrf2": 20.0},
                "ru-en": {"bleu": 10.0, "chrf2": 20.0},
            }
            candidate = {
                "en-ru": {"bleu": 11.0, "chrf2": 21.0},
                "ru-en": {"bleu": 12.0, "chrf2": 19.0},
            }
            (target / "exp1.json").write_text(json.dumps(baseline))
            (target / "exp2.json").write_text(json.dumps(candidate))
            config = {"pairs": [{"id": "en_ru", "languages": ["en", "ru"]}]}
            with mock.patch.object(flow, "PROJECT_ROOT", root):
                report = flow.gate(config, "en_ru")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(
            [item["passed"] for item in report["directions"]], [True, False]
        )


if __name__ == "__main__":
    unittest.main()
