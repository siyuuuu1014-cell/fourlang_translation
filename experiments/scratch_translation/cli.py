"""Small CLI, deliberately independent of scripts/pipeline_v* entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import Timeout

from .common import GROUPS, PACKAGE, load_config, read_json, suite_root


def main(group=None):
    parser = argparse.ArgumentParser(
        description="Isolated, random-initialized four-language Transformer experiments"
    )
    parser.add_argument(
        "action",
        choices=("train", "evaluate") if group else ("prepare", "status", "compare"),
    )
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.toml")
    parser.add_argument(
        "--suite",
        help="New identifier for a changed config; both arms must use the same identifier",
    )
    if group:
        parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        parser.add_argument(
            "--stop-after-steps",
            type=int,
            help="Train at most N additional steps and save a resumable checkpoint (smoke test)",
        )
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.suite)
        if args.action == "prepare":
            from .data import prepare

            prepare(config)
        elif args.action == "status":
            root = suite_root(config)
            status = {
                "suite": config["experiment"]["suite"],
                "prepared": (root / "prepared/manifest.json").is_file(),
            }
            for arm in GROUPS:
                run = root / "runs" / arm
                # This is a disk snapshot, not a claim that a process is currently alive.
                status[arm] = {
                    name: read_json(run / name) if (run / name).is_file() else None
                    for name in ("latest.json", "train_report.json")
                }
            print(json.dumps(status, ensure_ascii=False, indent=2))
        elif args.action == "compare":
            from .engine import compare

            compare(config)
        elif args.action == "train":
            from .engine import train

            print(
                json.dumps(
                    train(config, group, args.device, args.stop_after_steps), indent=2
                )
            )
        else:
            if args.stop_after_steps is not None:
                parser.error("--stop-after-steps is only valid with train")
            from .engine import evaluate

            evaluate(config, group, args.device)
    except KeyboardInterrupt:
        parser.exit(
            130,
            "\nInterrupted. Re-run the SAME train command to resume its last committed checkpoint.\n",
        )
    except Timeout:
        parser.exit(
            2,
            "Another process holds this preparation/run lock. Do not launch the same arm twice.\n",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
