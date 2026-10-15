from __future__ import annotations

import argparse
import json
from pathlib import Path

from filelock import Timeout

from .common import PACKAGE, load_config, read_json, suite_root


def main():
    parser = argparse.ArgumentParser(
        description="Isolated scratch Transformer v2 (never writes mainline or v1 artifacts)"
    )
    parser.add_argument("action", choices=("diagnose-v1", "train", "evaluate", "status"))
    parser.add_argument("--config", type=Path, default=PACKAGE / "config.toml")
    parser.add_argument("--suite", help="Use a new identifier whenever config/code/data changes")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--stop-after-steps", type=int,
                        help="Commit at most N additional updates, then pause safely")
    args = parser.parse_args()
    try:
        config = load_config(args.config, args.suite)
        if args.action == "train":
            from .engine import train
            result = train(config, args.device, args.stop_after_steps)
        elif args.action == "evaluate":
            if args.stop_after_steps is not None:
                parser.error("--stop-after-steps is valid only for train")
            from .engine import evaluate
            result = evaluate(config, args.device)
        elif args.action == "diagnose-v1":
            if args.stop_after_steps is not None:
                parser.error("--stop-after-steps is valid only for train")
            from .diagnostics import diagnose_v1
            result = diagnose_v1(config, args.device)
        else:
            run = suite_root(config) / "run"
            result = {"suite": config["experiment"]["suite"],
                      "v2_only": True,
                      "latest": read_json(run / "latest.json") if (run / "latest.json").is_file() else None,
                      "train_report": read_json(run / "train_report.json") if (run / "train_report.json").is_file() else None,
                      "v1_diagnostics": (suite_root(config) / "v1_diagnostics/comparison.json").is_file()}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        parser.exit(130, "\nInterrupted; repeat the identical train command to resume the last checkpoint.\n")
    except Timeout:
        parser.exit(2, "This v2 run is already locked by another process.\n")
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")
