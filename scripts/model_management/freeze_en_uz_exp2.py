from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


SOURCE_RELATIVE = (
    "results/student/small100/"
    "exp2_distillation_v1/best_model"
)

DESTINATION_RELATIVE = (
    "models/final_specialists/"
    "en_uz_small100_v1"
)


def project_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Freeze the already accepted EN<->UZ "
            "SMaLL-100 Exp2 into the deployment model area."
        )
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
    )

    return p.parse_args()


def main():
    args = parse_args()

    root = project_root()

    source = (
        root
        / SOURCE_RELATIVE
    ).resolve()

    destination = (
        root
        / DESTINATION_RELATIVE
    ).resolve()

    print("=" * 100)
    print("FREEZE EN<->UZ SMALL100 EXP2")
    print("=" * 100)

    print("\nSource:")
    print(source)

    print("\nDestination:")
    print(destination)

    if not source.exists():
        raise FileNotFoundError(
            source
        )

    required_files = {
        "config.json",
        "model.safetensors",
        "tokenization_small100.py",
    }

    missing = {
        name
        for name
        in required_files
        if not (
            source
            /
            name
        ).exists()
    }

    if missing:
        raise RuntimeError(
            "Source Exp2 model is incomplete. Missing: "
            f"{sorted(missing)}"
        )

    if destination.exists():
        if not args.overwrite:
            raise RuntimeError(
                "\nDestination already exists:\n"
                f"{destination}\n\n"
                "Use --overwrite only if you intentionally "
                "want to recreate the frozen deployment copy."
            )

        shutil.rmtree(
            destination
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copytree(
        source,
        destination,
    )

    model_card = {
        "model_name": "en_uz_small100_exp2_v1",
        "architecture": "SMaLL-100",
        "directions": [
            "en_uz",
            "uz_en",
        ],
        "source_experiment": SOURCE_RELATIVE,
        "frozen_model_path": DESTINATION_RELATIVE,
        "training_stage": "Exp2",
        "final_decision": "EXP2_ACCEPT",
        "status": "production_candidate",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    (
        destination
        /
        "model_card.json"
    ).write_text(
        json.dumps(
            model_card,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFrozen deployment model created.")
    print(destination)

    print(
        "\nNo tracked configuration file was modified. "
        "The registry already points to this deployment path; "
        "runtime availability is detected from path existence."
    )


if __name__ == "__main__":
    main()
