from __future__ import annotations

import json
from pathlib import Path

from scripts.synthetic.renderer_v054 import (
    V054Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.4"
FROZEN_PARENT = "0.5.3"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v054.json"
)

ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v054.json"
)

OUTPUT_FILE = (
    RESOURCE_DIR
    / "activation_v054.json"
)


ENABLED_VERBS = {
    "NEED",
    "LOSE",
}

ENABLED_FRAMES = {
    "NEED_OBJECT",
    "LOSE_OBJECT",
}

BLOCKED_VERBS = {
    "RETURN",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "RETURN_PLACE",
    "LEAVE_PLACE",
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected dict: {path}"
        )

    return data


def write_json(
    path: Path,
    data: dict,
) -> None:

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


def main() -> None:

    print("=" * 100)
    print("V0.5.4 ACTIVATION VALIDATOR")
    print("=" * 100)

    verbs_data = read_json(
        VERB_FILE
    )

    arguments_data = read_json(
        ARGUMENT_FILE
    )

    errors = []
    warnings = []

    verbs = verbs_data.get(
        "verbs",
        {},
    )

    contracts = arguments_data.get(
        "frame_contracts_v054",
        {},
    )

    objects = arguments_data.get(
        "objects",
        {},
    )

    # =========================================================================
    # Verbs
    # =========================================================================

    for verb_id in ENABLED_VERBS:
        if verb_id not in verbs:
            errors.append(
                f"Missing verb realization: "
                f"{verb_id}"
            )

    # =========================================================================
    # Frames
    # =========================================================================

    for frame_id in ENABLED_FRAMES:
        if frame_id not in contracts:
            errors.append(
                f"Missing frame contract: "
                f"{frame_id}"
            )

    if (
        contracts.get(
            "NEED_OBJECT",
            {},
        ).get("verb")
        != "NEED"
    ):
        errors.append(
            "NEED_OBJECT fixed verb mismatch."
        )

    if (
        contracts.get(
            "LOSE_OBJECT",
            {},
        ).get("verb")
        != "LOSE"
    ):
        errors.append(
            "LOSE_OBJECT fixed verb mismatch."
        )

    # =========================================================================
    # LOSE must remain future-only
    # =========================================================================

    lose_contract = contracts.get(
        "LOSE_OBJECT",
        {},
    )

    if (
        lose_contract.get(
            "allowed_tenses"
        )
        != ["future"]
    ):
        errors.append(
            "LOSE_OBJECT must be future-only."
        )

    lose_resource = verbs.get(
        "LOSE",
        {},
    )

    if (
        lose_resource.get(
            "generation_tenses"
        )
        != ["future"]
    ):
        errors.append(
            "LOSE realization generation_tenses "
            "must equal ['future']."
        )

    # =========================================================================
    # NEED morphology
    # =========================================================================

    for object_id, obj in objects.items():

        for field in (
            "ru_nom",
            "ru_number",
            "ru_need_agreement",
            "uz_nom",
        ):
            if field not in obj:
                errors.append(
                    f"{object_id}: missing {field}"
                )

        agreement = obj.get(
            "ru_need_agreement"
        )

        if agreement not in {
            "m",
            "f",
            "n",
            "pl",
        }:
            errors.append(
                f"{object_id}: invalid "
                f"ru_need_agreement="
                f"{agreement}"
            )

    # =========================================================================
    # Renderer smoke
    # =========================================================================

    try:
        renderer = V054Renderer()

        need = renderer.render(
            frame_id="NEED_OBJECT",
            slots={
                "subject": "PERSON_I",
                "verb": "NEED",
                "object": "ID_CARD",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        )

        lose = renderer.render(
            frame_id="LOSE_OBJECT",
            slots={
                "subject": "PERSON_HE",
                "verb": "LOSE",
                "object": "ID_CARD",
            },
            features={
                "tense": "future",
                "polarity": "neg",
            },
        )

        if set(need) != {
            "zh",
            "en",
            "ru",
            "uz",
        }:
            errors.append(
                "NEED renderer incomplete."
            )

        if set(lose) != {
            "zh",
            "en",
            "ru",
            "uz",
        }:
            errors.append(
                "LOSE renderer incomplete."
            )

        # present LOSE MUST fail
        try:
            renderer.render(
                frame_id="LOSE_OBJECT",
                slots={
                    "subject": "PERSON_I",
                    "verb": "LOSE",
                    "object": "ID_CARD",
                },
                features={
                    "tense": "present",
                    "polarity": "pos",
                },
            )

        except Exception:
            pass

        else:
            errors.append(
                "LOSE present tense was "
                "unexpectedly accepted."
            )

    except Exception as exc:
        errors.append(
            f"Renderer smoke failed: {exc}"
        )

    # =========================================================================
    # Activation
    # =========================================================================

    activation = {
        "activation_version":
            VERSION,

        "frozen_parent":
            FROZEN_PARENT,

        "status":
            (
                "validated"
                if not errors
                else "failed"
            ),

        "enabled_verbs":
            sorted(
                ENABLED_VERBS
            ),

        "enabled_frames":
            sorted(
                ENABLED_FRAMES
            ),

        "blocked_verbs":
            sorted(
                BLOCKED_VERBS
            ),

        "blocked_frames":
            sorted(
                BLOCKED_FRAMES
            ),

        "constraints": {
            "NEED_OBJECT": [
                "language_specific_need_construction",
                "ru_experiencer_dative",
                "ru_object_nominative",
                "ru_predicative_agreement",
                "uz_experiencer_dative",
                "uz_object_bare",
            ],

            "LOSE_OBJECT": [
                "future_only_generation",
                "present_generation_not_allowed",
                "past_not_yet_enabled",
                "ru_object_accusative",
                "uz_object_accusative",
            ],
        },

        "tense_policy": {
            "NEED_OBJECT": [
                "present",
                "future",
            ],

            "LOSE_OBJECT": [
                "future",
            ],
        },

        "generation_policy": {
            "v054_generation_allowed":
                not errors,

            "only_enabled_frames":
                True,

            "past_generation_allowed":
                False,

            "lose_present_generation_allowed":
                False,

            "lose_future_generation_allowed":
                True,
        },

        "warnings":
            warnings,

        "errors":
            errors,
    }

    write_json(
        OUTPUT_FILE,
        activation,
    )

    print(
        "Frozen parent:",
        FROZEN_PARENT,
    )

    print(
        "Enabled verbs:",
        sorted(ENABLED_VERBS),
    )

    print(
        "Enabled frames:",
        sorted(ENABLED_FRAMES),
    )

    print(
        "Blocked verbs:",
        sorted(BLOCKED_VERBS),
    )

    print()

    print(
        "NEED allowed tenses:",
        activation[
            "tense_policy"
        ][
            "NEED_OBJECT"
        ],
    )

    print(
        "LOSE allowed tenses:",
        activation[
            "tense_policy"
        ][
            "LOSE_OBJECT"
        ],
    )

    print()

    print(
        "Warnings:",
        len(warnings),
    )

    print(
        "Errors:",
        len(errors),
    )

    if errors:
        print()
        print("ERRORS")
        print("-" * 80)

        for error in errors:
            print(
                "ERROR:",
                error,
            )

    print()
    print(
        "Activation artifact:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 100)

    if errors:
        print(
            "V0.5.4 ACTIVATION "
            "VALIDATION FAILED"
        )

        print("=" * 100)

        raise SystemExit(1)

    print(
        "V0.5.4 ACTIVATION "
        "VALIDATION PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()