from __future__ import annotations

import json
from pathlib import Path

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.3"
FROZEN_PARENT = "0.5.2"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v053.json"
)

ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v053.json"
)

OUTPUT_FILE = (
    RESOURCE_DIR
    / "activation_v053.json"
)


ENABLED_VERBS = {
    "GIVE",
    "BRING",
}

ENABLED_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}

BLOCKED_VERBS = {
    "LEAVE",
    "LOSE",
    "NEED",
    "RETURN",
}

BLOCKED_FRAMES = {
    "LEAVE_PLACE",
    "LOSE_OBJECT",
    "NEED_OBJECT",
    "RETURN_PLACE",
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
        return json.load(f)


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
    print(
        "V0.5.3 ACTIVATION VALIDATOR"
    )
    print("=" * 100)

    verb_data = read_json(
        VERB_FILE
    )

    arg_data = read_json(
        ARGUMENT_FILE
    )

    errors = []
    warnings = []

    verbs = verb_data.get(
        "verbs",
        {},
    )

    contracts = arg_data.get(
        "frame_contracts_v053",
        {},
    )

    places = arg_data.get(
        "place_targets",
        {},
    )

    # -------------------------------------------------------------------------
    # Verb realization
    # -------------------------------------------------------------------------

    for verb_id in ENABLED_VERBS:

        if verb_id not in verbs:
            errors.append(
                f"Missing verb realization: "
                f"{verb_id}"
            )

    # -------------------------------------------------------------------------
    # Frame contracts
    # -------------------------------------------------------------------------

    for frame_id in ENABLED_FRAMES:

        if frame_id not in contracts:
            errors.append(
                f"Missing frame contract: "
                f"{frame_id}"
            )

    if (
        contracts.get(
            "GIVE_OBJECT_PERSON",
            {},
        ).get("verb")
        != "GIVE"
    ):
        errors.append(
            "GIVE_OBJECT_PERSON "
            "verb contract mismatch."
        )

    if (
        contracts.get(
            "BRING_OBJECT_DESTINATION",
            {},
        ).get("verb")
        != "BRING"
    ):
        errors.append(
            "BRING_OBJECT_DESTINATION "
            "verb contract mismatch."
        )

    # -------------------------------------------------------------------------
    # Directional places
    # -------------------------------------------------------------------------

    for place_id, record in (
        places.items()
    ):

        for key in (
            "zh_dir",
            "en_dir",
            "ru_dir",
            "uz_dir",
        ):

            if not record.get(key):

                errors.append(
                    f"{place_id}: "
                    f"missing {key}"
                )

    # -------------------------------------------------------------------------
    # Renderer smoke test
    # -------------------------------------------------------------------------

    try:
        renderer = (
            V053Renderer()
        )

        give = renderer.render(
            frame_id=(
                "GIVE_OBJECT_PERSON"
            ),
            slots={
                "subject": "PERSON_HE",
                "verb": "GIVE",
                "object": "ID_CARD",
                "person": "PERSON_YOU",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        )

        bring = renderer.render(
            frame_id=(
                "BRING_OBJECT_DESTINATION"
            ),
            slots={
                "subject": "PERSON_SHE",
                "verb": "BRING",
                "object": "LUGGAGE",
                "destination": "HOTEL",
            },
            features={
                "tense": "future",
                "polarity": "pos",
            },
        )

        if len(give) != 4:
            errors.append(
                "GIVE renderer incomplete."
            )

        if len(bring) != 4:
            errors.append(
                "BRING renderer incomplete."
            )

    except Exception as exc:

        errors.append(
            f"Renderer smoke failed: "
            f"{exc}"
        )

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
            "GIVE_OBJECT_PERSON": [
                "subject_must_not_equal_person",
                "speaker_group_overlap_not_allowed",
                "object_role_ru_accusative",
                "person_role_ru_dative",
                "object_role_uz_accusative",
                "person_role_uz_dative",
                "object_person_roles_must_not_swap",
            ],

            "BRING_OBJECT_DESTINATION": [
                "object_role_ru_accusative",
                "object_role_uz_accusative",
                "destination_role_directional",
                "object_destination_roles_must_not_swap",
            ],
        },

        "generation_policy": {
            "v053_generation_allowed":
                not errors,

            "only_enabled_frames":
                True,

            "reflexive_generation_allowed":
                False,

            "speaker_group_overlap_allowed":
                False,

            "role_swap_allowed":
                False,
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

    print(
        "Directional targets:",
        len(places),
    )

    print(
        "Warnings:",
        len(warnings),
    )

    print(
        "Errors:",
        len(errors),
    )

    if warnings:

        print()
        print("WARNINGS")
        print("-" * 80)

        for warning in warnings:
            print(
                "WARN:",
                warning,
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
            "V0.5.3 ACTIVATION "
            "VALIDATION FAILED"
        )

        print("=" * 100)

        raise SystemExit(1)

    print(
        "V0.5.3 ACTIVATION "
        "VALIDATION PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()