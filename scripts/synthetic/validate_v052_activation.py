from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.2"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ARGUMENT_RESOURCE = (
    RESOURCE_DIR
    / "argument_realization_v052.json"
)

VERB_RESOURCE = (
    RESOURCE_DIR
    / "verb_realization_v052.json"
)

RESOURCE_MANIFEST = (
    RESOURCE_DIR
    / "manifest_v052.json"
)

OUTPUT_ACTIVATION = (
    RESOURCE_DIR
    / "activation_v052.json"
)


# ============================================================
# Activation policy
# ============================================================

FROZEN_PARENT_VERSION = "0.5.1"

NEW_ENABLED_VERBS = {
    "CALL",
    "WAIT",
}

NEW_ENABLED_FRAMES = {
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
}


# These capabilities may already exist at resource level,
# but MUST NOT become generation-enabled in V0.5.2.
BLOCKED_VERBS = {
    "GIVE",
    "BRING",
    "LOSE",
    "RETURN",
    "NEED",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "LOSE_OBJECT",
    "RETURN_PLACE",
    "NEED_OBJECT",
    "LEAVE_PLACE",
}


FRAME_TO_VERB = {
    "CALL_PERSON": "CALL",
    "WAIT_PERSON": "WAIT",
    "WAIT_AT_PLACE": "WAIT",
}


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(data, dict):

        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return data


def write_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


# ============================================================
# Validation
# ============================================================

def main() -> None:

    print("=" * 100)
    print("V0.5.2 ACTIVATION VALIDATOR")
    print("=" * 100)

    errors = []
    warnings = []

    verbs_resource = read_json(
        VERB_RESOURCE
    )

    arguments_resource = read_json(
        ARGUMENT_RESOURCE
    )

    manifest = read_json(
        RESOURCE_MANIFEST
    )

    verbs = verbs_resource.get(
        "verbs",
        {},
    )

    frame_contracts = (
        arguments_resource.get(
            "frame_contracts_v052",
            {},
        )
    )

    person_targets = (
        arguments_resource.get(
            "person_targets",
            {},
        )
    )

    place_targets = (
        arguments_resource.get(
            "place_targets",
            {},
        )
    )

    # ========================================================
    # 1. Version integrity
    # ========================================================

    if (
        verbs_resource.get(
            "resource_version"
        )
        != VERSION
    ):

        errors.append(
            "Verb resource version is not 0.5.2."
        )

    if (
        arguments_resource.get(
            "resource_version"
        )
        != VERSION
    ):

        errors.append(
            "Argument resource version is not 0.5.2."
        )

    if (
        manifest.get(
            "version"
        )
        != VERSION
    ):

        errors.append(
            "Manifest version is not 0.5.2."
        )

    if (
        manifest.get(
            "frozen_parent"
        )
        != FROZEN_PARENT_VERSION
    ):

        errors.append(
            "Frozen parent must be 0.5.1."
        )

    # ========================================================
    # 2. New verb resources exist
    # ========================================================

    for verb_id in sorted(
        NEW_ENABLED_VERBS
    ):

        if verb_id not in verbs:

            errors.append(
                f"Enabled verb missing from "
                f"verb resource: {verb_id}"
            )

    # ========================================================
    # 3. Frame contracts exist
    # ========================================================

    for frame_id in sorted(
        NEW_ENABLED_FRAMES
    ):

        contract = frame_contracts.get(
            frame_id
        )

        if not isinstance(
            contract,
            dict,
        ):

            errors.append(
                f"Missing frame contract: "
                f"{frame_id}"
            )

            continue

        expected_verb = (
            FRAME_TO_VERB[
                frame_id
            ]
        )

        actual_verb = contract.get(
            "verb"
        )

        if actual_verb != expected_verb:

            errors.append(
                f"{frame_id}: expected fixed "
                f"verb {expected_verb}, "
                f"got {actual_verb}"
            )

    # ========================================================
    # 4. Required slot contracts
    # ========================================================

    expected_required_slots = {

        "CALL_PERSON": {
            "subject",
            "verb",
            "person",
        },

        "WAIT_PERSON": {
            "subject",
            "verb",
            "person",
        },

        "WAIT_AT_PLACE": {
            "subject",
            "verb",
            "place",
        },
    }

    for frame_id, expected in (
        expected_required_slots.items()
    ):

        contract = frame_contracts.get(
            frame_id,
            {},
        )

        actual = set(
            contract.get(
                "required_slots",
                [],
            )
        )

        if actual != expected:

            errors.append(
                f"{frame_id}: required_slots "
                f"mismatch. "
                f"expected={sorted(expected)}, "
                f"actual={sorted(actual)}"
            )

    # ========================================================
    # 5. Reflexive protection
    # ========================================================

    for frame_id in (
        "CALL_PERSON",
        "WAIT_PERSON",
    ):

        contract = frame_contracts.get(
            frame_id,
            {},
        )

        constraints = set(
            contract.get(
                "constraints",
                [],
            )
        )

        if (
            "subject_must_not_equal_person"
            not in constraints
        ):

            errors.append(
                f"{frame_id}: missing "
                "subject_must_not_equal_person "
                "constraint."
            )

    # ========================================================
    # 6. Argument realization resources
    # ========================================================

    if len(
        person_targets
    ) < 6:

        errors.append(
            "Expected at least 6 person targets."
        )

    if len(
        place_targets
    ) < 11:

        errors.append(
            "Expected at least 11 place targets."
        )

    required_person_fields = {
        "zh",
        "en_object",
        "ru_acc",
        "ru_dat",
        "uz_acc",
        "uz_dat",
    }

    for person_id, data in (
        person_targets.items()
    ):

        missing = (
            required_person_fields
            - set(data)
        )

        if missing:

            errors.append(
                f"{person_id}: missing "
                f"person realization fields "
                f"{sorted(missing)}"
            )

    required_place_fields = {
        "zh_loc",
        "en_loc",
        "ru_loc",
        "uz_loc",
    }

    for place_id, data in (
        place_targets.items()
    ):

        missing = (
            required_place_fields
            - set(data)
        )

        if missing:

            errors.append(
                f"{place_id}: missing "
                f"place realization fields "
                f"{sorted(missing)}"
            )

    # ========================================================
    # 7. Import actual Renderer
    # ========================================================

    try:

        from scripts.synthetic.renderer_v052 import (
            V052Renderer,
            NEW_FRAMES as RENDERER_NEW_FRAMES,
        )

        renderer = V052Renderer()

    except Exception as exc:

        errors.append(
            "Renderer V0.5.2 failed to load: "
            f"{exc}"
        )

        renderer = None
        RENDERER_NEW_FRAMES = set()

    # ========================================================
    # 8. Renderer activation consistency
    # ========================================================

    if set(
        RENDERER_NEW_FRAMES
    ) != NEW_ENABLED_FRAMES:

        errors.append(
            "Renderer NEW_FRAMES mismatch. "
            f"expected="
            f"{sorted(NEW_ENABLED_FRAMES)}, "
            f"actual="
            f"{sorted(RENDERER_NEW_FRAMES)}"
        )

    # ========================================================
    # 9. Positive smoke rendering
    # ========================================================

    positive_cases = [

        (
            "CALL_PERSON",
            {
                "subject":
                    "PERSON_I",

                "verb":
                    "CALL",

                "person":
                    "PERSON_HE",
            },
        ),

        (
            "WAIT_PERSON",
            {
                "subject":
                    "PERSON_I",

                "verb":
                    "WAIT",

                "person":
                    "PERSON_HE",
            },
        ),

        (
            "WAIT_AT_PLACE",
            {
                "subject":
                    "PERSON_I",

                "verb":
                    "WAIT",

                "place":
                    "STATION",
            },
        ),
    ]

    if renderer is not None:

        for frame_id, slots in (
            positive_cases
        ):

            try:

                texts = renderer.render(
                    frame_id=frame_id,
                    slots=slots,
                    features={
                        "tense":
                            "present",

                        "polarity":
                            "pos",
                    },
                )

                if set(
                    texts.keys()
                ) != {
                    "zh",
                    "en",
                    "ru",
                    "uz",
                }:

                    errors.append(
                        f"{frame_id}: renderer "
                        "did not return all four "
                        "languages."
                    )

            except Exception as exc:

                errors.append(
                    f"{frame_id}: activation "
                    f"render failed: {exc}"
                )

    # ========================================================
    # 10. Reflexive negative test
    # ========================================================

    if renderer is not None:

        reflexive_cases = [

            (
                "CALL_PERSON",
                {
                    "subject":
                        "PERSON_I",

                    "verb":
                        "CALL",

                    "person":
                        "PERSON_I",
                },
            ),

            (
                "WAIT_PERSON",
                {
                    "subject":
                        "PERSON_HE",

                    "verb":
                        "WAIT",

                    "person":
                        "PERSON_HE",
                },
            ),
        ]

        for frame_id, slots in (
            reflexive_cases
        ):

            try:

                renderer.render(
                    frame_id=frame_id,
                    slots=slots,
                    features={
                        "tense":
                            "present",

                        "polarity":
                            "pos",
                    },
                )

            except Exception:
                pass

            else:

                errors.append(
                    f"{frame_id}: reflexive "
                    "leakage was not blocked."
                )

    # ========================================================
    # 11. Block future batches
    # ========================================================

    for verb_id in sorted(
        BLOCKED_VERBS
    ):

        if verb_id in NEW_ENABLED_VERBS:

            errors.append(
                f"Blocked verb accidentally "
                f"enabled: {verb_id}"
            )

    for frame_id in sorted(
        BLOCKED_FRAMES
    ):

        if frame_id in NEW_ENABLED_FRAMES:

            errors.append(
                f"Blocked frame accidentally "
                f"enabled: {frame_id}"
            )

    # ========================================================
    # 12. V0.5.1 fallback sanity check
    # ========================================================

    if renderer is not None:

        try:

            see_texts = renderer.render(
                frame_id="SEE_OBJECT",

                slots={
                    "subject":
                        "PERSON_I",

                    "verb":
                        "SEE",

                    "object":
                        "LUGGAGE",
                },

                features={
                    "tense":
                        "present",

                    "polarity":
                        "pos",
                },
            )

            expected = {
                "zh":
                    "我看见行李。",

                "en":
                    "I see the luggage.",

                "ru":
                    "Я вижу багаж.",

                "uz":
                    "Men bagajni ko'raman.",
            }

            if see_texts != expected:

                errors.append(
                    "Frozen V0.5.1 SEE fallback "
                    "changed."
                )

        except Exception as exc:

            errors.append(
                "Frozen V0.5.1 fallback failed: "
                f"{exc}"
            )

    # ========================================================
    # Report
    # ========================================================

    print(
        "Frozen parent:",
        FROZEN_PARENT_VERSION,
    )

    print(
        "Enabled verbs:",
        sorted(
            NEW_ENABLED_VERBS
        ),
    )

    print(
        "Enabled frames:",
        sorted(
            NEW_ENABLED_FRAMES
        ),
    )

    print(
        "Blocked verbs:",
        sorted(
            BLOCKED_VERBS
        ),
    )

    print(
        "Person targets:",
        len(
            person_targets
        ),
    )

    print(
        "Place targets:",
        len(
            place_targets
        ),
    )

    print(
        "Warnings:",
        len(
            warnings
        ),
    )

    print(
        "Errors:",
        len(
            errors
        ),
    )

    if warnings:

        print()
        print("WARNINGS")
        print("-" * 100)

        for warning in warnings:

            print(
                "WARN:",
                warning,
            )

    if errors:

        print()
        print("ERRORS")
        print("-" * 100)

        for error in errors:

            print(
                "ERROR:",
                error,
            )

        print()
        print("=" * 100)
        print(
            "V0.5.2 ACTIVATION VALIDATION FAILED"
        )
        print("=" * 100)

        raise SystemExit(1)

    # ========================================================
    # Only create activation artifact after successful gate.
    # ========================================================

    activation = {

        "activation_version":
            VERSION,

        "frozen_parent":
            FROZEN_PARENT_VERSION,

        "status":
            "validated",

        "enabled_verbs":
            sorted(
                NEW_ENABLED_VERBS
            ),

        "enabled_frames":
            sorted(
                NEW_ENABLED_FRAMES
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
            "CALL_PERSON": [
                "subject_must_not_equal_person",
                "person_role_ru_dative",
                "person_role_uz_dative",
            ],

            "WAIT_PERSON": [
                "subject_must_not_equal_person",
                "person_role_ru_accusative",
                "person_role_uz_accusative",
                "english_wait_for",
            ],

            "WAIT_AT_PLACE": [
                "language_specific_locative",
            ],
        },

        "generation_policy": {
            "v052_generation_allowed":
                True,

            "only_enabled_frames":
                True,

            "reflexive_generation_allowed":
                False,
        },
    }

    write_json(
        OUTPUT_ACTIVATION,
        activation,
    )

    print()
    print(
        "Activation artifact:"
    )

    print(
        OUTPUT_ACTIVATION
    )

    print()
    print("=" * 100)
    print(
        "V0.5.2 ACTIVATION VALIDATION PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()