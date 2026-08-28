from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.hard_semantic_validator_v051 import (
    build_renderer,
    validate_new_row,
)


def make_valid_row() -> dict:

    return {
        "semantic_id":
            "test_v051",

        "scenario":
            "daily",

        "frame_id":
            "SEE_OBJECT",

        "slots": {
            "subject":
                "PERSON_I",

            "verb":
                "SEE",

            "object":
                "LUGGAGE",
        },

        "features": {
            "tense":
                "present",

            "polarity":
                "pos",
        },

        "texts": {
            "zh":
                "我看见行李。",

            "en":
                "I see the luggage.",

            "ru":
                "Я вижу багаж.",

            "uz":
                "Men bagajni ko'raman.",
        },

        "compatibility_validation": {
            "accept":
                True,

            "violations":
                [],
        },
    }


def assert_pass(
    name: str,
    row: dict,
    renderer,
) -> None:

    accept, violations = (
        validate_new_row(
            row,
            renderer,
        )
    )

    if not accept:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"{violations}"
        )

    print(
        f"PASS: {name}"
    )


def assert_reject(
    name: str,
    row: dict,
    expected_type: str,
    renderer,
) -> None:

    accept, violations = (
        validate_new_row(
            row,
            renderer,
        )
    )

    if accept:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            "Expected reject."
        )

    types = {
        violation.get(
            "type"
        )
        for violation in violations
        if isinstance(
            violation,
            dict,
        )
    }

    if expected_type not in types:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"Expected: {expected_type}\n"
            f"Actual: {violations}"
        )

    print(
        f"PASS: {name}"
    )


def main() -> None:

    renderer = build_renderer()

    passed = 0
    total = 0

    # ========================================================
    # 1. Valid row
    # ========================================================

    total += 1

    row = make_valid_row()

    assert_pass(
        "valid SEE row",
        row,
        renderer,
    )

    passed += 1

    # ========================================================
    # 2. English text changed
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "texts"
    ][
        "en"
    ] = "I take the luggage."

    assert_reject(
        "English semantic mismatch",
        row,
        "TEXT_RENDER_MISMATCH",
        renderer,
    )

    passed += 1

    # ========================================================
    # 3. Russian text changed
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "texts"
    ][
        "ru"
    ] = "Я беру багаж."

    assert_reject(
        "Russian semantic mismatch",
        row,
        "TEXT_RENDER_MISMATCH",
        renderer,
    )

    passed += 1

    # ========================================================
    # 4. Uzbek object/meaning changed
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "texts"
    ][
        "uz"
    ] = "Men pulni ko'raman."

    assert_reject(
        "Uzbek semantic mismatch",
        row,
        "TEXT_RENDER_MISMATCH",
        renderer,
    )

    passed += 1

    # ========================================================
    # 5. Wrong frame verb
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "slots"
    ][
        "verb"
    ] = "TAKE"

    assert_reject(
        "frame verb mismatch",
        row,
        "FRAME_VERB_MISMATCH",
        renderer,
    )

    passed += 1

    # ========================================================
    # 6. Missing object
    # ========================================================

    total += 1

    row = make_valid_row()

    del row[
        "slots"
    ][
        "object"
    ]

    assert_reject(
        "missing object blocked",
        row,
        "MISSING_REQUIRED_SLOT",
        renderer,
    )

    passed += 1

    # ========================================================
    # 7. Illegal event_type leakage
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "features"
    ][
        "event_type"
    ] = "habitual"

    assert_reject(
        "unexpected event type blocked",
        row,
        "UNEXPECTED_EVENT_TYPE",
        renderer,
    )

    passed += 1

    # ========================================================
    # 8. Compatibility rejection cannot enter
    # ========================================================

    total += 1

    row = make_valid_row()

    row[
        "compatibility_validation"
    ][
        "accept"
    ] = False

    assert_reject(
        "compatibility reject blocked",
        row,
        "COMPATIBILITY_NOT_ACCEPTED",
        renderer,
    )

    passed += 1

    print()

    print(
        "=" * 90
    )

    print(
        f"V0.5.1 HARD SEMANTIC REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print(
        "=" * 90
    )

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()