from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.semantic_compatibility_validator_v051 import (
    V05_ARGUMENTS,
    V05_COMPATIBILITY,
    read_json,
    validate_new_row,
)


def assert_pass(
    name: str,
    row: dict,
    compatibility: dict,
    arguments: dict,
) -> None:

    accept, violations, warnings = (
        validate_new_row(
            row,
            compatibility,
            arguments,
        )
    )

    if not accept:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"violations={violations}"
        )

    print(
        f"PASS: {name}"
    )


def assert_reject(
    name: str,
    row: dict,
    expected_type: str,
    compatibility: dict,
    arguments: dict,
) -> None:

    accept, violations, warnings = (
        validate_new_row(
            row,
            compatibility,
            arguments,
        )
    )

    if accept:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"Expected reject but accepted."
        )

    actual_types = {
        v.get(
            "type"
        )
        for v in violations
        if isinstance(
            v,
            dict,
        )
    }

    if expected_type not in actual_types:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"Expected violation="
            f"{expected_type}\n"
            f"Actual={violations}"
        )

    print(
        f"PASS: {name}"
    )


def make_row(
    *,
    frame: str,
    verb: str,
    object_id: str,
) -> dict:

    return {
        "semantic_id":
            "test",

        "frame_id":
            frame,

        "slots": {
            "subject":
                "PERSON_I",

            "verb":
                verb,

            "object":
                object_id,
        },

        "features": {
            "tense":
                "present",

            "polarity":
                "pos",
        },

        "texts": {
            "zh": "TEST",
            "en": "TEST",
            "ru": "TEST",
            "uz": "TEST",
        },
    }


def main() -> None:

    compatibility = read_json(
        V05_COMPATIBILITY
    )

    arguments = read_json(
        V05_ARGUMENTS
    )

    passed = 0
    total = 0

    # ========================================================
    # 1. SEE + ROOM is legal
    # ========================================================

    total += 1

    assert_pass(
        "SEE + ROOM allowed",
        make_row(
            frame="SEE_OBJECT",
            verb="SEE",
            object_id="ROOM",
        ),
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 2. SEE + TABLE is legal
    # ========================================================

    total += 1

    assert_pass(
        "SEE + TABLE allowed",
        make_row(
            frame="SEE_OBJECT",
            verb="SEE",
            object_id="TABLE",
        ),
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 3. TAKE + LUGGAGE is legal
    # ========================================================

    total += 1

    assert_pass(
        "TAKE + LUGGAGE allowed",
        make_row(
            frame="TAKE_OBJECT",
            verb="TAKE",
            object_id="LUGGAGE",
        ),
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 4. TAKE + ROOM must be rejected
    # ========================================================

    total += 1

    assert_reject(
        "TAKE + ROOM blocked",
        make_row(
            frame="TAKE_OBJECT",
            verb="TAKE",
            object_id="ROOM",
        ),
        "OBJECT_CLASS_NOT_ALLOWED",
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 5. TAKE + TABLE must be rejected
    # ========================================================

    total += 1

    assert_reject(
        "TAKE + TABLE blocked",
        make_row(
            frame="TAKE_OBJECT",
            verb="TAKE",
            object_id="TABLE",
        ),
        "OBJECT_CLASS_NOT_ALLOWED",
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 6. SEE + MONEY is not enabled by current compatibility
    # ========================================================

    total += 1

    assert_reject(
        "SEE + MONEY blocked",
        make_row(
            frame="SEE_OBJECT",
            verb="SEE",
            object_id="MONEY",
        ),
        "OBJECT_CLASS_NOT_ALLOWED",
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 7. Wrong verb inside SEE frame
    # ========================================================

    total += 1

    assert_reject(
        "SEE_OBJECT + TAKE blocked",
        make_row(
            frame="SEE_OBJECT",
            verb="TAKE",
            object_id="LUGGAGE",
        ),
        "FRAME_VERB_MISMATCH",
        compatibility,
        arguments,
    )

    passed += 1

    # ========================================================
    # 8. LOSE must remain blocked
    # ========================================================

    total += 1

    assert_reject(
        "LOSE_OBJECT remains blocked",
        make_row(
            frame="LOSE_OBJECT",
            verb="LOSE",
            object_id="LUGGAGE",
        ),
        "BLOCKED_FRAME",
        compatibility,
        arguments,
    )

    passed += 1

    print()
    print("=" * 90)

    print(
        f"V0.5.1 COMPATIBILITY REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print("=" * 90)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()