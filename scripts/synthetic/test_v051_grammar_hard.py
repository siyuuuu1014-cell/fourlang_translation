from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.grammar_hard_validator_v051 import (
    DEFAULT_ARGUMENT_REALIZATION,
    DEFAULT_VERB_REALIZATION,
    V051GrammarChecker,
    read_json,
)


def make_valid_see_she() -> dict:

    return {
        "semantic_id":
            "test",

        "frame_id":
            "SEE_OBJECT",

        "slots": {
            "subject":
                "PERSON_SHE",

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
                "她看见行李。",

            "en":
                "She sees the luggage.",

            "ru":
                "Она видит багаж.",

            "uz":
                "U bagajni ko'radi.",
        },
    }


def make_valid_take_she() -> dict:

    return {
        "semantic_id":
            "test_take",

        "frame_id":
            "TAKE_OBJECT",

        "slots": {
            "subject":
                "PERSON_SHE",

            "verb":
                "TAKE",

            "object":
                "CLOTHES",
        },

        "features": {
            "tense":
                "present",

            "polarity":
                "pos",
        },

        "texts": {
            "zh":
                "她拿衣服。",

            "en":
                "She takes clothes.",

            "ru":
                "Она берёт одежду.",

            "uz":
                "U kiyimni oladi.",
        },
    }


def assert_pass(
    name: str,
    checker: V051GrammarChecker,
    row: dict,
) -> None:

    accept, violations = (
        checker.validate(
            row
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


def assert_reject_language(
    name: str,
    checker: V051GrammarChecker,
    row: dict,
    language: str,
) -> None:

    accept, violations = (
        checker.validate(
            row
        )
    )

    if accept:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            "Expected rejection."
        )

    matches = [
        violation
        for violation in violations
        if (
            isinstance(
                violation,
                dict,
            )
            and violation.get(
                "type"
            )
            == "GRAMMAR_FORM_MISMATCH"
            and violation.get(
                "language"
            )
            == language
        )
    ]

    if not matches:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"Expected {language} mismatch.\n"
            f"Actual={violations}"
        )

    print(
        f"PASS: {name}"
    )


def main() -> None:

    checker = V051GrammarChecker(
        verb_resource=read_json(
            DEFAULT_VERB_REALIZATION
        ),

        argument_resource=read_json(
            DEFAULT_ARGUMENT_REALIZATION
        ),
    )

    passed = 0
    total = 0

    # ========================================================
    # 1. Valid SEE 3sg
    # ========================================================

    total += 1

    assert_pass(
        "valid SEE 3sg morphology",
        checker,
        make_valid_see_she(),
    )

    passed += 1

    # ========================================================
    # 2. English 3sg wrong: see instead of sees
    # ========================================================

    total += 1

    row = make_valid_see_she()

    row[
        "texts"
    ][
        "en"
    ] = "She see the luggage."

    assert_reject_language(
        "EN 3sg agreement blocked",
        checker,
        row,
        "en",
    )

    passed += 1

    # ========================================================
    # 3. Russian person wrong
    # ========================================================

    total += 1

    row = make_valid_see_she()

    row[
        "texts"
    ][
        "ru"
    ] = "Она вижу багаж."

    assert_reject_language(
        "RU person agreement blocked",
        checker,
        row,
        "ru",
    )

    passed += 1

    # ========================================================
    # 4. Uzbek person wrong
    # ========================================================

    total += 1

    row = make_valid_see_she()

    row[
        "texts"
    ][
        "uz"
    ] = "U bagajni ko'raman."

    assert_reject_language(
        "UZ person agreement blocked",
        checker,
        row,
        "uz",
    )

    passed += 1

    # ========================================================
    # 5. Russian accusative wrong
    # ========================================================

    total += 1

    row = make_valid_take_she()

    row[
        "texts"
    ][
        "ru"
    ] = "Она берёт одежда."

    assert_reject_language(
        "RU object accusative blocked",
        checker,
        row,
        "ru",
    )

    passed += 1

    # ========================================================
    # 6. Uzbek -ni missing
    # ========================================================

    total += 1

    row = make_valid_see_she()

    row[
        "texts"
    ][
        "uz"
    ] = "U bagaj ko'radi."

    assert_reject_language(
        "UZ accusative -ni blocked",
        checker,
        row,
        "uz",
    )

    passed += 1

    # ========================================================
    # 7. English negative 3sg wrong
    # ========================================================

    total += 1

    row = make_valid_see_she()

    row[
        "features"
    ][
        "polarity"
    ] = "neg"

    row[
        "texts"
    ] = {
        "zh":
            "她没看见行李。",

        "en":
            "She do not see the luggage.",

        "ru":
            "Она не видит багаж.",

        "uz":
            "U bagajni ko'rmaydi.",
    }

    assert_reject_language(
        "EN negative 3sg blocked",
        checker,
        row,
        "en",
    )

    passed += 1

    # ========================================================
    # 8. Valid TAKE object case
    # ========================================================

    total += 1

    assert_pass(
        "valid TAKE case morphology",
        checker,
        make_valid_take_she(),
    )

    passed += 1

    print()
    print("=" * 90)

    print(
        f"V0.5.1 GRAMMAR HARD REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print("=" * 90)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()