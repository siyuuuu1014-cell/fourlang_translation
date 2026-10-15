from __future__ import annotations

from scripts.synthetic.semantic_compatibility_validator_v052 import (
    validate_row,
    load_resources,
)


def expect_accept(
    name,
    row,
    person_targets,
    place_targets,
):

    violations, warnings = validate_row(
        row,
        person_targets=person_targets,
        place_targets=place_targets,
    )

    if violations:
        raise AssertionError(
            f"{name}: unexpected violations "
            f"{violations}"
        )

    print(
        "PASS:",
        name,
    )


def expect_reject(
    name,
    row,
    person_targets,
    place_targets,
):

    violations, warnings = validate_row(
        row,
        person_targets=person_targets,
        place_targets=place_targets,
    )

    if not violations:

        raise AssertionError(
            f"{name}: expected reject"
        )

    print(
        "PASS:",
        name,
    )


def make_row(
    frame_id,
    slots,
):

    return {
        "semantic_id":
            "test",

        "scenario":
            "daily",

        "frame_id":
            frame_id,

        "slots":
            slots,

        "features": {
            "tense":
                "present",

            "polarity":
                "pos",
        },

        "texts": {
            "zh": "测试。",
            "en": "Test.",
            "ru": "Тест.",
            "uz": "Test.",
        },

        "metadata": {
            "generator_version":
                "0.5.2",

            "generation_source":
                "v052_new_frame",
        },
    }


def main():

    (
        activation,
        person_targets,
        place_targets,
        frame_contracts,
    ) = load_resources()

    tests = 0

    # --------------------------------------------------------
    # 1 valid CALL
    # --------------------------------------------------------

    expect_accept(
        "valid CALL_PERSON",

        make_row(
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

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 2 CALL reflexive blocked
    # --------------------------------------------------------

    expect_reject(
        "CALL reflexive blocked",

        make_row(
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

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 3 CALL wrong verb
    # --------------------------------------------------------

    expect_reject(
        "CALL wrong verb blocked",

        make_row(
            "CALL_PERSON",

            {
                "subject":
                    "PERSON_I",

                "verb":
                    "WAIT",

                "person":
                    "PERSON_HE",
            },
        ),

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 4 CALL missing target
    # --------------------------------------------------------

    expect_reject(
        "CALL missing person blocked",

        make_row(
            "CALL_PERSON",

            {
                "subject":
                    "PERSON_I",

                "verb":
                    "CALL",
            },
        ),

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 5 valid WAIT_PERSON
    # --------------------------------------------------------

    expect_accept(
        "valid WAIT_PERSON",

        make_row(
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

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 6 WAIT reflexive
    # --------------------------------------------------------

    expect_reject(
        "WAIT reflexive blocked",

        make_row(
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

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 7 valid WAIT place
    # --------------------------------------------------------

    expect_accept(
        "valid WAIT_AT_PLACE",

        make_row(
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

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 8 unknown place
    # --------------------------------------------------------

    expect_reject(
        "WAIT unknown place blocked",

        make_row(
            "WAIT_AT_PLACE",

            {
                "subject":
                    "PERSON_I",

                "verb":
                    "WAIT",

                "place":
                    "UNKNOWN_PLACE",
            },
        ),

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 9 WAIT person/place confusion
    # --------------------------------------------------------

    expect_reject(
        "WAIT_AT_PLACE person leakage blocked",

        make_row(
            "WAIT_AT_PLACE",

            {
                "subject":
                    "PERSON_I",

                "verb":
                    "WAIT",

                "place":
                    "STATION",

                "person":
                    "PERSON_HE",
            },
        ),

        person_targets,
        place_targets,
    )

    tests += 1

    # --------------------------------------------------------
    # 10 blocked GIVE
    # --------------------------------------------------------

    expect_reject(
        "GIVE remains blocked",

        make_row(
            "GIVE_OBJECT_PERSON",

            {
                "subject":
                    "PERSON_I",

                "verb":
                    "GIVE",

                "object":
                    "BOOK",

                "person":
                    "PERSON_HE",
            },
        ),

        person_targets,
        place_targets,
    )

    tests += 1

    print()
    print("=" * 100)

    print(
        f"V0.5.2 COMPATIBILITY REGRESSION: "
        f"{tests}/{tests} PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()