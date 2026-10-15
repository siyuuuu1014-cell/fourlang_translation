from __future__ import annotations

from scripts.synthetic.semantic_compatibility_validator_v053 import (
    load_resources,
    validate_new_row,
)


def check_blocked(
    name,
    row,
    expected_type,
):
    (
        _,
        objects,
        persons,
        places,
    ) = load_resources()

    result = validate_new_row(
        row,
        objects=objects,
        persons=persons,
        places=places,
    )

    types = {
        x["type"]
        for x in result[
            "violations"
        ]
    }

    if expected_type not in types:
        raise AssertionError(
            f"{name}\n"
            f"expected={expected_type}\n"
            f"actual={types}"
        )

    print(
        f"PASS: {name}"
    )


def main():

    base_texts = {
        "zh": "测试。",
        "en": "Test.",
        "ru": "Тест.",
        "uz": "Test.",
    }

    # -------------------------------------------------------------------------
    # GIVE participant overlap
    # -------------------------------------------------------------------------

    check_blocked(
        "GIVE I -> WE blocked",
        {
            "frame_id":
                "GIVE_OBJECT_PERSON",

            "slots": {
                "subject":
                    "PERSON_I",

                "verb":
                    "GIVE",

                "object":
                    "MONEY",

                "person":
                    "PERSON_WE",
            },

            "features": {
                "tense":
                    "present",

                "polarity":
                    "pos",
            },

            "texts":
                base_texts,
        },
        "PARTICIPANT_OVERLAP_NOT_ENABLED",
    )

    # -------------------------------------------------------------------------
    # GIVE exact reflexive
    # -------------------------------------------------------------------------

    check_blocked(
        "GIVE reflexive blocked",
        {
            "frame_id":
                "GIVE_OBJECT_PERSON",

            "slots": {
                "subject":
                    "PERSON_HE",

                "verb":
                    "GIVE",

                "object":
                    "MONEY",

                "person":
                    "PERSON_HE",
            },

            "features": {
                "tense":
                    "present",

                "polarity":
                    "pos",
            },

            "texts":
                base_texts,
        },
        "REFLEXIVE_SEMANTICS_NOT_ENABLED",
    )

    # -------------------------------------------------------------------------
    # GIVE wrong verb
    # -------------------------------------------------------------------------

    check_blocked(
        "GIVE fixed verb mismatch",
        {
            "frame_id":
                "GIVE_OBJECT_PERSON",

            "slots": {
                "subject":
                    "PERSON_HE",

                "verb":
                    "TAKE",

                "object":
                    "MONEY",

                "person":
                    "PERSON_YOU",
            },

            "features": {
                "tense":
                    "present",

                "polarity":
                    "pos",
            },

            "texts":
                base_texts,
        },
        "FRAME_VERB_MISMATCH",
    )

    # -------------------------------------------------------------------------
    # BRING wrong destination type
    # -------------------------------------------------------------------------

    check_blocked(
        "BRING object used as destination blocked",
        {
            "frame_id":
                "BRING_OBJECT_DESTINATION",

            "slots": {
                "subject":
                    "PERSON_I",

                "verb":
                    "BRING",

                "object":
                    "LUGGAGE",

                "destination":
                    "MONEY",
            },

            "features": {
                "tense":
                    "present",

                "polarity":
                    "pos",
            },

            "texts":
                base_texts,
        },
        "UNKNOWN_DESTINATION",
    )

    # -------------------------------------------------------------------------
    # BRING missing destination
    # -------------------------------------------------------------------------

    check_blocked(
        "BRING missing destination blocked",
        {
            "frame_id":
                "BRING_OBJECT_DESTINATION",

            "slots": {
                "subject":
                    "PERSON_I",

                "verb":
                    "BRING",

                "object":
                    "LUGGAGE",
            },

            "features": {
                "tense":
                    "present",

                "polarity":
                    "pos",
            },

            "texts":
                base_texts,
        },
        "MISSING_DESTINATION",
    )

    print()
    print("=" * 100)
    print(
        "V0.5.3 COMPATIBILITY "
        "REGRESSION: 5/5 PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()