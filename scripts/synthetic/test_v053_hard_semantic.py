from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.hard_semantic_validator_v053 import (
    validate_new_row,
)

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


def main():
    renderer = V053Renderer()

    passed = 0

    # =====================================================================
    # Valid GIVE
    # =====================================================================

    row = {
        "frame_id":
            "GIVE_OBJECT_PERSON",

        "slots": {
            "subject":
                "PERSON_HE",

            "verb":
                "GIVE",

            "object":
                "ID_CARD",

            "person":
                "PERSON_YOU",
        },

        "features": {
            "tense":
                "present",

            "polarity":
                "pos",
        },
    }

    row["texts"] = renderer.render(
        frame_id=row["frame_id"],
        slots=row["slots"],
        features=row["features"],
    )

    result = validate_new_row(
        row,
        renderer,
    )

    assert result["accept"]

    print(
        "PASS: valid GIVE"
    )
    passed += 1

    # =====================================================================
    # GIVE swapped text
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["en"] = (
        "He gives the ID card him."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert any(
        x["type"]
        == "TEXT_RENDER_MISMATCH"
        for x in result["violations"]
    )

    print(
        "PASS: GIVE role/render mismatch blocked"
    )
    passed += 1

    # =====================================================================
    # GIVE overlap
    # =====================================================================

    bad = deepcopy(row)

    bad["slots"] = {
        "subject":
            "PERSON_I",

        "verb":
            "GIVE",

        "object":
            "MONEY",

        "person":
            "PERSON_WE",
    }

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    print(
        "PASS: GIVE participant overlap blocked"
    )
    passed += 1

    # =====================================================================
    # Valid BRING
    # =====================================================================

    row = {
        "frame_id":
            "BRING_OBJECT_DESTINATION",

        "slots": {
            "subject":
                "PERSON_SHE",

            "verb":
                "BRING",

            "object":
                "LUGGAGE",

            "destination":
                "HOTEL",
        },

        "features": {
            "tense":
                "future",

            "polarity":
                "pos",
        },
    }

    row["texts"] = renderer.render(
        frame_id=row["frame_id"],
        slots=row["slots"],
        features=row["features"],
    )

    result = validate_new_row(
        row,
        renderer,
    )

    assert result["accept"]

    print(
        "PASS: valid BRING"
    )
    passed += 1

    # =====================================================================
    # BRING locative instead of directional
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["ru"] = (
        "Она принесёт багаж в отеле."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    print(
        "PASS: BRING wrong directional blocked"
    )
    passed += 1

    # =====================================================================
    # BRING wrong verb
    # =====================================================================

    bad = deepcopy(row)

    bad["slots"][
        "verb"
    ] = "GO"

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    print(
        "PASS: BRING fixed verb mismatch blocked"
    )
    passed += 1

    print()
    print("=" * 100)
    print(
        f"V0.5.3 HARD SEMANTIC "
        f"REGRESSION: {passed}/6 PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()