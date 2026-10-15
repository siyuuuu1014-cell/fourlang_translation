from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.hard_semantic_validator_v052 import (
    validate_new_row,
)
from scripts.synthetic.renderer_v052 import (
    V052Renderer,
)


def make_valid(
    renderer,
    frame_id,
    slots,
    features=None,
):

    if features is None:

        features = {
            "tense": "present",
            "polarity": "pos",
        }

    texts = renderer.render(
        frame_id=frame_id,
        slots=slots,
        features=features,
    )

    return {
        "semantic_id": "test",
        "scenario": "daily",
        "frame_id": frame_id,
        "slots": deepcopy(slots),
        "features": deepcopy(features),
        "texts": deepcopy(texts),

        "metadata": {
            "generator_version": "0.5.2",
            "generation_source": "v052_new_frame",
        },

        "compatibility_validation": {
            "validator_version": "0.5.2",
            "accept": True,
            "violations": [],
            "warnings": [],
        },
    }


def expect_accept(
    renderer,
    name,
    row,
):

    result = validate_new_row(
        row,
        renderer=renderer,
    )

    if not result["accept"]:

        raise AssertionError(
            f"{name}: unexpected reject "
            f"{result['violations']}"
        )

    print(
        "PASS:",
        name,
    )


def expect_reject(
    renderer,
    name,
    row,
    expected_type=None,
):

    result = validate_new_row(
        row,
        renderer=renderer,
    )

    if result["accept"]:

        raise AssertionError(
            f"{name}: expected reject"
        )

    types = {
        item.get("type")
        for item in result.get(
            "violations",
            [],
        )
    }

    if (
        expected_type is not None
        and expected_type not in types
    ):

        raise AssertionError(
            f"{name}: expected "
            f"{expected_type}, "
            f"got {types}"
        )

    print(
        "PASS:",
        name,
    )


def main():

    renderer = V052Renderer()

    total = 0

    # ========================================================
    # 1 valid CALL
    # ========================================================

    row = make_valid(
        renderer,

        "CALL_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "CALL",
            "person": "PERSON_HE",
        },
    )

    expect_accept(
        renderer,
        "valid CALL_PERSON",
        row,
    )

    total += 1

    # ========================================================
    # 2 CALL Russian dative corrupted
    # ========================================================

    row = make_valid(
        renderer,

        "CALL_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "CALL",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "ru"
    ] = "Я звоню его."

    expect_reject(
        renderer,
        "CALL Russian dative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 3 CALL Uzbek dative corrupted
    # ========================================================

    row = make_valid(
        renderer,

        "CALL_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "CALL",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "uz"
    ] = "Men uni qo'ng'iroq qilaman."

    expect_reject(
        renderer,
        "CALL Uzbek dative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 4 CALL target semantic mismatch
    # ========================================================

    row = make_valid(
        renderer,

        "CALL_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "CALL",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "en"
    ] = "I call her."

    expect_reject(
        renderer,
        "CALL English target mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 5 valid WAIT person
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "person": "PERSON_HE",
        },
    )

    expect_accept(
        renderer,
        "valid WAIT_PERSON",
        row,
    )

    total += 1

    # ========================================================
    # 6 WAIT Russian wrong case
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "ru"
    ] = "Я жду ему."

    expect_reject(
        renderer,
        "WAIT Russian accusative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 7 WAIT Uzbek wrong case
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "uz"
    ] = "Men unga kutaman."

    expect_reject(
        renderer,
        "WAIT Uzbek accusative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 8 WAIT English missing for
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_PERSON",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "person": "PERSON_HE",
        },
    )

    row[
        "texts"
    ][
        "en"
    ] = "I wait him."

    expect_reject(
        renderer,
        "WAIT English wait-for mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 9 valid WAIT place
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_AT_PLACE",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "place": "STATION",
        },
    )

    expect_accept(
        renderer,
        "valid WAIT_AT_PLACE",
        row,
    )

    total += 1

    # ========================================================
    # 10 Russian destination instead of locative
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_AT_PLACE",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "place": "STATION",
        },
    )

    row[
        "texts"
    ][
        "ru"
    ] = "Я жду на вокзал."

    expect_reject(
        renderer,
        "WAIT Russian locative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 11 Uzbek destination instead of locative
    # ========================================================

    row = make_valid(
        renderer,

        "WAIT_AT_PLACE",

        {
            "subject": "PERSON_I",
            "verb": "WAIT",
            "place": "STATION",
        },
    )

    row[
        "texts"
    ][
        "uz"
    ] = "Men vokzalga kutaman."

    expect_reject(
        renderer,
        "WAIT Uzbek locative mismatch blocked",
        row,
        "TEXT_RENDER_MISMATCH",
    )

    total += 1

    # ========================================================
    # 12 reflexive semantic leak
    # Build manually because Renderer itself blocks it.
    # ========================================================

    row = {
        "semantic_id": "test",
        "scenario": "communication",
        "frame_id": "CALL_PERSON",

        "slots": {
            "subject": "PERSON_I",
            "verb": "CALL",
            "person": "PERSON_I",
        },

        "features": {
            "tense": "present",
            "polarity": "pos",
        },

        "texts": {
            "zh": "我给我打电话。",
            "en": "I call me.",
            "ru": "Я звоню мне.",
            "uz": "Men menga qo'ng'iroq qilaman.",
        },

        "compatibility_validation": {
            "validator_version": "0.5.2",
            "accept": True,
            "violations": [],
            "warnings": [],
        },
    }

    expect_reject(
        renderer,
        "reflexive semantic leakage blocked",
        row,
        "REFLEXIVE_SEMANTICS_NOT_ENABLED",
    )

    total += 1

    print()
    print("=" * 100)

    print(
        f"V0.5.2 HARD SEMANTIC REGRESSION: "
        f"{total}/{total} PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()