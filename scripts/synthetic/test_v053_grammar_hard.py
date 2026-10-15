from __future__ import annotations

from copy import deepcopy

from scripts.synthetic.grammar_hard_validator_v053 import (
    validate_new_row,
)

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


def make_row(
    renderer,
    *,
    frame_id,
    slots,
    features,
):

    texts = renderer.render(
        frame_id=frame_id,
        slots=slots,
        features=features,
    )

    return {
        "semantic_id":
            "test_v053",

        "frame_id":
            frame_id,

        "slots":
            slots,

        "features":
            features,

        "texts":
            texts,

        "compatibility_validation": {
            "validator_version":
                "0.5.3",

            "accept":
                True,

            "violations":
                [],

            "warnings":
                [],
        },

        "hard_semantic_validation": {
            "validator_version":
                "0.5.3",

            "accept":
                True,

            "violations":
                [],
        },
    }


def has_type(
    result,
    expected_type,
):

    return any(
        violation.get("type")
        == expected_type
        for violation
        in result.get(
            "violations",
            [],
        )
    )


def main():

    renderer = V053Renderer()

    passed = 0

    # =====================================================================
    # 1. Valid GIVE
    # =====================================================================

    row = make_row(
        renderer,
        frame_id="GIVE_OBJECT_PERSON",
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
    # 2. English 3sg error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["en"] = (
        "He give you the ID card."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "EN_AGREEMENT_OR_TENSE",
    )

    print(
        "PASS: GIVE English 3sg blocked"
    )

    passed += 1

    # =====================================================================
    # 3. Russian agreement error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["ru"] = (
        "Он даю тебе удостоверение личности."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "RU_AGREEMENT_OR_TENSE",
    )

    print(
        "PASS: GIVE Russian agreement blocked"
    )

    passed += 1

    # =====================================================================
    # 4. Uzbek agreement error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["uz"] = (
        "U senga shaxsiy guvohnomani beraman."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "UZ_AGREEMENT_OR_TENSE",
    )

    print(
        "PASS: GIVE Uzbek agreement blocked"
    )

    passed += 1

    # =====================================================================
    # 5. Valid BRING
    # =====================================================================

    row = make_row(
        renderer,
        frame_id="BRING_OBJECT_DESTINATION",
        slots={
            "subject": "PERSON_WE",
            "verb": "BRING",
            "object": "LUGGAGE",
            "destination": "HOTEL",
        },
        features={
            "tense": "future",
            "polarity": "neg",
        },
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
    # 6. Russian future/person error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["ru"] = (
        "Мы не принесёт багаж в отель."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "RU_AGREEMENT_OR_TENSE",
    )

    print(
        "PASS: BRING Russian future/person blocked"
    )

    passed += 1

    # =====================================================================
    # 7. Uzbek person error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["uz"] = (
        "Biz bagajni mehmonxonaga olib bormaydi."
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "UZ_AGREEMENT_OR_TENSE",
    )

    print(
        "PASS: BRING Uzbek agreement blocked"
    )

    passed += 1

    # =====================================================================
    # 8. Chinese future/negation error
    # =====================================================================

    bad = deepcopy(row)

    bad["texts"]["zh"] = (
        "我们把行李带到酒店。"
    )

    result = validate_new_row(
        bad,
        renderer,
    )

    assert not result["accept"]

    assert has_type(
        result,
        "ZH_TENSE_OR_WORD_ORDER",
    )

    print(
        "PASS: BRING Chinese future/negation blocked"
    )

    passed += 1

    print()
    print("=" * 100)

    print(
        f"V0.5.3 GRAMMAR HARD "
        f"REGRESSION: {passed}/8 PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()