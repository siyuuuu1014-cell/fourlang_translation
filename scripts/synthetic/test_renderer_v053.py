from __future__ import annotations

from scripts.synthetic.renderer_v052 import (
    V052Renderer,
)

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


def assert_equal(
    actual,
    expected,
    name,
):
    if actual != expected:
        raise AssertionError(
            f"\n{name}\n"
            f"expected: {expected}\n"
            f"actual:   {actual}"
        )

    print(
        f"PASS: {name}"
    )


def assert_blocked(
    fn,
    name,
):
    try:
        fn()

    except Exception:
        print(
            f"PASS: {name}"
        )
        return

    raise AssertionError(
        f"{name}: expected block"
    )


def main():

    r = V053Renderer()

    passed = 0

    # =====================================================================
    # GIVE present positive
    # =====================================================================

    out = r.render(
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

    expected = {
        "zh":
            "他给你身份证。",

        "en":
            "He gives you the ID card.",

        "ru":
            "Он даёт тебе удостоверение личности.",

        "uz":
            "U senga shaxsiy guvohnomani beradi.",
    }

    assert_equal(
        out,
        expected,
        "GIVE present positive",
    )
    passed += 1

    # =====================================================================
    # GIVE future negative
    # =====================================================================

    out = r.render(
        frame_id="GIVE_OBJECT_PERSON",
        slots={
            "subject": "PERSON_I",
            "verb": "GIVE",
            "object": "MONEY",
            "person": "PERSON_HE",
        },
        features={
            "tense": "future",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "我不会给他钱。",

        "en":
            "I will not give him money.",

        "ru":
            "Я не дам ему деньги.",

        "uz":
            "Men unga pulni bermayman.",
    }

    assert_equal(
        out,
        expected,
        "GIVE future negative",
    )
    passed += 1

    # =====================================================================
    # GIVE Russian dative + object accusative
    # =====================================================================

    assert (
        "тебе"
        in r.render(
            frame_id="GIVE_OBJECT_PERSON",
            slots={
                "subject": "PERSON_I",
                "verb": "GIVE",
                "object": "CHARGER",
                "person": "PERSON_YOU",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        )["ru"]
    )

    print(
        "PASS: GIVE RU dative recipient"
    )
    passed += 1

    # =====================================================================
    # GIVE Uzbek dative + accusative
    # =====================================================================

    uz = r.render(
        frame_id="GIVE_OBJECT_PERSON",
        slots={
            "subject": "PERSON_I",
            "verb": "GIVE",
            "object": "CHARGER",
            "person": "PERSON_YOU",
        },
        features={
            "tense": "present",
            "polarity": "pos",
        },
    )["uz"]

    if (
        "senga" not in uz
        or "quvvatlagichni"
        not in uz
    ):
        raise AssertionError(uz)

    print(
        "PASS: GIVE UZ dative + accusative"
    )
    passed += 1

    # =====================================================================
    # participant overlap
    # =====================================================================

    assert_blocked(
        lambda: r.render(
            frame_id="GIVE_OBJECT_PERSON",
            slots={
                "subject": "PERSON_I",
                "verb": "GIVE",
                "object": "MONEY",
                "person": "PERSON_WE",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        ),
        "GIVE I->WE overlap blocked",
    )
    passed += 1

    assert_blocked(
        lambda: r.render(
            frame_id="GIVE_OBJECT_PERSON",
            slots={
                "subject": "PERSON_WE",
                "verb": "GIVE",
                "object": "MONEY",
                "person": "PERSON_I",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        ),
        "GIVE WE->I overlap blocked",
    )
    passed += 1

    # =====================================================================
    # fixed verb
    # =====================================================================

    assert_blocked(
        lambda: r.render(
            frame_id="GIVE_OBJECT_PERSON",
            slots={
                "subject": "PERSON_I",
                "verb": "TAKE",
                "object": "MONEY",
                "person": "PERSON_HE",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        ),
        "GIVE fixed verb mismatch blocked",
    )
    passed += 1

    # =====================================================================
    # BRING present positive
    # =====================================================================

    out = r.render(
        frame_id="BRING_OBJECT_DESTINATION",
        slots={
            "subject": "PERSON_SHE",
            "verb": "BRING",
            "object": "LUGGAGE",
            "destination": "HOTEL",
        },
        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她把行李带到酒店。",

        "en":
            "She brings the luggage to the hotel.",

        "ru":
            "Она приносит багаж в отель.",

        "uz":
            "U bagajni mehmonxonaga olib boradi.",
    }

    assert_equal(
        out,
        expected,
        "BRING present positive",
    )
    passed += 1

    # =====================================================================
    # BRING future negative
    # =====================================================================

    out = r.render(
        frame_id="BRING_OBJECT_DESTINATION",
        slots={
            "subject": "PERSON_WE",
            "verb": "BRING",
            "object": "CHARGER",
            "destination": "STATION",
        },
        features={
            "tense": "future",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "我们不会把充电器带到车站。",

        "en":
            "We will not bring the charger to the station.",

        "ru":
            "Мы не принесём зарядное устройство на вокзал.",

        "uz":
            "Biz quvvatlagichni vokzalga olib bormaymiz.",
    }

    assert_equal(
        out,
        expected,
        "BRING future negative",
    )
    passed += 1

    # =====================================================================
    # Directional target
    # =====================================================================

    bring = r.render(
        frame_id="BRING_OBJECT_DESTINATION",
        slots={
            "subject": "PERSON_I",
            "verb": "BRING",
            "object": "ID_CARD",
            "destination": "AIRPORT",
        },
        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    if (
        "в аэропорт"
        not in bring["ru"]
    ):
        raise AssertionError(
            bring["ru"]
        )

    if (
        "aeroportga"
        not in bring["uz"]
    ):
        raise AssertionError(
            bring["uz"]
        )

    print(
        "PASS: BRING directional case"
    )
    passed += 1

    # =====================================================================
    # Wrong verb
    # =====================================================================

    assert_blocked(
        lambda: r.render(
            frame_id="BRING_OBJECT_DESTINATION",
            slots={
                "subject": "PERSON_I",
                "verb": "GO",
                "object": "LUGGAGE",
                "destination": "HOTEL",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        ),
        "BRING fixed verb mismatch blocked",
    )
    passed += 1

    # =====================================================================
    # Missing destination
    # =====================================================================

    assert_blocked(
        lambda: r.render(
            frame_id="BRING_OBJECT_DESTINATION",
            slots={
                "subject": "PERSON_I",
                "verb": "BRING",
                "object": "LUGGAGE",
            },
            features={
                "tense": "present",
                "polarity": "pos",
            },
        ),
        "BRING missing destination blocked",
    )
    passed += 1

    # =====================================================================
    # Frozen V0.5.2 fallback
    # =====================================================================

    frozen = V052Renderer()

    slots = {
        "subject": "PERSON_HE",
        "verb": "CALL",
        "person": "PERSON_YOU",
    }

    features = {
        "tense": "future",
        "polarity": "neg",
    }

    old_out = frozen.render(
        frame_id="CALL_PERSON",
        slots=slots,
        features=features,
    )

    new_out = r.render(
        frame_id="CALL_PERSON",
        slots=slots,
        features=features,
    )

    assert_equal(
        new_out,
        old_out,
        "V0.5.2 CALL fallback unchanged",
    )
    passed += 1

    slots = {
        "subject": "PERSON_I",
        "verb": "WAIT",
        "place": "AIRPORT",
    }

    features = {
        "tense": "present",
        "polarity": "pos",
    }

    old_out = frozen.render(
        frame_id="WAIT_AT_PLACE",
        slots=slots,
        features=features,
    )

    new_out = r.render(
        frame_id="WAIT_AT_PLACE",
        slots=slots,
        features=features,
    )

    assert_equal(
        new_out,
        old_out,
        "V0.5.2 WAIT fallback unchanged",
    )
    passed += 1

    print()
    print("=" * 100)
    print(
        f"V0.5.3 RENDERER REGRESSION: "
        f"{passed}/14 PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()