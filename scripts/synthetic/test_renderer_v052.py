from __future__ import annotations

from scripts.synthetic.renderer_v052 import (
    V052Renderer,
)


def assert_render(
    *,
    renderer,
    name,
    frame_id,
    slots,
    features,
    expected,
):

    actual = renderer.render(
        frame_id=frame_id,
        slots=slots,
        features=features,
    )

    if actual != expected:

        print()
        print("=" * 100)
        print("FAIL:", name)
        print("=" * 100)

        print(
            "Expected:"
        )

        for lang, text in expected.items():
            print(
                lang.upper(),
                text
            )

        print()

        print(
            "Actual:"
        )

        for lang, text in actual.items():
            print(
                lang.upper(),
                text
            )

        raise AssertionError(
            name
        )

    print(
        "PASS:",
        name,
    )


def assert_raises(
    *,
    renderer,
    name,
    frame_id,
    slots,
    features,
):

    try:

        renderer.render(
            frame_id=frame_id,
            slots=slots,
            features=features,
        )

    except Exception:

        print(
            "PASS:",
            name,
        )

        return

    raise AssertionError(
        f"{name}: expected exception"
    )


def main():

    renderer = V052Renderer()

    passed = 0
    total = 0

    # ========================================================
    # 1 CALL present
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="CALL present positive",
        frame_id="CALL_PERSON",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "CALL",

            "person":
                "PERSON_HE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "我给他打电话。",

            "en":
                "I call him.",

            "ru":
                "Я звоню ему.",

            "uz":
                "Men unga qo'ng'iroq qilaman.",
        },
    )

    passed += 1

    # ========================================================
    # 2 CALL 3sg
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="CALL English 3sg and RU dative",
        frame_id="CALL_PERSON",

        slots={
            "subject":
                "PERSON_SHE",

            "verb":
                "CALL",

            "person":
                "PERSON_HE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "她给他打电话。",

            "en":
                "She calls him.",

            "ru":
                "Она звонит ему.",

            "uz":
                "U unga qo'ng'iroq qiladi.",
        },
    )

    passed += 1

    # ========================================================
    # 3 CALL future negative
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="CALL future negative",
        frame_id="CALL_PERSON",

        slots={
            "subject":
                "PERSON_WE",

            "verb":
                "CALL",

            "person":
                "PERSON_THEY",
        },

        features={
            "tense":
                "future",

            "polarity":
                "neg",
        },

        expected={
            "zh":
                "我们不会给他们打电话。",

            "en":
                "We will not call them.",

            "ru":
                "Мы не позвоним им.",

            "uz":
                "Biz ularga qo'ng'iroq qilmaymiz.",
        },
    )

    passed += 1

    # ========================================================
    # 4 WAIT person
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT person present positive",
        frame_id="WAIT_PERSON",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "WAIT",

            "person":
                "PERSON_HE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "我等他。",

            "en":
                "I wait for him.",

            "ru":
                "Я жду его.",

            "uz":
                "Men uni kutaman.",
        },
    )

    passed += 1

    # ========================================================
    # 5 WAIT person future
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT person future positive",
        frame_id="WAIT_PERSON",

        slots={
            "subject":
                "PERSON_THEY",

            "verb":
                "WAIT",

            "person":
                "PERSON_SHE",
        },

        features={
            "tense":
                "future",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "他们会等她。",

            "en":
                "They will wait for her.",

            "ru":
                "Они будут ждать её.",

            "uz":
                "Ular uni kutadilar.",
        },
    )

    passed += 1

    # ========================================================
    # 6 WAIT person negative
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT person present negative",
        frame_id="WAIT_PERSON",

        slots={
            "subject":
                "PERSON_HE",

            "verb":
                "WAIT",

            "person":
                "PERSON_WE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "neg",
        },

        expected={
            "zh":
                "他不等我们。",

            "en":
                "He does not wait for us.",

            "ru":
                "Он не ждёт нас.",

            "uz":
                "U bizni kutmaydi.",
        },
    )

    passed += 1

    # ========================================================
    # 7 WAIT local place
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT at station",
        frame_id="WAIT_AT_PLACE",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "WAIT",

            "place":
                "STATION",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "我在车站等。",

            "en":
                "I wait at the station.",

            "ru":
                "Я жду на вокзале.",

            "uz":
                "Men vokzalda kutaman.",
        },
    )

    passed += 1

    # ========================================================
    # 8 WAIT city location
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT in Tashkent",
        frame_id="WAIT_AT_PLACE",

        slots={
            "subject":
                "PERSON_WE",

            "verb":
                "WAIT",

            "place":
                "TASHKENT",
        },

        features={
            "tense":
                "future",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "我们会在塔什干等。",

            "en":
                "We will wait in Tashkent.",

            "ru":
                "Мы будем ждать в Ташкенте.",

            "uz":
                "Biz Toshkentda kutamiz.",
        },
    )

    passed += 1

    # ========================================================
    # 9 WAIT negative locative
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="WAIT at airport future negative",
        frame_id="WAIT_AT_PLACE",

        slots={
            "subject":
                "PERSON_SHE",

            "verb":
                "WAIT",

            "place":
                "AIRPORT",
        },

        features={
            "tense":
                "future",

            "polarity":
                "neg",
        },

        expected={
            "zh":
                "她不会在机场等。",

            "en":
                "She will not wait at the airport.",

            "ru":
                "Она не будет ждать в аэропорту.",

            "uz":
                "U aeroportda kutmaydi.",
        },
    )

    passed += 1

    # ========================================================
    # 10 CALL reflexive blocked
    # ========================================================

    total += 1

    assert_raises(
        renderer=renderer,
        name="CALL reflexive leakage blocked",
        frame_id="CALL_PERSON",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "CALL",

            "person":
                "PERSON_I",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },
    )

    passed += 1

    # ========================================================
    # 11 WAIT reflexive blocked
    # ========================================================

    total += 1

    assert_raises(
        renderer=renderer,
        name="WAIT reflexive leakage blocked",
        frame_id="WAIT_PERSON",

        slots={
            "subject":
                "PERSON_HE",

            "verb":
                "WAIT",

            "person":
                "PERSON_HE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },
    )

    passed += 1

    # ========================================================
    # 12 fixed verb mismatch
    # ========================================================

    total += 1

    assert_raises(
        renderer=renderer,
        name="CALL frame fixed verb mismatch blocked",
        frame_id="CALL_PERSON",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "WAIT",

            "person":
                "PERSON_HE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },
    )

    passed += 1

    # ========================================================
    # 13 unknown place
    # ========================================================

    total += 1

    assert_raises(
        renderer=renderer,
        name="unknown WAIT place blocked",
        frame_id="WAIT_AT_PLACE",

        slots={
            "subject":
                "PERSON_I",

            "verb":
                "WAIT",

            "place":
                "UNKNOWN_PLACE",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },
    )

    passed += 1

    # ========================================================
    # 14 V0.5.1 fallback SEE
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="V0.5.1 SEE fallback unchanged",
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

        expected={
            "zh":
                "我看见行李。",

            "en":
                "I see the luggage.",

            "ru":
                "Я вижу багаж.",

            "uz":
                "Men bagajni ko'raman.",
        },
    )

    passed += 1

    # ========================================================
    # 15 V0.5.1 fallback TAKE
    # ========================================================

    total += 1

    assert_render(
        renderer=renderer,
        name="V0.5.1 TAKE fallback unchanged",
        frame_id="TAKE_OBJECT",

        slots={
            "subject":
                "PERSON_SHE",

            "verb":
                "TAKE",

            "object":
                "CLOTHES",
        },

        features={
            "tense":
                "present",

            "polarity":
                "pos",
        },

        expected={
            "zh":
                "她拿衣服。",

            "en":
                "She takes clothes.",

            "ru":
                "Она берёт одежду.",

            "uz":
                "U kiyimni oladi.",
        },
    )

    passed += 1

    print()
    print("=" * 100)

    print(
        f"V0.5.2 RENDERER REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print("=" * 100)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()