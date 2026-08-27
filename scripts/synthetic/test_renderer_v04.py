from __future__ import annotations

from scripts.synthetic.renderer_v04 import (
    RenderError,
    V04Renderer,
)


def assert_texts(
    *,
    name: str,
    actual: dict,
    expected: dict,
) -> None:

    for lang in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        got = actual[
            lang
        ]

        want = expected[
            lang
        ]

        if got != want:

            raise AssertionError(
                f"\n{name}\n"
                f"Language: {lang}\n"
                f"Expected: {want}\n"
                f"Actual:   {got}"
            )

    print(
        f"PASS: {name}"
    )


def assert_raises(
    *,
    name: str,
    func,
) -> None:

    try:

        func()

    except RenderError:

        print(
            f"PASS: {name}"
        )

        return

    raise AssertionError(
        f"{name}: expected RenderError"
    )


def main() -> None:

    renderer = V04Renderer()

    passed = 0

    total = 10

    # ========================================================
    # 1. Existing DRINK
    # ========================================================

    actual = renderer.render(
        frame_id="TRANSITIVE_TIME",

        slots={
            "subject": "PERSON_WE",
            "time": "TIME_TODAY",
            "verb": "DRINK",
            "object": "WATER",
        },

        features={
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "我们今天不喝水。",

        "en":
            "We do not drink water today.",

        "ru":
            "Мы сегодня не пьём воду.",

        "uz":
            "Biz bugun suvni ichmaymiz.",
    }

    assert_texts(
        name="existing DRINK present negative",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 2. FIND future positive
    # ========================================================

    actual = renderer.render(
        frame_id="TRANSITIVE_TIME",

        slots={
            "subject": "PERSON_SHE",
            "time": "TIME_TOMORROW",
            "verb": "FIND",
            "object": "PASSPORT",
        },

        features={
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她明天会找到护照。",

        "en":
            "She will find a passport tomorrow.",

        "ru":
            "Она завтра найдёт паспорт.",

        "uz":
            "U ertaga pasportni topadi.",
    }

    assert_texts(
        name="FIND future positive",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 3. GO future negative
    # ========================================================

    actual = renderer.render(
        frame_id="MOTION_TIME",

        slots={
            "subject": "PERSON_SHE",
            "time": "TIME_TOMORROW",
            "verb": "GO",
            "destination": "AIRPORT",
        },

        features={
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "她明天不会去机场。",

        "en":
            (
                "She will not go "
                "to the airport tomorrow."
            ),

        "ru":
            (
                "Она завтра не будет ехать "
                "в аэропорт."
            ),

        "uz":
            (
                "U ertaga aeroportga "
                "bormaydi."
            ),
    }

    assert_texts(
        name="existing GO future negative",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 4. New COME future
    # ========================================================

    actual = renderer.render(
        frame_id="MOTION_TIME",

        slots={
            "subject": "PERSON_SHE",
            "time": "TIME_TOMORROW",
            "verb": "COME",
            "destination": "AIRPORT",
        },

        features={
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她明天会来机场。",

        "en":
            (
                "She will come "
                "to the airport tomorrow."
            ),

        "ru":
            (
                "Она завтра придёт "
                "в аэропорт."
            ),

        "uz":
            (
                "U ertaga aeroportga "
                "keladi."
            ),
    }

    assert_texts(
        name="new COME future",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 5. ARRIVE airport
    # ========================================================

    actual = renderer.render(
        frame_id="MOTION_TIME",

        slots={
            "subject": "PERSON_SHE",
            "time": "TIME_TOMORROW",
            "verb": "ARRIVE",
            "destination": "AIRPORT",
        },

        features={
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她明天会到达机场。",

        "en":
            (
                "She will arrive "
                "at the airport tomorrow."
            ),

        "ru":
            (
                "Она завтра прибудет "
                "в аэропорт."
            ),

        "uz":
            (
                "U ertaga aeroportga "
                "yetib keladi."
            ),
    }

    assert_texts(
        name="new ARRIVE airport",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 6. ARRIVE city:
    #    English must use IN, not AT.
    # ========================================================

    actual = renderer.render(
        frame_id="MOTION_TIME",

        slots={
            "subject": "PERSON_SHE",
            "time": "TIME_TOMORROW",
            "verb": "ARRIVE",
            "destination": "MOSCOW",
        },

        features={
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她明天会到达莫斯科。",

        "en":
            (
                "She will arrive "
                "in Moscow tomorrow."
            ),

        "ru":
            (
                "Она завтра прибудет "
                "в Москву."
            ),

        "uz":
            (
                "U ertaga Moskvaga "
                "yetib keladi."
            ),
    }

    assert_texts(
        name="ARRIVE city preposition",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 7. WANT positive
    # ========================================================

    actual = renderer.render(
        frame_id="WANT_OBJECT",

        slots={
            "subject": "PERSON_I",
            "object": "WATER",
        },

        features={
            "polarity": "pos",
            "tense": "present",
        },
    )

    expected = {
        "zh":
            "我想要水。",

        "en":
            "I want water.",

        "ru":
            "Я хочу воду.",

        "uz":
            "Men suvni xohlayman.",
    }

    assert_texts(
        name="new WANT positive",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 8. WANT negative / 3sg
    # ========================================================

    actual = renderer.render(
        frame_id="WANT_OBJECT",

        slots={
            "subject": "PERSON_SHE",
            "object": "COFFEE",
        },

        features={
            "polarity": "neg",
            "tense": "present",
        },
    )

    expected = {
        "zh":
            "她不想要咖啡。",

        "en":
            "She does not want coffee.",

        "ru":
            "Она не хочет кофе.",

        "uz":
            "U qahvani xohlamaydi.",
    }

    assert_texts(
        name="new WANT negative",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 9. WHERE_PLACE
    # ========================================================

    actual = renderer.render(
        frame_id="WHERE_PLACE",

        slots={
            "place": "AIRPORT",
        },
    )

    expected = {
        "zh":
            "机场在哪里？",

        "en":
            "Where is the airport?",

        "ru":
            "Где аэропорт?",

        "uz":
            "Aeroport qayerda?",
    }

    assert_texts(
        name="WHERE_PLACE",
        actual=actual,
        expected=expected,
    )

    passed += 1

    # ========================================================
    # 10. Disabled resources must not leak
    # ========================================================

    assert_raises(
        name="disabled NEED frame blocked",

        func=lambda: renderer.render(
            frame_id="NEED_OBJECT",

            slots={
                "subject": "PERSON_I",
                "object": "MEDICINE",
            },
        ),
    )

    assert_raises(
        name="disabled LEAVE verb blocked",

        func=lambda: renderer.render(
            frame_id="MOTION_DESTINATION",

            slots={
                "subject": "PERSON_I",
                "verb": "LEAVE",
                "destination": "HOTEL",
            },
        ),
    )

    # Two guards count as one regression category.
    passed += 1

    print()

    print(
        "=" * 80
    )

    print(
        f"V0.4 RENDERER REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()