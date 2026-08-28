from __future__ import annotations

from scripts.synthetic.renderer_v0441 import (
    V0441Renderer,
)

from scripts.synthetic.renderer_v051 import (
    V051Renderer,
    RenderError,
)


def assert_equal(
    name: str,
    actual,
    expected,
) -> None:

    if actual != expected:

        raise AssertionError(
            f"\nFAIL: {name}\n"
            f"EXPECTED:\n{expected}\n"
            f"ACTUAL:\n{actual}"
        )

    print(
        f"PASS: {name}"
    )


def main() -> None:

    renderer = V051Renderer()

    passed = 0
    total = 0

    # ========================================================
    # 1. SEE / present / positive
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="SEE_OBJECT",

        slots={
            "subject": "PERSON_I",
            "verb": "SEE",
            "object": "ID_CARD",
        },

        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "我看见身份证。",

        "en":
            "I see the ID card.",

        "ru":
            "Я вижу удостоверение личности.",

        "uz":
            "Men shaxsiy guvohnomani ko'raman.",
    }

    assert_equal(
        "SEE present positive",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 2. SEE / present / negative
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="SEE_OBJECT",

        slots={
            "subject": "PERSON_SHE",
            "verb": "SEE",
            "object": "TABLE",
        },

        features={
            "tense": "present",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "她没看见桌子。",

        "en":
            "She does not see the table.",

        "ru":
            "Она не видит стол.",

        "uz":
            "U stolni ko'rmaydi.",
    }

    assert_equal(
        "SEE present negative",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 3. SEE / future / positive
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="SEE_OBJECT",

        slots={
            "subject": "PERSON_THEY",
            "verb": "SEE",
            "object": "ROOM",
        },

        features={
            "tense": "future",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "他们会看见房间。",

        "en":
            "They will see the room.",

        "ru":
            "Они увидят комнату.",

        "uz":
            "Ular xonani ko'radilar.",
    }

    assert_equal(
        "SEE future positive",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 4. TAKE / present / positive
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="TAKE_OBJECT",

        slots={
            "subject": "PERSON_HE",
            "verb": "TAKE",
            "object": "MONEY",
        },

        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "他拿钱。",

        "en":
            "He takes money.",

        "ru":
            "Он берёт деньги.",

        "uz":
            "U pulni oladi.",
    }

    assert_equal(
        "TAKE present positive",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 5. TAKE / present / negative
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="TAKE_OBJECT",

        slots={
            "subject": "PERSON_YOU",
            "verb": "TAKE",
            "object": "LUGGAGE",
        },

        features={
            "tense": "present",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "你不拿行李。",

        "en":
            "You do not take the luggage.",

        "ru":
            "Ты не берёшь багаж.",

        "uz":
            "Sen bagajni olmaysan.",
    }

    assert_equal(
        "TAKE present negative",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 6. TAKE / future / negative
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="TAKE_OBJECT",

        slots={
            "subject": "PERSON_WE",
            "verb": "TAKE",
            "object": "CHARGER",
        },

        features={
            "tense": "future",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "我们不会拿充电器。",

        "en":
            "We will not take the charger.",

        "ru":
            "Мы не возьмём зарядное устройство.",

        "uz":
            "Biz quvvatlagichni olmaymiz.",
    }

    assert_equal(
        "TAKE future negative",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 7. LOSE / present / positive
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="LOSE_OBJECT",

        slots={
            "subject": "PERSON_SHE",
            "verb": "LOSE",
            "object": "ID_CARD",
        },

        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她丢了身份证。",

        "en":
            "She loses the ID card.",

        "ru":
            "Она теряет удостоверение личности.",

        "uz":
            "U shaxsiy guvohnomani yo'qotadi.",
    }

    assert_equal(
        "LOSE present positive",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 8. LOSE / present / negative
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="LOSE_OBJECT",

        slots={
            "subject": "PERSON_I",
            "verb": "LOSE",
            "object": "MONEY",
        },

        features={
            "tense": "present",
            "polarity": "neg",
        },
    )

    expected = {
        "zh":
            "我没丢钱。",

        "en":
            "I do not lose money.",

        "ru":
            "Я не теряю деньги.",

        "uz":
            "Men pulni yo'qotmayman.",
    }

    assert_equal(
        "LOSE present negative",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 9. LOSE / future / positive
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="LOSE_OBJECT",

        slots={
            "subject": "PERSON_THEY",
            "verb": "LOSE",
            "object": "LUGGAGE",
        },

        features={
            "tense": "future",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "他们会丢行李。",

        "en":
            "They will lose the luggage.",

        "ru":
            "Они потеряют багаж.",

        "uz":
            "Ular bagajni yo'qotadilar.",
    }

    assert_equal(
        "LOSE future positive",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 10. 3sg morphology
    # ========================================================

    total += 1

    actual = renderer.render(
        frame_id="TAKE_OBJECT",

        slots={
            "subject": "PERSON_SHE",
            "verb": "TAKE",
            "object": "CLOTHES",
        },

        features={
            "tense": "present",
            "polarity": "pos",
        },
    )

    expected = {
        "zh":
            "她拿衣服。",

        "en":
            "She takes clothes.",

        "ru":
            "Она берёт одежду.",

        "uz":
            "U kiyimni oladi.",
    }

    assert_equal(
        "3sg morphology and object case",
        actual,
        expected,
    )

    passed += 1

    # ========================================================
    # 11. Wrong verb must fail
    # ========================================================

    total += 1

    try:

        renderer.render(
            frame_id="SEE_OBJECT",

            slots={
                "subject": "PERSON_I",
                "verb": "TAKE",
                "object": "ID_CARD",
            },

            features={
                "tense": "present",
                "polarity": "pos",
            },
        )

    except RenderError:

        print(
            "PASS: frame-fixed verb mismatch blocked"
        )

        passed += 1

    else:

        raise AssertionError(
            "FAIL: frame-fixed verb mismatch "
            "was not blocked."
        )

    # ========================================================
    # 12. Frozen V0.4.4.1 fallback
    # ========================================================

    total += 1

    old_renderer = (
        V0441Renderer()
    )

    old_result = old_renderer.render(
        frame_id="WHERE_PLACE",

        slots={
            "place": "BANK",
        },

        features={
            "polarity": "pos",
        },
    )

    new_result = renderer.render(
        frame_id="WHERE_PLACE",

        slots={
            "place": "BANK",
        },

        features={
            "polarity": "pos",
        },
    )

    assert_equal(
        "V0.4.4.1 fallback unchanged",
        new_result,
        old_result,
    )

    passed += 1

    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "=" * 90
    )

    print(
        f"V0.5.1 RENDERER REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print(
        "=" * 90
    )

    if passed != total:

        raise SystemExit(1)


if __name__ == "__main__":
    main()