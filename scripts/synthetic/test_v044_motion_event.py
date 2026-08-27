from __future__ import annotations

from scripts.synthetic.generate_synthetic_v04 import (
    DEFAULT_COMPATIBILITY,
    DEFAULT_FRAMES,
    PROJECT_ROOT,
    V04GenerationResources,
)

from scripts.synthetic.generate_synthetic_v044 import (
    SyntheticGeneratorV044,
)

from scripts.synthetic.renderer_v044 import (
    V044Renderer,
)


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v044.jsonl"
)

POLICY = (
    RESOURCE_DIR
    / "generation_policy_v044.json"
)


def assert_true(
    name: str,
    condition: bool,
) -> None:

    if not condition:
        raise AssertionError(
            name
        )

    print(
        f"PASS: {name}"
    )


def make_generator():

    resources = V04GenerationResources(
        concepts_path=CONCEPTS,
        frames_path=DEFAULT_FRAMES,
        compatibility_path=DEFAULT_COMPATIBILITY,
        policy_path=POLICY,
    )

    renderer = V044Renderer(
        concepts_path=CONCEPTS,
        frames_path=DEFAULT_FRAMES,
    )

    generator = SyntheticGeneratorV044(
        resources=resources,
        renderer=renderer,
        seed=2051,
    )

    return generator


def main() -> None:

    generator = make_generator()

    passed = 0
    total = 6

    # ========================================================
    # 1. GO + TIME_NOW -> habitual + TIME_EVERY_DAY
    # ========================================================

    candidate = {
        "frame_id": "MOTION_TIME",

        "slots": {
            "subject": "PERSON_I",
            "time": "TIME_NOW",
            "verb": "GO",
            "destination": "HOSPITAL",
        },

        "features": {
            "tense": "present",
            "polarity": "pos",
        },

        "computed": {},

        "texts": {},
    }

    candidate = (
        generator.apply_motion_event_cleanup(
            candidate
        )
    )

    assert_true(
        "GO NOW converts to habitual EVERY_DAY",
        (
            candidate[
                "slots"
            ][
                "time"
            ]
            == "TIME_EVERY_DAY"

            and

            candidate[
                "features"
            ][
                "tense"
            ]
            == "present"

            and

            candidate[
                "features"
            ][
                "event_type"
            ]
            == "habitual"
        ),
    )

    passed += 1

    # ========================================================
    # 2. Russian GO habitual morphology
    #
    # Я каждый день езжу в больницу.
    # ========================================================

    assert_true(
        "RU GO habitual uses езжу",
        candidate[
            "texts"
        ][
            "ru"
        ]
        == "Я каждый день езжу в больницу.",
    )

    passed += 1

    # ========================================================
    # 3. English habitual
    #
    # I go to the hospital every day.
    # ========================================================

    assert_true(
        "EN GO habitual",
        candidate[
            "texts"
        ][
            "en"
        ]
        == "I go to the hospital every day.",
    )

    passed += 1

    # ========================================================
    # 4. COME + NOW -> habitual
    # ========================================================

    candidate = {
        "frame_id": "MOTION_TIME",

        "slots": {
            "subject": "PERSON_I",
            "time": "TIME_NOW",
            "verb": "COME",
            "destination": "HOSPITAL",
        },

        "features": {
            "tense": "present",
            "polarity": "pos",
        },

        "computed": {},

        "texts": {},
    }

    candidate = (
        generator.apply_motion_event_cleanup(
            candidate
        )
    )

    assert_true(
        "COME NOW converts to habitual",
        (
            candidate[
                "slots"
            ][
                "time"
            ]
            == "TIME_EVERY_DAY"

            and

            candidate[
                "features"
            ][
                "event_type"
            ]
            == "habitual"

            and

            candidate[
                "texts"
            ][
                "ru"
            ]
            == "Я каждый день прихожу в больницу."
        ),
    )

    passed += 1

    # ========================================================
    # 5. Bare GO present -> planned future
    #
    # Old:
    # I go to the bank.
    #
    # New:
    # I will go to the bank.
    # ========================================================

    candidate = {
        "frame_id": "MOTION_DESTINATION",

        "slots": {
            "subject": "PERSON_I",
            "verb": "GO",
            "destination": "BANK",
        },

        "features": {
            "tense": "present",
            "polarity": "pos",
        },

        "computed": {},

        "texts": {},
    }

    candidate = (
        generator.apply_motion_event_cleanup(
            candidate
        )
    )

    assert_true(
        "bare GO present converts to planned future",
        (
            candidate[
                "features"
            ][
                "tense"
            ]
            == "future"

            and

            candidate[
                "features"
            ][
                "event_type"
            ]
            == "planned"

            and

            candidate[
                "texts"
            ][
                "en"
            ]
            == "I will go to the bank."
        ),
    )

    passed += 1

    # ========================================================
    # 6. Generate a real 100-row batch and make sure:
    #
    # every GO / COME present row must be:
    #
    # tense = present
    # event_type = habitual
    # TIME_EVERY_DAY
    #
    # No:
    # TIME_NOW + present
    # TIME_TODAY + present
    # NO_TIME + present
    # ========================================================

    rows, _ = generator.generate(
        n=100,
        max_attempts=10000,
    )

    bad = []

    present_motion = 0

    for row in rows:

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        verb = slots.get(
            "verb"
        )

        if verb not in {
            "GO",
            "COME",
        }:
            continue

        if features.get(
            "tense"
        ) != "present":
            continue

        present_motion += 1

        time_id = (
            slots.get(
                "time"
            )
            or slots.get(
                "day"
            )
        )

        event_type = features.get(
            "event_type"
        )

        if (
            time_id
            != "TIME_EVERY_DAY"
            or event_type
            != "habitual"
        ):

            bad.append({
                "semantic_id":
                    row.get(
                        "semantic_id"
                    ),

                "verb":
                    verb,

                "frame":
                    row.get(
                        "frame_id"
                    ),

                "time":
                    time_id,

                "tense":
                    features.get(
                        "tense"
                    ),

                "event_type":
                    event_type,

                "texts":
                    row.get(
                        "texts"
                    ),
            })

    assert_true(
        "generated present motion is habitual-only",
        len(
            bad
        ) == 0,
    )

    passed += 1

    print()

    print(
        "Present motion samples in test batch:",
        present_motion,
    )

    print(
        "Bad present motion samples:",
        len(
            bad
        ),
    )

    if bad:

        print()

        print(
            "Bad examples:"
        )

        for row in bad[
            :10
        ]:

            print(
                row
            )

    print()

    print(
        "=" * 90
    )

    print(
        f"V0.4.4 MOTION EVENT "
        f"REGRESSION: "
        f"{passed}/{total} PASS"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()