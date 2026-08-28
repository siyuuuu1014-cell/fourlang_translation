from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
    / "verb_realization_v051.json"
)

REQUIRED_VERBS = {
    "SEE",
    "TAKE",
    "LOSE",
}

PERSON_CODES = {
    "1sg",
    "2sg",
    "3sg",
    "1pl",
    "2pl",
    "3pl",
}


def main() -> None:

    if not RESOURCE.exists():

        raise FileNotFoundError(
            f"Missing resource: {RESOURCE}"
        )

    with RESOURCE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    errors = []

    verbs = data.get(
        "verbs",
        {},
    )

    # ========================================================
    # Basic verbs
    # ========================================================

    missing_verbs = (
        REQUIRED_VERBS
        - set(verbs)
    )

    if missing_verbs:

        errors.append(
            "Missing verbs: "
            + ", ".join(
                sorted(missing_verbs)
            )
        )

    # ========================================================
    # Validate each verb
    # ========================================================

    for verb_id in sorted(
        REQUIRED_VERBS
    ):

        verb = verbs.get(
            verb_id,
            {},
        )

        # ----------------------------------------------------
        # Chinese
        # ----------------------------------------------------

        zh = verb.get(
            "zh",
            {},
        )

        for key in (
            "present_positive",
            "present_negative",
            "future_positive",
            "future_negative",
        ):

            if not zh.get(key):

                errors.append(
                    f"{verb_id}.zh missing {key}"
                )

        # ----------------------------------------------------
        # English
        # ----------------------------------------------------

        en = verb.get(
            "en",
            {},
        )

        for group in (
            "present",
            "present_negative",
        ):

            forms = en.get(
                group,
                {},
            )

            missing = (
                PERSON_CODES
                - set(forms)
            )

            if missing:

                errors.append(
                    f"{verb_id}.en.{group} "
                    f"missing persons: "
                    f"{sorted(missing)}"
                )

        for key in (
            "future",
            "future_negative",
        ):

            if not en.get(key):

                errors.append(
                    f"{verb_id}.en missing {key}"
                )

        # ----------------------------------------------------
        # Russian
        # ----------------------------------------------------

        ru = verb.get(
            "ru",
            {},
        )

        for group in (
            "present",
            "future",
        ):

            forms = ru.get(
                group,
                {},
            )

            missing = (
                PERSON_CODES
                - set(forms)
            )

            if missing:

                errors.append(
                    f"{verb_id}.ru.{group} "
                    f"missing persons: "
                    f"{sorted(missing)}"
                )

        # ----------------------------------------------------
        # Uzbek
        # ----------------------------------------------------

        uz = verb.get(
            "uz",
            {},
        )

        for group in (
            "present_future",
            "negative",
        ):

            forms = uz.get(
                group,
                {},
            )

            missing = (
                PERSON_CODES
                - set(forms)
            )

            if missing:

                errors.append(
                    f"{verb_id}.uz.{group} "
                    f"missing persons: "
                    f"{sorted(missing)}"
                )

    # ========================================================
    # Print
    # ========================================================

    print("=" * 90)
    print("V0.5.1 VERB REALIZATION VALIDATOR")
    print("=" * 90)

    print(
        "Resource:",
        RESOURCE,
    )

    print(
        "Version:",
        data.get("version"),
    )

    print(
        "Enabled verbs:",
        data.get(
            "enabled_verbs"
        ),
    )

    print(
        "Errors:",
        len(errors),
    )

    if errors:

        print()
        print("ERRORS")
        print("-" * 90)

        for error in errors:

            print(
                "ERROR:",
                error,
            )

        raise SystemExit(1)

    print()
    print("=" * 90)
    print("V0.5.1 VERB REALIZATION VALIDATION PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()