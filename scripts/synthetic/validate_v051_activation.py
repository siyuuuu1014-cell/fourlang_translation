from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

POLICY_FILE = (
    RESOURCE_DIR
    / "generation_policy_v051.json"
)

VERB_RESOURCE = (
    RESOURCE_DIR
    / "verb_realization_v051.json"
)

FRAME_RESOURCE = (
    RESOURCE_DIR
    / "frames_v05.json"
)


EXPECTED_ENABLED_VERBS = {
    "SEE",
    "TAKE",
}

EXPECTED_ENABLED_FRAMES = {
    "SEE_OBJECT",
    "TAKE_OBJECT",
}


def read_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing resource: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main() -> None:

    policy = read_json(
        POLICY_FILE
    )

    verb_resource = read_json(
        VERB_RESOURCE
    )

    frame_resource = read_json(
        FRAME_RESOURCE
    )

    errors = []
    warnings = []

    enabled_verbs = set(
        policy.get(
            "enabled_new_verbs",
            [],
        )
    )

    enabled_frames = set(
        policy.get(
            "enabled_new_frames",
            [],
        )
    )

    # ========================================================
    # Exact Batch-1A activation
    # ========================================================

    if enabled_verbs != EXPECTED_ENABLED_VERBS:

        errors.append(
            "Enabled verbs must be exactly "
            f"{sorted(EXPECTED_ENABLED_VERBS)}, "
            f"got {sorted(enabled_verbs)}"
        )

    if enabled_frames != EXPECTED_ENABLED_FRAMES:

        errors.append(
            "Enabled frames must be exactly "
            f"{sorted(EXPECTED_ENABLED_FRAMES)}, "
            f"got {sorted(enabled_frames)}"
        )

    # ========================================================
    # Renderer support
    # ========================================================

    renderer_verbs = set(
        verb_resource.get(
            "enabled_verbs",
            [],
        )
    )

    for verb_id in enabled_verbs:

        if verb_id not in renderer_verbs:

            errors.append(
                f"Enabled verb has no V0.5.1 "
                f"renderer resource: {verb_id}"
            )

    # ========================================================
    # Frame existence
    # ========================================================

    known_frames = {
        frame.get("id")
        for frame in frame_resource.get(
            "frames",
            [],
        )
        if frame.get("id")
    }

    for frame_id in enabled_frames:

        if frame_id not in known_frames:

            errors.append(
                f"Enabled frame not defined: "
                f"{frame_id}"
            )

    # ========================================================
    # Critical LOSE guard
    # ========================================================

    if "LOSE" in enabled_verbs:

        errors.append(
            "LOSE must remain generation-disabled "
            "until resultative/past semantics exist."
        )

    if "LOSE_OBJECT" in enabled_frames:

        errors.append(
            "LOSE_OBJECT must remain disabled."
        )

    lose_guard = (
        policy
        .get(
            "semantic_guards",
            {},
        )
        .get(
            "LOSE",
            {},
        )
    )

    if lose_guard.get(
        "generation_allowed"
    ) is not False:

        errors.append(
            "LOSE semantic guard is missing "
            "generation_allowed=false."
        )

    # ========================================================
    # Tense / polarity
    # ========================================================

    for verb_id in enabled_verbs:

        tenses = set(
            policy
            .get(
                "allowed_tenses",
                {},
            )
            .get(
                verb_id,
                [],
            )
        )

        if tenses != {
            "present",
            "future",
        }:

            errors.append(
                f"{verb_id}: expected tenses "
                f"present/future, got {sorted(tenses)}"
            )

        polarities = set(
            policy
            .get(
                "allowed_polarities",
                {},
            )
            .get(
                verb_id,
                [],
            )
        )

        if polarities != {
            "pos",
            "neg",
        }:

            errors.append(
                f"{verb_id}: expected polarities "
                f"pos/neg, got {sorted(polarities)}"
            )

    # ========================================================
    # Information warnings
    # ========================================================

    warnings.append(
        "LOSE renderer exists but generation is "
        "intentionally blocked."
    )

    warnings.append(
        "V0.5.1 Generator should activate only "
        "SEE_OBJECT and TAKE_OBJECT."
    )

    # ========================================================
    # Report
    # ========================================================

    print(
        "=" * 90
    )

    print(
        "V0.5.1 ACTIVATION VALIDATOR"
    )

    print(
        "=" * 90
    )

    print(
        "Enabled verbs:",
        sorted(enabled_verbs),
    )

    print(
        "Enabled frames:",
        sorted(enabled_frames),
    )

    print(
        "Warnings:",
        len(warnings),
    )

    print(
        "Errors:",
        len(errors),
    )

    print()

    if warnings:

        print(
            "WARNINGS"
        )

        print(
            "-" * 90
        )

        for warning in warnings:

            print(
                "WARN:",
                warning,
            )

    if errors:

        print()

        print(
            "ERRORS"
        )

        print(
            "-" * 90
        )

        for error in errors:

            print(
                "ERROR:",
                error,
            )

        print()

        print(
            "V0.5.1 ACTIVATION VALIDATION FAILED"
        )

        raise SystemExit(1)

    print()

    print(
        "=" * 90
    )

    print(
        "V0.5.1 ACTIVATION VALIDATION PASS"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()