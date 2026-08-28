from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "semantic_v051_raw.jsonl"
)


ALLOWED_NEW = {
    "SEE_OBJECT": "SEE",
    "TAKE_OBJECT": "TAKE",
}

BLOCKED_VERBS = {
    "LOSE",
    "CALL",
    "WAIT",
    "GIVE",
    "BRING",
    "RETURN",
    "NEED",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "LOSE_OBJECT",
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "RETURN_PLACE",
    "NEED_OBJECT",
    "LEAVE_PLACE",
    "WHERE_OBJECT",
    "WHERE_PERSON",
}


def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_no}"
                ) from exc

            rows.append(row)

    return rows


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    args = parser.parse_args()

    rows = read_jsonl(
        Path(args.input)
    )

    errors = []

    semantic_ids = [
        row.get("semantic_id")
        for row in rows
    ]

    if len(semantic_ids) != len(
        set(semantic_ids)
    ):
        errors.append(
            "Duplicate semantic IDs found."
        )

    new_frame_counter = Counter()
    new_verb_counter = Counter()

    blocked_verb_count = 0
    blocked_frame_count = 0
    malformed_new_count = 0

    for row in rows:

        semantic_id = row.get(
            "semantic_id"
        )

        frame_id = row.get(
            "frame_id"
        )

        slots = row.get(
            "slots",
            {},
        )

        verb_id = slots.get(
            "verb"
        )

        # ====================================================
        # Disabled V0.5 abilities must never leak.
        # ====================================================

        if verb_id in BLOCKED_VERBS:

            blocked_verb_count += 1

            errors.append(
                f"{semantic_id}: "
                f"blocked verb leaked: "
                f"{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:

            blocked_frame_count += 1

            errors.append(
                f"{semantic_id}: "
                f"blocked frame leaked: "
                f"{frame_id}"
            )

        # ====================================================
        # New V0.5.1 contract
        # ====================================================

        if frame_id in ALLOWED_NEW:

            expected_verb = (
                ALLOWED_NEW[
                    frame_id
                ]
            )

            new_frame_counter[
                frame_id
            ] += 1

            new_verb_counter[
                verb_id
            ] += 1

            if verb_id != expected_verb:

                malformed_new_count += 1

                errors.append(
                    f"{semantic_id}: "
                    f"{frame_id} expects "
                    f"{expected_verb}, "
                    f"got {verb_id}"
                )

            if not slots.get(
                "subject"
            ):

                malformed_new_count += 1

                errors.append(
                    f"{semantic_id}: "
                    f"missing subject"
                )

            if not slots.get(
                "object"
            ):

                malformed_new_count += 1

                errors.append(
                    f"{semantic_id}: "
                    f"missing object"
                )

            features = row.get(
                "features",
                {},
            )

            if features.get(
                "tense"
            ) not in {
                "present",
                "future",
            }:

                malformed_new_count += 1

                errors.append(
                    f"{semantic_id}: "
                    f"invalid tense"
                )

            if features.get(
                "polarity"
            ) not in {
                "pos",
                "neg",
            }:

                malformed_new_count += 1

                errors.append(
                    f"{semantic_id}: "
                    f"invalid polarity"
                )

            texts = row.get(
                "texts",
                {},
            )

            for language in (
                "zh",
                "en",
                "ru",
                "uz",
            ):

                text = texts.get(
                    language
                )

                if not isinstance(
                    text,
                    str,
                ) or not text.strip():

                    malformed_new_count += 1

                    errors.append(
                        f"{semantic_id}: "
                        f"missing {language} text"
                    )

    # ========================================================
    # Required new capabilities
    # ========================================================

    for frame_id, verb_id in (
        ALLOWED_NEW.items()
    ):

        if new_frame_counter[
            frame_id
        ] == 0:

            errors.append(
                f"{frame_id} count = 0"
            )

        if new_verb_counter[
            verb_id
        ] == 0:

            errors.append(
                f"{verb_id} count = 0"
            )

    # ========================================================
    # Report
    # ========================================================

    print("=" * 90)
    print("V0.5.1 GENERATION CONTRACT TEST")
    print("=" * 90)

    print(
        "Rows:",
        len(rows),
    )

    print(
        "New frames:",
        dict(new_frame_counter),
    )

    print(
        "New verbs:",
        dict(new_verb_counter),
    )

    print(
        "Blocked verb count:",
        blocked_verb_count,
    )

    print(
        "Blocked frame count:",
        blocked_frame_count,
    )

    print(
        "Malformed new rows:",
        malformed_new_count,
    )

    print(
        "Errors:",
        len(errors),
    )

    if errors:

        print()
        print("ERRORS")
        print("-" * 90)

        for error in errors[:50]:
            print(
                "ERROR:",
                error,
            )

        raise SystemExit(1)

    print()
    print("=" * 90)
    print("V0.5.1 GENERATION CONTRACT: PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()