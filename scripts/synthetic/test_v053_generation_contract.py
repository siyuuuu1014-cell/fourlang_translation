from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


VERSION = "0.5.3"

NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}

NEW_VERBS = {
    "GIVE",
    "BRING",
}

BLOCKED_VERBS = {
    "LEAVE",
    "LOSE",
    "NEED",
    "RETURN",
}

BLOCKED_FRAMES = {
    "LEAVE_PLACE",
    "LOSE_OBJECT",
    "NEED_OBJECT",
    "RETURN_PLACE",
}

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    rows = read_jsonl(
        args.input
    )

    renderer = V053Renderer()

    errors = []

    semantic_ids = set()

    new_rows = []

    frame_counter = Counter()

    verb_counter = Counter()

    overlap_count = 0
    malformed_count = 0
    role_render_mismatch = 0
    blocked_verb_count = 0
    blocked_frame_count = 0

    for row in rows:

        sid = row.get(
            "semantic_id"
        )

        frame_id = row.get(
            "frame_id"
        )

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        texts = row.get(
            "texts",
            {},
        )

        metadata = row.get(
            "metadata",
            {},
        )

        # ---------------------------------------------------------------------
        # semantic_id
        # ---------------------------------------------------------------------

        if not sid:

            errors.append(
                "Missing semantic_id"
            )

        elif sid in semantic_ids:

            errors.append(
                f"{sid}: duplicate semantic_id"
            )

        else:
            semantic_ids.add(
                sid
            )

        # ---------------------------------------------------------------------
        # Block future capabilities
        # ---------------------------------------------------------------------

        verb_id = (
            slots.get(
                "verb"
            )
            if isinstance(
                slots,
                dict,
            )
            else None
        )

        if verb_id in BLOCKED_VERBS:

            blocked_verb_count += 1

            errors.append(
                f"{sid}: "
                f"blocked verb leaked: "
                f"{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:

            blocked_frame_count += 1

            errors.append(
                f"{sid}: "
                f"blocked frame leaked: "
                f"{frame_id}"
            )

        # ---------------------------------------------------------------------
        # New V0.5.3
        # ---------------------------------------------------------------------

        if frame_id not in NEW_FRAMES:
            continue

        new_rows.append(
            row
        )

        frame_counter[
            frame_id
        ] += 1

        verb_counter[
            verb_id
        ] += 1

        if (
            metadata.get(
                "generation_source"
            )
            != "v053_new_frame"
        ):

            errors.append(
                f"{sid}: "
                "new generation_source mismatch"
            )

        # ---------------------------------------------------------------------
        # GIVE
        # ---------------------------------------------------------------------

        if frame_id == (
            "GIVE_OBJECT_PERSON"
        ):

            expected_keys = {
                "subject",
                "verb",
                "object",
                "person",
            }

            if (
                set(
                    slots.keys()
                )
                != expected_keys
            ):

                malformed_count += 1

                errors.append(
                    f"{sid}: "
                    f"malformed GIVE slots: "
                    f"{slots}"
                )

            if verb_id != "GIVE":

                errors.append(
                    f"{sid}: "
                    "GIVE verb mismatch"
                )

            subject_id = slots.get(
                "subject"
            )

            person_id = slots.get(
                "person"
            )

            if (
                not subject_id
                or not person_id
                or subject_id == person_id
                or (
                    subject_id,
                    person_id,
                )
                in PARTICIPANT_OVERLAP_PAIRS
            ):

                overlap_count += 1

                errors.append(
                    f"{sid}: "
                    "GIVE participant overlap "
                    f"{subject_id}->{person_id}"
                )

        # ---------------------------------------------------------------------
        # BRING
        # ---------------------------------------------------------------------

        elif frame_id == (
            "BRING_OBJECT_DESTINATION"
        ):

            expected_keys = {
                "subject",
                "verb",
                "object",
                "destination",
            }

            if (
                set(
                    slots.keys()
                )
                != expected_keys
            ):

                malformed_count += 1

                errors.append(
                    f"{sid}: "
                    f"malformed BRING slots: "
                    f"{slots}"
                )

            if verb_id != "BRING":

                errors.append(
                    f"{sid}: "
                    "BRING verb mismatch"
                )

        # ---------------------------------------------------------------------
        # Tense / polarity
        # ---------------------------------------------------------------------

        if features.get(
            "tense"
        ) not in {
            "present",
            "future",
        }:

            errors.append(
                f"{sid}: invalid tense"
            )

        if features.get(
            "polarity"
        ) not in {
            "pos",
            "neg",
        }:

            errors.append(
                f"{sid}: invalid polarity"
            )

        # ---------------------------------------------------------------------
        # Languages
        # ---------------------------------------------------------------------

        if set(
            texts.keys()
        ) != {
            "zh",
            "en",
            "ru",
            "uz",
        }:

            errors.append(
                f"{sid}: "
                "language set mismatch"
            )

        # ---------------------------------------------------------------------
        # Deterministic rerender
        #
        # This catches stale text, accidental role swap, wrong case forms,
        # wrong directional realization, etc.
        # ---------------------------------------------------------------------

        try:

            expected_texts = (
                renderer.render(
                    frame_id=frame_id,
                    slots=slots,
                    features=features,
                )
            )

        except Exception as exc:

            errors.append(
                f"{sid}: rerender failed: "
                f"{exc}"
            )

            continue

        if expected_texts != texts:

            role_render_mismatch += 1

            errors.append(
                f"{sid}: "
                "deterministic render mismatch"
            )

    # =========================================================================
    # Aggregate checks
    # =========================================================================

    new_count = len(
        new_rows
    )

    if new_count == 0:

        errors.append(
            "No V0.5.3 new rows found."
        )

    expected_give = (
        new_count // 2
    )

    expected_bring = (
        new_count - expected_give
    )

    if (
        frame_counter[
            "GIVE_OBJECT_PERSON"
        ]
        != expected_give
    ):

        errors.append(
            "GIVE frame count mismatch: "
            f"{frame_counter['GIVE_OBJECT_PERSON']} "
            f"!= {expected_give}"
        )

    if (
        frame_counter[
            "BRING_OBJECT_DESTINATION"
        ]
        != expected_bring
    ):

        errors.append(
            "BRING frame count mismatch: "
            f"{frame_counter['BRING_OBJECT_DESTINATION']} "
            f"!= {expected_bring}"
        )

    if set(
        verb_counter.keys()
    ) != NEW_VERBS:

        errors.append(
            "New verb set mismatch: "
            f"{set(verb_counter.keys())}"
        )

    # =========================================================================
    # Report
    # =========================================================================

    print("=" * 100)
    print(
        "V0.5.3 GENERATION CONTRACT TEST"
    )
    print("=" * 100)

    print(
        "Input:",
        args.input,
    )

    print(
        "Rows:",
        len(rows),
    )

    print(
        "V0.5.3 new rows:",
        new_count,
    )

    print()

    print(
        "New frames:",
        dict(
            frame_counter
        ),
    )

    print(
        "New verbs:",
        dict(
            verb_counter
        ),
    )

    print()

    print(
        "Participant overlap count:",
        overlap_count,
    )

    print(
        "Malformed new rows:",
        malformed_count,
    )

    print(
        "Role/render mismatches:",
        role_render_mismatch,
    )

    print(
        "Blocked verb count:",
        blocked_verb_count,
    )

    print(
        "Blocked frame count:",
        blocked_frame_count,
    )

    print()

    print(
        "Errors:",
        len(errors),
    )

    if errors:

        print()
        print("ERROR DETAILS")
        print("-" * 100)

        for error in errors[
            :100
        ]:
            print(
                "ERROR:",
                error,
            )

    print()
    print("=" * 100)

    if errors:

        print(
            "V0.5.3 GENERATION CONTRACT: FAIL"
        )

        print("=" * 100)

        raise SystemExit(1)

    print(
        "V0.5.3 GENERATION CONTRACT: PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()