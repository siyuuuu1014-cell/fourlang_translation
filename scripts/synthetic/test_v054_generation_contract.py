from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.synthetic.renderer_v054 import (
    V054Renderer,
)


NEW_FRAMES = {
    "NEED_OBJECT",
    "LOSE_OBJECT",
}

NEW_VERBS = {
    "NEED",
    "LOSE",
}

BLOCKED_VERBS = {
    "RETURN",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "RETURN_PLACE",
    "LEAVE_PLACE",
}


# =============================================================================
# IO
# =============================================================================

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

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from exc

    return rows


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "V0.5.4 Generation "
            "Contract Regression"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    rows = read_jsonl(
        args.input
    )

    renderer = V054Renderer()

    errors = []

    semantic_ids = set()

    new_rows = []

    frame_counter = Counter()

    verb_counter = Counter()

    feature_counter = Counter()

    blocked_verb_count = 0
    blocked_frame_count = 0

    malformed_new_rows = 0

    rerender_mismatch_count = 0

    lose_present_count = 0

    lose_past_count = 0

    # =========================================================================
    # Rows
    # =========================================================================

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
                "MISSING_SEMANTIC_ID"
            )

        elif sid in semantic_ids:
            errors.append(
                f"{sid}: "
                "DUPLICATE_SEMANTIC_ID"
            )

        else:
            semantic_ids.add(
                sid
            )

        if not isinstance(
            slots,
            dict,
        ):
            errors.append(
                f"{sid}: "
                "INVALID_SLOTS"
            )
            continue

        verb_id = slots.get(
            "verb"
        )

        # ---------------------------------------------------------------------
        # Future capabilities must remain blocked
        # ---------------------------------------------------------------------

        if verb_id in BLOCKED_VERBS:

            blocked_verb_count += 1

            errors.append(
                f"{sid}: "
                f"BLOCKED_VERB:{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:

            blocked_frame_count += 1

            errors.append(
                f"{sid}: "
                f"BLOCKED_FRAME:{frame_id}"
            )

        # Only inspect new V0.5.4 rows below.
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

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        feature_counter[
            (
                frame_id,
                tense,
                polarity,
            )
        ] += 1

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        if (
            metadata.get(
                "generation_source"
            )
            != "v054_new_frame"
        ):
            errors.append(
                f"{sid}: "
                "GENERATION_SOURCE_MISMATCH"
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
                "LANGUAGE_SET_MISMATCH"
            )

        # ---------------------------------------------------------------------
        # NEED
        # ---------------------------------------------------------------------

        if frame_id == "NEED_OBJECT":

            expected_slots = {
                "subject",
                "verb",
                "object",
            }

            if set(
                slots.keys()
            ) != expected_slots:

                malformed_new_rows += 1

                errors.append(
                    f"{sid}: "
                    "MALFORMED_NEED_SLOTS"
                )

            if verb_id != "NEED":
                errors.append(
                    f"{sid}: "
                    "NEED_VERB_MISMATCH"
                )

            if tense not in {
                "present",
                "future",
            }:
                errors.append(
                    f"{sid}: "
                    f"NEED_INVALID_TENSE:"
                    f"{tense}"
                )

            if polarity not in {
                "pos",
                "neg",
            }:
                errors.append(
                    f"{sid}: "
                    f"NEED_INVALID_POLARITY:"
                    f"{polarity}"
                )

        # ---------------------------------------------------------------------
        # LOSE
        # ---------------------------------------------------------------------

        elif frame_id == "LOSE_OBJECT":

            expected_slots = {
                "subject",
                "verb",
                "object",
            }

            if set(
                slots.keys()
            ) != expected_slots:

                malformed_new_rows += 1

                errors.append(
                    f"{sid}: "
                    "MALFORMED_LOSE_SLOTS"
                )

            if verb_id != "LOSE":
                errors.append(
                    f"{sid}: "
                    "LOSE_VERB_MISMATCH"
                )

            if tense == "present":

                lose_present_count += 1

                errors.append(
                    f"{sid}: "
                    "LOSE_PRESENT_LEAKAGE"
                )

            elif tense == "past":

                lose_past_count += 1

                errors.append(
                    f"{sid}: "
                    "LOSE_PAST_LEAKAGE"
                )

            elif tense != "future":

                errors.append(
                    f"{sid}: "
                    f"LOSE_INVALID_TENSE:"
                    f"{tense}"
                )

            if polarity not in {
                "pos",
                "neg",
            }:
                errors.append(
                    f"{sid}: "
                    f"LOSE_INVALID_POLARITY:"
                    f"{polarity}"
                )

        # ---------------------------------------------------------------------
        # Deterministic rerender
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
                f"{sid}: "
                f"RERENDER_FAILED:{exc}"
            )

            continue

        if expected_texts != texts:

            rerender_mismatch_count += 1

            errors.append(
                f"{sid}: "
                "RERENDER_MISMATCH"
            )

    # =========================================================================
    # Aggregate checks
    # =========================================================================

    new_count = len(
        new_rows
    )

    if new_count == 0:
        errors.append(
            "NO_V054_NEW_ROWS"
        )

    expected_need = (
        new_count // 2
    )

    expected_lose = (
        new_count - expected_need
    )

    if frame_counter[
        "NEED_OBJECT"
    ] != expected_need:

        errors.append(
            "NEED_COUNT_MISMATCH:"
            f"{frame_counter['NEED_OBJECT']}"
            f"!={expected_need}"
        )

    if frame_counter[
        "LOSE_OBJECT"
    ] != expected_lose:

        errors.append(
            "LOSE_COUNT_MISMATCH:"
            f"{frame_counter['LOSE_OBJECT']}"
            f"!={expected_lose}"
        )

    if set(
        verb_counter.keys()
    ) != NEW_VERBS:

        errors.append(
            "NEW_VERB_SET_MISMATCH:"
            f"{set(verb_counter.keys())}"
        )

    # -------------------------------------------------------------------------
    # NEED feature coverage
    # -------------------------------------------------------------------------

    if frame_counter[
        "NEED_OBJECT"
    ] >= 4:

        for tense in (
            "present",
            "future",
        ):
            for polarity in (
                "pos",
                "neg",
            ):

                count = feature_counter[
                    (
                        "NEED_OBJECT",
                        tense,
                        polarity,
                    )
                ]

                if count == 0:
                    errors.append(
                        "NEED_FEATURE_MISSING:"
                        f"{tense}:{polarity}"
                    )

    # -------------------------------------------------------------------------
    # LOSE feature coverage
    # -------------------------------------------------------------------------

    if frame_counter[
        "LOSE_OBJECT"
    ] >= 2:

        for polarity in (
            "pos",
            "neg",
        ):

            count = feature_counter[
                (
                    "LOSE_OBJECT",
                    "future",
                    polarity,
                )
            ]

            if count == 0:
                errors.append(
                    "LOSE_FEATURE_MISSING:"
                    f"future:{polarity}"
                )

    # =========================================================================
    # Report
    # =========================================================================

    print("=" * 100)
    print(
        "V0.5.4 GENERATION CONTRACT TEST"
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
        "V0.5.4 new rows:",
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
        "Feature distribution:"
    )

    for key, count in sorted(
        feature_counter.items()
    ):
        print(
            f"{str(key):<45}"
            f"{count}"
        )

    print()

    print(
        "LOSE present count:",
        lose_present_count,
    )

    print(
        "LOSE past count:",
        lose_past_count,
    )

    print()

    print(
        "Malformed new rows:",
        malformed_new_rows,
    )

    print(
        "Rerender mismatches:",
        rerender_mismatch_count,
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
            "V0.5.4 GENERATION CONTRACT: FAIL"
        )

        print("=" * 100)

        raise SystemExit(1)

    print(
        "V0.5.4 GENERATION CONTRACT: PASS"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()