from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.3"

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_audit_200"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v053_targeted_40.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v053_targeted_40_summary.json"
)


TARGETS = {
    "GIVE_OBJECT_PERSON": 20,
    "BRING_OBJECT_DESTINATION": 20,
}

NEW_FRAMES = set(TARGETS)


def read_jsonl(path: Path) -> list[dict]:

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


def write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.write("\n")


def coverage_tokens(
    row: dict,
) -> set[str]:

    frame = row.get(
        "frame_id",
        "NO_FRAME",
    )

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    tense = features.get(
        "tense",
        "NO_TENSE",
    )

    polarity = features.get(
        "polarity",
        "NO_POLARITY",
    )

    tokens = {
        f"frame:{frame}",
        f"subject:{slots.get('subject')}",
        f"object:{slots.get('object')}",
        f"tense:{tense}",
        f"polarity:{polarity}",
        f"tp:{tense}:{polarity}",
        f"frame_tp:{frame}:{tense}:{polarity}",
    }

    if frame == "GIVE_OBJECT_PERSON":

        tokens.add(
            f"person:{slots.get('person')}"
        )

        tokens.add(
            "give_roles:"
            f"{slots.get('object')}:"
            f"{slots.get('person')}"
        )

    elif frame == "BRING_OBJECT_DESTINATION":

        tokens.add(
            f"destination:{slots.get('destination')}"
        )

        tokens.add(
            "bring_roles:"
            f"{slots.get('object')}:"
            f"{slots.get('destination')}"
        )

    return tokens


def select_diverse(
    rows: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:

    if len(rows) < n:
        raise RuntimeError(
            f"Not enough candidates: "
            f"available={len(rows)}, "
            f"required={n}"
        )

    remaining = list(rows)

    rng.shuffle(
        remaining
    )

    selected = []

    covered = Counter()

    while len(selected) < n:

        best_index = None
        best_score = None

        for index, row in enumerate(
            remaining
        ):

            tokens = coverage_tokens(
                row
            )

            score = sum(
                1.0 / (
                    1.0
                    + covered[token]
                )
                for token in tokens
            )

            features = row.get(
                "features",
                {},
            )

            combo = (
                f"frame_tp:"
                f"{row.get('frame_id')}:"
                f"{features.get('tense')}:"
                f"{features.get('polarity')}"
            )

            # Strongly encourage every frame × tense × polarity cell.
            if covered[combo] == 0:
                score += 10.0

            # Prefer unseen subject.
            subject_token = (
                f"subject:"
                f"{row.get('slots', {}).get('subject')}"
            )

            if covered[
                subject_token
            ] == 0:
                score += 3.0

            # Prefer unseen object.
            object_token = (
                f"object:"
                f"{row.get('slots', {}).get('object')}"
            )

            if covered[
                object_token
            ] == 0:
                score += 3.0

            # Mild random tiebreak.
            score += (
                rng.random()
                * 1e-6
            )

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_index = index

        row = remaining.pop(
            best_index
        )

        selected.append(
            row
        )

        for token in coverage_tokens(
            row
        ):
            covered[token] += 1

    return selected


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "V0.5.3 GIVE/BRING "
            "Targeted Qwen Audit Sampler"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2058,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    rows = read_jsonl(
        args.input
    )

    candidates = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    ]

    print("=" * 100)
    print(
        "V0.5.3 TARGETED "
        "LINGUISTIC AUDIT SAMPLER"
    )
    print("=" * 100)

    print(
        "Input rows:",
        len(rows),
    )

    print(
        "V0.5.3 candidates:",
        len(candidates),
    )

    rng = random.Random(
        args.seed
    )

    selected = []

    for frame_id, target in (
        TARGETS.items()
    ):

        pool = [
            row
            for row in candidates
            if row.get(
                "frame_id"
            ) == frame_id
        ]

        print(
            f"{frame_id:<30}"
            f"available={len(pool):<5}"
            f"target={target}"
        )

        selected.extend(
            select_diverse(
                pool,
                target,
                rng,
            )
        )

    rng.shuffle(
        selected
    )

    output_rows = []

    for row in selected:

        output = copy.deepcopy(
            row
        )

        slots = output.get(
            "slots",
            {},
        )

        features = output.get(
            "features",
            {},
        )

        frame_id = output.get(
            "frame_id"
        )

        output[
            "audit_metadata"
        ] = {
            "audit_version":
                VERSION,

            "audit_group":
                frame_id,

            "audit_reason":
                (
                    "V0.5.3 GIVE/BRING "
                    "targeted linguistic audit"
                ),

            "subject":
                slots.get(
                    "subject"
                ),

            "object":
                slots.get(
                    "object"
                ),

            "person":
                slots.get(
                    "person"
                ),

            "destination":
                slots.get(
                    "destination"
                ),

            "tense":
                features.get(
                    "tense"
                ),

            "polarity":
                features.get(
                    "polarity"
                ),
        }

        output_rows.append(
            output
        )

    # =========================================================================
    # Stats
    # =========================================================================

    frame_counter = Counter(
        row.get(
            "frame_id"
        )
        for row in output_rows
    )

    tense_counter = Counter(
        row.get(
            "features",
            {},
        ).get(
            "tense"
        )
        for row in output_rows
    )

    polarity_counter = Counter(
        row.get(
            "features",
            {},
        ).get(
            "polarity"
        )
        for row in output_rows
    )

    subject_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "subject"
        )
        for row in output_rows
    )

    object_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "object"
        )
        for row in output_rows
    )

    person_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "person"
        )
        for row in output_rows
        if row.get(
            "frame_id"
        ) == "GIVE_OBJECT_PERSON"
    )

    destination_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "destination"
        )
        for row in output_rows
        if row.get(
            "frame_id"
        )
        == "BRING_OBJECT_DESTINATION"
    )

    combo_counter = Counter(
        (
            row.get("frame_id"),
            row.get(
                "features",
                {},
            ).get("tense"),
            row.get(
                "features",
                {},
            ).get("polarity"),
        )
        for row in output_rows
    )

    write_jsonl(
        args.output,
        output_rows,
    )

    summary = {
        "audit_version":
            VERSION,

        "seed":
            args.seed,

        "input_rows":
            len(rows),

        "candidate_rows":
            len(candidates),

        "selected":
            len(output_rows),

        "frame_distribution":
            dict(frame_counter),

        "tense_distribution":
            dict(tense_counter),

        "polarity_distribution":
            dict(polarity_counter),

        "subject_distribution":
            dict(subject_counter),

        "object_distribution":
            dict(object_counter),

        "person_distribution":
            dict(person_counter),

        "destination_distribution":
            dict(destination_counter),

        "frame_tense_polarity": {
            "|".join(
                str(x)
                for x in key
            ): value
            for key, value
            in combo_counter.items()
        },
    }

    write_json(
        args.summary,
        summary,
    )

    print()
    print("=" * 100)
    print(
        "V0.5.3 TARGETED "
        "AUDIT SAMPLE COMPLETE"
    )
    print("=" * 100)

    print(
        "Selected:",
        len(output_rows),
    )

    print()

    print(
        "Frames:",
        dict(frame_counter),
    )

    print(
        "Tense:",
        dict(tense_counter),
    )

    print(
        "Polarity:",
        dict(polarity_counter),
    )

    print(
        "Subjects:",
        dict(subject_counter),
    )

    print(
        "Objects:",
        dict(object_counter),
    )

    print(
        "GIVE persons:",
        dict(person_counter),
    )

    print(
        "BRING destinations:",
        dict(destination_counter),
    )

    print()

    print(
        "Output:",
        args.output,
    )

    print(
        "Summary:",
        args.summary,
    )


if __name__ == "__main__":
    main()