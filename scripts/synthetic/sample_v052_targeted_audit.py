from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_audit_200"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v052_targeted_40.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v052_targeted_40_summary.json"
)


TARGETS = {
    "CALL_PERSON": 14,
    "WAIT_PERSON": 13,
    "WAIT_AT_PLACE": 13,
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
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL "
                    f"{path}:{line_no}"
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
    obj: dict,
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
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


def get_group_tokens(
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

    tokens = {
        f"frame:{frame}",
        f"tense:{features.get('tense')}",
        f"polarity:{features.get('polarity')}",
        f"subject:{slots.get('subject')}",
    }

    if frame in {
        "CALL_PERSON",
        "WAIT_PERSON",
    }:
        tokens.add(
            f"person:{slots.get('person')}"
        )

    if frame == "WAIT_AT_PLACE":
        tokens.add(
            f"place:{slots.get('place')}"
        )

    # Combined high-value coverage dimensions
    tokens.add(
        "tp:"
        f"{features.get('tense')}:"
        f"{features.get('polarity')}"
    )

    tokens.add(
        "frame_tp:"
        f"{frame}:"
        f"{features.get('tense')}:"
        f"{features.get('polarity')}"
    )

    return tokens


def select_group(
    rows: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:

    if len(rows) < n:
        raise RuntimeError(
            f"Not enough rows: "
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

            tokens = get_group_tokens(
                row
            )

            # Prefer dimensions not yet covered.
            novelty = sum(
                1.0 / (
                    1.0
                    + covered[token]
                )
                for token in tokens
            )

            # Strongly favor uncovered
            # tense × polarity combinations.
            features = row.get(
                "features",
                {},
            )

            frame = row.get(
                "frame_id"
            )

            combo = (
                "frame_tp:"
                f"{frame}:"
                f"{features.get('tense')}:"
                f"{features.get('polarity')}"
            )

            if covered[combo] == 0:
                novelty += 5.0

            # Mild random tiebreak.
            score = (
                novelty
                + rng.random() * 1e-4
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

        for token in get_group_tokens(
            row
        ):
            covered[token] += 1

    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V0.5.2 Targeted "
            "Linguistic Audit Sampler"
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
        default=2054,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    rows = read_jsonl(
        args.input
    )

    # Only V0.5.2 new capabilities.
    candidates = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    ]

    print("=" * 100)
    print(
        "V0.5.2 TARGETED "
        "LINGUISTIC AUDIT SAMPLER"
    )
    print("=" * 100)

    print(
        "Input rows:",
        len(rows),
    )

    print(
        "V0.5.2 candidates:",
        len(candidates),
    )

    candidate_counts = Counter(
        row.get(
            "frame_id"
        )
        for row in candidates
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
            f"{frame_id:<20}"
            f" available={len(pool):<4}"
            f" target={target}"
        )

        picked = select_group(
            pool,
            target,
            rng,
        )

        selected.extend(
            picked
        )

    rng.shuffle(
        selected
    )

    # Add audit metadata without
    # changing semantic content.
    output_rows = []

    for row in selected:

        row = json.loads(
            json.dumps(
                row,
                ensure_ascii=False,
            )
        )

        frame_id = row[
            "frame_id"
        ]

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        row[
            "audit_metadata"
        ] = {
            "audit_version":
                "0.5.2",

            "audit_group":
                frame_id,

            "audit_reason":
                (
                    "V0.5.2 CALL/WAIT "
                    "targeted linguistic audit"
                ),

            "tense":
                features.get(
                    "tense"
                ),

            "polarity":
                features.get(
                    "polarity"
                ),

            "subject":
                slots.get(
                    "subject"
                ),

            "target":
                (
                    slots.get("person")
                    or slots.get("place")
                ),
        }

        output_rows.append(
            row
        )

    write_jsonl(
        args.output,
        output_rows,
    )

    # ========================================================
    # Statistics
    # ========================================================

    frame_counts = Counter(
        r["frame_id"]
        for r in output_rows
    )

    tense_counts = Counter(
        r.get(
            "features",
            {},
        ).get(
            "tense"
        )
        for r in output_rows
    )

    polarity_counts = Counter(
        r.get(
            "features",
            {},
        ).get(
            "polarity"
        )
        for r in output_rows
    )

    subject_counts = Counter(
        r.get(
            "slots",
            {},
        ).get(
            "subject"
        )
        for r in output_rows
    )

    combo_counts = Counter(
        (
            r["frame_id"],
            r.get(
                "features",
                {},
            ).get(
                "tense"
            ),
            r.get(
                "features",
                {},
            ).get(
                "polarity"
            ),
        )
        for r in output_rows
    )

    target_counts = Counter(
        (
            r.get(
                "slots",
                {},
            ).get("person")
            or r.get(
                "slots",
                {},
            ).get("place")
        )
        for r in output_rows
    )

    summary = {
        "audit_version":
            "0.5.2",

        "seed":
            args.seed,

        "input_rows":
            len(rows),

        "candidate_rows":
            len(candidates),

        "selected":
            len(output_rows),

        "frame_distribution":
            dict(
                frame_counts
            ),

        "tense_distribution":
            dict(
                tense_counts
            ),

        "polarity_distribution":
            dict(
                polarity_counts
            ),

        "subject_distribution":
            dict(
                subject_counts
            ),

        "target_distribution":
            dict(
                target_counts
            ),

        "frame_tense_polarity":
            {
                "|".join(
                    str(x)
                    for x in key
                ): value
                for key, value
                in combo_counts.items()
            },
    }

    write_json(
        args.summary,
        summary,
    )

    print()
    print("=" * 100)
    print("AUDIT SAMPLE COMPLETE")
    print("=" * 100)

    print(
        "Selected:",
        len(output_rows),
    )

    print()
    print(
        "Frame distribution:"
    )

    for key, value in (
        frame_counts.items()
    ):
        print(
            f"{key:<20}{value}"
        )

    print()
    print(
        "Tense:",
        dict(tense_counts),
    )

    print(
        "Polarity:",
        dict(polarity_counts),
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