from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v0441_regression_200_fix2"
    / "03_grammar_hard_v0441"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v0441_targeted_80.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v0441_targeted_80_summary.json"
)

LEXICALIZATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "ru_motion_lexicalization_v0441.json"
)


TARGET_N = 80
SEED = 2056


# ============================================================
# IO
# ============================================================

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

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

    return rows


def read_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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


# ============================================================
# Group classifier
# ============================================================

def build_destination_lookup(
    lexicalization: dict,
) -> dict[str, str]:

    result = {}

    classes = lexicalization.get(
        "destination_classes",
        {},
    )

    for class_name, destination_ids in classes.items():

        for destination_id in destination_ids:

            result[
                destination_id
            ] = class_name

    return result


def classify_group(
    row: dict,
    destination_lookup: dict[str, str],
) -> str:

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

    verb = slots.get(
        "verb"
    )

    destination = slots.get(
        "destination"
    )

    event_type = features.get(
        "event_type"
    )

    # --------------------------------------------------------
    # Habitual motion
    # --------------------------------------------------------

    if (
        event_type == "habitual"
        and verb in {
            "GO",
            "COME",
        }
    ):

        destination_class = (
            destination_lookup.get(
                destination,
                "no_destination",
            )
        )

        return (
            f"HABITUAL_"
            f"{verb}_"
            f"{destination_class.upper()}"
        )

    # --------------------------------------------------------
    # Planned GO / COME
    # --------------------------------------------------------

    if (
        event_type == "planned"
        and verb == "GO"
    ):
        return "PLANNED_GO"

    if (
        event_type == "planned"
        and verb == "COME"
    ):
        return "PLANNED_COME"

    # --------------------------------------------------------
    # ARRIVE
    # --------------------------------------------------------

    if verb == "ARRIVE":
        return "ARRIVE"

    # --------------------------------------------------------
    # Clock
    # --------------------------------------------------------

    if (
        isinstance(
            frame_id,
            str,
        )
        and frame_id.endswith(
            "_CLOCK"
        )
    ):
        return "CLOCK"

    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    if frame_id == "WHERE_PLACE":
        return "WHERE_PLACE"

    return "OTHER"


# ============================================================
# Main sampler
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    parser.add_argument(
        "--summary",
        default=str(
            DEFAULT_SUMMARY
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=TARGET_N,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    rng = random.Random(
        args.seed
    )

    rows = read_jsonl(
        Path(
            args.input
        )
    )

    lexicalization = read_json(
        LEXICALIZATION_FILE
    )

    destination_lookup = (
        build_destination_lookup(
            lexicalization
        )
    )

    groups = defaultdict(
        list
    )

    for row in rows:

        group = classify_group(
            row,
            destination_lookup,
        )

        groups[
            group
        ].append(
            row
        )

    # ========================================================
    # Desired emphasis
    #
    # These are targets rather than hard requirements.
    # Current 200-row set may not contain enough samples for
    # every group.
    # ========================================================

    quotas = {
        "HABITUAL_GO_LOCAL": 8,
        "HABITUAL_GO_TRAVEL": 8,

        "HABITUAL_COME_LOCAL": 8,
        "HABITUAL_COME_TRAVEL": 8,

        "PLANNED_GO": 10,
        "PLANNED_COME": 10,

        "ARRIVE": 10,

        "CLOCK": 10,

        "WHERE_PLACE": 8,
    }

    selected = []
    selected_ids = set()

    selected_group_counter = Counter()

    # --------------------------------------------------------
    # First pass:
    # take quota from each important group
    # --------------------------------------------------------

    for group_name, quota in quotas.items():

        candidates = list(
            groups.get(
                group_name,
                [],
            )
        )

        rng.shuffle(
            candidates
        )

        picked = candidates[
            :quota
        ]

        for row in picked:

            semantic_id = row.get(
                "semantic_id"
            )

            if semantic_id in selected_ids:
                continue

            selected.append(
                row
            )

            selected_ids.add(
                semantic_id
            )

            selected_group_counter[
                group_name
            ] += 1

    # --------------------------------------------------------
    # Second pass:
    # backfill to target N with remaining targeted examples
    # --------------------------------------------------------

    targeted_group_names = set(
        quotas
    )

    remaining_targeted = []

    for group_name in targeted_group_names:

        for row in groups.get(
            group_name,
            []
        ):

            if row.get(
                "semantic_id"
            ) not in selected_ids:

                remaining_targeted.append(
                    row
                )

    rng.shuffle(
        remaining_targeted
    )

    for row in remaining_targeted:

        if len(
            selected
        ) >= args.n:
            break

        semantic_id = row.get(
            "semantic_id"
        )

        if semantic_id in selected_ids:
            continue

        selected.append(
            row
        )

        selected_ids.add(
            semantic_id
        )

        group_name = classify_group(
            row,
            destination_lookup,
        )

        selected_group_counter[
            group_name
        ] += 1

    # --------------------------------------------------------
    # Third pass:
    # still not enough → use OTHER rows for general regression.
    # --------------------------------------------------------

    if len(
        selected
    ) < args.n:

        other_rows = [
            row
            for row in rows
            if (
                row.get(
                    "semantic_id"
                )
                not in selected_ids
            )
        ]

        rng.shuffle(
            other_rows
        )

        for row in other_rows:

            if len(
                selected
            ) >= args.n:
                break

            semantic_id = row.get(
                "semantic_id"
            )

            selected.append(
                row
            )

            selected_ids.add(
                semantic_id
            )

            group_name = classify_group(
                row,
                destination_lookup,
            )

            selected_group_counter[
                group_name
            ] += 1

    # --------------------------------------------------------
    # Add audit metadata without touching semantics
    # --------------------------------------------------------

    output_rows = []

    for index, row in enumerate(
        selected,
        start=1,
    ):

        output_row = dict(
            row
        )

        output_row[
            "audit_metadata"
        ] = {
            "audit_version":
                "v0441_targeted",

            "audit_index":
                index,

            "audit_group":
                classify_group(
                    row,
                    destination_lookup,
                ),
        }

        output_rows.append(
            output_row
        )

    summary = {
        "source_rows":
            len(
                rows
            ),

        "requested":
            args.n,

        "selected":
            len(
                output_rows
            ),

        "seed":
            args.seed,

        "available_groups": {
            key:
                len(
                    value
                )
            for key, value in sorted(
                groups.items()
            )
        },

        "selected_groups":
            dict(
                selected_group_counter
                .most_common()
            ),

        "quotas":
            quotas,
    }

    write_jsonl(
        Path(
            args.output
        ),
        output_rows,
    )

    write_json(
        Path(
            args.summary
        ),
        summary,
    )

    print(
        "=" * 90
    )

    print(
        "V0.4.4.1 TARGETED LINGUISTIC AUDIT SAMPLER"
    )

    print(
        "=" * 90
    )

    print(
        "Source:",
        len(
            rows
        ),
    )

    print(
        "Selected:",
        len(
            output_rows
        ),
    )

    print()

    print(
        "Available / Selected:"
    )

    print(
        "-" * 70
    )

    all_groups = sorted(
        set(
            groups
        )
        | set(
            selected_group_counter
        )
    )

    for group_name in all_groups:

        print(
            f"{group_name:<30}"
            f"available="
            f"{len(groups.get(group_name, [])):<5}"
            f"selected="
            f"{selected_group_counter.get(group_name, 0)}"
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

    print()

    print(
        "AUDIT SAMPLE COMPLETE"
    )


if __name__ == "__main__":
    main()