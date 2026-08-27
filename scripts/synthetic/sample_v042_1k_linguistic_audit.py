from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v042_pilot_1k"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v042_1k_linguistic_150.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v042_1k_linguistic_150_summary.json"
)


# ============================================================
# Sampling targets
# ============================================================
#
# New V0.4 abilities receive most of the audit budget:
#
# COME       30
# ARRIVE     30
# WANT       30
#
# Existing verbs:
# GO          8
# BUY         8
# FIND        8
# EAT         7
# DRINK       7
# READ        7
#
# WHERE      15
#
# TOTAL     150
# ============================================================

TARGETS = {
    "COME": 30,
    "ARRIVE": 30,
    "WANT": 30,

    "GO": 8,
    "BUY": 8,
    "FIND": 8,
    "EAT": 7,
    "DRINK": 7,
    "READ": 7,

    "__WHERE_PLACE__": 15,
}

TOTAL_TARGET = sum(
    TARGETS.values()
)


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

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

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):

                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must be JSON object."
                )

            rows.append(
                row
            )

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


# ============================================================
# Row helpers
# ============================================================

def get_verb(
    row: dict,
) -> str | None:

    return (
        row
        .get(
            "slots",
            {},
        )
        .get(
            "verb"
        )
    )


def get_frame(
    row: dict,
) -> str:

    return str(
        row.get(
            "frame_id",
            "NONE",
        )
    )


def group_name(
    row: dict,
) -> str | None:

    frame_id = get_frame(
        row
    )

    if frame_id == "WHERE_PLACE":

        return "__WHERE_PLACE__"

    return get_verb(
        row
    )


# ============================================================
# Coverage tokens
# ============================================================

def coverage_tokens(
    row: dict,
) -> set[str]:

    tokens = set()

    frame_id = get_frame(
        row
    )

    tokens.add(
        f"frame={frame_id}"
    )

    scenario = row.get(
        "scenario"
    )

    if scenario:

        tokens.add(
            f"scenario={scenario}"
        )

    slots = row.get(
        "slots",
        {},
    )

    if isinstance(
        slots,
        dict,
    ):

        for key in (
            "subject",
            "verb",
            "object",
            "destination",
            "time",
            "day",
        ):

            value = slots.get(
                key
            )

            if value:

                tokens.add(
                    f"{key}={value}"
                )

    features = row.get(
        "features",
        {},
    )

    if isinstance(
        features,
        dict,
    ):

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        if tense:

            tokens.add(
                f"tense={tense}"
            )

        if polarity:

            tokens.add(
                f"polarity={polarity}"
            )

    computed = row.get(
        "computed",
        {},
    )

    if isinstance(
        computed,
        dict,
    ):

        clock = computed.get(
            "clock"
        )

        if clock:

            # We want clock examples,
            # but exact HH:MM should not dominate
            # the diversity score.
            tokens.add(
                "has_clock=yes"
            )

    return tokens


# ============================================================
# Greedy diverse selection
# ============================================================

def greedy_diverse_sample(
    rows: list[dict],
    *,
    k: int,
    rng: random.Random,
) -> list[dict]:

    if len(
        rows
    ) < k:

        raise RuntimeError(
            f"Need {k} rows, "
            f"but only {len(rows)} available."
        )

    candidates = list(
        rows
    )

    rng.shuffle(
        candidates
    )

    selected = []

    covered = set()

    # Frequency of individual coverage tokens.
    token_frequency = Counter()

    for row in candidates:

        token_frequency.update(
            coverage_tokens(
                row
            )
        )

    while len(
        selected
    ) < k:

        best_index = None
        best_score = None

        for index, row in enumerate(
            candidates
        ):

            tokens = coverage_tokens(
                row
            )

            new_tokens = (
                tokens
                - covered
            )

            # ------------------------------------------------
            # Primary objective:
            # gain unseen dimensions
            # ------------------------------------------------

            new_score = sum(
                1.0
                / max(
                    token_frequency[token],
                    1,
                )
                for token in new_tokens
            )

            # ------------------------------------------------
            # Secondary:
            # prefer rows containing multiple dimensions
            # ------------------------------------------------

            diversity_score = (
                0.001
                * len(
                    tokens
                )
            )

            # ------------------------------------------------
            # Small random component resolves exact ties
            # deterministically under seed.
            # ------------------------------------------------

            tie_break = (
                rng.random()
                * 1e-6
            )

            score = (
                new_score
                + diversity_score
                + tie_break
            )

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_index = index

        if best_index is None:

            raise RuntimeError(
                "Unable to select audit sample."
            )

        row = candidates.pop(
            best_index
        )

        selected.append(
            row
        )

        covered.update(
            coverage_tokens(
                row
            )
        )

    return selected


# ============================================================
# Summary
# ============================================================

def build_summary(
    selected: list[dict],
    available: list[dict],
    seed: int,
) -> dict:

    verb_counter = Counter()

    frame_counter = Counter()

    scenario_counter = Counter()

    tense_counter = Counter()

    polarity_counter = Counter()

    subject_counter = Counter()

    object_counter = Counter()

    destination_counter = Counter()

    group_counter = Counter()

    for row in selected:

        group = group_name(
            row
        )

        if group:

            group_counter[
                group
            ] += 1

        verb = get_verb(
            row
        )

        if verb:

            verb_counter[
                verb
            ] += 1

        frame_counter[
            get_frame(
                row
            )
        ] += 1

        scenario = row.get(
            "scenario"
        )

        if scenario:

            scenario_counter[
                scenario
            ] += 1

        slots = row.get(
            "slots",
            {},
        )

        if isinstance(
            slots,
            dict,
        ):

            subject = slots.get(
                "subject"
            )

            object_id = slots.get(
                "object"
            )

            destination = slots.get(
                "destination"
            )

            if subject:

                subject_counter[
                    subject
                ] += 1

            if object_id:

                object_counter[
                    object_id
                ] += 1

            if destination:

                destination_counter[
                    destination
                ] += 1

        features = row.get(
            "features",
            {},
        )

        if isinstance(
            features,
            dict,
        ):

            tense = features.get(
                "tense"
            )

            polarity = features.get(
                "polarity"
            )

            if tense:

                tense_counter[
                    tense
                ] += 1

            if polarity:

                polarity_counter[
                    polarity
                ] += 1

    return {
        "seed":
            seed,

        "source_samples":
            len(
                available
            ),

        "selected_samples":
            len(
                selected
            ),

        "target_samples":
            TOTAL_TARGET,

        "requested_group_targets":
            TARGETS,

        "actual_group_counts":
            dict(
                group_counter
            ),

        "verb_distribution":
            dict(
                verb_counter.most_common()
            ),

        "frame_distribution":
            dict(
                frame_counter.most_common()
            ),

        "scenario_distribution":
            dict(
                scenario_counter.most_common()
            ),

        "tense_distribution":
            dict(
                tense_counter.most_common()
            ),

        "polarity_distribution":
            dict(
                polarity_counter.most_common()
            ),

        "subject_distribution":
            dict(
                subject_counter.most_common()
            ),

        "object_distribution":
            dict(
                object_counter.most_common()
            ),

        "destination_distribution":
            dict(
                destination_counter.most_common()
            ),
    }


# ============================================================
# Main sampling
# ============================================================

def sample_audit(
    rows: list[dict],
    *,
    seed: int,
) -> list[dict]:

    rng = random.Random(
        seed
    )

    groups: dict[
        str,
        list[dict],
    ] = {
        key: []
        for key in TARGETS
    }

    for row in rows:

        group = group_name(
            row
        )

        if group in groups:

            groups[
                group
            ].append(
                row
            )

    selected = []

    print(
        "=" * 90
    )

    print(
        "V0.4.2 1K LINGUISTIC AUDIT SAMPLER"
    )

    print(
        "=" * 90
    )

    print(
        "Input rows:",
        len(
            rows
        ),
    )

    print()

    for group, target in (
        TARGETS.items()
    ):

        available = groups[
            group
        ]

        print(
            f"{group:<20}"
            f"available={len(available):<5}"
            f" target={target}"
        )

        if len(
            available
        ) < target:

            raise RuntimeError(
                f"Group {group}: "
                f"available={len(available)}, "
                f"target={target}"
            )

        chosen = (
            greedy_diverse_sample(
                available,
                k=target,
                rng=rng,
            )
        )

        selected.extend(
            chosen
        )

    if len(
        selected
    ) != TOTAL_TARGET:

        raise RuntimeError(
            f"Expected {TOTAL_TARGET} selected, "
            f"got {len(selected)}"
        )

    semantic_ids = [
        row.get(
            "semantic_id"
        )
        for row in selected
    ]

    if len(
        semantic_ids
    ) != len(
        set(
            semantic_ids
        )
    ):

        raise RuntimeError(
            "Duplicate semantic_id "
            "in final audit selection."
        )

    rng.shuffle(
        selected
    )

    return selected


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Stratified 150-sample linguistic "
            "audit sampler for V0.4.2 1K pilot."
        )
    )

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
        "--seed",
        type=int,
        default=2045,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    summary_path = Path(
        args.summary
    )

    rows = read_jsonl(
        input_path
    )

    selected = sample_audit(
        rows,
        seed=args.seed,
    )

    summary = build_summary(
        selected,
        rows,
        args.seed,
    )

    write_jsonl(
        output_path,
        selected,
    )

    write_json(
        summary_path,
        summary,
    )

    print()

    print(
        "=" * 90
    )

    print(
        "AUDIT SAMPLE COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "Selected:",
        len(
            selected
        ),
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print()

    print(
        "Group distribution:"
    )

    for key, value in (
        summary[
            "actual_group_counts"
        ].items()
    ):

        print(
            f"{key:<20}"
            f"{value}"
        )

    print()

    print(
        "Tense:",
        summary[
            "tense_distribution"
        ],
    )

    print(
        "Polarity:",
        summary[
            "polarity_distribution"
        ],
    )


if __name__ == "__main__":

    main()