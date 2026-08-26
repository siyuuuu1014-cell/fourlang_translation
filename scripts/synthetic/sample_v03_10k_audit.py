from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v03_10k"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v03_10k_audit_300.jsonl"
)


DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v03_10k_audit_300_summary.json"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found:\n{path}"
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


# ============================================================
# Helpers
# ============================================================

def get_slots(row: dict) -> dict:

    slots = row.get(
        "slots",
        {},
    )

    if not isinstance(slots, dict):
        return {}

    return slots


def get_features(row: dict) -> dict:

    features = row.get(
        "features",
        {},
    )

    if not isinstance(features, dict):
        return {}

    return features


def get_sample_id(row: dict) -> str:

    return str(
        row.get("semantic_id")
        or row.get("id")
        or ""
    )


# ============================================================
# Feature tokens
# ============================================================

def extract_tokens(row: dict) -> set[str]:

    slots = get_slots(row)
    features = get_features(row)

    tokens = set()

    frame = row.get(
        "frame_id",
        "UNKNOWN",
    )

    tokens.add(
        f"frame::{frame}"
    )


    verb = slots.get(
        "verb"
    )

    if verb:
        tokens.add(
            f"verb::{verb}"
        )
    else:
        tokens.add(
            "verb::NO_VERB"
        )


    subject = slots.get(
        "subject"
    )

    if subject:
        tokens.add(
            f"subject::{subject}"
        )


    obj = slots.get(
        "object"
    )

    if obj:
        tokens.add(
            f"object::{obj}"
        )


    destination = slots.get(
        "destination"
    )

    if destination:
        tokens.add(
            f"destination::{destination}"
        )


    tense = features.get(
        "tense",
        "UNKNOWN",
    )

    tokens.add(
        f"tense::{tense}"
    )


    polarity = features.get(
        "polarity",
        "UNKNOWN",
    )

    tokens.add(
        f"polarity::{polarity}"
    )


    computed = row.get(
        "computed",
        {},
    )

    if not isinstance(computed, dict):
        computed = {}

    if computed.get("clock"):

        tokens.add(
            "clock::YES"
        )

    else:

        tokens.add(
            "clock::NO"
        )


    return tokens


# ============================================================
# Main stratum
# ============================================================

def get_stratum(row: dict) -> tuple:

    slots = get_slots(row)
    features = get_features(row)

    return (
        row.get(
            "frame_id",
            "UNKNOWN",
        ),

        slots.get(
            "verb"
        )
        or "NO_VERB",

        features.get(
            "tense",
            "UNKNOWN",
        ),

        features.get(
            "polarity",
            "UNKNOWN",
        ),
    )


# ============================================================
# Coverage score
# ============================================================

def build_global_token_counts(
    rows: list[dict],
) -> Counter:

    counts = Counter()

    for row in rows:

        for token in extract_tokens(row):

            counts[token] += 1

    return counts


def rarity_score(
    row: dict,
    global_counts: Counter,
) -> float:

    """
    Rare concepts receive slightly higher priority.

    1 / sqrt(freq)
    avoids letting extremely rare concepts dominate.
    """

    score = 0.0

    for token in extract_tokens(row):

        freq = global_counts[token]

        if freq > 0:

            score += (
                1.0
                / math.sqrt(freq)
            )

    return score


# ============================================================
# Balanced selection
# ============================================================

def select_audit_rows(
    rows: list[dict],
    target: int,
    seed: int,
) -> list[dict]:

    if target <= 0:
        raise ValueError(
            "target must be > 0"
        )

    if target > len(rows):
        raise ValueError(
            f"target={target} "
            f"> rows={len(rows)}"
        )


    rng = random.Random(seed)

    global_counts = (
        build_global_token_counts(
            rows
        )
    )


    # ========================================================
    # Group by:
    #
    # frame × verb × tense × polarity
    # ========================================================

    groups = defaultdict(list)

    for row in rows:

        groups[
            get_stratum(row)
        ].append(row)


    for group_rows in groups.values():

        rng.shuffle(
            group_rows
        )


    selected = []

    selected_ids = set()

    coverage = Counter()


    # ========================================================
    # Phase 1:
    # Try to cover each major stratum.
    # ========================================================

    group_items = list(
        groups.items()
    )

    # Rare strata first.
    group_items.sort(
        key=lambda kv: len(kv[1])
    )


    for (
        _,
        group_rows,
    ) in group_items:

        if len(selected) >= target:
            break

        candidate = max(
            group_rows,
            key=lambda r: rarity_score(
                r,
                global_counts,
            ),
        )

        sid = get_sample_id(
            candidate
        )

        if sid in selected_ids:
            continue

        selected.append(
            candidate
        )

        selected_ids.add(
            sid
        )

        for token in extract_tokens(
            candidate
        ):

            coverage[token] += 1


    # ========================================================
    # Phase 2:
    # Greedy coverage of underrepresented features.
    # ========================================================

    remaining = [
        row
        for row in rows
        if get_sample_id(row)
        not in selected_ids
    ]

    rng.shuffle(
        remaining
    )


    while (
        len(selected) < target
        and remaining
    ):

        best_index = None
        best_score = None

        # 10K x 300 is still small enough.
        for i, row in enumerate(
            remaining
        ):

            tokens = extract_tokens(
                row
            )

            score = 0.0

            for token in tokens:

                # Prefer tokens that have been
                # selected fewer times.
                selected_count = (
                    coverage[token]
                )

                freq = (
                    global_counts[token]
                )

                rarity = (
                    1.0
                    / math.sqrt(freq)
                    if freq
                    else 0.0
                )

                score += (
                    (1.0 / (1 + selected_count))
                    + rarity
                )


            # Tiny deterministic random tie breaker.
            score += (
                rng.random()
                * 1e-6
            )


            if (
                best_score is None
                or
                score > best_score
            ):

                best_score = score
                best_index = i


        candidate = remaining.pop(
            best_index
        )

        sid = get_sample_id(
            candidate
        )

        if sid in selected_ids:
            continue


        selected.append(
            candidate
        )

        selected_ids.add(
            sid
        )


        for token in extract_tokens(
            candidate
        ):

            coverage[token] += 1


    if len(selected) != target:

        raise RuntimeError(
            f"Expected {target}, "
            f"got {len(selected)}"
        )


    # Avoid ordering effects during Qwen audit.
    rng.shuffle(
        selected
    )

    return selected


# ============================================================
# Distribution report
# ============================================================

def build_distribution(
    rows: list[dict],
) -> dict:

    frame = Counter()
    verb = Counter()
    subject = Counter()
    obj = Counter()
    destination = Counter()
    tense = Counter()
    polarity = Counter()
    clock = Counter()


    for row in rows:

        slots = get_slots(row)
        features = get_features(row)

        frame[
            row.get(
                "frame_id",
                "UNKNOWN",
            )
        ] += 1


        verb[
            slots.get(
                "verb"
            )
            or "NO_VERB"
        ] += 1


        if slots.get("subject"):

            subject[
                slots["subject"]
            ] += 1


        if slots.get("object"):

            obj[
                slots["object"]
            ] += 1


        if slots.get(
            "destination"
        ):

            destination[
                slots["destination"]
            ] += 1


        tense[
            features.get(
                "tense",
                "UNKNOWN",
            )
        ] += 1


        polarity[
            features.get(
                "polarity",
                "UNKNOWN",
            )
        ] += 1


        computed = row.get(
            "computed",
            {},
        )

        if not isinstance(
            computed,
            dict,
        ):

            computed = {}


        clock[
            (
                "YES"
                if computed.get("clock")
                else "NO"
            )
        ] += 1


    return {
        "frame":
            dict(
                frame.most_common()
            ),

        "verb":
            dict(
                verb.most_common()
            ),

        "subject":
            dict(
                subject.most_common()
            ),

        "object":
            dict(
                obj.most_common()
            ),

        "destination":
            dict(
                destination.most_common()
            ),

        "tense":
            dict(
                tense.most_common()
            ),

        "polarity":
            dict(
                polarity.most_common()
            ),

        "clock":
            dict(
                clock.most_common()
            ),
    }


# ============================================================
# Console
# ============================================================

def print_counter(
    title: str,
    values: dict,
) -> None:

    print()
    print(title)
    print("-" * 80)

    if not values:

        print("None")
        return


    for key, value in values.items():

        print(
            f"{str(key):<35}"
            f"{value}"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build stratified audit sample "
            "from Synthetic V0.3 10K corpus."
        )
    )


    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )


    parser.add_argument(
        "--output",
        type=str,
        default=str(
            DEFAULT_OUTPUT
        ),
    )


    parser.add_argument(
        "--report",
        type=str,
        default=str(
            DEFAULT_REPORT
        ),
    )


    parser.add_argument(
        "--n",
        type=int,
        default=300,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2029,
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )

    output_file = Path(
        args.output
    )

    report_file = Path(
        args.report
    )


    rows = read_jsonl(
        input_file
    )


    selected = (
        select_audit_rows(
            rows=rows,
            target=args.n,
            seed=args.seed,
        )
    )


    write_jsonl(
        output_file,
        selected,
    )


    distribution = (
        build_distribution(
            selected
        )
    )


    report = {
        "version":
            "v03_10k_audit_300",

        "source":
            str(input_file),

        "source_rows":
            len(rows),

        "audit_rows":
            len(selected),

        "seed":
            args.seed,

        "selection_policy":
            (
                "stratified_rarity_coverage"
            ),

        "distribution":
            distribution,
    }


    report_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    with report_file.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("=" * 80)
    print("V0.3 10K STRATIFIED AUDIT SAMPLE")
    print("=" * 80)

    print(
        "Source rows:",
        len(rows)
    )

    print(
        "Audit rows:",
        len(selected)
    )

    print(
        "Seed:",
        args.seed
    )

    print(
        "Output:",
        output_file
    )

    print(
        "Report:",
        report_file
    )


    for name in [
        "frame",
        "verb",
        "tense",
        "polarity",
        "subject",
        "object",
        "destination",
        "clock",
    ]:

        print_counter(
            name.upper(),
            distribution[name],
        )


if __name__ == "__main__":
    main()