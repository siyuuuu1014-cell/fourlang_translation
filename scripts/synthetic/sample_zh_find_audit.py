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
    / "v031_10k"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v031_zh_find_100.jsonl"
)

DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "v031_zh_find_100_summary.json"
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
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


def get_group(row: dict) -> tuple:
    features = row.get(
        "features",
        {},
    )

    slots = row.get(
        "slots",
        {},
    )

    computed = row.get(
        "computed",
        {},
    )

    return (
        features.get(
            "tense",
            "UNKNOWN",
        ),
        features.get(
            "polarity",
            "UNKNOWN",
        ),
        slots.get(
            "subject",
            "UNKNOWN",
        ),
        (
            "CLOCK"
            if computed.get("clock")
            else "NO_CLOCK"
        ),
    )


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
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2032,
    )

    args = parser.parse_args()

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

    # 只保留 FIND
    find_rows = [
        row
        for row in rows
        if (
            row.get(
                "slots",
                {},
            ).get(
                "verb"
            )
            == "FIND"
        )
    ]

    print(
        "Total rows:",
        len(rows)
    )

    print(
        "FIND rows:",
        len(find_rows)
    )

    if len(find_rows) < args.n:
        raise RuntimeError(
            f"Only {len(find_rows)} FIND rows, "
            f"cannot sample {args.n}"
        )

    rng = random.Random(
        args.seed
    )

    groups = defaultdict(
        list
    )

    for row in find_rows:
        groups[
            get_group(row)
        ].append(row)

    for group_rows in groups.values():
        rng.shuffle(
            group_rows
        )

    selected = []
    selected_ids = set()

    # 第一轮：每个组合至少抽一个
    group_items = list(
        groups.items()
    )

    rng.shuffle(
        group_items
    )

    for _, group_rows in group_items:
        if len(selected) >= args.n:
            break

        row = group_rows[0]

        sid = row.get(
            "semantic_id"
        )

        if sid in selected_ids:
            continue

        selected.append(
            row
        )

        selected_ids.add(
            sid
        )

    # 第二轮：补足 100
    remaining = [
        row
        for row in find_rows
        if row.get(
            "semantic_id"
        )
        not in selected_ids
    ]

    rng.shuffle(
        remaining
    )

    need = (
        args.n
        - len(selected)
    )

    selected.extend(
        remaining[:need]
    )

    rng.shuffle(
        selected
    )

    if len(selected) != args.n:
        raise RuntimeError(
            f"Expected {args.n}, "
            f"got {len(selected)}"
        )

    write_jsonl(
        output_path,
        selected,
    )

    distribution = Counter()

    for row in selected:
        features = row.get(
            "features",
            {},
        )

        computed = row.get(
            "computed",
            {},
        )

        key = (
            features.get(
                "tense"
            ),
            features.get(
                "polarity"
            ),
            (
                "clock"
                if computed.get(
                    "clock"
                )
                else "no_clock"
            ),
        )

        distribution[
            str(key)
        ] += 1

    summary = {
        "source_rows":
            len(rows),

        "find_rows":
            len(find_rows),

        "sampled":
            len(selected),

        "seed":
            args.seed,

        "distribution":
            dict(
                distribution
            ),
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 80)
    print("ZH FIND AUDIT SAMPLE COMPLETE")
    print("=" * 80)

    print(
        "Sampled:",
        len(selected)
    )

    print(
        "Output:",
        output_path
    )

    print(
        "Summary:",
        summary_path
    )

    print()

    for key, value in distribution.items():
        print(
            key,
            value
        )


if __name__ == "__main__":
    main()