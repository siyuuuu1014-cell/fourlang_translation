from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "semantic_v01_valid.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "synthetic_v01_audit_100.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "synthetic_v01_audit_100_summary.json"
)


def read_jsonl(path: Path):

    rows = []

    with open(
        path,
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
    rows,
):

    with open(
        path,
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


def get_stratum(row):

    frame_id = row.get(
        "frame_id",
        "UNKNOWN"
    )

    features = row.get(
        "features",
        {}
    )

    polarity = features.get(
        "polarity",
        "none"
    )

    tense = features.get(
        "tense",
        "none"
    )

    computed = row.get(
        "computed",
        {}
    )

    has_clock = (
        "clock"
        if computed.get("clock")
        else "no_clock"
    )

    return (
        f"{frame_id}"
        f"|{polarity}"
        f"|{tense}"
        f"|{has_clock}"
    )


def stratified_sample(
    rows,
    target_n,
    seed,
):

    rng = random.Random(seed)

    groups = defaultdict(list)

    for row in rows:

        groups[
            get_stratum(row)
        ].append(row)


    for group in groups.values():

        rng.shuffle(group)


    selected = []

    selected_ids = set()


    # ========================================================
    # 第一轮：
    # 每个 stratum 至少拿 1 条
    # ========================================================

    group_names = sorted(
        groups.keys()
    )

    for name in group_names:

        if (
            len(selected)
            >= target_n
        ):
            break

        row = groups[name][0]

        selected.append(row)

        selected_ids.add(
            row["semantic_id"]
        )


    # ========================================================
    # 第二轮：
    # 从所有剩余样本随机补足
    # ========================================================

    remaining = [
        row
        for row in rows
        if row[
            "semantic_id"
        ] not in selected_ids
    ]

    rng.shuffle(
        remaining
    )


    need = (
        target_n
        - len(selected)
    )


    if need > 0:

        selected.extend(
            remaining[:need]
        )


    # 最后打乱
    rng.shuffle(
        selected
    )


    return selected


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()


    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            INPUT_FILE
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    rows = read_jsonl(
        INPUT_FILE
    )


    if args.n > len(rows):

        raise ValueError(
            f"要求抽取 {args.n} 条，"
            f"但总数据只有 {len(rows)} 条。"
        )


    selected = stratified_sample(
        rows,
        args.n,
        args.seed,
    )


    write_jsonl(
        OUTPUT_FILE,
        selected,
    )


    frame_counter = Counter(
        row.get(
            "frame_id",
            "UNKNOWN"
        )
        for row in selected
    )


    polarity_counter = Counter(
        row.get(
            "features",
            {}
        ).get(
            "polarity",
            "none"
        )
        for row in selected
    )


    tense_counter = Counter(
        row.get(
            "features",
            {}
        ).get(
            "tense",
            "none"
        )
        for row in selected
    )


    clock_counter = Counter(
        "clock"
        if row.get(
            "computed",
            {}
        ).get(
            "clock"
        )
        else "no_clock"
        for row in selected
    )


    summary = {
        "source_total":
            len(rows),

        "sample_size":
            len(selected),

        "seed":
            args.seed,

        "frame_distribution":
            dict(
                frame_counter
            ),

        "polarity_distribution":
            dict(
                polarity_counter
            ),

        "tense_distribution":
            dict(
                tense_counter
            ),

        "clock_distribution":
            dict(
                clock_counter
            ),
    }


    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("=" * 80)
    print("Synthetic Audit Sample")
    print("=" * 80)

    print(
        "Source:",
        len(rows)
    )

    print(
        "Selected:",
        len(selected)
    )

    print(
        "\nFrame distribution:"
    )

    for k, v in sorted(
        frame_counter.items()
    ):

        print(
            f"{k:<24} {v}"
        )


    print(
        "\nPolarity:"
    )

    for k, v in sorted(
        polarity_counter.items()
    ):

        print(
            f"{k:<12} {v}"
        )


    print(
        "\nTense:"
    )

    for k, v in sorted(
        tense_counter.items()
    ):

        print(
            f"{k:<12} {v}"
        )


    print(
        "\nClock:"
    )

    for k, v in sorted(
        clock_counter.items()
    ):

        print(
            f"{k:<12} {v}"
        )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()