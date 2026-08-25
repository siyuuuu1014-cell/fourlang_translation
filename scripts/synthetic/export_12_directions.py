from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATED_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
)

QWEN_FILE = (
    VALIDATED_DIR
    / "semantic_v01_qwen_accepted.jsonl"
)

RULE_FILE = (
    VALIDATED_DIR
    / "semantic_v01_valid.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "student"
    / "v01"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "train.jsonl"
)

VALID_FILE = (
    OUTPUT_DIR
    / "valid.jsonl"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test.jsonl"
)

STATS_FILE = (
    OUTPUT_DIR
    / "stats.json"
)


LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


def read_jsonl(
    path,
):

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
                    json.loads(
                        line
                    )
                )

    return rows


def stable_split(
    semantic_id,
):

    digest = hashlib.sha1(
        semantic_id.encode(
            "utf-8"
        )
    ).hexdigest()


    value = (
        int(
            digest[:8],
            16,
        )
        /
        0xFFFFFFFF
    )


    if value < 0.01:
        return "test"

    if value < 0.02:
        return "valid"

    return "train"


def expand_sample(
    row,
):

    texts = row[
        "texts"
    ]

    records = []


    for src in LANGUAGES:

        for tgt in LANGUAGES:

            if src == tgt:
                continue


            records.append({
                "semantic_id":
                    row[
                        "semantic_id"
                    ],

                "frame_id":
                    row[
                        "frame_id"
                    ],

                "src_lang":
                    src,

                "tgt_lang":
                    tgt,

                "source":
                    texts[src],

                "target":
                    texts[tgt],

                "input_text":
                    f"<2{tgt}> "
                    f"{texts[src]}",

                "source_type":
                    "grammar_synthetic",

                "resource_version":
                    row.get(
                        "resource_version",
                        "0.1",
                    ),

                "features":
                    row.get(
                        "features",
                        {},
                    ),
            })


    return records


def write_jsonl(
    path,
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


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # Qwen审核后的数据优先
    if QWEN_FILE.exists():

        input_file = QWEN_FILE

        print(
            "Using Qwen-validated corpus."
        )

    else:

        input_file = RULE_FILE

        print(
            "Qwen corpus not found."
        )

        print(
            "Using rule-validated corpus."
        )


    rows = read_jsonl(
        input_file
    )


    splits = {
        "train": [],
        "valid": [],
        "test": [],
    }


    direction_counter = Counter()


    for row in rows:

        split = stable_split(
            row[
                "semantic_id"
            ]
        )


        expanded = expand_sample(
            row
        )


        splits[
            split
        ].extend(
            expanded
        )


        for item in expanded:

            direction = (
                f"{item['src_lang']}"
                f"_{item['tgt_lang']}"
            )

            direction_counter[
                direction
            ] += 1


    write_jsonl(
        TRAIN_FILE,
        splits["train"],
    )

    write_jsonl(
        VALID_FILE,
        splits["valid"],
    )

    write_jsonl(
        TEST_FILE,
        splits["test"],
    )


    stats = {
        "semantic_samples":
            len(rows),

        "train_pairs":
            len(
                splits["train"]
            ),

        "valid_pairs":
            len(
                splits["valid"]
            ),

        "test_pairs":
            len(
                splits["test"]
            ),

        "total_pairs":
            sum(
                len(v)
                for v in splits.values()
            ),

        "direction_distribution":
            dict(
                sorted(
                    direction_counter.items()
                )
            ),
    }


    with open(
        STATS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("=" * 80)
    print("12-direction export complete")
    print("=" * 80)

    print(
        "Semantic samples:",
        len(rows),
    )

    print(
        "Train:",
        len(
            splits["train"]
        ),
    )

    print(
        "Valid:",
        len(
            splits["valid"]
        ),
    )

    print(
        "Test:",
        len(
            splits["test"]
        ),
    )


    print(
        "\nDirections:"
    )

    for key, value in sorted(
        direction_counter.items()
    ):

        print(
            f"{key:<10}"
            f"{value}"
        )


if __name__ == "__main__":
    main()