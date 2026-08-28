from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
    / "03_grammar_hard"
    / "grammar_accepted.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
)

OUTPUT = (
    OUTPUT_DIR
    / "v051_targeted_40.jsonl"
)

SUMMARY = (
    OUTPUT_DIR
    / "v051_targeted_40_summary.json"
)

SEED = 2061
OLD_TARGET = 20

NEW_FRAMES = {
    "SEE_OBJECT",
    "TAKE_OBJECT",
}


def read_jsonl(path: Path) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input not found: {path}"
        )

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


def old_signature(row: dict) -> tuple:

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    return (
        row.get("frame_id"),
        slots.get("verb", "NO_VERB"),
        features.get("tense", "NO_TENSE"),
        features.get("polarity", "NO_POLARITY"),
        features.get("event_type", "NO_EVENT_TYPE"),
    )


def diversity_tokens(row: dict) -> set[str]:

    slots = row.get(
        "slots",
        {},
    )

    features = row.get(
        "features",
        {},
    )

    tokens = {
        f"frame:{row.get('frame_id')}",
        f"verb:{slots.get('verb')}",
        f"subject:{slots.get('subject')}",
        f"object:{slots.get('object')}",
        f"destination:{slots.get('destination')}",
        f"time:{slots.get('time')}",
        f"day:{slots.get('day')}",
        f"tense:{features.get('tense')}",
        f"polarity:{features.get('polarity')}",
        f"event:{features.get('event_type')}",
    }

    return {
        token
        for token in tokens
        if not token.endswith(":None")
    }


def select_diverse_old(
    rows: list[dict],
    target: int,
    seed: int,
) -> list[dict]:

    rng = random.Random(seed)

    pool = list(rows)

    rng.shuffle(pool)

    selected = []
    covered = set()

    while pool and len(selected) < target:

        best_index = None
        best_score = None

        signature_counts = Counter(
            old_signature(row)
            for row in selected
        )

        for index, row in enumerate(pool):

            tokens = diversity_tokens(row)

            new_token_count = len(
                tokens - covered
            )

            signature = old_signature(row)

            signature_penalty = (
                signature_counts[
                    signature
                ]
            )

            # More new coverage is better.
            # Repeated structural signatures are penalized.
            score = (
                new_token_count * 100
                - signature_penalty * 15
                + rng.random()
            )

            if (
                best_score is None
                or score > best_score
            ):

                best_score = score
                best_index = index

        chosen = pool.pop(
            best_index
        )

        selected.append(
            chosen
        )

        covered.update(
            diversity_tokens(chosen)
        )

    return selected


def attach_audit_metadata(
    row: dict,
    group: str,
) -> dict:

    output = dict(row)

    metadata = output.get(
        "audit_metadata",
        {},
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    metadata = dict(metadata)

    metadata.update({
        "audit_version":
            "0.5.1",

        "audit_group":
            group,

        "audit_reason":
            (
                "V0.5.1 SEE/TAKE new capability"
                if group == "V051_NEW"
                else "Frozen V0.4.4.1 regression control"
            ),
    })

    output[
        "audit_metadata"
    ] = metadata

    return output


def main() -> None:

    rows = read_jsonl(
        INPUT
    )

    new_rows = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    ]

    old_rows = [
        row
        for row in rows
        if row.get(
            "frame_id"
        ) not in NEW_FRAMES
    ]

    # V0.5.1 smoke currently contains exactly 20 new rows.
    # Keep every new sample.
    selected_new = [
        attach_audit_metadata(
            row,
            "V051_NEW",
        )
        for row in new_rows
    ]

    selected_old_raw = (
        select_diverse_old(
            old_rows,
            target=min(
                OLD_TARGET,
                len(old_rows),
            ),
            seed=SEED,
        )
    )

    selected_old = [
        attach_audit_metadata(
            row,
            "FROZEN_CONTROL",
        )
        for row in selected_old_raw
    ]

    selected = (
        selected_new
        + selected_old
    )

    random.Random(
        SEED + 1
    ).shuffle(
        selected
    )

    write_jsonl(
        OUTPUT,
        selected,
    )

    new_verb_distribution = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb"
        )
        for row in selected_new
    )

    new_tense_distribution = Counter(
        row.get(
            "features",
            {},
        ).get(
            "tense"
        )
        for row in selected_new
    )

    new_polarity_distribution = Counter(
        row.get(
            "features",
            {},
        ).get(
            "polarity"
        )
        for row in selected_new
    )

    old_frame_distribution = Counter(
        row.get(
            "frame_id"
        )
        for row in selected_old
    )

    old_verb_distribution = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb",
            "NO_VERB",
        )
        for row in selected_old
    )

    summary = {
        "version":
            "0.5.1",

        "source":
            str(INPUT),

        "seed":
            SEED,

        "source_rows":
            len(rows),

        "new_available":
            len(new_rows),

        "old_available":
            len(old_rows),

        "selected":
            len(selected),

        "selected_new":
            len(selected_new),

        "selected_old":
            len(selected_old),

        "new_verb_distribution":
            dict(
                new_verb_distribution
            ),

        "new_tense_distribution":
            dict(
                new_tense_distribution
            ),

        "new_polarity_distribution":
            dict(
                new_polarity_distribution
            ),

        "old_frame_distribution":
            dict(
                old_frame_distribution
            ),

        "old_verb_distribution":
            dict(
                old_verb_distribution
            ),
    }

    write_json(
        SUMMARY,
        summary,
    )

    print("=" * 100)
    print("V0.5.1 TARGETED LINGUISTIC AUDIT SAMPLER")
    print("=" * 100)

    print(
        "Source rows:",
        len(rows),
    )

    print(
        "New SEE/TAKE available:",
        len(new_rows),
    )

    print(
        "Frozen control available:",
        len(old_rows),
    )

    print()

    print(
        "Selected new:",
        len(selected_new),
    )

    print(
        "Selected old:",
        len(selected_old),
    )

    print(
        "Selected total:",
        len(selected),
    )

    print()

    print(
        "New verb distribution:",
        dict(
            new_verb_distribution
        ),
    )

    print(
        "New tense distribution:",
        dict(
            new_tense_distribution
        ),
    )

    print(
        "New polarity distribution:",
        dict(
            new_polarity_distribution
        ),
    )

    print()

    print(
        "Output:",
        OUTPUT,
    )

    print(
        "Summary:",
        SUMMARY,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "V0.5.1 TARGETED AUDIT SAMPLE COMPLETE"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()