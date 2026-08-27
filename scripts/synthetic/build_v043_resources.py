from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

SOURCE_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v04.jsonl"
)

SOURCE_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v042_1k.json"
)

OUTPUT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v043.jsonl"
)

OUTPUT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v043.json"
)


VERSION = "0.4.3"


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

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must be JSON object."
                )

            rows.append(row)

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

        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"{path} root must be object."
        )

    return data


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
# Concepts V0.4.3
# ============================================================

def build_concepts_v043(
    source_rows: list[dict],
) -> list[dict]:

    rows = deepcopy(
        source_rows
    )

    arrive_found = False

    for concept in rows:

        if concept.get("id") != "ARRIVE":
            continue

        arrive_found = True

        features = concept.setdefault(
            "features",
            {},
        )

        # ----------------------------------------------------
        # V0.4.3 policy:
        #
        # ARRIVE is resultative.
        # Current synthetic tense system only has:
        # present / future
        #
        # "present negative" causes cross-language
        # aspect mismatch:
        #
        # 她没到达...
        # She does not arrive...
        #
        # Until past/perfect/result-state support exists,
        # only future is considered safe.
        # ----------------------------------------------------

        features[
            "allowed_tenses"
        ] = [
            "future"
        ]

        meta = concept.setdefault(
            "meta",
            {},
        )

        meta[
            "version_modified"
        ] = VERSION

        meta[
            "linguistic_cleanup"
        ] = [
            "future_only_until_resultative_aspect_support"
        ]

    if not arrive_found:

        raise RuntimeError(
            "ARRIVE concept not found."
        )

    return rows


# ============================================================
# Policy V0.4.3
# ============================================================

def build_policy_v043(
    source_policy: dict,
) -> dict:

    policy = deepcopy(
        source_policy
    )

    policy[
        "version"
    ] = VERSION

    policy[
        "description"
    ] = (
        "V0.4.3 linguistic cleanup policy: "
        "ARRIVE future-only, clock frames "
        "scheduled-future, verbless WHERE_PLACE "
        "without tense."
    )

    clock_policy = policy.setdefault(
        "clock_policy",
        {},
    )

    clock_policy[
        "force_future"
    ] = True

    clock_policy[
        "allowed_day_ids"
    ] = [
        "TIME_TOMORROW"
    ]

    meta = policy.setdefault(
        "meta",
        {},
    )

    meta[
        "version"
    ] = VERSION

    meta[
        "previous_policy_version"
    ] = source_policy.get(
        "version",
        "unknown",
    )

    meta[
        "purpose"
    ] = "linguistic_cleanup"

    meta[
        "changes"
    ] = [
        "ARRIVE future-only",
        "clock frames force future",
        "clock frames use TIME_TOMORROW",
        "WHERE_PLACE removes tense metadata",
    ]

    return policy


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 90
    )

    print(
        "BUILD V0.4.3 RESOURCES"
    )

    print(
        "=" * 90
    )

    source_concepts = read_jsonl(
        SOURCE_CONCEPTS
    )

    source_policy = read_json(
        SOURCE_POLICY
    )

    concepts_v043 = (
        build_concepts_v043(
            source_concepts
        )
    )

    policy_v043 = (
        build_policy_v043(
            source_policy
        )
    )

    write_jsonl(
        OUTPUT_CONCEPTS,
        concepts_v043,
    )

    write_json(
        OUTPUT_POLICY,
        policy_v043,
    )

    arrive = next(
        row
        for row in concepts_v043
        if row.get("id") == "ARRIVE"
    )

    print(
        "Concepts:",
        len(
            concepts_v043
        ),
    )

    print(
        "ARRIVE allowed_tenses:",
        arrive[
            "features"
        ][
            "allowed_tenses"
        ],
    )

    print(
        "Clock force future:",
        policy_v043[
            "clock_policy"
        ][
            "force_future"
        ],
    )

    print(
        "Clock days:",
        policy_v043[
            "clock_policy"
        ][
            "allowed_day_ids"
        ],
    )

    print()

    print(
        "Files:"
    )

    print(
        OUTPUT_CONCEPTS
    )

    print(
        OUTPUT_POLICY
    )

    print()

    print(
        "=" * 90
    )

    print(
        "V0.4.3 RESOURCE BUILD PASS"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":
    main()