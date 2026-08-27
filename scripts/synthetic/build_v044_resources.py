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
    / "concepts_v043.jsonl"
)

SOURCE_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v043.json"
)

OUTPUT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v044.jsonl"
)

OUTPUT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v044.json"
)

VERSION = "0.4.4"


def read_jsonl(path: Path) -> list[dict]:

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


def build_time_every_day(
    concepts: list[dict],
) -> dict:

    # Reuse the existing TIME_TODAY schema
    # so that we don't invent a second time-concept format.
    prototype = next(
        (
            row
            for row in concepts
            if row.get("id")
            == "TIME_TODAY"
        ),
        None,
    )

    if prototype is None:

        raise RuntimeError(
            "TIME_TODAY not found; "
            "cannot build TIME_EVERY_DAY."
        )

    concept = deepcopy(
        prototype
    )

    concept[
        "id"
    ] = "TIME_EVERY_DAY"

    forms = concept.setdefault(
        "forms",
        {},
    )

    forms[
        "zh"
    ] = {
        "surface": "每天",
        "base": "每天",
    }

    forms[
        "en"
    ] = {
        "surface": "every day",
        "base": "every day",
    }

    forms[
        "ru"
    ] = {
        "surface": "каждый день",
        "base": "каждый день",
    }

    forms[
        "uz"
    ] = {
        "surface": "har kuni",
        "base": "har kuni",
    }

    time_features = concept.setdefault(
        "time_features",
        {},
    )

    time_features[
        "tense_hint"
    ] = "present"

    time_features[
        "event_type"
    ] = "habitual"

    meta = concept.setdefault(
        "meta",
        {},
    )

    meta[
        "enabled"
    ] = True

    meta[
        "version_added"
    ] = VERSION

    meta[
        "purpose"
    ] = (
        "habitual_present_motion"
    )

    return concept


def build_concepts(
    source: list[dict],
) -> list[dict]:

    rows = deepcopy(
        source
    )

    existing = {
        row.get("id")
        for row in rows
    }

    if "TIME_EVERY_DAY" not in existing:

        rows.append(
            build_time_every_day(
                rows
            )
        )

    return rows


def build_policy(
    source: dict,
) -> dict:

    policy = deepcopy(
        source
    )

    policy[
        "version"
    ] = VERSION

    policy[
        "description"
    ] = (
        "V0.4.4 motion-event cleanup: "
        "GO/COME present is habitual-only; "
        "TIME_NOW progressive motion is deferred."
    )

    motion_policy = policy.setdefault(
        "motion_event_policy",
        {},
    )

    motion_policy.update({
        "target_verbs": [
            "GO",
            "COME",
        ],

        "present_event_type":
            "habitual",

        "habitual_time_id":
            "TIME_EVERY_DAY",

        "time_now_policy":
            "convert_to_habitual",

        "time_today_policy":
            "convert_to_future_planned",

        "no_time_present_policy":
            "convert_to_future_planned",

        "progressive_enabled":
            False,
    })

    meta = policy.setdefault(
        "meta",
        {},
    )

    meta[
        "version"
    ] = VERSION

    meta[
        "previous_version"
    ] = source.get(
        "version",
        "unknown",
    )

    meta[
        "purpose"
    ] = (
        "motion_aspect_cleanup"
    )

    return policy


def main() -> None:

    concepts = read_jsonl(
        SOURCE_CONCEPTS
    )

    policy = read_json(
        SOURCE_POLICY
    )

    concepts = build_concepts(
        concepts
    )

    policy = build_policy(
        policy
    )

    write_jsonl(
        OUTPUT_CONCEPTS,
        concepts,
    )

    write_json(
        OUTPUT_POLICY,
        policy,
    )

    every_day = next(
        row
        for row in concepts
        if row.get("id")
        == "TIME_EVERY_DAY"
    )

    print(
        "=" * 90
    )

    print(
        "BUILD V0.4.4 RESOURCES"
    )

    print(
        "=" * 90
    )

    print(
        "Concepts:",
        len(
            concepts
        ),
    )

    print(
        "Added:",
        every_day[
            "id"
        ],
    )

    print(
        "ZH:",
        every_day[
            "forms"
        ][
            "zh"
        ],
    )

    print(
        "EN:",
        every_day[
            "forms"
        ][
            "en"
        ],
    )

    print(
        "RU:",
        every_day[
            "forms"
        ][
            "ru"
        ],
    )

    print(
        "UZ:",
        every_day[
            "forms"
        ][
            "uz"
        ],
    )

    print()

    print(
        "Progressive enabled:",
        policy[
            "motion_event_policy"
        ][
            "progressive_enabled"
        ],
    )

    print()

    print(
        OUTPUT_CONCEPTS
    )

    print(
        OUTPUT_POLICY
    )

    print()

    print(
        "V0.4.4 RESOURCE BUILD PASS"
    )


if __name__ == "__main__":
    main()