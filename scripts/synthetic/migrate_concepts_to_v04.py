from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OLD_CONCEPTS = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)

DEFAULT_VERB_POLICIES = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "verb_policies_v2.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "concepts_v04.jsonl"
)


# ============================================================
# Semantic-class mappings
# ============================================================

OBJECT_CLASS_MAP = {
    "ticket": [
        "TRAVEL_DOCUMENT",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],

    "drink": [
        "DRINKABLE",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],

    "food": [
        "EDIBLE",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],

    "device": [
        "DEVICE",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],

    "document": [
        "IDENTITY_DOCUMENT",
        "PHYSICAL_OBJECT",
    ],

    "book": [
        "READABLE_TEXT",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],

    "medicine": [
        "MEDICINE",
        "PURCHASABLE",
        "PHYSICAL_OBJECT",
    ],
}


VERB_SEMANTIC_CLASSES = {
    "GO": [
        "MOTION_ACTION",
    ],

    "BUY": [
        "COMMERCE_ACTION",
    ],

    "FIND": [
        "INFORMATION_ACTION",
    ],

    "EAT": [
        "CONSUMPTION_ACTION",
    ],

    "DRINK": [
        "CONSUMPTION_ACTION",
    ],

    "READ": [
        "INFORMATION_ACTION",
    ],
}


SCENARIO_MAP = {
    "GO": [
        "daily",
        "travel",
        "transport",
        "hotel",
        "medical",
    ],

    "BUY": [
        "daily",
        "shopping",
        "restaurant",
        "travel",
    ],

    "FIND": [
        "daily",
        "travel",
        "hotel",
        "transport",
    ],

    "EAT": [
        "daily",
        "restaurant",
    ],

    "DRINK": [
        "daily",
        "restaurant",
    ],

    "READ": [
        "daily",
        "work_study",
    ],
}


PLACE_SCENARIOS = {
    "AIRPORT": [
        "travel",
        "transport",
    ],

    "HOTEL": [
        "travel",
        "hotel",
    ],

    "STATION": [
        "travel",
        "transport",
    ],

    "HOSPITAL": [
        "medical",
        "travel",
    ],

    "RESTAURANT": [
        "restaurant",
        "travel",
    ],

    "BANK": [
        "daily",
        "travel",
    ],

    "OFFICE": [
        "daily",
        "work_study",
    ],

    "SCHOOL": [
        "work_study",
        "daily",
    ],

    "TASHKENT": [
        "travel",
    ],

    "MOSCOW": [
        "travel",
    ],

    "BEIJING": [
        "travel",
    ],
}


OBJECT_SCENARIOS = {
    "TICKET": [
        "travel",
        "transport",
        "shopping",
    ],

    "WATER": [
        "daily",
        "restaurant",
        "medical",
    ],

    "FOOD": [
        "daily",
        "restaurant",
    ],

    "PHONE": [
        "daily",
        "communication",
        "shopping",
    ],

    "PASSPORT": [
        "travel",
        "hotel",
    ],

    "BOOK": [
        "daily",
        "work_study",
        "shopping",
    ],

    "COFFEE": [
        "daily",
        "restaurant",
    ],

    "MEDICINE": [
        "medical",
        "shopping",
    ],
}


# ============================================================
# IO
# ============================================================

def read_jsonl(path: Path) -> list[dict]:
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
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
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


# ============================================================
# Helpers
# ============================================================

def normalize_person(
    old: dict,
) -> dict:

    cid = old["id"]

    person_code = (
        old
        .get("meta", {})
        .get("person")
    )

    if not person_code:
        raise RuntimeError(
            f"{cid}: missing meta.person"
        )

    person = int(
        person_code[0]
    )

    number = (
        "plural"
        if person_code.endswith("pl")
        else "singular"
    )

    gender = None

    if cid == "PERSON_HE":
        gender = "male"

    elif cid == "PERSON_SHE":
        gender = "female"

    forms = deepcopy(
        old["forms"]
    )

    for lang in (
        "ru",
        "uz",
    ):
        forms[lang][
            "person_code"
        ] = person_code

    return {
        "id": cid,

        "concept_type":
            "person",

        "semantic_classes": [
            "PERSON"
        ],

        "scenario_tags": [
            "daily",
            "travel",
            "shopping",
            "restaurant",
            "hotel",
            "transport",
            "medical",
            "communication",
            "work_study",
        ],

        "person_features": {
            "person":
                person,

            "number":
                number,

            "gender":
                gender,
        },

        "forms":
            forms,

        "legacy": {
            "pos":
                old.get("pos"),

            "semantic_type":
                old.get(
                    "semantic_type"
                ),
        },

        "meta": {
            "enabled":
                True,

            "version_added":
                "0.4",

            "source_version":
                "0.3.1",
        },
    }


def normalize_time(
    old: dict,
) -> dict:

    cid = old["id"]

    old_meta = old.get(
        "meta",
        {},
    )

    tense = old_meta.get(
        "tense"
    )

    if tense not in {
        "present",
        "future",
    }:
        raise RuntimeError(
            f"{cid}: invalid tense "
            f"{tense!r}"
        )

    time_features = {
        "tense_hint":
            tense,
    }

    if cid == "TIME_NOW":

        time_features.update({
            "relative_day":
                0,

            "aspect_hint":
                "current",
        })

    elif cid == "TIME_TODAY":

        time_features.update({
            "relative_day":
                0,
        })

    elif cid == "TIME_TOMORROW":

        time_features.update({
            "relative_day":
                1,
        })

    elif cid == "TIME_NEXT_WEEK":

        time_features.update({
            "relative_week":
                1,
        })

    return {
        "id":
            cid,

        "concept_type":
            "time",

        "semantic_classes": [
            (
                "FUTURE_TIME"
                if tense == "future"
                else "PRESENT_TIME"
            )
        ],

        "scenario_tags": [
            "daily",
            "travel",
            "shopping",
            "restaurant",
            "hotel",
            "transport",
            "medical",
            "communication",
            "work_study",
        ],

        "time_features":
            time_features,

        "forms":
            deepcopy(
                old["forms"]
            ),

        "legacy": {
            "pos":
                old.get("pos"),

            "semantic_type":
                old.get(
                    "semantic_type"
                ),
        },

        "meta": {
            "enabled":
                True,

            "version_added":
                "0.4",

            "source_version":
                "0.3.1",
        },
    }


def normalize_place(
    old: dict,
) -> dict:

    cid = old["id"]

    forms = deepcopy(
        old["forms"]
    )

    # Preserve current tested destination surfaces.
    # Also expose a generic surface field.
    for lang in (
        "zh",
        "en",
        "ru",
        "uz",
    ):
        base = forms[
            lang
        ].get(
            "base"
        )

        if base:
            forms[
                lang
            ][
                "surface"
            ] = base

    return {
        "id":
            cid,

        "concept_type":
            "place",

        "semantic_classes": [
            "PLACE"
        ],

        "place_class":
            (
                "TRANSPORT_HUB"
                if cid in {
                    "AIRPORT",
                    "STATION",
                }
                else "GENERAL_PLACE"
            ),

        "scenario_tags":
            PLACE_SCENARIOS.get(
                cid,
                [
                    "daily",
                    "travel",
                ],
            ),

        "forms":
            forms,

        "legacy": {
            "pos":
                old.get("pos"),

            "semantic_type":
                old.get(
                    "semantic_type"
                ),
        },

        "meta": {
            "enabled":
                True,

            "version_added":
                "0.4",

            "source_version":
                "0.3.1",
        },
    }


def normalize_object(
    old: dict,
) -> dict:

    cid = old["id"]

    old_meta = old.get(
        "meta",
        {},
    )

    object_type = old_meta.get(
        "object_type"
    )

    classes = OBJECT_CLASS_MAP.get(
        object_type
    )

    if not classes:
        raise RuntimeError(
            f"{cid}: unknown object_type "
            f"{object_type!r}"
        )

    forms = deepcopy(
        old["forms"]
    )

    for lang in (
        "zh",
        "en",
        "ru",
        "uz",
    ):
        base = forms[
            lang
        ].get(
            "base"
        )

        if base:
            forms[
                lang
            ][
                "surface"
            ] = base

    return {
        "id":
            cid,

        "concept_type":
            "object",

        "semantic_classes":
            classes,

        "scenario_tags":
            OBJECT_SCENARIOS.get(
                cid,
                ["daily"],
            ),

        "object_features": {
            "legacy_object_type":
                object_type,
        },

        "forms":
            forms,

        "legacy": {
            "pos":
                old.get("pos"),

            "semantic_type":
                old.get(
                    "semantic_type"
                ),

            "meta":
                deepcopy(
                    old_meta
                ),
        },

        "meta": {
            "enabled":
                True,

            "version_added":
                "0.4",

            "source_version":
                "0.3.1",
        },
    }


def infer_transitivity(
    old: dict,
) -> str:

    semantic_type = old.get(
        "semantic_type"
    )

    if semantic_type == "motion_verb":
        return "intransitive"

    if semantic_type == "transitive_verb":
        return "transitive"

    raise RuntimeError(
        f"{old['id']}: unknown "
        f"verb semantic_type "
        f"{semantic_type!r}"
    )


def normalize_verb(
    old: dict,
    verb_policies: dict,
) -> dict:

    cid = old["id"]

    old_forms = deepcopy(
        old["forms"]
    )

    policy = (
        verb_policies
        .get(
            "verbs",
            {}
        )
        .get(
            cid,
            {}
        )
    )

    # ========================================================
    # Chinese
    # ========================================================

    zh_old = old_forms.get(
        "zh",
        {},
    )

    zh = deepcopy(
        zh_old
    )

    if cid == "FIND":

        # IMPORTANT:
        # Preserve V0.3.1 renderer behaviour explicitly
        # in V0.4 resource schema.

        zh.update({
            "base":
                "找到",

            "present_positive":
                "找到了",

            "present_negative":
                "没找到",

            "future_positive":
                "会找到",

            "future_negative":
                "不会找到",
        })

    else:

        base = zh.get(
            "base"
        )

        if base:

            zh.setdefault(
                "positive",
                base,
            )

            zh.setdefault(
                "negative",
                "不" + base,
            )

    # ========================================================
    # English
    # ========================================================

    en = deepcopy(
        old_forms.get(
            "en",
            {}
        )
    )

    # ========================================================
    # Russian
    # ========================================================

    old_ru = old_forms.get(
        "ru",
        {},
    )

    ru = {
        "infinitive":
            old_ru.get(
                "inf"
            ),

        "present": {
            key.replace(
                "present_",
                ""
            ):
                value

            for key, value
            in old_ru.items()

            if key.startswith(
                "present_"
            )
        },
    }

    # Merge tested FIND aspect policy.
    if policy.get("ru"):

        ru_policy = policy["ru"]

        ru[
            "aspect_policy"
        ] = deepcopy(
            ru_policy.get(
                "aspect_policy",
                {}
            )
        )

        imperfective = (
            ru_policy.get(
                "imperfective"
            )
        )

        if imperfective:

            ru[
                "imperfective"
            ] = deepcopy(
                imperfective
            )

        perfective = (
            ru_policy.get(
                "perfective"
            )
        )

        if perfective:

            ru[
                "perfective"
            ] = deepcopy(
                perfective
            )

        # FIND uses perfective future.
        if cid == "FIND":

            ru[
                "future_strategy"
            ] = "perfective"

    else:

        ru[
            "future_strategy"
        ] = "analytic"

    # ========================================================
    # Uzbek
    # ========================================================

    old_uz = old_forms.get(
        "uz",
        {},
    )

    uz = {
        "present_future": {
            key.replace(
                "finite_pos_",
                ""
            ):
                value

            for key, value
            in old_uz.items()

            if key.startswith(
                "finite_pos_"
            )
        },

        "negative_present_future": {
            key.replace(
                "finite_neg_",
                ""
            ):
                value

            for key, value
            in old_uz.items()

            if key.startswith(
                "finite_neg_"
            )
        },
    }

    # Preserve a readable base/lemma.
    # Do not invent an Uzbek infinitive if
    # the old verified resource did not have one.
    first_form = next(
        iter(
            uz[
                "present_future"
            ].values()
        ),
        None,
    )

    uz[
        "base"
    ] = first_form or cid.lower()

    # ========================================================
    # Argument schema
    # ========================================================

    transitivity = (
        infer_transitivity(
            old
        )
    )

    argument_schema = {
        "subject": {
            "required":
                True,

            "semantic_classes": [
                "PERSON"
            ],
        }
    }

    if transitivity == "transitive":

        argument_schema[
            "object"
        ] = {
            "required":
                True,

            "semantic_classes": [],
        }

    if cid == "GO":

        argument_schema[
            "destination"
        ] = {
            "required":
                False,

            "semantic_classes": [
                "PLACE"
            ],
        }

    return {
        "id":
            cid,

        "concept_type":
            "verb",

        "semantic_classes":
            VERB_SEMANTIC_CLASSES[
                cid
            ],

        "scenario_tags":
            SCENARIO_MAP[
                cid
            ],

        "features": {
            "transitivity":
                transitivity,

            "resultative":
                cid == "FIND",

            "modal":
                False,
        },

        "argument_schema":
            argument_schema,

        "forms": {
            "zh":
                zh,

            "en":
                en,

            "ru":
                ru,

            "uz":
                uz,
        },

        "legacy": {
            "pos":
                old.get("pos"),

            "semantic_type":
                old.get(
                    "semantic_type"
                ),

            "meta":
                deepcopy(
                    old.get(
                        "meta",
                        {}
                    )
                ),
        },

        "meta": {
            "enabled":
                True,

            "version_added":
                "0.4",

            "source_version":
                "0.3.1",
        },
    }


# ============================================================
# Main migration
# ============================================================

def migrate(
    old_rows: list[dict],
    verb_policies: dict,
) -> list[dict]:

    output = []

    for old in old_rows:

        pos = old.get(
            "pos"
        )

        semantic_type = old.get(
            "semantic_type"
        )

        if pos == "pronoun":

            new = normalize_person(
                old
            )

        elif pos == "time":

            new = normalize_time(
                old
            )

        elif (
            pos == "noun"
            and semantic_type
            == "destination"
        ):

            new = normalize_place(
                old
            )

        elif (
            pos == "noun"
            and semantic_type
            == "object"
        ):

            new = normalize_object(
                old
            )

        elif pos == "verb":

            new = normalize_verb(
                old,
                verb_policies,
            )

        else:

            raise RuntimeError(
                f"Cannot migrate concept "
                f"{old.get('id')}: "
                f"pos={pos!r}, "
                f"semantic_type="
                f"{semantic_type!r}"
            )

        output.append(
            new
        )

    return output


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_OLD_CONCEPTS
        ),
    )

    parser.add_argument(
        "--verb-policies",
        default=str(
            DEFAULT_VERB_POLICIES
        ),
    )

    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_OUTPUT
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    verb_policy_path = Path(
        args.verb_policies
    )

    output_path = Path(
        args.output
    )

    old_rows = read_jsonl(
        input_path
    )

    with verb_policy_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        verb_policies = json.load(
            f
        )

    migrated = migrate(
        old_rows,
        verb_policies,
    )

    ids = [
        row["id"]
        for row in migrated
    ]

    if len(ids) != len(
        set(ids)
    ):
        raise RuntimeError(
            "Duplicate concept IDs "
            "after migration."
        )

    write_jsonl(
        output_path,
        migrated,
    )

    type_counts = {}

    for row in migrated:

        ctype = row[
            "concept_type"
        ]

        type_counts[
            ctype
        ] = (
            type_counts.get(
                ctype,
                0,
            )
            + 1
        )

    print(
        "=" * 80
    )

    print(
        "V0.3.1 -> V0.4 CONCEPT MIGRATION"
    )

    print(
        "=" * 80
    )

    print(
        "Input:",
        input_path,
    )

    print(
        "Verb policies:",
        verb_policy_path,
    )

    print(
        "Output:",
        output_path,
    )

    print()

    print(
        "Input concepts:",
        len(old_rows),
    )

    print(
        "Migrated concepts:",
        len(migrated),
    )

    print(
        "Type counts:",
        type_counts,
    )

    print()

    print(
        "FIND policy merged:",
        any(
            row["id"] == "FIND"
            and row[
                "forms"
            ][
                "zh"
            ].get(
                "future_positive"
            )
            == "会找到"

            for row in migrated
        ),
    )

    print()

    print(
        "MIGRATION COMPLETE"
    )


if __name__ == "__main__":
    main()