from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ============================================================
# Reuse the stable V0.1 generator
# ============================================================

from scripts.synthetic import generate_synthetic_v01 as v01

from scripts.synthetic.renderer_v2 import (
    load_verb_policies,
    render_zh_verb_v2,
    render_ru_verb_v2,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
)

COMPATIBILITY_FILE = (
    RESOURCE_DIR
    / "semantic_compatibility.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "semantic_v03_raw.jsonl"
)

STATS_FILE = (
    OUTPUT_DIR
    / "semantic_v03_stats.json"
)


GENERATOR_VERSION = "0.3"
RENDERER_VERSION = "v2"


# ============================================================
# Load resources
# ============================================================

def load_compatibility_resource() -> dict:

    if not COMPATIBILITY_FILE.exists():

        raise FileNotFoundError(
            "Semantic compatibility resource "
            "not found:\n"
            f"{COMPATIBILITY_FILE}"
        )

    with COMPATIBILITY_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            "semantic_compatibility.json "
            "must contain a JSON object."
        )

    if not isinstance(
        data.get("concept_classes"),
        dict,
    ):

        raise ValueError(
            "semantic_compatibility.json "
            "missing concept_classes."
        )

    if not isinstance(
        data.get("verb_rules"),
        dict,
    ):

        raise ValueError(
            "semantic_compatibility.json "
            "missing verb_rules."
        )

    return data


COMPATIBILITY = (
    load_compatibility_resource()
)

VERB_POLICIES = (
    load_verb_policies()
)


# ============================================================
# Concept normalization
# ============================================================

def normalize_concept_id(
    value: Any,
) -> str:

    if value is None:
        return ""

    value = (
        str(value)
        .strip()
        .upper()
    )

    value = re.sub(
        r"[\s\-]+",
        "_",
        value,
    )

    return value


def concept_matches_pattern(
    concept_id: str,
    pattern: str,
) -> bool:

    """
    Supports both:

        FOOD
        OBJECT_FOOD
        ITEM_FOOD

    while avoiding unrestricted substring matching.
    """

    concept_id = normalize_concept_id(
        concept_id
    )

    pattern = normalize_concept_id(
        pattern
    )

    if (
        not concept_id
        or
        not pattern
    ):
        return False

    if concept_id == pattern:
        return True

    tokens = [
        token
        for token
        in concept_id.split("_")
        if token
    ]

    return pattern in tokens


# ============================================================
# Semantic classes
# ============================================================

def classify_concept(
    concept: dict,
) -> set[str]:

    """
    Return all semantic compatibility classes
    for a Concept.
    """

    concept_id = concept.get(
        "id"
    )

    if not concept_id:
        return set()

    classes: set[str] = set()

    class_map = (
        COMPATIBILITY.get(
            "concept_classes",
            {},
        )
    )

    for (
        class_name,
        patterns,
    ) in class_map.items():

        if not isinstance(
            patterns,
            list,
        ):
            continue

        for pattern in patterns:

            if concept_matches_pattern(
                str(concept_id),
                str(pattern),
            ):

                classes.add(
                    class_name
                )

                break

    return classes


# ============================================================
# Verb rule lookup
# ============================================================

def find_verb_rule(
    verb: dict | None,
) -> tuple[str | None, dict | None]:

    if not verb:
        return None, None

    verb_id = verb.get(
        "id"
    )

    if not verb_id:
        return None, None

    rules = (
        COMPATIBILITY.get(
            "verb_rules",
            {},
        )
    )

    for (
        rule_verb,
        rule,
    ) in rules.items():

        if concept_matches_pattern(
            str(verb_id),
            str(rule_verb),
        ):

            return (
                rule_verb,
                rule,
            )

    return None, None


# ============================================================
# Explicit forbidden pairs
# ============================================================

def is_explicitly_forbidden(
    verb: dict | None,
    obj: dict | None,
) -> bool:

    if (
        not verb
        or
        not obj
    ):
        return False

    verb_id = verb.get(
        "id"
    )

    object_id = obj.get(
        "id"
    )

    if (
        not verb_id
        or
        not object_id
    ):
        return False

    pairs = (
        COMPATIBILITY.get(
            "explicit_forbidden_pairs",
            [],
        )
    )

    for pair in pairs:

        pair_verb = pair.get(
            "verb"
        )

        pair_object = pair.get(
            "object"
        )

        if (
            not pair_verb
            or
            not pair_object
        ):
            continue

        verb_match = (
            concept_matches_pattern(
                str(verb_id),
                str(pair_verb),
            )
        )

        object_match = (
            concept_matches_pattern(
                str(object_id),
                str(pair_object),
            )
        )

        if (
            verb_match
            and
            object_match
        ):
            return True

    return False


# ============================================================
# Build ordinary candidate pool
# ============================================================

def build_base_pool(
    concepts: list[dict],
    spec: dict,
) -> list[dict]:

    return [
        concept
        for concept in concepts
        if v01.match_filter(
            concept,
            spec,
        )
    ]


# ============================================================
# Object compatibility
# ============================================================

def build_object_pool(
    *,
    concepts: list[dict],
    spec: dict,
    verb: dict,
) -> list[dict]:

    base_pool = build_base_pool(
        concepts,
        spec,
    )

    (
        rule_name,
        rule,
    ) = find_verb_rule(
        verb
    )

    if rule is None:

        raise RuntimeError(
            "Generator V0.3 found a verb "
            "without a semantic compatibility rule: "
            f"{verb.get('id')}"
        )

    allowed_classes = set(
        rule.get(
            "allowed_object_classes",
            [],
        )
    )

    if not allowed_classes:

        raise RuntimeError(
            "Verb requires object sampling "
            "but has no allowed_object_classes: "
            f"{verb.get('id')} "
            f"(rule={rule_name})"
        )

    compatible_pool = []

    for concept in base_pool:

        # ----------------------------------------------------
        # Explicit forbidden combination
        # ----------------------------------------------------

        if is_explicitly_forbidden(
            verb,
            concept,
        ):

            continue

        # ----------------------------------------------------
        # Semantic class
        # ----------------------------------------------------

        concept_classes = (
            classify_concept(
                concept
            )
        )

        if not concept_classes:

            # Generator V0.3 is intentionally stricter
            # than the audit validator.
            #
            # Validator V1.1 may WARN about an unknown
            # concept, but a production generator should
            # not randomly emit concepts whose compatibility
            # is unknown.
            continue

        if concept_classes.intersection(
            allowed_classes
        ):

            compatible_pool.append(
                concept
            )

    if not compatible_pool:

        raise RuntimeError(
            "No compatible object candidates.\n"
            f"verb={verb.get('id')}\n"
            f"allowed_classes="
            f"{sorted(allowed_classes)}\n"
            f"base_pool="
            f"{[c.get('id') for c in base_pool]}"
        )

    return compatible_pool


# ============================================================
# Destination compatibility
# ============================================================

def build_destination_pool(
    *,
    concepts: list[dict],
    spec: dict,
    verb: dict,
) -> list[dict]:

    base_pool = build_base_pool(
        concepts,
        spec,
    )

    (
        rule_name,
        rule,
    ) = find_verb_rule(
        verb
    )

    if rule is None:

        raise RuntimeError(
            "Generator V0.3 found a verb "
            "without a semantic compatibility rule: "
            f"{verb.get('id')}"
        )

    allowed_classes = set(
        rule.get(
            "allowed_destination_classes",
            [],
        )
    )

    if not allowed_classes:

        raise RuntimeError(
            "Verb requires destination sampling "
            "but has no allowed_destination_classes: "
            f"{verb.get('id')} "
            f"(rule={rule_name})"
        )

    compatible_pool = []

    for concept in base_pool:

        concept_classes = (
            classify_concept(
                concept
            )
        )

        if not concept_classes:
            continue

        if concept_classes.intersection(
            allowed_classes
        ):

            compatible_pool.append(
                concept
            )

    if not compatible_pool:

        raise RuntimeError(
            "No compatible destination candidates.\n"
            f"verb={verb.get('id')}\n"
            f"allowed_classes="
            f"{sorted(allowed_classes)}\n"
            f"base_pool="
            f"{[c.get('id') for c in base_pool]}"
        )

    return compatible_pool


# ============================================================
# Validate already-selected fixed slots
# ============================================================

def validate_fixed_semantic_slots(
    selected: dict,
) -> None:

    """
    Fixed frame slots also have to obey compatibility.

    Fail fast if resources define an impossible
    fixed combination.
    """

    verb = selected.get(
        "verb"
    )

    if not verb:
        # WHERE_PLACE and similar verbless frames.
        return

    (
        _,
        rule,
    ) = find_verb_rule(
        verb
    )

    if rule is None:
        return

    obj = selected.get(
        "object"
    )

    if obj:

        if is_explicitly_forbidden(
            verb,
            obj,
        ):

            raise RuntimeError(
                "Fixed frame slots contain "
                "an explicitly forbidden pair: "
                f"{verb.get('id')} + "
                f"{obj.get('id')}"
            )

        allowed_classes = set(
            rule.get(
                "allowed_object_classes",
                [],
            )
        )

        if allowed_classes:

            obj_classes = (
                classify_concept(
                    obj
                )
            )

            if not obj_classes.intersection(
                allowed_classes
            ):

                raise RuntimeError(
                    "Fixed object violates "
                    "semantic compatibility: "
                    f"verb={verb.get('id')} "
                    f"object={obj.get('id')} "
                    f"classes={sorted(obj_classes)} "
                    f"allowed={sorted(allowed_classes)}"
                )

    destination = selected.get(
        "destination"
    )

    if destination:

        allowed_classes = set(
            rule.get(
                "allowed_destination_classes",
                [],
            )
        )

        if allowed_classes:

            destination_classes = (
                classify_concept(
                    destination
                )
            )

            if not (
                destination_classes
                .intersection(
                    allowed_classes
                )
            ):

                raise RuntimeError(
                    "Fixed destination violates "
                    "semantic compatibility: "
                    f"verb={verb.get('id')} "
                    f"destination="
                    f"{destination.get('id')} "
                    f"classes="
                    f"{sorted(destination_classes)} "
                    f"allowed="
                    f"{sorted(allowed_classes)}"
                )


# ============================================================
# V0.3 Compatibility-aware slot selection
# ============================================================

def select_slots_v03(
    frame,
    concepts,
    concepts_by_id,
    rng,
):

    selected = {}

    # ========================================================
    # 1. Fixed slots
    # ========================================================

    for (
        slot_name,
        concept_id,
    ) in frame.get(
        "fixed_slots",
        {},
    ).items():

        if concept_id not in concepts_by_id:

            raise RuntimeError(
                "Unknown fixed concept ID: "
                f"{concept_id}"
            )

        selected[
            slot_name
        ] = concepts_by_id[
            concept_id
        ]

    # ========================================================
    # 2. Dynamic slots
    # ========================================================

    for slot_name in frame.get(
        "slot_order",
        [],
    ):

        # If the same slot was already fixed,
        # never overwrite it.
        if slot_name in selected:
            continue

        spec = frame[
            "slots"
        ][slot_name]

        # ----------------------------------------------------
        # OBJECT
        # ----------------------------------------------------

        if (
            slot_name == "object"
            and
            "verb" in selected
        ):

            pool = build_object_pool(
                concepts=concepts,
                spec=spec,
                verb=selected["verb"],
            )

            selected[
                slot_name
            ] = rng.choice(
                pool
            )

            continue

        # ----------------------------------------------------
        # DESTINATION
        # ----------------------------------------------------

        if (
            slot_name == "destination"
            and
            "verb" in selected
        ):

            pool = (
                build_destination_pool(
                    concepts=concepts,
                    spec=spec,
                    verb=selected["verb"],
                )
            )

            selected[
                slot_name
            ] = rng.choice(
                pool
            )

            continue

        # ----------------------------------------------------
        # All other slots keep V0.1 behavior
        # ----------------------------------------------------

        selected[
            slot_name
        ] = v01.choose_concept(
            concepts,
            spec,
            rng,
        )

    # ========================================================
    # 3. Validate any fixed semantic slots
    # ========================================================

    validate_fixed_semantic_slots(
        selected
    )

    return selected


# ============================================================
# Renderer V0.3
#
# Reuses V0.1 renderer, then applies Renderer V2 policies.
# ============================================================

ORIGINAL_RENDER_VERB = (
    v01.render_verb
)


def render_verb_v03(
    concept,
    lang,
    tense,
    polarity,
    person,
):

    # First preserve all already-working V0.1 forms.
    original_surface = (
        ORIGINAL_RENDER_VERB(
            concept,
            lang,
            tense,
            polarity,
            person,
        )
    )

    verb_id = str(
        concept.get(
            "id",
            "",
        )
    )

    # --------------------------------------------------------
    # Chinese V2:
    # FIND + negative
    #
    # 不找到 -> 没找到
    # --------------------------------------------------------

    if lang == "zh":

        return render_zh_verb_v2(
            verb_id=verb_id,
            original_surface=original_surface,
            tense=tense,
            polarity=polarity,
            policies=VERB_POLICIES,
        )

    # --------------------------------------------------------
    # Russian V2:
    # FIND + future
    #
    # будет находить -> найдёт
    # --------------------------------------------------------

    if lang == "ru":

        return render_ru_verb_v2(
            verb_id=verb_id,
            original_surface=original_surface,
            person=person,
            tense=tense,
            polarity=polarity,
            policies=VERB_POLICIES,
        )

    # EN / UZ unchanged
    return original_surface


# ============================================================
# Post-process metadata
# ============================================================

def update_output_metadata() -> None:

    if not OUTPUT_FILE.exists():

        raise FileNotFoundError(
            "V0.3 generation output "
            "was not created:\n"
            f"{OUTPUT_FILE}"
        )

    rows = []

    with OUTPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(
                line
            )

            row[
                "resource_version"
            ] = GENERATOR_VERSION

            row[
                "generator_version"
            ] = GENERATOR_VERSION

            row[
                "renderer_version"
            ] = RENDERER_VERSION

            row[
                "compatibility_version"
            ] = COMPATIBILITY.get(
                "version",
                "unknown",
            )

            row[
                "generation_policy"
            ] = (
                "compatibility_aware"
            )

            rows.append(
                row
            )

    with OUTPUT_FILE.open(
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


def update_stats_metadata() -> None:

    if not STATS_FILE.exists():
        return

    try:

        with STATS_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            stats = json.load(
                f
            )

    except Exception:

        # Stats are secondary.
        # Never corrupt generation because
        # an old stats format cannot be read.
        return

    stats[
        "generator_version"
    ] = GENERATOR_VERSION

    stats[
        "renderer_version"
    ] = RENDERER_VERSION

    stats[
        "compatibility_version"
    ] = COMPATIBILITY.get(
        "version",
        "unknown",
    )

    stats[
        "generation_policy"
    ] = "compatibility_aware"

    stats[
        "output_file"
    ] = str(
        OUTPUT_FILE
    )

    with STATS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Install V0.3 overrides into V0.1 engine
# ============================================================

def install_v03_overrides() -> None:

    # Output isolation:
    # V0.1 data is NEVER overwritten.
    v01.OUTPUT_DIR = OUTPUT_DIR
    v01.OUTPUT_FILE = OUTPUT_FILE
    v01.STATS_FILE = STATS_FILE

    # Compatibility-aware generation.
    v01.select_slots = (
        select_slots_v03
    )

    # Renderer V2 fixes.
    v01.render_verb = (
        render_verb_v03
    )


# ============================================================
# Diagnostics
# ============================================================

def print_v03_config() -> None:

    print("=" * 90)
    print(
        "SYNTHETIC GENERATOR V0.3"
    )
    print("=" * 90)

    print(
        "Compatibility:",
        COMPATIBILITY_FILE,
    )

    print(
        "Compatibility version:",
        COMPATIBILITY.get(
            "version"
        ),
    )

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Policy:",
        "compatibility-aware sampling",
    )

    print(
        "Renderer:",
        "Renderer V2",
    )

    print("=" * 90)
    print()


# ============================================================
# Main
# ============================================================

def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_v03_config()

    install_v03_overrides()

    # Reuse the already-tested V0.1 main generation loop.
    v01.main()

    # Rewrite version/provenance metadata.
    update_output_metadata()

    update_stats_metadata()

    print()
    print("=" * 90)
    print(
        "GENERATOR V0.3 COMPLETE"
    )
    print("=" * 90)

    print(
        "Output:",
        OUTPUT_FILE,
    )

    print(
        "Stats:",
        STATS_FILE,
    )

    print(
        "Generator version:",
        GENERATOR_VERSION,
    )

    print(
        "Compatibility:",
        COMPATIBILITY.get(
            "version"
        ),
    )

    print(
        "Renderer:",
        RENDERER_VERSION,
    )


if __name__ == "__main__":
    main()