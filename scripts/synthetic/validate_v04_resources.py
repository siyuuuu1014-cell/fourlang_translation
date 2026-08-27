from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

CONCEPTS_FILE = RESOURCE_DIR / "concepts_v04.jsonl"
FRAMES_FILE = RESOURCE_DIR / "frames_v04.json"
COMPAT_FILE = RESOURCE_DIR / "semantic_compatibility_v04.json"
POLICY_FILE = RESOURCE_DIR / "generation_policy_v04.json"


LANGUAGES = {
    "zh",
    "en",
    "ru",
    "uz",
}

VALID_CONCEPT_TYPES = {
    "person",
    "verb",
    "object",
    "place",
    "time",
}

VALID_POLARITIES = {
    "pos",
    "neg",
}

VALID_TENSES = {
    "present",
    "future",
}

VALID_RU_CASE_ROLES = {
    "nom",
    "acc",
    "gen",
    "dat",
    "prep",
    "ins",
    "destination",
    "location",
    "source",
}

VALID_UZ_CASE_ROLES = {
    "nom",
    "acc",
    "dat",
    "loc",
    "abl",
    "destination",
    "location",
    "source",
}


class ResourceValidationError(Exception):
    pass


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise ResourceValidationError(
            f"Missing file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise ResourceValidationError(
            f"Missing file: {path}"
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
                raise ResourceValidationError(
                    f"Invalid JSONL in {path.name} "
                    f"at line {line_no}: {exc}"
                ) from exc

            if not isinstance(row, dict):
                raise ResourceValidationError(
                    f"{path.name} line {line_no} "
                    f"is not a JSON object."
                )

            rows.append(row)

    return rows


def require(
    condition: bool,
    message: str,
    errors: list[str],
) -> None:
    if not condition:
        errors.append(message)


# ============================================================
# General helpers
# ============================================================

def is_enabled(item: dict) -> bool:
    """
    Resource is enabled by default.

    meta.enabled = false means planned / disabled.
    """

    meta = item.get(
        "meta",
        {},
    )

    if not isinstance(
        meta,
        dict,
    ):
        return True

    return bool(
        meta.get(
            "enabled",
            True,
        )
    )


def semantic_classes_of(
    concept: dict,
) -> set[str]:

    classes = concept.get(
        "semantic_classes"
    )

    # Backward-compatible fallback
    if classes is None:
        single = concept.get(
            "semantic_class"
        )

        if single:
            classes = [
                single
            ]

    if not classes:
        return set()

    if isinstance(
        classes,
        str,
    ):
        classes = [
            classes
        ]

    return {
        str(x)
        for x in classes
    }


# ============================================================
# Concepts
# ============================================================

def validate_concepts(
    concepts: list[dict],
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict]:

    index: dict[
        str,
        dict,
    ] = {}

    type_counter = Counter()

    for i, concept in enumerate(
        concepts,
        start=1,
    ):

        cid = concept.get(
            "id"
        )

        ctype = concept.get(
            "concept_type"
        )

        require(
            isinstance(
                cid,
                str,
            )
            and bool(
                cid.strip()
            ),
            (
                f"Concept line {i}: "
                f"missing valid id."
            ),
            errors,
        )

        if not isinstance(
            cid,
            str,
        ):
            continue

        require(
            cid not in index,
            (
                f"Duplicate concept id: "
                f"{cid}"
            ),
            errors,
        )

        if cid in index:
            continue

        index[
            cid
        ] = concept

        require(
            ctype
            in VALID_CONCEPT_TYPES,
            (
                f"Concept {cid}: invalid "
                f"concept_type={ctype!r}"
            ),
            errors,
        )

        if ctype:
            type_counter[
                ctype
            ] += 1

        # ----------------------------------------------------
        # Planned / disabled concept
        # ----------------------------------------------------

        if not is_enabled(
            concept
        ):
            warnings.append(
                (
                    f"Planned/disabled concept: "
                    f"{cid}"
                )
            )

            continue

        # ----------------------------------------------------
        # Forms
        # ----------------------------------------------------

        forms = concept.get(
            "forms"
        )

        require(
            isinstance(
                forms,
                dict,
            ),
            (
                f"Concept {cid}: "
                f"forms must be an object."
            ),
            errors,
        )

        if not isinstance(
            forms,
            dict,
        ):
            continue

        for lang in LANGUAGES:

            require(
                lang in forms,
                (
                    f"Concept {cid}: "
                    f"missing forms.{lang}"
                ),
                errors,
            )

        # ====================================================
        # Person
        # ====================================================

        if ctype == "person":

            pf = concept.get(
                "person_features",
                {},
            )

            require(
                isinstance(
                    pf,
                    dict,
                ),
                (
                    f"Person {cid}: "
                    f"person_features must be object."
                ),
                errors,
            )

            if isinstance(
                pf,
                dict,
            ):

                require(
                    pf.get(
                        "person"
                    )
                    in {
                        1,
                        2,
                        3,
                    },
                    (
                        f"Person {cid}: invalid "
                        f"person_features.person"
                    ),
                    errors,
                )

                require(
                    pf.get(
                        "number"
                    )
                    in {
                        "singular",
                        "plural",
                    },
                    (
                        f"Person {cid}: invalid "
                        f"person_features.number"
                    ),
                    errors,
                )

            ru = forms.get(
                "ru",
                {},
            )

            uz = forms.get(
                "uz",
                {},
            )

            require(
                isinstance(
                    ru,
                    dict,
                )
                and "person_code" in ru,
                (
                    f"Person {cid}: "
                    f"missing ru.person_code"
                ),
                errors,
            )

            require(
                isinstance(
                    uz,
                    dict,
                )
                and "person_code" in uz,
                (
                    f"Person {cid}: "
                    f"missing uz.person_code"
                ),
                errors,
            )

        # ====================================================
        # Verb
        # ====================================================

        if ctype == "verb":

            features = concept.get(
                "features",
                {},
            )

            require(
                isinstance(
                    features,
                    dict,
                ),
                (
                    f"Verb {cid}: "
                    f"features must be object."
                ),
                errors,
            )

            if isinstance(
                features,
                dict,
            ):

                require(
                    features.get(
                        "transitivity"
                    )
                    in {
                        "transitive",
                        "intransitive",
                        "ambitransitive",
                        "modal",
                    },
                    (
                        f"Verb {cid}: invalid "
                        f"or missing "
                        f"features.transitivity"
                    ),
                    errors,
                )

            # English
            en = forms.get(
                "en",
                {},
            )

            require(
                isinstance(
                    en,
                    dict,
                )
                and "base" in en,
                (
                    f"Verb {cid}: "
                    f"missing en.base"
                ),
                errors,
            )

            # Chinese
            zh = forms.get(
                "zh",
                {},
            )

            require(
                isinstance(
                    zh,
                    dict,
                )
                and "base" in zh,
                (
                    f"Verb {cid}: "
                    f"missing zh.base"
                ),
                errors,
            )

            # Russian
            ru = forms.get(
                "ru",
                {},
            )

            require(
                isinstance(
                    ru,
                    dict,
                )
                and "infinitive" in ru,
                (
                    f"Verb {cid}: "
                    f"missing ru.infinitive"
                ),
                errors,
            )

            # Uzbek
            uz = forms.get(
                "uz",
                {},
            )

            require(
                isinstance(
                    uz,
                    dict,
                )
                and (
                    "base" in uz
                    or "lemma" in uz
                ),
                (
                    f"Verb {cid}: "
                    f"missing uz.base/uz.lemma"
                ),
                errors,
            )

        # ====================================================
        # Object
        # ====================================================

        if ctype == "object":

            classes = (
                semantic_classes_of(
                    concept
                )
            )

            require(
                bool(
                    classes
                ),
                (
                    f"Object {cid}: "
                    f"no semantic_classes."
                ),
                errors,
            )

        # ====================================================
        # Place
        # ====================================================

        if ctype == "place":

            classes = (
                semantic_classes_of(
                    concept
                )
            )

            require(
                "PLACE" in classes,
                (
                    f"Place {cid}: "
                    f"must include "
                    f"semantic class PLACE."
                ),
                errors,
            )

        # ====================================================
        # Time
        # ====================================================

        if ctype == "time":

            tf = concept.get(
                "time_features",
                {},
            )

            require(
                isinstance(
                    tf,
                    dict,
                ),
                (
                    f"Time {cid}: "
                    f"time_features must "
                    f"be object."
                ),
                errors,
            )

            if isinstance(
                tf,
                dict,
            ):

                tense_hint = tf.get(
                    "tense_hint"
                )

                if tense_hint is not None:

                    require(
                        tense_hint
                        in VALID_TENSES,
                        (
                            f"Time {cid}: "
                            f"invalid tense_hint="
                            f"{tense_hint!r}"
                        ),
                        errors,
                    )

    print(
        "Concept types:",
        dict(
            type_counter
        ),
    )

    return index


# ============================================================
# Semantic classes
# ============================================================

def collect_known_semantic_classes(
    concepts: list[dict],
    compat: dict,
) -> set[str]:

    known = set(
        compat.get(
            "semantic_classes",
            {},
        ).keys()
    )

    for concept in concepts:

        known.update(
            semantic_classes_of(
                concept
            )
        )

    return known


# ============================================================
# Compatibility
# ============================================================

def validate_compatibility(
    compat: dict,
    concept_index: dict[str, dict],
    known_classes: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:

    unknown_policy = compat.get(
        "unknown_policy"
    )

    require(
        unknown_policy
        in {
            "reject",
            "warn",
            "allow",
        },
        (
            "semantic_compatibility_v04.json: "
            "unknown_policy must be "
            "reject/warn/allow"
        ),
        errors,
    )

    verb_rules = compat.get(
        "verb_rules",
        {},
    )

    require(
        isinstance(
            verb_rules,
            dict,
        ),
        (
            "verb_rules must "
            "be an object."
        ),
        errors,
    )

    if not isinstance(
        verb_rules,
        dict,
    ):
        return

    # ========================================================
    # Verb compatibility rules
    # ========================================================

    for verb_id, rules in (
        verb_rules.items()
    ):

        # ----------------------------------------------------
        # Future/planned verb
        # ----------------------------------------------------

        if verb_id not in concept_index:

            warnings.append(
                (
                    "Planned compatibility rule "
                    "references unavailable verb: "
                    f"{verb_id}"
                )
            )

            # Do not validate internal compatibility
            # against a concept that does not exist yet.
            continue

        verb_concept = (
            concept_index[
                verb_id
            ]
        )

        # ----------------------------------------------------
        # Existing but disabled verb
        # ----------------------------------------------------

        if not is_enabled(
            verb_concept
        ):

            warnings.append(
                (
                    "Compatibility rule references "
                    "disabled verb: "
                    f"{verb_id}"
                )
            )

            continue

        # ----------------------------------------------------
        # Existing active verb must really be verb
        # ----------------------------------------------------

        require(
            verb_concept.get(
                "concept_type"
            )
            == "verb",
            (
                f"Compatibility rule "
                f"{verb_id}: "
                f"concept is not verb."
            ),
            errors,
        )

        if not isinstance(
            rules,
            dict,
        ):
            errors.append(
                (
                    f"Compatibility rule "
                    f"{verb_id} "
                    f"must be object."
                )
            )

            continue

        for slot_name, rule in (
            rules.items()
        ):

            if not isinstance(
                rule,
                dict,
            ):

                errors.append(
                    (
                        f"{verb_id}."
                        f"{slot_name}: "
                        f"rule must be object."
                    )
                )

                continue

            allowed_classes = (
                rule.get(
                    "allowed_classes",
                    [],
                )
            )

            require(
                isinstance(
                    allowed_classes,
                    list,
                ),
                (
                    f"{verb_id}."
                    f"{slot_name}: "
                    f"allowed_classes "
                    f"must be list."
                ),
                errors,
            )

            if not isinstance(
                allowed_classes,
                list,
            ):
                continue

            for cls in (
                allowed_classes
            ):

                require(
                    cls in known_classes,
                    (
                        f"{verb_id}."
                        f"{slot_name}: "
                        f"unknown semantic "
                        f"class {cls}"
                    ),
                    errors,
                )

    # ========================================================
    # Explicit forbidden pairs
    # ========================================================

    forbidden = compat.get(
        "explicit_forbidden",
        [],
    )

    require(
        isinstance(
            forbidden,
            list,
        ),
        (
            "explicit_forbidden "
            "must be a list."
        ),
        errors,
    )

    if not isinstance(
        forbidden,
        list,
    ):
        return

    for item in forbidden:

        if not isinstance(
            item,
            dict,
        ):

            errors.append(
                (
                    "Invalid "
                    "explicit_forbidden item."
                )
            )

            continue

        verb_id = item.get(
            "verb"
        )

        obj_id = item.get(
            "object"
        )

        # ----------------------------------------------------
        # Planned missing verb
        # ----------------------------------------------------

        if verb_id not in concept_index:

            warnings.append(
                (
                    "Planned forbidden rule "
                    "references unavailable verb: "
                    f"{verb_id}"
                )
            )

        else:

            concept = (
                concept_index[
                    verb_id
                ]
            )

            if (
                is_enabled(
                    concept
                )
                and concept.get(
                    "concept_type"
                )
                != "verb"
            ):

                errors.append(
                    (
                        "Forbidden rule verb "
                        f"{verb_id} "
                        "is not a verb concept."
                    )
                )

        # ----------------------------------------------------
        # Planned missing object
        # ----------------------------------------------------

        if (
            obj_id is not None
            and obj_id
            not in concept_index
        ):

            warnings.append(
                (
                    "Planned forbidden rule "
                    "references unavailable object: "
                    f"{obj_id}"
                )
            )


# ============================================================
# Frames
# ============================================================

def validate_frames(
    frames_data: dict,
    concept_index: dict[str, dict],
    known_classes: set[str],
    enabled_scenarios: set[str],
    errors: list[str],
    warnings: list[str],
) -> tuple[
    int,
    int,
]:

    frames = frames_data.get(
        "frames",
        [],
    )

    require(
        isinstance(
            frames,
            list,
        ),
        (
            "frames must "
            "be a list."
        ),
        errors,
    )

    if not isinstance(
        frames,
        list,
    ):
        return (
            0,
            0,
        )

    frame_ids = set()

    active_count = 0
    disabled_count = 0

    for frame in frames:

        if not isinstance(
            frame,
            dict,
        ):

            errors.append(
                (
                    "Frame entry "
                    "must be object."
                )
            )

            continue

        fid = frame.get(
            "id"
        )

        require(
            isinstance(
                fid,
                str,
            )
            and bool(
                fid.strip()
            ),
            (
                "Frame missing "
                "valid id."
            ),
            errors,
        )

        if not isinstance(
            fid,
            str,
        ):
            continue

        require(
            fid not in frame_ids,
            (
                f"Duplicate frame id: "
                f"{fid}"
            ),
            errors,
        )

        if fid in frame_ids:
            continue

        frame_ids.add(
            fid
        )

        # ----------------------------------------------------
        # Disabled / planned frame
        # ----------------------------------------------------

        if not is_enabled(
            frame
        ):

            disabled_count += 1

            warnings.append(
                (
                    f"Planned/disabled frame: "
                    f"{fid}"
                )
            )

            # IMPORTANT:
            # Do not validate missing fixed concepts,
            # because planned frame may depend on future
            # V0.4 concepts.
            continue

        active_count += 1

        # ----------------------------------------------------
        # Weight
        # ----------------------------------------------------

        weight = frame.get(
            "weight",
            1.0,
        )

        require(
            isinstance(
                weight,
                (int, float),
            )
            and not isinstance(
                weight,
                bool,
            )
            and weight > 0,
            (
                f"Frame {fid}: "
                f"weight must be > 0."
            ),
            errors,
        )

        # ----------------------------------------------------
        # Scenario tags
        # ----------------------------------------------------

        scenario_tags = (
            frame.get(
                "scenario_tags",
                [],
            )
        )

        require(
            isinstance(
                scenario_tags,
                list,
            ),
            (
                f"Frame {fid}: "
                f"scenario_tags "
                f"must be list."
            ),
            errors,
        )

        if isinstance(
            scenario_tags,
            list,
        ):

            for scenario in (
                scenario_tags
            ):

                require(
                    scenario
                    in enabled_scenarios,
                    (
                        f"Frame {fid}: "
                        f"unknown scenario "
                        f"{scenario}"
                    ),
                    errors,
                )

        # ----------------------------------------------------
        # Slots
        # ----------------------------------------------------

        slots = frame.get(
            "slots",
            {},
        )

        require(
            isinstance(
                slots,
                dict,
            ),
            (
                f"Frame {fid}: "
                f"slots must be object."
            ),
            errors,
        )

        if not isinstance(
            slots,
            dict,
        ):
            continue

        for slot_name, slot in (
            slots.items()
        ):

            if not isinstance(
                slot,
                dict,
            ):

                errors.append(
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: "
                        f"must be object."
                    )
                )

                continue

            # ================================================
            # Fixed concept
            # ================================================

            fixed_id = slot.get(
                "fixed_concept_id"
            )

            if fixed_id is not None:

                require(
                    fixed_id
                    in concept_index,
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: "
                        f"missing fixed concept "
                        f"{fixed_id}"
                    ),
                    errors,
                )

                if fixed_id in concept_index:

                    fixed_concept = (
                        concept_index[
                            fixed_id
                        ]
                    )

                    require(
                        is_enabled(
                            fixed_concept
                        ),
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: "
                            f"fixed concept "
                            f"{fixed_id} "
                            f"is disabled."
                        ),
                        errors,
                    )

            # ================================================
            # Concept types
            # ================================================

            concept_types = (
                slot.get(
                    "concept_types",
                    [],
                )
            )

            if concept_types:

                require(
                    isinstance(
                        concept_types,
                        list,
                    ),
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: "
                        f"concept_types "
                        f"must be list."
                    ),
                    errors,
                )

                if isinstance(
                    concept_types,
                    list,
                ):

                    for ctype in (
                        concept_types
                    ):

                        require(
                            ctype
                            in VALID_CONCEPT_TYPES,
                            (
                                f"Frame {fid} slot "
                                f"{slot_name}: "
                                f"invalid concept_type "
                                f"{ctype}"
                            ),
                            errors,
                        )

            # ================================================
            # Semantic classes
            # ================================================

            semantic_classes = (
                slot.get(
                    "semantic_classes",
                    [],
                )
            )

            if isinstance(
                semantic_classes,
                str,
            ):
                semantic_classes = [
                    semantic_classes
                ]

            require(
                isinstance(
                    semantic_classes,
                    list,
                ),
                (
                    f"Frame {fid} slot "
                    f"{slot_name}: "
                    f"semantic_classes "
                    f"must be list."
                ),
                errors,
            )

            if isinstance(
                semantic_classes,
                list,
            ):

                for cls in (
                    semantic_classes
                ):

                    require(
                        cls in known_classes,
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: "
                            f"unknown semantic "
                            f"class {cls}"
                        ),
                        errors,
                    )

            # ================================================
            # Case role
            # ================================================

            case_role = slot.get(
                "case_role",
                {},
            )

            if case_role:

                require(
                    isinstance(
                        case_role,
                        dict,
                    ),
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: "
                        f"case_role "
                        f"must be object."
                    ),
                    errors,
                )

            if isinstance(
                case_role,
                dict,
            ):

                ru_role = (
                    case_role.get(
                        "ru"
                    )
                )

                uz_role = (
                    case_role.get(
                        "uz"
                    )
                )

                if ru_role is not None:

                    require(
                        ru_role
                        in VALID_RU_CASE_ROLES,
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: "
                            f"invalid RU role "
                            f"{ru_role}"
                        ),
                        errors,
                    )

                if uz_role is not None:

                    require(
                        uz_role
                        in VALID_UZ_CASE_ROLES,
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: "
                            f"invalid UZ role "
                            f"{uz_role}"
                        ),
                        errors,
                    )

        # ----------------------------------------------------
        # Render templates
        # ----------------------------------------------------

        templates = frame.get(
            "render_template",
            {},
        )

        require(
            isinstance(
                templates,
                dict,
            ),
            (
                f"Frame {fid}: "
                f"render_template "
                f"must be object."
            ),
            errors,
        )

        if isinstance(
            templates,
            dict,
        ):

            for lang in LANGUAGES:

                require(
                    lang in templates,
                    (
                        f"Frame {fid}: "
                        f"missing "
                        f"render_template.{lang}"
                    ),
                    errors,
                )

                if lang in templates:

                    require(
                        isinstance(
                            templates[
                                lang
                            ],
                            str,
                        )
                        and bool(
                            templates[
                                lang
                            ].strip()
                        ),
                        (
                            f"Frame {fid}: "
                            f"invalid "
                            f"render_template."
                            f"{lang}"
                        ),
                        errors,
                    )

    return (
        active_count,
        disabled_count,
    )


# ============================================================
# Generation policy
# ============================================================

def validate_policy(
    policy: dict,
    errors: list[str],
) -> set[str]:

    langs = set(
        policy.get(
            "languages",
            [],
        )
    )

    require(
        langs == LANGUAGES,
        (
            "generation_policy languages "
            "must be exactly "
            "zh/en/ru/uz."
        ),
        errors,
    )

    scenarios = set(
        policy.get(
            "enabled_scenarios",
            [],
        )
    )

    require(
        bool(
            scenarios
        ),
        (
            "No enabled scenarios."
        ),
        errors,
    )

    # ========================================================
    # Scenario weights
    # ========================================================

    scenario_weights = (
        policy.get(
            "scenario_weights",
            {},
        )
    )

    require(
        isinstance(
            scenario_weights,
            dict,
        ),
        (
            "scenario_weights "
            "must be object."
        ),
        errors,
    )

    if isinstance(
        scenario_weights,
        dict,
    ):

        require(
            set(
                scenario_weights.keys()
            )
            == scenarios,
            (
                "scenario_weights keys "
                "must match "
                "enabled_scenarios."
            ),
            errors,
        )

        if scenario_weights:

            total = sum(
                float(v)
                for v
                in scenario_weights.values()
            )

            require(
                abs(
                    total - 1.0
                )
                < 1e-6,
                (
                    "scenario_weights "
                    "must sum to 1.0, "
                    f"got {total}"
                ),
                errors,
            )

    # ========================================================
    # Polarity weights
    # ========================================================

    polarity = policy.get(
        "polarity_weights",
        {},
    )

    require(
        isinstance(
            polarity,
            dict,
        ),
        (
            "polarity_weights "
            "must be object."
        ),
        errors,
    )

    if isinstance(
        polarity,
        dict,
    ):

        require(
            set(
                polarity.keys()
            )
            == VALID_POLARITIES,
            (
                "polarity_weights must "
                "contain exactly "
                "pos and neg."
            ),
            errors,
        )

        if polarity:

            total = sum(
                float(v)
                for v
                in polarity.values()
            )

            require(
                abs(
                    total - 1.0
                )
                < 1e-6,
                (
                    "polarity_weights "
                    "must sum to 1.0, "
                    f"got {total}"
                ),
                errors,
            )

    # ========================================================
    # Tense weights
    # ========================================================

    tense = policy.get(
        "tense_weights",
        {},
    )

    require(
        isinstance(
            tense,
            dict,
        ),
        (
            "tense_weights "
            "must be object."
        ),
        errors,
    )

    if isinstance(
        tense,
        dict,
    ):

        require(
            set(
                tense.keys()
            )
            == VALID_TENSES,
            (
                "tense_weights "
                "must contain exactly "
                "present and future."
            ),
            errors,
        )

        if tense:

            total = sum(
                float(v)
                for v
                in tense.values()
            )

            require(
                abs(
                    total - 1.0
                )
                < 1e-6,
                (
                    "tense_weights "
                    "must sum to 1.0, "
                    f"got {total}"
                ),
                errors,
            )

    return scenarios


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "V0.4 RESOURCE VALIDATOR"
    )

    print(
        "=" * 80
    )

    errors: list[
        str
    ] = []

    warnings: list[
        str
    ] = []

    # ========================================================
    # Load
    # ========================================================

    try:

        policy = load_json(
            POLICY_FILE
        )

        frames = load_json(
            FRAMES_FILE
        )

        compat = load_json(
            COMPAT_FILE
        )

        concepts = load_jsonl(
            CONCEPTS_FILE
        )

    except ResourceValidationError as exc:

        print()

        print(
            "FATAL:"
        )

        print(
            exc
        )

        raise SystemExit(
            1
        )

    # ========================================================
    # Validate policy
    # ========================================================

    enabled_scenarios = (
        validate_policy(
            policy,
            errors,
        )
    )

    # ========================================================
    # Validate concepts
    # ========================================================

    concept_index = (
        validate_concepts(
            concepts,
            errors,
            warnings,
        )
    )

    # ========================================================
    # Semantic classes
    # ========================================================

    known_classes = (
        collect_known_semantic_classes(
            concepts,
            compat,
        )
    )

    # ========================================================
    # Compatibility
    # ========================================================

    validate_compatibility(
        compat,
        concept_index,
        known_classes,
        errors,
        warnings,
    )

    # ========================================================
    # Frames
    # ========================================================

    (
        active_frame_count,
        disabled_frame_count,
    ) = validate_frames(
        frames,
        concept_index,
        known_classes,
        enabled_scenarios,
        errors,
        warnings,
    )

    all_frames = frames.get(
        "frames",
        [],
    )

    # ========================================================
    # Summary
    # ========================================================

    print()

    print(
        "Concepts:",
        len(
            concepts
        ),
    )

    print(
        "Frames defined:",
        len(
            all_frames
        )
        if isinstance(
            all_frames,
            list,
        )
        else 0,
    )

    print(
        "Frames active:",
        active_frame_count,
    )

    print(
        "Frames planned/disabled:",
        disabled_frame_count,
    )

    print(
        "Semantic classes:",
        len(
            known_classes
        ),
    )

    print(
        "Warnings:",
        len(
            warnings
        ),
    )

    print(
        "Errors:",
        len(
            errors
        ),
    )

    # ========================================================
    # Warnings
    # ========================================================

    if warnings:

        print()

        print(
            "WARNINGS"
        )

        print(
            "-" * 80
        )

        for warning in warnings:

            print(
                "WARN:",
                warning
            )

    # ========================================================
    # Errors
    # ========================================================

    if errors:

        print()

        print(
            "ERRORS"
        )

        print(
            "-" * 80
        )

        for error in errors:

            print(
                "ERROR:",
                error
            )

        print()

        print(
            "=" * 80
        )

        print(
            "RESOURCE VALIDATION FAILED"
        )

        print(
            "=" * 80
        )

        raise SystemExit(
            1
        )

    # ========================================================
    # Pass
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "RESOURCE VALIDATION PASS"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()