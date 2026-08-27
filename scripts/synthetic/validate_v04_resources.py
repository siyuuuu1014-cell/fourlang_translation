from __future__ import annotations

import json
from collections import Counter, defaultdict
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


def semantic_classes_of(
    concept: dict,
) -> set[str]:

    classes = concept.get(
        "semantic_classes"
    )

    if classes is None:
        single = concept.get(
            "semantic_class"
        )

        if single:
            classes = [single]

    if not classes:
        return set()

    return {
        str(x)
        for x in classes
    }


def validate_concepts(
    concepts: list[dict],
    errors: list[str],
    warnings: list[str],
) -> dict[str, dict]:

    index: dict[str, dict] = {}
    type_counter = Counter()

    for i, concept in enumerate(
        concepts,
        start=1,
    ):
        cid = concept.get("id")
        ctype = concept.get(
            "concept_type"
        )

        require(
            isinstance(cid, str)
            and bool(cid.strip()),
            f"Concept line {i}: missing valid id.",
            errors,
        )

        if not isinstance(cid, str):
            continue

        require(
            cid not in index,
            f"Duplicate concept id: {cid}",
            errors,
        )

        index[cid] = concept

        require(
            ctype in VALID_CONCEPT_TYPES,
            (
                f"Concept {cid}: invalid "
                f"concept_type={ctype!r}"
            ),
            errors,
        )

        if ctype:
            type_counter[ctype] += 1

        forms = concept.get(
            "forms"
        )

        require(
            isinstance(forms, dict),
            f"Concept {cid}: forms must be an object.",
            errors,
        )

        if not isinstance(forms, dict):
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

        meta = concept.get(
            "meta",
            {}
        )

        if meta.get(
            "enabled",
            True,
        ) is False:
            warnings.append(
                f"Concept {cid} is disabled."
            )

        # ----------------------------------------------------
        # Person-specific checks
        # ----------------------------------------------------

        if ctype == "person":

            pf = concept.get(
                "person_features",
                {}
            )

            require(
                pf.get("person")
                in {1, 2, 3},
                (
                    f"Person {cid}: invalid "
                    f"person_features.person"
                ),
                errors,
            )

            require(
                pf.get("number")
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

            if isinstance(forms, dict):

                ru = forms.get(
                    "ru",
                    {}
                )

                uz = forms.get(
                    "uz",
                    {}
                )

                require(
                    "person_code" in ru,
                    (
                        f"Person {cid}: "
                        f"missing ru.person_code"
                    ),
                    errors,
                )

                require(
                    "person_code" in uz,
                    (
                        f"Person {cid}: "
                        f"missing uz.person_code"
                    ),
                    errors,
                )

        # ----------------------------------------------------
        # Verb-specific checks
        # ----------------------------------------------------

        if ctype == "verb":

            features = concept.get(
                "features",
                {}
            )

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
                    f"Verb {cid}: invalid or missing "
                    f"features.transitivity"
                ),
                errors,
            )

            en = forms.get(
                "en",
                {}
            )

            require(
                "base" in en,
                f"Verb {cid}: missing en.base",
                errors,
            )

            zh = forms.get(
                "zh",
                {}
            )

            require(
                "base" in zh,
                f"Verb {cid}: missing zh.base",
                errors,
            )

            ru = forms.get(
                "ru",
                {}
            )

            require(
                "infinitive" in ru,
                f"Verb {cid}: missing ru.infinitive",
                errors,
            )

            uz = forms.get(
                "uz",
                {}
            )

            require(
                (
                    "base" in uz
                    or
                    "lemma" in uz
                ),
                (
                    f"Verb {cid}: missing "
                    f"uz.base/uz.lemma"
                ),
                errors,
            )

        # ----------------------------------------------------
        # Object checks
        # ----------------------------------------------------

        if ctype == "object":

            classes = semantic_classes_of(
                concept
            )

            require(
                bool(classes),
                (
                    f"Object {cid}: "
                    f"no semantic_classes."
                ),
                errors,
            )

        # ----------------------------------------------------
        # Place checks
        # ----------------------------------------------------

        if ctype == "place":

            require(
                (
                    "PLACE"
                    in semantic_classes_of(
                        concept
                    )
                ),
                (
                    f"Place {cid}: "
                    f"must include semantic class PLACE."
                ),
                errors,
            )

        # ----------------------------------------------------
        # Time checks
        # ----------------------------------------------------

        if ctype == "time":

            tf = concept.get(
                "time_features",
                {}
            )

            tense_hint = tf.get(
                "tense_hint"
            )

            if tense_hint is not None:

                require(
                    tense_hint
                    in VALID_TENSES,
                    (
                        f"Time {cid}: invalid "
                        f"tense_hint={tense_hint!r}"
                    ),
                    errors,
                )

    print(
        "Concept types:",
        dict(type_counter),
    )

    return index


def collect_known_semantic_classes(
    concepts: list[dict],
    compat: dict,
) -> set[str]:

    known = set(
        compat.get(
            "semantic_classes",
            {}
        ).keys()
    )

    for concept in concepts:
        known.update(
            semantic_classes_of(
                concept
            )
        )

    return known


def validate_compatibility(
    compat: dict,
    concept_index: dict[str, dict],
    known_classes: set[str],
    errors: list[str],
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
            "unknown_policy must be reject/warn/allow"
        ),
        errors,
    )

    verb_rules = compat.get(
        "verb_rules",
        {}
    )

    require(
        isinstance(
            verb_rules,
            dict,
        ),
        "verb_rules must be an object.",
        errors,
    )

    for verb_id, rules in verb_rules.items():

        require(
            verb_id in concept_index,
            (
                f"Compatibility rule references "
                f"missing verb: {verb_id}"
            ),
            errors,
        )

        if verb_id in concept_index:

            require(
                concept_index[
                    verb_id
                ].get(
                    "concept_type"
                )
                == "verb",
                (
                    f"Compatibility rule {verb_id}: "
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
                    f"{verb_id} must be object."
                )
            )
            continue

        for slot_name, rule in rules.items():

            if not isinstance(
                rule,
                dict,
            ):
                errors.append(
                    (
                        f"{verb_id}.{slot_name}: "
                        f"rule must be object."
                    )
                )
                continue

            allowed_classes = rule.get(
                "allowed_classes",
                [],
            )

            require(
                isinstance(
                    allowed_classes,
                    list,
                ),
                (
                    f"{verb_id}.{slot_name}: "
                    f"allowed_classes must be list."
                ),
                errors,
            )

            for cls in allowed_classes:

                require(
                    cls in known_classes,
                    (
                        f"{verb_id}.{slot_name}: "
                        f"unknown semantic class {cls}"
                    ),
                    errors,
                )

    forbidden = compat.get(
        "explicit_forbidden",
        [],
    )

    require(
        isinstance(
            forbidden,
            list,
        ),
        "explicit_forbidden must be a list.",
        errors,
    )

    if isinstance(
        forbidden,
        list,
    ):
        for item in forbidden:

            if not isinstance(
                item,
                dict,
            ):
                errors.append(
                    "Invalid explicit_forbidden item."
                )
                continue

            verb_id = item.get(
                "verb"
            )

            obj_id = item.get(
                "object"
            )

            require(
                verb_id in concept_index,
                (
                    f"Forbidden rule missing verb: "
                    f"{verb_id}"
                ),
                errors,
            )

            require(
                obj_id in concept_index,
                (
                    f"Forbidden rule missing object: "
                    f"{obj_id}"
                ),
                errors,
            )


def validate_frames(
    frames_data: dict,
    concept_index: dict[str, dict],
    known_classes: set[str],
    enabled_scenarios: set[str],
    errors: list[str],
) -> None:

    frames = frames_data.get(
        "frames",
        []
    )

    require(
        isinstance(
            frames,
            list,
        ),
        "frames must be a list.",
        errors,
    )

    frame_ids = set()

    for frame in frames:

        if not isinstance(
            frame,
            dict,
        ):
            errors.append(
                "Frame entry must be object."
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
            and bool(fid.strip()),
            "Frame missing valid id.",
            errors,
        )

        if not isinstance(
            fid,
            str,
        ):
            continue

        require(
            fid not in frame_ids,
            f"Duplicate frame id: {fid}",
            errors,
        )

        frame_ids.add(
            fid
        )

        weight = frame.get(
            "weight",
            1.0,
        )

        require(
            isinstance(
                weight,
                (int, float),
            )
            and weight > 0,
            (
                f"Frame {fid}: "
                f"weight must be > 0."
            ),
            errors,
        )

        for scenario in frame.get(
            "scenario_tags",
            [],
        ):

            require(
                scenario
                in enabled_scenarios,
                (
                    f"Frame {fid}: unknown "
                    f"scenario {scenario}"
                ),
                errors,
            )

        slots = frame.get(
            "slots",
            {}
        )

        require(
            isinstance(
                slots,
                dict,
            ),
            (
                f"Frame {fid}: slots "
                f"must be object."
            ),
            errors,
        )

        if not isinstance(
            slots,
            dict,
        ):
            continue

        for slot_name, slot in slots.items():

            if not isinstance(
                slot,
                dict,
            ):
                errors.append(
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: must be object."
                    )
                )
                continue

            fixed_id = slot.get(
                "fixed_concept_id"
            )

            if fixed_id is not None:

                require(
                    fixed_id
                    in concept_index,
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: missing "
                        f"fixed concept {fixed_id}"
                    ),
                    errors,
                )

            for ctype in slot.get(
                "concept_types",
                [],
            ):

                require(
                    ctype
                    in VALID_CONCEPT_TYPES,
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: invalid "
                        f"concept_type {ctype}"
                    ),
                    errors,
                )

            for cls in slot.get(
                "semantic_classes",
                [],
            ):

                require(
                    cls in known_classes,
                    (
                        f"Frame {fid} slot "
                        f"{slot_name}: unknown "
                        f"semantic class {cls}"
                    ),
                    errors,
                )

            case_role = slot.get(
                "case_role",
                {},
            )

            if isinstance(
                case_role,
                dict,
            ):

                ru_role = case_role.get(
                    "ru"
                )

                uz_role = case_role.get(
                    "uz"
                )

                if ru_role is not None:

                    require(
                        ru_role
                        in VALID_RU_CASE_ROLES,
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: invalid "
                            f"RU role {ru_role}"
                        ),
                        errors,
                    )

                if uz_role is not None:

                    require(
                        uz_role
                        in VALID_UZ_CASE_ROLES,
                        (
                            f"Frame {fid} slot "
                            f"{slot_name}: invalid "
                            f"UZ role {uz_role}"
                        ),
                        errors,
                    )

        # ----------------------------------------------------
        # Render templates
        # ----------------------------------------------------

        templates = frame.get(
            "render_template",
            {}
        )

        for lang in LANGUAGES:

            require(
                lang in templates,
                (
                    f"Frame {fid}: missing "
                    f"render_template.{lang}"
                ),
                errors,
            )


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
            "must be exactly zh/en/ru/uz."
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
        bool(scenarios),
        "No enabled scenarios.",
        errors,
    )

    scenario_weights = policy.get(
        "scenario_weights",
        {}
    )

    require(
        set(
            scenario_weights.keys()
        )
        == scenarios,
        (
            "scenario_weights keys must match "
            "enabled_scenarios."
        ),
        errors,
    )

    if scenario_weights:

        total = sum(
            float(v)
            for v in scenario_weights.values()
        )

        require(
            abs(total - 1.0) < 1e-6,
            (
                "scenario_weights must sum "
                f"to 1.0, got {total}"
            ),
            errors,
        )

    polarity = policy.get(
        "polarity_weights",
        {}
    )

    require(
        set(
            polarity.keys()
        )
        == VALID_POLARITIES,
        (
            "polarity_weights must contain "
            "pos and neg."
        ),
        errors,
    )

    if polarity:

        total = sum(
            float(v)
            for v in polarity.values()
        )

        require(
            abs(total - 1.0) < 1e-6,
            (
                "polarity_weights must sum "
                f"to 1.0, got {total}"
            ),
            errors,
        )

    tense = policy.get(
        "tense_weights",
        {}
    )

    require(
        set(
            tense.keys()
        )
        == VALID_TENSES,
        (
            "tense_weights must contain "
            "present and future."
        ),
        errors,
    )

    if tense:

        total = sum(
            float(v)
            for v in tense.values()
        )

        require(
            abs(total - 1.0) < 1e-6,
            (
                "tense_weights must sum "
                f"to 1.0, got {total}"
            ),
            errors,
        )

    return scenarios


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

    errors: list[str] = []
    warnings: list[str] = []

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
        raise SystemExit(1)

    enabled_scenarios = (
        validate_policy(
            policy,
            errors,
        )
    )

    concept_index = (
        validate_concepts(
            concepts,
            errors,
            warnings,
        )
    )

    known_classes = (
        collect_known_semantic_classes(
            concepts,
            compat,
        )
    )

    validate_compatibility(
        compat,
        concept_index,
        known_classes,
        errors,
    )

    validate_frames(
        frames,
        concept_index,
        known_classes,
        enabled_scenarios,
        errors,
    )

    print()
    print(
        "Concepts:",
        len(concepts),
    )

    print(
        "Frames:",
        len(
            frames.get(
                "frames",
                [],
            )
        ),
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
            "RESOURCE VALIDATION FAILED"
        )

        raise SystemExit(1)

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