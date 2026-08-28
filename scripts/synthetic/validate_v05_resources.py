from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

V04_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

V05_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

BASE_CONCEPTS = (
    V04_DIR
    / "concepts_v044.jsonl"
)

CONCEPTS = (
    V05_DIR
    / "concepts_v05.jsonl"
)

FRAMES = (
    V05_DIR
    / "frames_v05.json"
)

COMPATIBILITY = (
    V05_DIR
    / "semantic_compatibility_v05.json"
)

POLICY = (
    V05_DIR
    / "generation_policy_v05.json"
)

MANIFEST = (
    V05_DIR
    / "manifest_v05.json"
)


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing: {path}"
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
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL "
                    f"{path}:{line_no}"
                ) from exc

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    errors = []
    warnings = []

    manifest = read_json(
        MANIFEST
    )

    base_rows = read_jsonl(
        BASE_CONCEPTS
    )

    delta_rows = read_jsonl(
        CONCEPTS
    )

    frames = read_json(
        FRAMES
    )

    compatibility = read_json(
        COMPATIBILITY
    )

    policy = read_json(
        POLICY
    )

    base_ids = {
        row.get("id")
        for row in base_rows
        if row.get("id")
    }

    # ========================================================
    # Concepts
    # ========================================================

    delta_ids = []

    type_counter = Counter()
    operation_counter = Counter()

    for row in delta_rows:

        concept_id = row.get(
            "id"
        )

        operation = row.get(
            "operation"
        )

        if not concept_id:

            errors.append(
                "Concept missing id."
            )

            continue

        delta_ids.append(
            concept_id
        )

        operation_counter[
            operation
        ] += 1

        if operation == "add":

            if concept_id in base_ids:

                errors.append(
                    f"ADD concept already exists "
                    f"in frozen core: {concept_id}"
                )

            concept_type = row.get(
                "type"
            )

            if not concept_type:

                errors.append(
                    f"{concept_id}: "
                    f"missing type."
                )

            else:

                type_counter[
                    concept_type
                ] += 1

            semantic_class = row.get(
                "semantic_class"
            )

            if not semantic_class:

                errors.append(
                    f"{concept_id}: "
                    f"missing semantic_class."
                )

            lang = row.get(
                "lang"
            )

            if not isinstance(
                lang,
                dict,
            ):

                errors.append(
                    f"{concept_id}: "
                    f"missing lang object."
                )

            else:

                for language in (
                    "zh",
                    "en",
                    "ru",
                    "uz",
                ):

                    lemma = (
                        lang
                        .get(
                            language,
                            {},
                        )
                        .get(
                            "lemma"
                        )
                    )

                    if not lemma:

                        errors.append(
                            f"{concept_id}: "
                            f"missing {language} lemma."
                        )

        elif operation == "override":

            if concept_id not in base_ids:

                errors.append(
                    f"OVERRIDE concept not found "
                    f"in frozen core: {concept_id}"
                )

        else:

            errors.append(
                f"{concept_id}: invalid operation "
                f"{operation!r}"
            )

        # ----------------------------------------------------
        # Resource layer must NOT accidentally become active.
        # ----------------------------------------------------

        if row.get(
            "generation_enabled"
        ) is True:

            errors.append(
                f"{concept_id}: generation_enabled=True "
                f"during resource-only phase."
            )

    # Duplicate IDs inside delta.
    duplicate_ids = [
        concept_id
        for concept_id, count in Counter(
            delta_ids
        ).items()
        if count > 1
    ]

    for concept_id in duplicate_ids:

        errors.append(
            f"Duplicate V0.5 delta concept: "
            f"{concept_id}"
        )

    merged_ids = (
        set(base_ids)
        | set(delta_ids)
    )

    # ========================================================
    # Frames
    # ========================================================

    frame_rows = frames.get(
        "frames",
        []
    )

    frame_ids = []

    for frame in frame_rows:

        frame_id = frame.get(
            "id"
        )

        if not frame_id:

            errors.append(
                "Frame missing id."
            )

            continue

        frame_ids.append(
            frame_id
        )

        if frame.get(
            "generation_enabled"
        ) is True:

            errors.append(
                f"{frame_id}: active before "
                f"Renderer V0.5 exists."
            )

        slots = frame.get(
            "slots"
        )

        if not isinstance(
            slots,
            list,
        ) or not slots:

            errors.append(
                f"{frame_id}: no slots."
            )

            continue

        slot_names = []

        for slot in slots:

            name = slot.get(
                "name"
            )

            if not name:

                errors.append(
                    f"{frame_id}: "
                    f"slot missing name."
                )

                continue

            slot_names.append(
                name
            )

            fixed = slot.get(
                "fixed"
            )

            if (
                fixed
                and fixed not in merged_ids
            ):

                errors.append(
                    f"{frame_id}: fixed concept "
                    f"does not exist: {fixed}"
                )

        if len(
            slot_names
        ) != len(
            set(slot_names)
        ):

            errors.append(
                f"{frame_id}: duplicate slot names."
            )

    for frame_id, count in Counter(
        frame_ids
    ).items():

        if count > 1:

            errors.append(
                f"Duplicate frame id: {frame_id}"
            )

    # ========================================================
    # Semantic Compatibility
    # ========================================================

    semantic_classes = (
        compatibility.get(
            "semantic_classes",
            {}
        )
    )

    for class_name, concept_ids in (
        semantic_classes.items()
    ):

        if not isinstance(
            concept_ids,
            list,
        ):

            errors.append(
                f"Semantic class {class_name} "
                f"is not a list."
            )

            continue

        for concept_id in concept_ids:

            if concept_id not in merged_ids:

                errors.append(
                    f"Semantic class {class_name} "
                    f"references unknown concept: "
                    f"{concept_id}"
                )

    verb_rules = compatibility.get(
        "verb_rules",
        {}
    )

    for verb_id, rule in (
        verb_rules.items()
    ):

        if verb_id not in merged_ids:

            errors.append(
                f"Compatibility references "
                f"unknown verb: {verb_id}"
            )

        for key in (
            "object_classes",
            "destination_classes",
            "place_classes",
        ):

            for class_name in rule.get(
                key,
                [],
            ):

                if class_name not in semantic_classes:

                    errors.append(
                        f"{verb_id}.{key} "
                        f"references unknown "
                        f"semantic class: "
                        f"{class_name}"
                    )

    # ========================================================
    # TIME leakage protection
    #
    # V0.4.4 taught us that global time concepts can
    # accidentally reach unrelated frames.
    #
    # Batch-1 must not introduce another unrestricted time.
    # ========================================================

    new_time_concepts = [
        row.get("id")
        for row in delta_rows
        if (
            row.get("operation") == "add"
            and row.get("type") == "time"
        )
    ]

    if new_time_concepts:

        errors.append(
            "Batch-1 unexpectedly introduces "
            f"new time concepts: "
            f"{new_time_concepts}"
        )

    # ========================================================
    # Policy
    # ========================================================

    scenario_targets = policy.get(
        "scenario_targets",
        {}
    )

    target_sum = sum(
        scenario_targets.values()
    )

    if abs(
        target_sum - 1.0
    ) > 1e-9:

        errors.append(
            "scenario_targets must sum to 1.0; "
            f"got {target_sum}"
        )

    if policy.get(
        "generation_enabled"
    ) is not False:

        errors.append(
            "V0.5 Batch-1 generation must "
            "remain disabled."
        )

    # ========================================================
    # Informative warnings
    # ========================================================

    planned_verbs = [
        row.get("id")
        for row in delta_rows
        if (
            row.get("operation") == "add"
            and row.get("type") == "verb"
        )
    ]

    if planned_verbs:

        warnings.append(
            "Planned verbs awaiting Renderer V0.5: "
            + ", ".join(planned_verbs)
        )

    warnings.append(
        "All V0.5 frames are intentionally "
        "generation-disabled during resource phase."
    )

    # ========================================================
    # Print
    # ========================================================

    print("=" * 90)
    print("V0.5 BATCH-1 RESOURCE VALIDATOR")
    print("=" * 90)

    print(
        "Frozen core:",
        manifest.get(
            "inherits",
            {},
        ).get(
            "synthetic_core"
        ),
    )

    print(
        "Frozen concepts:",
        len(base_rows),
    )

    print(
        "V0.5 delta concepts:",
        len(delta_rows),
    )

    print(
        "Operations:",
        dict(operation_counter),
    )

    print(
        "New concept types:",
        dict(type_counter),
    )

    print(
        "New frames:",
        len(frame_rows),
    )

    print(
        "Semantic classes:",
        len(semantic_classes),
    )

    print(
        "Verb compatibility rules:",
        len(verb_rules),
    )

    print(
        "Warnings:",
        len(warnings),
    )

    print(
        "Errors:",
        len(errors),
    )

    if warnings:

        print()
        print("WARNINGS")
        print("-" * 90)

        for warning in warnings:
            print(
                "WARN:",
                warning,
            )

    if errors:

        print()
        print("ERRORS")
        print("-" * 90)

        for error in errors:
            print(
                "ERROR:",
                error,
            )

        print()
        print("=" * 90)
        print("V0.5 RESOURCE VALIDATION FAILED")
        print("=" * 90)

        raise SystemExit(1)

    print()
    print("=" * 90)
    print("V0.5 BATCH-1 RESOURCE VALIDATION PASS")
    print("=" * 90)


if __name__ == "__main__":
    main()