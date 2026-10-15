from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


# =============================================================================
# Versions / Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.3"
FROZEN_PARENT = "0.5.2"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ACTIVATION_FILE = (
    RESOURCE_DIR
    / "activation_v053.json"
)

COMPATIBILITY_FILE = (
    RESOURCE_DIR
    / "semantic_compatibility_v05.json"
)

CONCEPT_FILE = (
    RESOURCE_DIR
    / "concepts_v05.jsonl"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v053_smoke_100"
)


# =============================================================================
# V0.5.3 capability boundary
# =============================================================================

NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}

NEW_VERBS = {
    "GIVE",
    "BRING",
}


# Capabilities that remain disabled after V0.5.3.
BLOCKED_VERBS = {
    "LEAVE",
    "LOSE",
    "NEED",
    "RETURN",
}

BLOCKED_FRAMES = {
    "LEAVE_PLACE",
    "LOSE_OBJECT",
    "NEED_OBJECT",
    "RETURN_PLACE",
}


# =============================================================================
# Participant overlap
# =============================================================================

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


def person_target_allowed(
    subject_id: str | None,
    person_id: str | None,
) -> bool:

    if not subject_id or not person_id:
        return False

    if subject_id == person_id:
        return False

    if (
        subject_id,
        person_id,
    ) in PARTICIPANT_OVERLAP_PAIRS:
        return False

    return True


# =============================================================================
# JSON IO
# =============================================================================

def read_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected JSON object: {path}"
        )

    return data


def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:

        for line_number, line in enumerate(
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
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
                ) from exc

            rows.append(row)

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


# =============================================================================
# Activation
# =============================================================================

def load_activation() -> dict:

    activation = read_json(
        ACTIVATION_FILE
    )

    if (
        activation.get(
            "activation_version"
        )
        != VERSION
    ):
        raise RuntimeError(
            "V0.5.3 activation version mismatch."
        )

    if (
        activation.get(
            "frozen_parent"
        )
        != FROZEN_PARENT
    ):
        raise RuntimeError(
            "V0.5.3 frozen parent mismatch."
        )

    if (
        activation.get("status")
        != "validated"
    ):
        raise RuntimeError(
            "V0.5.3 activation has not "
            "passed validation."
        )

    enabled_verbs = set(
        activation.get(
            "enabled_verbs",
            [],
        )
    )

    enabled_frames = set(
        activation.get(
            "enabled_frames",
            [],
        )
    )

    if enabled_verbs != NEW_VERBS:
        raise RuntimeError(
            "Enabled verb whitelist mismatch. "
            f"expected={sorted(NEW_VERBS)}, "
            f"actual={sorted(enabled_verbs)}"
        )

    if enabled_frames != NEW_FRAMES:
        raise RuntimeError(
            "Enabled frame whitelist mismatch. "
            f"expected={sorted(NEW_FRAMES)}, "
            f"actual={sorted(enabled_frames)}"
        )

    policy = activation.get(
        "generation_policy",
        {},
    )

    if not policy.get(
        "v053_generation_allowed",
        False,
    ):
        raise RuntimeError(
            "V0.5.3 generation is disabled."
        )

    if policy.get(
        "role_swap_allowed",
        True,
    ):
        raise RuntimeError(
            "V0.5.3 requires "
            "role_swap_allowed=false."
        )

    return activation


# =============================================================================
# Concept / compatibility helpers
# =============================================================================

def concept_identifier(
    row: dict,
) -> str | None:

    for key in (
        "concept_id",
        "id",
        "semantic_id",
        "name",
    ):
        value = row.get(key)

        if isinstance(value, str) and value:
            return value

    return None


def concept_classes(
    row: dict,
) -> set[str]:

    values: set[str] = set()

    keys = (
        "semantic_class",
        "semantic_classes",
        "class",
        "classes",
        "semantic_type",
    )

    for key in keys:

        value = row.get(key)

        if isinstance(value, str):
            values.add(
                value.upper()
            )

        elif isinstance(value, list):
            values.update(
                str(x).upper()
                for x in value
                if x
            )

    return values


def load_concepts() -> dict[str, dict]:

    rows = read_jsonl(
        CONCEPT_FILE
    )

    result = {}

    for row in rows:

        cid = concept_identifier(
            row
        )

        if cid:
            result[cid] = row

    return result


def find_verb_rule(
    obj,
    verb_id: str,
):

    """
    Locate the compatibility rule for a verb without depending on one
    particular historical JSON nesting layout.
    """

    if isinstance(obj, dict):

        direct = obj.get(
            verb_id
        )

        if isinstance(direct, dict):
            return direct

        if (
            str(
                obj.get("verb", "")
            ).upper()
            == verb_id
        ):
            return obj

        if (
            str(
                obj.get("verb_id", "")
            ).upper()
            == verb_id
        ):
            return obj

        for value in obj.values():

            found = find_verb_rule(
                value,
                verb_id,
            )

            if found is not None:
                return found

    elif isinstance(obj, list):

        for value in obj:

            found = find_verb_rule(
                value,
                verb_id,
            )

            if found is not None:
                return found

    return None


def _strings_from_value(
    value,
) -> set[str]:

    result = set()

    if isinstance(value, str):
        result.add(
            value.upper()
        )

    elif isinstance(value, list):

        for item in value:

            if isinstance(item, str):
                result.add(
                    item.upper()
                )

    return result


def extract_allowed_object_classes(
    rule: dict | None,
) -> set[str]:

    if not isinstance(rule, dict):
        return set()

    preferred_keys = (
        "allowed_object_classes",
        "object_classes",
        "allowed_object_semantic_classes",
        "object_semantic_classes",
    )

    for key in preferred_keys:

        if key in rule:

            result = _strings_from_value(
                rule[key]
            )

            if result:
                return result

    # Common nested structure:
    #
    # "object": {
    #     "allowed_classes": [...]
    # }
    object_rule = rule.get(
        "object"
    )

    if isinstance(
        object_rule,
        dict,
    ):

        for key in (
            "allowed_classes",
            "classes",
            "semantic_classes",
            "allowed_semantic_classes",
        ):

            result = _strings_from_value(
                object_rule.get(key)
            )

            if result:
                return result

    # Last-resort recursive inspection.
    for key, value in rule.items():

        normalized = str(
            key
        ).lower()

        if (
            "object" in normalized
            and (
                "class" in normalized
                or "semantic" in normalized
            )
        ):

            result = _strings_from_value(
                value
            )

            if result:
                return result

    return set()


def compatible_object_ids(
    *,
    available_object_ids: list[str],
    verb_id: str,
    concepts: dict[str, dict],
    compatibility: dict,
) -> list[str]:

    rule = find_verb_rule(
        compatibility,
        verb_id,
    )

    allowed_classes = (
        extract_allowed_object_classes(
            rule
        )
    )

    # Preferred path: derive from compatibility classes.
    if allowed_classes:

        compatible = []

        for object_id in available_object_ids:

            concept = concepts.get(
                object_id,
                {},
            )

            classes = concept_classes(
                concept
            )

            if (
                classes
                and classes.intersection(
                    allowed_classes
                )
            ):
                compatible.append(
                    object_id
                )

        if compatible:

            print(
                f"{verb_id} allowed object classes:",
                sorted(allowed_classes),
            )

            print(
                f"{verb_id} compatible objects:",
                compatible,
            )

            return compatible

    # -------------------------------------------------------------------------
    # Conservative fallback.
    #
    # This is intentionally only used if historical resource schemas do not
    # expose semantic classes in a machine-readable way.
    # -------------------------------------------------------------------------

    if verb_id == "GIVE":

        safe_ids = {
            "MONEY",
            "BOOK",
            "PHONE",
            "CHARGER",
            "TICKET",
            "KEY",
            "ID_CARD",
            "PASSPORT",
            "LUGGAGE",
        }

    elif verb_id == "BRING":

        safe_ids = {
            "LUGGAGE",
            "BOOK",
            "PHONE",
            "CHARGER",
            "TICKET",
            "KEY",
            "ID_CARD",
            "PASSPORT",
            "UMBRELLA",
            "MONEY",
        }

    else:
        safe_ids = set()

    result = [
        object_id
        for object_id
        in available_object_ids
        if object_id in safe_ids
    ]

    if not result:

        raise RuntimeError(
            f"Could not derive compatible "
            f"object pool for {verb_id}."
        )

    print(
        f"WARNING: {verb_id} compatibility "
        "class extraction unavailable; "
        "using conservative fallback IDs."
    )

    print(
        f"{verb_id} fallback objects:",
        result,
    )

    return result


# =============================================================================
# Frozen V0.5.2 generation
# =============================================================================

def generate_frozen_parent(
    *,
    n: int,
    seed: int,
    output_dir: Path,
) -> list[dict]:

    if output_dir.exists():
        shutil.rmtree(
            output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.generate_synthetic_v052",
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--preview",
        "0",
    ]

    print()
    print("=" * 100)
    print(
        "RUNNING FROZEN V0.5.2 GENERATOR"
    )
    print("=" * 100)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    expected = (
        output_dir
        / "semantic_v052_raw.jsonl"
    )

    if expected.exists():
        source = expected

    else:

        candidates = sorted(
            output_dir.rglob(
                "semantic_*_raw.jsonl"
            )
        )

        if not candidates:
            candidates = sorted(
                output_dir.rglob(
                    "semantic*raw.jsonl"
                )
            )

        if not candidates:
            raise RuntimeError(
                "Frozen V0.5.2 generation "
                "completed but raw JSONL "
                "was not found."
            )

        source = candidates[0]

    rows = read_jsonl(
        source
    )

    if len(rows) != n:
        raise RuntimeError(
            "Frozen V0.5.2 row count mismatch: "
            f"expected={n}, "
            f"actual={len(rows)}"
        )

    print(
        "Frozen parent rows:",
        len(rows),
    )

    print(
        "Frozen parent file:",
        source,
    )

    return rows


def normalize_parent_rows(
    rows: list[dict],
) -> list[dict]:

    result = []

    for row in rows:

        output = copy.deepcopy(
            row
        )

        old_metadata = output.get(
            "metadata",
            {},
        )

        if not isinstance(
            old_metadata,
            dict,
        ):
            old_metadata = {}

        metadata = dict(
            old_metadata
        )

        metadata[
            "parent_generator_version"
        ] = old_metadata.get(
            "generator_version",
            FROZEN_PARENT,
        )

        metadata[
            "parent_generation_source"
        ] = old_metadata.get(
            "generation_source"
        )

        metadata[
            "generator_version"
        ] = VERSION

        metadata[
            "frozen_parent_version"
        ] = FROZEN_PARENT

        metadata[
            "generation_source"
        ] = "frozen_parent_v052"

        output[
            "metadata"
        ] = metadata

        result.append(
            output
        )

    return result


# =============================================================================
# Sampling
# =============================================================================

def choose_tense(
    rng: random.Random,
) -> str:

    return rng.choice(
        [
            "present",
            "future",
        ]
    )


def choose_polarity(
    rng: random.Random,
) -> str:

    # Keep enough negative examples without making the corpus negative-heavy.
    return (
        "pos"
        if rng.random() < 0.65
        else "neg"
    )


def semantic_signature(
    *,
    frame_id: str,
    slots: dict,
    features: dict,
) -> str:

    return json.dumps(
        {
            "frame_id": frame_id,
            "slots": slots,
            "features": features,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def frame_targets(
    total: int,
) -> dict[str, int]:

    give = (
        total // 2
    )

    bring = (
        total - give
    )

    return {
        "GIVE_OBJECT_PERSON":
            give,

        "BRING_OBJECT_DESTINATION":
            bring,
    }


# =============================================================================
# Generate V0.5.3 new rows
# =============================================================================

def build_new_rows(
    *,
    total_new: int,
    seed: int,
) -> tuple[
    list[dict],
    int,
]:

    from scripts.synthetic.renderer_v053 import (
        V053Renderer,
    )

    renderer = V053Renderer()

    rng = random.Random(
        seed
    )

    activation = load_activation()

    compatibility = read_json(
        COMPATIBILITY_FILE
    )

    concepts = load_concepts()

    subject_ids = sorted(
        renderer.subjects.keys()
    )

    person_ids = sorted(
        renderer.person_targets.keys()
    )

    destination_ids = sorted(
        renderer.place_targets.keys()
    )

    available_objects = sorted(
        renderer.objects.keys()
    )

    give_objects = compatible_object_ids(
        available_object_ids=available_objects,
        verb_id="GIVE",
        concepts=concepts,
        compatibility=compatibility,
    )

    bring_objects = compatible_object_ids(
        available_object_ids=available_objects,
        verb_id="BRING",
        concepts=concepts,
        compatibility=compatibility,
    )

    if not subject_ids:
        raise RuntimeError(
            "No subjects available."
        )

    if not person_ids:
        raise RuntimeError(
            "No person targets available."
        )

    if not destination_ids:
        raise RuntimeError(
            "No destinations available."
        )

    targets = frame_targets(
        total_new
    )

    frame_counter = Counter()

    rows: list[dict] = []

    signatures: set[str] = set()

    attempts = 0

    max_attempts = max(
        5000,
        total_new * 300,
    )

    while (
        len(rows) < total_new
        and attempts < max_attempts
    ):

        attempts += 1

        unfinished = [
            frame_id
            for frame_id, target
            in targets.items()
            if frame_counter[
                frame_id
            ] < target
        ]

        if not unfinished:
            break

        # Prefer the frame furthest below its target ratio.
        unfinished.sort(
            key=lambda frame_id: (
                frame_counter[
                    frame_id
                ]
                / max(
                    targets[
                        frame_id
                    ],
                    1,
                ),
                rng.random(),
            )
        )

        frame_id = (
            unfinished[0]
        )

        subject_id = rng.choice(
            subject_ids
        )

        tense = choose_tense(
            rng
        )

        polarity = choose_polarity(
            rng
        )

        features = {
            "tense":
                tense,

            "polarity":
                polarity,
        }

        # =====================================================================
        # GIVE_OBJECT_PERSON
        # =====================================================================

        if frame_id == (
            "GIVE_OBJECT_PERSON"
        ):

            valid_people = [
                person_id
                for person_id in person_ids
                if person_target_allowed(
                    subject_id,
                    person_id,
                )
            ]

            if not valid_people:
                continue

            object_id = rng.choice(
                give_objects
            )

            person_id = rng.choice(
                valid_people
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "GIVE",

                "object":
                    object_id,

                "person":
                    person_id,
            }

            scenario = rng.choice(
                [
                    "daily",
                    "shopping",
                    "communication",
                    "travel",
                ]
            )

        # =====================================================================
        # BRING_OBJECT_DESTINATION
        # =====================================================================

        elif frame_id == (
            "BRING_OBJECT_DESTINATION"
        ):

            object_id = rng.choice(
                bring_objects
            )

            destination_id = rng.choice(
                destination_ids
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "BRING",

                "object":
                    object_id,

                "destination":
                    destination_id,
            }

            scenario = rng.choice(
                [
                    "travel",
                    "hotel",
                    "transport",
                    "daily",
                ]
            )

        else:
            raise RuntimeError(
                f"Unexpected frame: {frame_id}"
            )

        signature = semantic_signature(
            frame_id=frame_id,
            slots=slots,
            features=features,
        )

        if signature in signatures:
            continue

        try:

            texts = renderer.render(
                frame_id=frame_id,
                slots=slots,
                features=features,
            )

        except Exception:
            continue

        if set(
            texts.keys()
        ) != {
            "zh",
            "en",
            "ru",
            "uz",
        }:
            continue

        if any(
            not isinstance(text, str)
            or not text.strip()
            for text in texts.values()
        ):
            continue

        row = {
            "semantic_id":
                None,

            "scenario":
                scenario,

            "frame_id":
                frame_id,

            "slots":
                slots,

            "features":
                features,

            "texts":
                texts,

            "metadata": {
                "synthetic_core_version":
                    "0.4.4.1",

                "frozen_parent_version":
                    FROZEN_PARENT,

                "generator_version":
                    VERSION,

                "renderer_version":
                    VERSION,

                "resource_version":
                    VERSION,

                "generation_source":
                    "v053_new_frame",

                "batch":
                    "Batch-3A",
            },
        }

        rows.append(
            row
        )

        signatures.add(
            signature
        )

        frame_counter[
            frame_id
        ] += 1

    if len(rows) != total_new:
        raise RuntimeError(
            "Could not generate requested "
            "V0.5.3 new rows. "
            f"requested={total_new}, "
            f"generated={len(rows)}, "
            f"attempts={attempts}"
        )

    return (
        rows,
        attempts,
    )


# =============================================================================
# Generation contract
# =============================================================================

def validate_final_contract(
    rows: list[dict],
    *,
    expected_total: int,
    expected_new: int,
) -> list[str]:

    errors = []

    if len(rows) != expected_total:
        errors.append(
            "TOTAL_COUNT_MISMATCH: "
            f"{len(rows)} != "
            f"{expected_total}"
        )

    ids = [
        row.get(
            "semantic_id"
        )
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "DUPLICATE_SEMANTIC_ID"
        )

    new_rows = []

    signatures = set()

    for row in rows:

        sid = row.get(
            "semantic_id",
            "UNKNOWN",
        )

        frame_id = row.get(
            "frame_id"
        )

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        metadata = row.get(
            "metadata",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):
            errors.append(
                f"{sid}: INVALID_SLOTS"
            )
            continue

        verb_id = slots.get(
            "verb"
        )

        if verb_id in BLOCKED_VERBS:
            errors.append(
                f"{sid}: BLOCKED_VERB_LEAKAGE:"
                f"{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:
            errors.append(
                f"{sid}: BLOCKED_FRAME_LEAKAGE:"
                f"{frame_id}"
            )

        # ---------------------------------------------------------------------
        # Language completeness
        # ---------------------------------------------------------------------

        texts = row.get(
            "texts",
            {},
        )

        if not isinstance(
            texts,
            dict,
        ):

            errors.append(
                f"{sid}: INVALID_TEXTS"
            )

        else:

            for language in (
                "zh",
                "en",
                "ru",
                "uz",
            ):

                value = texts.get(
                    language
                )

                if (
                    not isinstance(
                        value,
                        str,
                    )
                    or not value.strip()
                ):

                    errors.append(
                        f"{sid}: "
                        f"MISSING_TEXT:"
                        f"{language}"
                    )

        # ---------------------------------------------------------------------
        # V0.5.3 new frame rules
        # ---------------------------------------------------------------------

        if frame_id in NEW_FRAMES:

            new_rows.append(
                row
            )

            if (
                metadata.get(
                    "generation_source"
                )
                != "v053_new_frame"
            ):

                errors.append(
                    f"{sid}: "
                    "NEW_FRAME_SOURCE_MISMATCH"
                )

            signature = semantic_signature(
                frame_id=frame_id,
                slots=slots,
                features=features,
            )

            if signature in signatures:

                errors.append(
                    f"{sid}: "
                    "DUPLICATE_NEW_SIGNATURE"
                )

            signatures.add(
                signature
            )

        # ---------------------------------------------------------------------
        # GIVE
        # ---------------------------------------------------------------------

        if frame_id == (
            "GIVE_OBJECT_PERSON"
        ):

            if verb_id != "GIVE":
                errors.append(
                    f"{sid}: "
                    "GIVE_FRAME_VERB_MISMATCH"
                )

            subject_id = slots.get(
                "subject"
            )

            person_id = slots.get(
                "person"
            )

            if not slots.get(
                "object"
            ):

                errors.append(
                    f"{sid}: "
                    "GIVE_MISSING_OBJECT"
                )

            if not person_id:

                errors.append(
                    f"{sid}: "
                    "GIVE_MISSING_PERSON"
                )

            if not person_target_allowed(
                subject_id,
                person_id,
            ):

                errors.append(
                    f"{sid}: "
                    "GIVE_PARTICIPANT_OVERLAP:"
                    f"{subject_id}->{person_id}"
                )

            for forbidden in (
                "destination",
                "place",
                "time",
                "day",
            ):

                if slots.get(
                    forbidden
                ) is not None:

                    errors.append(
                        f"{sid}: "
                        "GIVE_FORBIDDEN_SLOT:"
                        f"{forbidden}"
                    )

        # ---------------------------------------------------------------------
        # BRING
        # ---------------------------------------------------------------------

        elif frame_id == (
            "BRING_OBJECT_DESTINATION"
        ):

            if verb_id != "BRING":

                errors.append(
                    f"{sid}: "
                    "BRING_FRAME_VERB_MISMATCH"
                )

            if not slots.get(
                "object"
            ):

                errors.append(
                    f"{sid}: "
                    "BRING_MISSING_OBJECT"
                )

            if not slots.get(
                "destination"
            ):

                errors.append(
                    f"{sid}: "
                    "BRING_MISSING_DESTINATION"
                )

            for forbidden in (
                "person",
                "place",
                "time",
                "day",
            ):

                if slots.get(
                    forbidden
                ) is not None:

                    errors.append(
                        f"{sid}: "
                        "BRING_FORBIDDEN_SLOT:"
                        f"{forbidden}"
                    )

    if len(new_rows) != expected_new:

        errors.append(
            "NEW_ROW_COUNT_MISMATCH: "
            f"{len(new_rows)} != "
            f"{expected_new}"
        )

    expected_targets = frame_targets(
        expected_new
    )

    actual_frames = Counter(
        row.get(
            "frame_id"
        )
        for row in new_rows
    )

    for frame_id, expected in (
        expected_targets.items()
    ):

        actual = actual_frames[
            frame_id
        ]

        if actual != expected:

            errors.append(
                f"FRAME_COUNT_MISMATCH:"
                f"{frame_id}:"
                f"{actual}!={expected}"
            )

    return errors


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Generator V0.5.3"
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2053,
    )

    parser.add_argument(
        "--new-ratio",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=15,
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    args = parse_args()

    load_activation()

    if args.n < 10:
        raise ValueError(
            "--n must be >= 10"
        )

    if not (
        0.05
        <= args.new_ratio
        <= 0.80
    ):
        raise ValueError(
            "--new-ratio must be between "
            "0.05 and 0.80"
        )

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        output_dir
        / "semantic_v053_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v053_stats.json"
    )

    frozen_dir = (
        output_dir
        / "_frozen_parent_v052"
    )

    new_n = max(
        2,
        round(
            args.n
            * args.new_ratio
        ),
    )

    if new_n >= args.n:
        new_n = (
            args.n - 1
        )

    parent_n = (
        args.n
        - new_n
    )

    print("=" * 100)
    print(
        "SYNTHETIC GENERATOR V0.5.3"
    )
    print("=" * 100)

    print(
        "Samples:",
        args.n,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Frozen V0.5.2 rows:",
        parent_n,
    )

    print(
        "V0.5.3 new rows:",
        new_n,
    )

    print(
        "New ratio:",
        f"{new_n / args.n:.2%}",
    )

    # =========================================================================
    # Frozen parent
    # =========================================================================

    frozen_raw = (
        generate_frozen_parent(
            n=parent_n,
            seed=args.seed,
            output_dir=frozen_dir,
        )
    )

    frozen_rows = (
        normalize_parent_rows(
            frozen_raw
        )
    )

    # =========================================================================
    # New rows
    # =========================================================================

    (
        new_rows,
        new_attempts,
    ) = build_new_rows(
        total_new=new_n,
        seed=args.seed + 1000,
    )

    rows = (
        frozen_rows
        + new_rows
    )

    rng = random.Random(
        args.seed + 2000
    )

    rng.shuffle(
        rows
    )

    # Re-index final consolidated corpus.
    for index, row in enumerate(
        rows,
        start=1,
    ):

        row[
            "semantic_id"
        ] = (
            f"sem_v053_"
            f"{index:08d}"
        )

    # =========================================================================
    # Contract
    # =========================================================================

    contract_errors = (
        validate_final_contract(
            rows,
            expected_total=args.n,
            expected_new=new_n,
        )
    )

    if contract_errors:

        print()
        print("=" * 100)
        print("CONTRACT ERRORS")
        print("=" * 100)

        for error in contract_errors[
            :100
        ]:
            print(
                "ERROR:",
                error,
            )

        raise RuntimeError(
            "V0.5.3 generation contract "
            f"failed: "
            f"{len(contract_errors)} errors."
        )

    # =========================================================================
    # Stats
    # =========================================================================

    new_frame_counter = Counter(
        row.get(
            "frame_id"
        )
        for row in new_rows
    )

    new_verb_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb"
        )
        for row in new_rows
    )

    tense_counter = Counter(
        row.get(
            "features",
            {},
        ).get(
            "tense"
        )
        for row in new_rows
    )

    polarity_counter = Counter(
        row.get(
            "features",
            {},
        ).get(
            "polarity"
        )
        for row in new_rows
    )

    write_jsonl(
        raw_file,
        rows,
    )

    stats = {
        "generator_version":
            VERSION,

        "frozen_parent":
            FROZEN_PARENT,

        "samples":
            args.n,

        "seed":
            args.seed,

        "frozen_parent_rows":
            parent_n,

        "new_rows":
            new_n,

        "new_ratio":
            new_n / args.n,

        "new_generation_attempts":
            new_attempts,

        "contract_errors":
            0,

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "new_verb_distribution":
            dict(
                new_verb_counter
            ),

        "new_tense_distribution":
            dict(
                tense_counter
            ),

        "new_polarity_distribution":
            dict(
                polarity_counter
            ),

        "participant_overlap_policy": {
            "exact_reflexive_allowed":
                False,

            "speaker_group_overlap_allowed":
                False,

            "blocked_pairs": [
                list(pair)
                for pair in sorted(
                    PARTICIPANT_OVERLAP_PAIRS
                )
            ],
        },

        "role_swap_allowed":
            False,
    }

    write_json(
        stats_file,
        stats,
    )

    # =========================================================================
    # Preview
    # =========================================================================

    if args.preview > 0:

        print()
        print("=" * 100)
        print("PREVIEW")
        print("=" * 100)

        for row in rows[
            :args.preview
        ]:

            print()
            print(
                row["semantic_id"],
                "|",
                row.get("scenario"),
                "|",
                row.get("frame_id"),
            )

            print(
                "slots:",
                row.get("slots"),
            )

            print(
                "features:",
                row.get("features"),
            )

            texts = row.get(
                "texts",
                {},
            )

            print(
                "ZH:",
                texts.get("zh"),
            )

            print(
                "EN:",
                texts.get("en"),
            )

            print(
                "RU:",
                texts.get("ru"),
            )

            print(
                "UZ:",
                texts.get("uz"),
            )

    print()
    print("=" * 100)
    print(
        "GENERATOR V0.5.3 COMPLETE"
    )
    print("=" * 100)

    print(
        "Generated:",
        len(rows),
    )

    print(
        "Frozen V0.5.2 rows:",
        parent_n,
    )

    print(
        "V0.5.3 new rows:",
        new_n,
    )

    print(
        "New generation attempts:",
        new_attempts,
    )

    print(
        "Contract errors:",
        0,
    )

    print()
    print(
        "New frame distribution:"
    )

    for frame_id, count in (
        new_frame_counter.most_common()
    ):
        print(
            f"{frame_id:<30}"
            f"{count}"
        )

    print()
    print(
        "New verb distribution:"
    )

    for verb_id, count in (
        new_verb_counter.most_common()
    ):
        print(
            f"{str(verb_id):<20}"
            f"{count}"
        )

    print()
    print(
        "New tense distribution:",
        dict(tense_counter),
    )

    print(
        "New polarity distribution:",
        dict(polarity_counter),
    )

    print()
    print("Files:")
    print(raw_file)
    print(stats_file)

    print()
    print("=" * 100)
    print(
        "V0.5.3 GENERATION CONTRACT PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()