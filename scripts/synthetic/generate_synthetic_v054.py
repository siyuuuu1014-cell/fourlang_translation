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
# Version / Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.4"
FROZEN_PARENT = "0.5.3"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ACTIVATION_FILE = (
    RESOURCE_DIR
    / "activation_v054.json"
)

COMPATIBILITY_FILE = (
    RESOURCE_DIR
    / "semantic_compatibility_v05.json"
)

CONCEPT_FILE = (
    RESOURCE_DIR
    / "concepts_v05.jsonl"
)

FRAMES_FILE = (
    RESOURCE_DIR
    / "frames_v05.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v054_smoke_100"
)


# =============================================================================
# V0.5.4 capability boundary
# =============================================================================

NEW_FRAMES = {
    "NEED_OBJECT",
    "LOSE_OBJECT",
}

NEW_VERBS = {
    "NEED",
    "LOSE",
}

BLOCKED_VERBS = {
    "RETURN",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "RETURN_PLACE",
    "LEAVE_PLACE",
}


# =============================================================================
# Conservative compatibility fallback
#
# These are used ONLY if semantic class extraction from the compatibility
# resources cannot be completed automatically.
# =============================================================================

FALLBACK_OBJECT_POOLS = {
    "NEED": {
        "ADDRESS",
        "CHARGER",
        "CLOTHES",
        "ID_CARD",
        "LUGGAGE",
        "MONEY",
    },

    "LOSE": {
        "CHARGER",
        "ID_CARD",
        "LUGGAGE",
        "MONEY",
    },
}


# =============================================================================
# IO
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

    rows: list[dict] = []

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
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_number}"
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
            "Activation version mismatch."
        )

    if (
        activation.get(
            "frozen_parent"
        )
        != FROZEN_PARENT
    ):
        raise RuntimeError(
            "Frozen parent mismatch."
        )

    if (
        activation.get(
            "status"
        )
        != "validated"
    ):
        raise RuntimeError(
            "V0.5.4 activation has not "
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
            "Enabled verbs mismatch. "
            f"expected={sorted(NEW_VERBS)}, "
            f"actual={sorted(enabled_verbs)}"
        )

    if enabled_frames != NEW_FRAMES:
        raise RuntimeError(
            "Enabled frames mismatch. "
            f"expected={sorted(NEW_FRAMES)}, "
            f"actual={sorted(enabled_frames)}"
        )

    generation_policy = activation.get(
        "generation_policy",
        {},
    )

    if not generation_policy.get(
        "v054_generation_allowed",
        False,
    ):
        raise RuntimeError(
            "V0.5.4 generation is disabled."
        )

    if generation_policy.get(
        "past_generation_allowed",
        True,
    ):
        raise RuntimeError(
            "V0.5.4 must not enable past tense."
        )

    if generation_policy.get(
        "lose_present_generation_allowed",
        True,
    ):
        raise RuntimeError(
            "LOSE present generation must "
            "remain disabled."
        )

    tense_policy = activation.get(
        "tense_policy",
        {},
    )

    if set(
        tense_policy.get(
            "NEED_OBJECT",
            [],
        )
    ) != {
        "present",
        "future",
    }:
        raise RuntimeError(
            "NEED_OBJECT tense policy mismatch."
        )

    if (
        tense_policy.get(
            "LOSE_OBJECT"
        )
        != ["future"]
    ):
        raise RuntimeError(
            "LOSE_OBJECT must be future-only."
        )

    return activation


# =============================================================================
# Frame scenarios
# =============================================================================

def load_frame_scenarios() -> dict[str, list[str]]:
    data = read_json(
        FRAMES_FILE
    )

    frames = data.get(
        "frames",
        [],
    )

    result = {}

    for frame in frames:
        if not isinstance(
            frame,
            dict,
        ):
            continue

        frame_id = frame.get(
            "id"
        )

        if frame_id not in NEW_FRAMES:
            continue

        scenarios = frame.get(
            "scenario_tags",
            [],
        )

        if isinstance(
            scenarios,
            list,
        ) and scenarios:
            result[
                frame_id
            ] = list(
                scenarios
            )

    result.setdefault(
        "NEED_OBJECT",
        [
            "daily",
            "travel",
            "medical",
            "shopping",
        ],
    )

    result.setdefault(
        "LOSE_OBJECT",
        [
            "daily",
            "travel",
        ],
    )

    return result


# =============================================================================
# Semantic compatibility helpers
# =============================================================================

def concept_identifier(
    row: dict,
) -> str | None:

    for key in (
        "id",
        "concept_id",
        "semantic_id",
        "name",
    ):
        value = row.get(
            key
        )

        if (
            isinstance(
                value,
                str,
            )
            and value
        ):
            return value

    return None


def concept_classes(
    row: dict,
) -> set[str]:

    result: set[str] = set()

    for key in (
        "semantic_class",
        "semantic_classes",
        "class",
        "classes",
        "semantic_type",
    ):
        value = row.get(
            key
        )

        if isinstance(
            value,
            str,
        ):
            result.add(
                value.upper()
            )

        elif isinstance(
            value,
            list,
        ):
            result.update(
                str(x).upper()
                for x in value
                if x
            )

    return result


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
            result[
                cid
            ] = row

    return result


def get_verb_rule(
    compatibility: dict,
    verb_id: str,
) -> dict | None:

    verb_rules = compatibility.get(
        "verb_rules",
        {},
    )

    if isinstance(
        verb_rules,
        dict,
    ):
        rule = verb_rules.get(
            verb_id
        )

        if isinstance(
            rule,
            dict,
        ):
            return rule

    return None


def compatible_object_ids(
    *,
    verb_id: str,
    available_object_ids: list[str],
    concepts: dict[str, dict],
    compatibility: dict,
) -> list[str]:

    rule = get_verb_rule(
        compatibility,
        verb_id,
    )

    allowed_classes: set[str] = set()

    if isinstance(
        rule,
        dict,
    ):
        raw = rule.get(
            "object_classes",
            [],
        )

        if isinstance(
            raw,
            list,
        ):
            allowed_classes = {
                str(x).upper()
                for x in raw
                if x
            }

    compatible = []

    if allowed_classes:

        for object_id in available_object_ids:

            concept = concepts.get(
                object_id
            )

            if not isinstance(
                concept,
                dict,
            ):
                continue

            classes = concept_classes(
                concept
            )

            if classes.intersection(
                allowed_classes
            ):
                compatible.append(
                    object_id
                )

    if compatible:

        print(
            f"{verb_id} allowed object classes:",
            sorted(
                allowed_classes
            ),
        )

        print(
            f"{verb_id} compatible objects:",
            compatible,
        )

        return sorted(
            compatible
        )

    # Conservative fallback.
    fallback = (
        FALLBACK_OBJECT_POOLS[
            verb_id
        ]
    )

    compatible = [
        object_id
        for object_id
        in available_object_ids
        if object_id in fallback
    ]

    if not compatible:
        raise RuntimeError(
            f"No compatible object pool "
            f"for {verb_id}."
        )

    print(
        f"WARNING: {verb_id} semantic-class "
        "matching produced no usable objects; "
        "using conservative fallback."
    )

    print(
        f"{verb_id} fallback objects:",
        compatible,
    )

    return sorted(
        compatible
    )


# =============================================================================
# Frozen V0.5.3 generation
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
        "scripts.synthetic.generate_synthetic_v053",
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
        "RUNNING FROZEN V0.5.3 GENERATOR"
    )
    print("=" * 100)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    expected = (
        output_dir
        / "semantic_v053_raw.jsonl"
    )

    if expected.exists():
        source = expected

    else:
        candidates = sorted(
            output_dir.rglob(
                "semantic*raw.jsonl"
            )
        )

        if not candidates:
            raise RuntimeError(
                "Frozen V0.5.3 generator "
                "completed but no raw JSONL "
                "was found."
            )

        source = candidates[0]

    rows = read_jsonl(
        source
    )

    if len(rows) != n:
        raise RuntimeError(
            "Frozen V0.5.3 row count mismatch: "
            f"expected={n}, "
            f"actual={len(rows)}"
        )

    print(
        "Frozen parent rows:",
        len(rows),
    )

    print(
        "Frozen parent source:",
        source,
    )

    return rows


def normalize_parent_rows(
    rows: list[dict],
) -> list[dict]:

    normalized = []

    for row in rows:
        output = copy.deepcopy(
            row
        )

        metadata = output.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        old_metadata = dict(
            metadata
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
        ] = "frozen_parent_v053"

        output[
            "metadata"
        ] = metadata

        normalized.append(
            output
        )

    return normalized


# =============================================================================
# Feature balancing
# =============================================================================

NEED_FEATURES = [
    ("present", "pos"),
    ("present", "neg"),
    ("future", "pos"),
    ("future", "neg"),
]

LOSE_FEATURES = [
    ("future", "pos"),
    ("future", "neg"),
]


def choose_balanced_feature(
    *,
    frame_id: str,
    counters: Counter,
    rng: random.Random,
) -> tuple[str, str]:

    if frame_id == "NEED_OBJECT":
        candidates = NEED_FEATURES

    elif frame_id == "LOSE_OBJECT":
        candidates = LOSE_FEATURES

    else:
        raise RuntimeError(
            f"Unknown frame: {frame_id}"
        )

    minimum = min(
        counters[
            (
                frame_id,
                tense,
                polarity,
            )
        ]
        for tense, polarity
        in candidates
    )

    least_used = [
        (
            tense,
            polarity,
        )
        for tense, polarity
        in candidates
        if counters[
            (
                frame_id,
                tense,
                polarity,
            )
        ] == minimum
    ]

    return rng.choice(
        least_used
    )


def frame_targets(
    total_new: int,
) -> dict[str, int]:

    need_count = (
        total_new // 2
    )

    lose_count = (
        total_new - need_count
    )

    return {
        "NEED_OBJECT":
            need_count,

        "LOSE_OBJECT":
            lose_count,
    }


def semantic_signature(
    *,
    frame_id: str,
    slots: dict,
    features: dict,
) -> str:

    return json.dumps(
        {
            "frame_id":
                frame_id,

            "slots":
                slots,

            "features":
                features,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


# =============================================================================
# Build V0.5.4 rows
# =============================================================================

def build_new_rows(
    *,
    total_new: int,
    seed: int,
) -> tuple[
    list[dict],
    int,
]:

    from scripts.synthetic.renderer_v054 import (
        V054Renderer,
    )

    renderer = V054Renderer()

    rng = random.Random(
        seed
    )

    compatibility = read_json(
        COMPATIBILITY_FILE
    )

    concepts = load_concepts()

    scenarios = load_frame_scenarios()

    subject_ids = sorted(
        renderer.subjects.keys()
    )

    available_object_ids = sorted(
        renderer.v054_objects.keys()
    )

    need_objects = compatible_object_ids(
        verb_id="NEED",
        available_object_ids=(
            available_object_ids
        ),
        concepts=concepts,
        compatibility=compatibility,
    )

    lose_objects = compatible_object_ids(
        verb_id="LOSE",
        available_object_ids=(
            available_object_ids
        ),
        concepts=concepts,
        compatibility=compatibility,
    )

    targets = frame_targets(
        total_new
    )

    frame_counter = Counter()

    feature_counter = Counter()

    signatures: set[str] = set()

    rows: list[dict] = []

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

        (
            tense,
            polarity,
        ) = choose_balanced_feature(
            frame_id=frame_id,
            counters=feature_counter,
            rng=rng,
        )

        features = {
            "tense":
                tense,

            "polarity":
                polarity,
        }

        # =====================================================================
        # NEED
        # =====================================================================

        if frame_id == "NEED_OBJECT":

            object_id = rng.choice(
                need_objects
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "NEED",

                "object":
                    object_id,
            }

            scenario = rng.choice(
                scenarios[
                    "NEED_OBJECT"
                ]
            )

        # =====================================================================
        # LOSE
        # =====================================================================

        elif frame_id == "LOSE_OBJECT":

            # Defense in depth:
            # LOSE V0.5.4 must NEVER generate present or past.
            if tense != "future":
                raise RuntimeError(
                    "INTERNAL ERROR: "
                    "LOSE non-future feature "
                    "reached generation."
                )

            object_id = rng.choice(
                lose_objects
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "LOSE",

                "object":
                    object_id,
            }

            scenario = rng.choice(
                scenarios[
                    "LOSE_OBJECT"
                ]
            )

        else:
            raise RuntimeError(
                f"Unexpected frame: "
                f"{frame_id}"
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
            not isinstance(
                text,
                str,
            )
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
                    "v054_new_frame",

                "batch":
                    "Batch-4A",
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

        feature_counter[
            (
                frame_id,
                tense,
                polarity,
            )
        ] += 1

    if len(rows) != total_new:
        raise RuntimeError(
            "Failed to generate requested "
            "V0.5.4 rows. "
            f"requested={total_new}, "
            f"generated={len(rows)}, "
            f"attempts={attempts}"
        )

    return (
        rows,
        attempts,
    )


# =============================================================================
# Contract
# =============================================================================

def validate_final_contract(
    rows: list[dict],
    *,
    expected_total: int,
    expected_new: int,
) -> list[str]:

    errors: list[str] = []

    if len(rows) != expected_total:
        errors.append(
            "TOTAL_COUNT_MISMATCH: "
            f"{len(rows)} != {expected_total}"
        )

    semantic_ids = [
        row.get(
            "semantic_id"
        )
        for row in rows
    ]

    if len(
        semantic_ids
    ) != len(
        set(semantic_ids)
    ):
        errors.append(
            "DUPLICATE_SEMANTIC_ID"
        )

    new_rows = []

    new_signatures = set()

    frame_counter = Counter()

    feature_counter = Counter()

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
                f"{sid}: BLOCKED_VERB:"
                f"{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:
            errors.append(
                f"{sid}: BLOCKED_FRAME:"
                f"{frame_id}"
            )

        if frame_id not in NEW_FRAMES:
            continue

        new_rows.append(
            row
        )

        frame_counter[
            frame_id
        ] += 1

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        feature_counter[
            (
                frame_id,
                tense,
                polarity,
            )
        ] += 1

        if (
            metadata.get(
                "generation_source"
            )
            != "v054_new_frame"
        ):
            errors.append(
                f"{sid}: "
                "NEW_SOURCE_MISMATCH"
            )

        signature = semantic_signature(
            frame_id=frame_id,
            slots=slots,
            features=features,
        )

        if signature in new_signatures:
            errors.append(
                f"{sid}: "
                "DUPLICATE_NEW_SIGNATURE"
            )

        new_signatures.add(
            signature
        )

        # ---------------------------------------------------------------------
        # NEED
        # ---------------------------------------------------------------------

        if frame_id == "NEED_OBJECT":

            if verb_id != "NEED":
                errors.append(
                    f"{sid}: "
                    "NEED_VERB_MISMATCH"
                )

            if set(
                slots.keys()
            ) != {
                "subject",
                "verb",
                "object",
            }:
                errors.append(
                    f"{sid}: "
                    "NEED_SLOT_STRUCTURE"
                )

            if tense not in {
                "present",
                "future",
            }:
                errors.append(
                    f"{sid}: "
                    "NEED_INVALID_TENSE:"
                    f"{tense}"
                )

            if polarity not in {
                "pos",
                "neg",
            }:
                errors.append(
                    f"{sid}: "
                    "NEED_INVALID_POLARITY:"
                    f"{polarity}"
                )

        # ---------------------------------------------------------------------
        # LOSE
        # ---------------------------------------------------------------------

        elif frame_id == "LOSE_OBJECT":

            if verb_id != "LOSE":
                errors.append(
                    f"{sid}: "
                    "LOSE_VERB_MISMATCH"
                )

            if set(
                slots.keys()
            ) != {
                "subject",
                "verb",
                "object",
            }:
                errors.append(
                    f"{sid}: "
                    "LOSE_SLOT_STRUCTURE"
                )

            if tense != "future":
                errors.append(
                    f"{sid}: "
                    "LOSE_NON_FUTURE_LEAK:"
                    f"{tense}"
                )

            if polarity not in {
                "pos",
                "neg",
            }:
                errors.append(
                    f"{sid}: "
                    "LOSE_INVALID_POLARITY:"
                    f"{polarity}"
                )

    if len(new_rows) != expected_new:
        errors.append(
            "NEW_COUNT_MISMATCH: "
            f"{len(new_rows)} "
            f"!= {expected_new}"
        )

    expected_frames = frame_targets(
        expected_new
    )

    for frame_id, expected in (
        expected_frames.items()
    ):
        actual = frame_counter[
            frame_id
        ]

        if actual != expected:
            errors.append(
                "FRAME_COUNT_MISMATCH:"
                f"{frame_id}:"
                f"{actual}!={expected}"
            )

    # NEED must cover all four combinations.
    if frame_counter[
        "NEED_OBJECT"
    ] >= 4:

        for tense, polarity in NEED_FEATURES:

            if feature_counter[
                (
                    "NEED_OBJECT",
                    tense,
                    polarity,
                )
            ] == 0:
                errors.append(
                    "NEED_FEATURE_COVERAGE_MISSING:"
                    f"{tense}:{polarity}"
                )

    # LOSE must cover positive and negative future.
    if frame_counter[
        "LOSE_OBJECT"
    ] >= 2:

        for tense, polarity in LOSE_FEATURES:

            if feature_counter[
                (
                    "LOSE_OBJECT",
                    tense,
                    polarity,
                )
            ] == 0:
                errors.append(
                    "LOSE_FEATURE_COVERAGE_MISSING:"
                    f"{tense}:{polarity}"
                )

    # Absolutely no LOSE present.
    lose_present = sum(
        count
        for (
            frame_id,
            tense,
            _,
        ), count
        in feature_counter.items()
        if (
            frame_id == "LOSE_OBJECT"
            and tense == "present"
        )
    )

    if lose_present != 0:
        errors.append(
            "LOSE_PRESENT_GENERATED"
        )

    return errors


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Generator V0.5.4"
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
        default=2059,
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
            "--new-ratio must be "
            "between 0.05 and 0.80"
        )

    output_dir = args.output_dir

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        output_dir
        / "semantic_v054_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v054_stats.json"
    )

    frozen_dir = (
        output_dir
        / "_frozen_parent_v053"
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
        "SYNTHETIC GENERATOR V0.5.4"
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
        "Frozen V0.5.3 rows:",
        parent_n,
    )

    print(
        "V0.5.4 new rows:",
        new_n,
    )

    print(
        "New ratio:",
        f"{new_n / args.n:.2%}",
    )

    print()

    print(
        "Tense policy:"
    )

    print(
        "NEED_OBJECT -> present/future"
    )

    print(
        "LOSE_OBJECT -> future ONLY"
    )

    # =========================================================================
    # Parent
    # =========================================================================

    frozen_raw = (
        generate_frozen_parent(
            n=parent_n,
            seed=args.seed,
            output_dir=frozen_dir,
        )
    )

    frozen_rows = normalize_parent_rows(
        frozen_raw
    )

    # =========================================================================
    # New
    # =========================================================================

    (
        new_rows,
        attempts,
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

    # Final consolidated semantic IDs.
    for index, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "semantic_id"
        ] = (
            f"sem_v054_"
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
            "V0.5.4 generation contract "
            f"failed with "
            f"{len(contract_errors)} errors."
        )

    # =========================================================================
    # Statistics
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

    feature_counter = Counter(
        (
            row.get(
                "frame_id"
            ),
            row.get(
                "features",
                {},
            ).get(
                "tense"
            ),
            row.get(
                "features",
                {},
            ).get(
                "polarity"
            ),
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
            attempts,

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

        "new_feature_distribution": {
            "|".join(
                str(x)
                for x in key
            ): value
            for key, value
            in feature_counter.items()
        },

        "tense_policy": {
            "NEED_OBJECT": [
                "present",
                "future",
            ],

            "LOSE_OBJECT": [
                "future",
            ],
        },

        "past_generation_allowed":
            False,

        "lose_present_generation_allowed":
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
                row[
                    "semantic_id"
                ],
                "|",
                row.get(
                    "scenario"
                ),
                "|",
                row.get(
                    "frame_id"
                ),
            )

            print(
                "slots:",
                row.get(
                    "slots"
                ),
            )

            print(
                "features:",
                row.get(
                    "features"
                ),
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

    # =========================================================================
    # Final report
    # =========================================================================

    print()
    print("=" * 100)
    print(
        "GENERATOR V0.5.4 COMPLETE"
    )
    print("=" * 100)

    print(
        "Generated:",
        len(rows),
    )

    print(
        "Frozen V0.5.3 rows:",
        parent_n,
    )

    print(
        "V0.5.4 new rows:",
        new_n,
    )

    print(
        "New generation attempts:",
        attempts,
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
            f"{frame_id:<25}"
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
        "Feature distribution:"
    )

    for key, count in sorted(
        feature_counter.items()
    ):
        print(
            f"{str(key):<45}"
            f"{count}"
        )

    print()

    print(
        "LOSE present generated:",
        0,
    )

    print()

    print("Files:")
    print(raw_file)
    print(stats_file)

    print()
    print("=" * 100)
    print(
        "V0.5.4 GENERATION CONTRACT PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()