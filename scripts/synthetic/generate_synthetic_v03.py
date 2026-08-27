from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


# ============================================================
# Stable V0.1 engine
# ============================================================

from scripts.synthetic import generate_synthetic_v01 as v01

from scripts.synthetic.renderer_v2 import (
    load_verb_policies,
    render_zh_verb_v2,
    render_ru_verb_v2,
)


# ============================================================
# Version
# ============================================================

GENERATOR_VERSION = "0.3.1"
RENDERER_VERSION = "v2.1"


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

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v03"
)


# ============================================================
# Basic IO
# ============================================================

def load_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if not isinstance(
        data,
        dict,
    ):

        raise ValueError(
            f"Expected JSON object:\n{path}"
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


# ============================================================
# Compatibility resource
# ============================================================

def load_compatibility_resource() -> dict:

    data = load_json(
        COMPATIBILITY_FILE
    )

    if not isinstance(
        data.get(
            "concept_classes"
        ),
        dict,
    ):

        raise ValueError(
            "semantic_compatibility.json "
            "missing concept_classes."
        )

    if not isinstance(
        data.get(
            "verb_rules"
        ),
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


# ============================================================
# Renderer V2 resource
# ============================================================

VERB_POLICIES = (
    load_verb_policies()
)


# ============================================================
# Keep original V0.1 renderer
# ============================================================

ORIGINAL_RENDER_VERB = (
    v01.render_verb
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

    concept_id = (
        normalize_concept_id(
            concept_id
        )
    )

    pattern = (
        normalize_concept_id(
            pattern
        )
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
# Semantic class
# ============================================================

def classify_concept(
    concept: dict,
) -> set[str]:

    concept_id = concept.get(
        "id"
    )

    if not concept_id:

        return set()

    class_map = (
        COMPATIBILITY.get(
            "concept_classes",
            {},
        )
    )

    result: set[str] = set()

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

                result.add(
                    class_name
                )

                break

    return result


# ============================================================
# Verb compatibility rule
# ============================================================

def find_verb_rule(
    verb: dict | None,
) -> tuple[
    str | None,
    dict | None,
]:

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

def find_explicit_forbidden_pair(
    verb: dict | None,
    obj: dict | None,
) -> dict | None:

    if (
        not verb
        or
        not obj
    ):

        return None

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

        return None

    pairs = (
        COMPATIBILITY.get(
            "explicit_forbidden_pairs",
            [],
        )
    )

    for pair in pairs:

        expected_verb = (
            pair.get(
                "verb"
            )
        )

        expected_object = (
            pair.get(
                "object"
            )
        )

        if (
            not expected_verb
            or
            not expected_object
        ):

            continue

        if (
            concept_matches_pattern(
                str(verb_id),
                str(expected_verb),
            )
            and
            concept_matches_pattern(
                str(object_id),
                str(expected_object),
            )
        ):

            return pair

    return None


def is_explicitly_forbidden(
    verb: dict | None,
    obj: dict | None,
) -> bool:

    return (
        find_explicit_forbidden_pair(
            verb,
            obj,
        )
        is not None
    )


# ============================================================
# Ordinary concept pool
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
# Compatibility-aware object pool
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
            "\nNo semantic compatibility "
            "rule for verb.\n"
            f"verb={verb.get('id')}"
        )

    allowed_classes = set(
        rule.get(
            "allowed_object_classes",
            [],
        )
    )

    if not allowed_classes:

        raise RuntimeError(
            "\nVerb requires an object "
            "but no allowed_object_classes "
            "were configured.\n"
            f"verb={verb.get('id')}\n"
            f"rule={rule_name}"
        )

    compatible_pool = []

    for concept in base_pool:

        # ----------------------------------------------------
        # Known forbidden pair
        # ----------------------------------------------------

        if is_explicitly_forbidden(
            verb,
            concept,
        ):

            continue

        # ----------------------------------------------------
        # Semantic class lookup
        # ----------------------------------------------------

        concept_classes = (
            classify_concept(
                concept
            )
        )

        # Production generator is strict:
        # unknown compatibility is NOT sampled.
        if not concept_classes:

            continue

        if (
            concept_classes
            & allowed_classes
        ):

            compatible_pool.append(
                concept
            )

    if not compatible_pool:

        raise RuntimeError(
            "\nNo compatible object "
            "candidates were found.\n"
            f"verb={verb.get('id')}\n"
            f"allowed_classes="
            f"{sorted(allowed_classes)}\n"
            f"base_pool="
            f"{[c.get('id') for c in base_pool]}"
        )

    return compatible_pool


# ============================================================
# Compatibility-aware destination pool
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
            "\nNo semantic compatibility "
            "rule for verb.\n"
            f"verb={verb.get('id')}"
        )

    allowed_classes = set(
        rule.get(
            "allowed_destination_classes",
            [],
        )
    )

    if not allowed_classes:

        raise RuntimeError(
            "\nVerb requires destination "
            "sampling but no "
            "allowed_destination_classes "
            "were configured.\n"
            f"verb={verb.get('id')}\n"
            f"rule={rule_name}"
        )

    compatible_pool = []

    for concept in base_pool:

        classes = (
            classify_concept(
                concept
            )
        )

        if not classes:

            continue

        if (
            classes
            & allowed_classes
        ):

            compatible_pool.append(
                concept
            )

    if not compatible_pool:

        raise RuntimeError(
            "\nNo compatible destination "
            "candidates were found.\n"
            f"verb={verb.get('id')}\n"
            f"allowed_classes="
            f"{sorted(allowed_classes)}\n"
            f"base_pool="
            f"{[c.get('id') for c in base_pool]}"
        )

    return compatible_pool


# ============================================================
# Fixed-slot compatibility validation
# ============================================================

def validate_selected_compatibility(
    selected: dict,
) -> None:

    verb = selected.get(
        "verb"
    )

    # WHERE_PLACE and similar verbless frames
    # are valid.
    if not verb:

        return

    (
        _,
        rule,
    ) = find_verb_rule(
        verb
    )

    if rule is None:

        raise RuntimeError(
            "\nSelected verb has no "
            "compatibility rule:\n"
            f"{verb.get('id')}"
        )

    # ========================================================
    # Object
    # ========================================================

    obj = selected.get(
        "object"
    )

    if obj:

        forbidden = (
            find_explicit_forbidden_pair(
                verb,
                obj,
            )
        )

        if forbidden is not None:

            raise RuntimeError(
                "\nSelected slots contain "
                "an explicitly forbidden "
                "verb-object pair.\n"
                f"verb={verb.get('id')}\n"
                f"object={obj.get('id')}\n"
                f"reason="
                f"{forbidden.get('reason')}"
            )

        allowed_classes = set(
            rule.get(
                "allowed_object_classes",
                [],
            )
        )

        if allowed_classes:

            object_classes = (
                classify_concept(
                    obj
                )
            )

            if not (
                object_classes
                & allowed_classes
            ):

                raise RuntimeError(
                    "\nObject violates semantic "
                    "compatibility.\n"
                    f"verb={verb.get('id')}\n"
                    f"object={obj.get('id')}\n"
                    f"object_classes="
                    f"{sorted(object_classes)}\n"
                    f"allowed_classes="
                    f"{sorted(allowed_classes)}"
                )

    # ========================================================
    # Destination
    # ========================================================

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
                & allowed_classes
            ):

                raise RuntimeError(
                    "\nDestination violates "
                    "semantic compatibility.\n"
                    f"verb={verb.get('id')}\n"
                    f"destination="
                    f"{destination.get('id')}\n"
                    f"destination_classes="
                    f"{sorted(destination_classes)}\n"
                    f"allowed_classes="
                    f"{sorted(allowed_classes)}"
                )


# ============================================================
# V0.3 slot selector
# ============================================================

def select_slots_v03(
    frame: dict,
    concepts: list[dict],
    concepts_by_id: dict,
    rng: random.Random,
) -> dict:

    selected: dict = {}

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
                "\nUnknown fixed concept ID.\n"
                f"frame={frame.get('id')}\n"
                f"slot={slot_name}\n"
                f"concept={concept_id}"
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

        # Fixed slots must never be overwritten.
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
        # Other slots:
        #
        # subject
        # verb
        # day/time concept
        # etc.
        #
        # Reuse stable V0.1 behavior.
        # ----------------------------------------------------

        selected[
            slot_name
        ] = v01.choose_concept(
            concepts,
            spec,
            rng,
        )

    # ========================================================
    # 3. Final deterministic compatibility assertion
    # ========================================================

    validate_selected_compatibility(
        selected
    )

    return selected


# ============================================================
# Renderer V0.3
# ============================================================

def render_verb_v03(
    concept: dict,
    lang: str,
    tense: str,
    polarity: str,
    person: str,
):

    # ========================================================
    # First obtain the already-working V0.1 surface.
    # ========================================================

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

    # ========================================================
    # Chinese Renderer V2
    #
    # Example fixed previously:
    #
    # FIND + negative
    # 不找到 -> 没找到
    # ========================================================

    if lang == "zh":

        return render_zh_verb_v2(
            verb_id=verb_id,
            original_surface=original_surface,
            tense=tense,
            polarity=polarity,
            policies=VERB_POLICIES,
        )

    # ========================================================
    # Russian Renderer V2
    #
    # Example fixed previously:
    #
    # FIND future:
    # будет находить -> найдёт
    # ========================================================

    if lang == "ru":

        return render_ru_verb_v2(
            verb_id=verb_id,
            original_surface=original_surface,
            person=person,
            tense=tense,
            polarity=polarity,
            policies=VERB_POLICIES,
        )

    # ========================================================
    # EN / UZ retain V0.1 forms
    # ========================================================

    return original_surface


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Generator V0.3 - "
            "compatibility-aware four-language "
            "parallel data generator"
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="Number of unique semantic samples.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
        help="Random seed.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Output directory. "
            "Relative paths are resolved "
            "from project root."
        ),
    )

    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=200,
        help=(
            "Maximum generation attempts = "
            "n * this value."
        ),
    )

    return parser.parse_args()


# ============================================================
# Resolve output
# ============================================================

def resolve_output_dir(
    value: str,
) -> Path:

    path = Path(
        value
    )

    if not path.is_absolute():

        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


# ============================================================
# Generate
# ============================================================

def generate_samples(
    *,
    n: int,
    seed: int,
    max_attempt_multiplier: int,
) -> tuple[
    list[dict],
    dict,
]:

    if n <= 0:

        raise ValueError(
            "--n must be > 0"
        )

    if max_attempt_multiplier <= 0:

        raise ValueError(
            "--max-attempt-multiplier "
            "must be > 0"
        )

    # ========================================================
    # Load the same resources as V0.1
    # ========================================================

    (
        concepts,
        concepts_by_id,
        frames,
    ) = v01.load_resources()

    rng = random.Random(
        seed
    )

    frame_weights = [
        frame.get(
            "weight",
            1,
        )
        for frame in frames
    ]

    generated: list[dict] = []

    signatures = set()

    frame_counter = Counter()

    verb_counter = Counter()

    object_counter = Counter()

    destination_counter = Counter()

    polarity_counter = Counter()

    tense_counter = Counter()

    max_attempts = (
        n
        * max_attempt_multiplier
    )

    attempts = 0

    # ========================================================
    # render_sample() belongs to generate_synthetic_v01.py.
    #
    # Its global render_verb symbol must temporarily point to
    # Renderer V0.3 while this generation is running.
    # ========================================================

    previous_render_verb = (
        v01.render_verb
    )

    v01.render_verb = (
        render_verb_v03
    )

    try:

        # ====================================================
        # Generation loop
        # ====================================================

        while (
            len(generated) < n
            and
            attempts < max_attempts
        ):

            attempts += 1

            # ------------------------------------------------
            # Frame sampling
            # ------------------------------------------------

            frame = rng.choices(
                frames,
                weights=frame_weights,
                k=1,
            )[0]

            # ------------------------------------------------
            # Compatibility-aware semantic slots
            # ------------------------------------------------

            selected = (
                select_slots_v03(
                    frame,
                    concepts,
                    concepts_by_id,
                    rng,
                )
            )

            # ------------------------------------------------
            # Same tense logic as V0.1
            # ------------------------------------------------

            tense = (
                v01.derive_tense(
                    selected
                )
            )

            # ------------------------------------------------
            # Same polarity sampling as V0.1
            # ------------------------------------------------

            polarity = (
                v01.choose_polarity(
                    frame,
                    rng,
                )
            )

            features = {
                "tense":
                    tense,

                "polarity":
                    polarity,
            }

            # ------------------------------------------------
            # Computed values
            # ------------------------------------------------

            computed = {}

            if (
                "clock"
                in frame.get(
                    "computed",
                    [],
                )
            ):

                computed[
                    "clock"
                ] = v01.build_clock(
                    rng
                )

            # ------------------------------------------------
            # Semantic signature / deduplication
            # ------------------------------------------------

            signature = (
                v01.semantic_signature(
                    frame,
                    selected,
                    features,
                    computed,
                )
            )

            if signature in signatures:

                continue

            signatures.add(
                signature
            )

            # ------------------------------------------------
            # Four-language rendering
            #
            # Uses V0.1 render_sample + patched render_verb_v03.
            # ------------------------------------------------

            (
                texts,
                traces,
            ) = v01.render_sample(
                frame,
                selected,
                features,
                computed,
            )

            # ------------------------------------------------
            # Output schema
            # ------------------------------------------------

            sample = {

                "semantic_id":
                    f"sem_{len(generated)+1:08d}",

                "frame_id":
                    frame["id"],

                "slots": {
                    key: value["id"]
                    for (
                        key,
                        value,
                    )
                    in selected.items()
                },

                "features":
                    features,

                "computed":
                    computed,

                "texts":
                    texts,

                "trace":
                    traces,

                "source_type":
                    "grammar_synthetic",

                "resource_version":
                    GENERATOR_VERSION,

                "generator_version":
                    GENERATOR_VERSION,

                "renderer_version":
                    RENDERER_VERSION,

                "compatibility_version":
                    COMPATIBILITY.get(
                        "version",
                        "unknown",
                    ),

                "generation_policy":
                    "compatibility_aware",
            }

            generated.append(
                sample
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            frame_counter[
                frame["id"]
            ] += 1

            tense_counter[
                tense
            ] += 1

            polarity_counter[
                polarity
            ] += 1

            verb_id = (
                sample.get(
                    "slots",
                    {},
                ).get(
                    "verb"
                )
            )

            if verb_id:

                verb_counter[
                    verb_id
                ] += 1

            object_id = (
                sample.get(
                    "slots",
                    {},
                ).get(
                    "object"
                )
            )

            if object_id:

                object_counter[
                    object_id
                ] += 1

            destination_id = (
                sample.get(
                    "slots",
                    {},
                ).get(
                    "destination"
                )
            )

            if destination_id:

                destination_counter[
                    destination_id
                ] += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                len(generated) % 1000 == 0
                or
                len(generated) == n
            ):

                print(
                    f"{len(generated)}/{n}"
                    f" | attempts={attempts}"
                    f" | unique="
                    f"{len(signatures)}"
                )

    finally:

        # Restore V0.1 module state.
        v01.render_verb = (
            previous_render_verb
        )

    # ========================================================
    # Guard
    # ========================================================

    if len(generated) < n:

        raise RuntimeError(
            "\nUnable to generate requested "
            "number of unique samples.\n"
            f"Generated: {len(generated)}\n"
            f"Target: {n}\n"
            f"Attempts: {attempts}\n"
            f"Max attempts: {max_attempts}\n"
        )

    # ========================================================
    # Stats
    # ========================================================

    stats = {

        "generator_version":
            GENERATOR_VERSION,

        "renderer_version":
            RENDERER_VERSION,

        "compatibility_version":
            COMPATIBILITY.get(
                "version",
                "unknown",
            ),

        "generation_policy":
            "compatibility_aware",

        "seed":
            seed,

        "target_samples":
            n,

        "generated_samples":
            len(generated),

        "attempts":
            attempts,

        "duplicate_or_retry_attempts":
            attempts
            - len(generated),

        "generation_efficiency":
            (
                len(generated)
                / attempts
                if attempts
                else 0
            ),

        "unique_signatures":
            len(signatures),

        "frame_distribution":
            dict(
                frame_counter
                .most_common()
            ),

        "verb_distribution":
            dict(
                verb_counter
                .most_common()
            ),

        "object_distribution":
            dict(
                object_counter
                .most_common()
            ),

        "destination_distribution":
            dict(
                destination_counter
                .most_common()
            ),

        "tense_distribution":
            dict(
                tense_counter
                .most_common()
            ),

        "polarity_distribution":
            dict(
                polarity_counter
                .most_common()
            ),
    }

    return (
        generated,
        stats,
    )


# ============================================================
# Diagnostics
# ============================================================

def print_configuration(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    output_file: Path,
    stats_file: Path,
) -> None:

    print("=" * 100)
    print("SYNTHETIC GENERATOR V0.3")
    print("=" * 100)

    print(
        "Project root:",
        PROJECT_ROOT,
    )

    print(
        "Compatibility:",
        COMPATIBILITY_FILE,
    )

    print(
        "Compatibility version:",
        COMPATIBILITY.get(
            "version",
            "unknown",
        ),
    )

    print(
        "Policy:",
        "compatibility-aware sampling",
    )

    print(
        "Renderer:",
        "Renderer V2",
    )

    print(
        "Samples:",
        args.n,
    )

    print(
        "Seed:",
        args.seed,
    )

    print(
        "Output dir:",
        output_dir,
    )

    print(
        "Output file:",
        output_file,
    )

    print(
        "Stats file:",
        stats_file,
    )

    print("=" * 100)


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    output_dir = (
        resolve_output_dir(
            args.output_dir
        )
    )

    output_file = (
        output_dir
        / "semantic_v03_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v03_stats.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_configuration(
        args=args,
        output_dir=output_dir,
        output_file=output_file,
        stats_file=stats_file,
    )

    # ========================================================
    # Generate
    # ========================================================

    (
        rows,
        stats,
    ) = generate_samples(
        n=args.n,
        seed=args.seed,
        max_attempt_multiplier=(
            args.max_attempt_multiplier
        ),
    )

    # ========================================================
    # IMPORTANT:
    #
    # V0.3 writes its own files.
    #
    # We NEVER call v01.main().
    #
    # Therefore semantic_v01_raw.jsonl cannot be overwritten
    # by V0.3.
    # ========================================================

    write_jsonl(
        output_file,
        rows,
    )

    stats[
        "output_dir"
    ] = str(
        output_dir
    )

    stats[
        "output_file"
    ] = str(
        output_file
    )

    stats[
        "stats_file"
    ] = str(
        stats_file
    )

    write_json(
        stats_file,
        stats,
    )

    # ========================================================
    # Final disk verification
    # ========================================================

    if not output_file.exists():

        raise FileNotFoundError(
            "\nGeneration completed but "
            "output file was not created:\n"
            f"{output_file}"
        )

    if not stats_file.exists():

        raise FileNotFoundError(
            "\nGeneration completed but "
            "stats file was not created:\n"
            f"{stats_file}"
        )

    # Count actual lines from disk.
    with output_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        actual_rows = sum(
            1
            for line in f
            if line.strip()
        )

    if actual_rows != args.n:

        raise RuntimeError(
            "\nOutput verification failed.\n"
            f"Expected rows: {args.n}\n"
            f"Actual rows: {actual_rows}\n"
            f"File: {output_file}"
        )

    # ========================================================
    # Complete
    # ========================================================

    print()
    print("=" * 100)
    print("GENERATOR V0.3 COMPLETE")
    print("=" * 100)

    print(
        "Generated:",
        actual_rows,
    )

    print(
        "Attempts:",
        stats.get(
            "attempts"
        ),
    )

    print(
        "Efficiency:",
        (
            f"{stats.get('generation_efficiency', 0):.2%}"
        ),
    )

    print(
        "Output:",
        output_file,
    )

    print(
        "Stats:",
        stats_file,
    )

    print(
        "Generator version:",
        GENERATOR_VERSION,
    )

    print(
        "Renderer version:",
        RENDERER_VERSION,
    )

    print(
        "Compatibility version:",
        COMPATIBILITY.get(
            "version",
            "unknown",
        ),
    )

    print()
    print(
        "IMPORTANT: V0.1 output was NOT used."
    )

    print("=" * 100)


if __name__ == "__main__":
    main()