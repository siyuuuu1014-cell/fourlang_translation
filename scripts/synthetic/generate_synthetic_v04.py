from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.synthetic.renderer_v04 import (
    RenderError,
    V04Renderer,
)


# ============================================================
# Project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v04.jsonl"
)

DEFAULT_FRAMES = (
    RESOURCE_DIR
    / "frames_v04.json"
)

DEFAULT_COMPATIBILITY = (
    RESOURCE_DIR
    / "semantic_compatibility_v04.json"
)

DEFAULT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v04.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v04_smoke_100"
)


GENERATOR_VERSION = "0.4.0"
RENDERER_VERSION = "0.4"
RESOURCE_VERSION = "0.4"


# ============================================================
# Special generation constraints
# ============================================================

# ARRIVE without a destination is grammatically possible,
# but this synthetic corpus deliberately uses ARRIVE as a
# destination/result frame to avoid weak templates like:
#
#   She arrives.
#   她到达。
#
# which are less useful for the target translation corpus.
VERBS_REQUIRING_DESTINATION = {
    "ARRIVE",
}


CLOCK_DAY_IDS = {
    "TIME_TODAY",
    "TIME_TOMORROW",
}


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"{path} root must be a JSON object."
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

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RuntimeError(
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    "must be JSON object."
                )

            rows.append(
                row
            )

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


# ============================================================
# Helpers
# ============================================================

def is_enabled(
    item: dict,
) -> bool:

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
        "semantic_classes",
        [],
    )

    if isinstance(
        classes,
        str,
    ):
        classes = [
            classes
        ]

    if not isinstance(
        classes,
        list,
    ):
        return set()

    return {
        str(item)
        for item in classes
    }


def weighted_choice(
    rng: random.Random,
    items: list[Any],
    weights: list[float],
) -> Any:

    if not items:
        raise RuntimeError(
            "weighted_choice received empty items."
        )

    if len(
        items
    ) != len(
        weights
    ):
        raise RuntimeError(
            "items / weights length mismatch."
        )

    total = sum(
        max(
            float(weight),
            0.0,
        )
        for weight in weights
    )

    if total <= 0:

        return rng.choice(
            items
        )

    target = (
        rng.random()
        * total
    )

    cumulative = 0.0

    for item, weight in zip(
        items,
        weights,
    ):

        cumulative += max(
            float(weight),
            0.0,
        )

        if target <= cumulative:

            return item

    return items[-1]


# ============================================================
# Resource container
# ============================================================

class V04GenerationResources:

    def __init__(
        self,
        *,
        concepts_path: Path,
        frames_path: Path,
        compatibility_path: Path,
        policy_path: Path,
    ) -> None:

        self.concept_rows = read_jsonl(
            concepts_path
        )

        self.frames_data = read_json(
            frames_path
        )

        self.compatibility = read_json(
            compatibility_path
        )

        self.policy = read_json(
            policy_path
        )

        self.concepts = {
            row["id"]: row
            for row
            in self.concept_rows
        }

        self.active_concepts = {
            cid: row
            for cid, row
            in self.concepts.items()
            if is_enabled(
                row
            )
        }

        raw_frames = (
            self.frames_data.get(
                "frames",
                [],
            )
        )

        self.active_frames = [
            frame
            for frame in raw_frames
            if (
                isinstance(
                    frame,
                    dict,
                )
                and is_enabled(
                    frame
                )
            )
        ]

        self.frames_by_id = {
            frame["id"]: frame
            for frame
            in self.active_frames
        }

        self.by_type: dict[
            str,
            list[dict],
        ] = defaultdict(
            list
        )

        for concept in (
            self.active_concepts.values()
        ):

            concept_type = (
                concept.get(
                    "concept_type"
                )
            )

            if concept_type:

                self.by_type[
                    concept_type
                ].append(
                    concept
                )

        self.class_parent = {}

        semantic_classes = (
            self.compatibility.get(
                "semantic_classes",
                {},
            )
        )

        if isinstance(
            semantic_classes,
            dict,
        ):

            for class_id, spec in (
                semantic_classes.items()
            ):

                parent = None

                if isinstance(
                    spec,
                    dict,
                ):

                    parent = spec.get(
                        "parent"
                    )

                self.class_parent[
                    class_id
                ] = parent

    # ========================================================
    # Semantic class inheritance
    # ========================================================

    def expanded_classes(
        self,
        concept: dict,
    ) -> set[str]:

        direct = (
            semantic_classes_of(
                concept
            )
        )

        expanded = set(
            direct
        )

        for class_id in list(
            direct
        ):

            current = class_id
            seen = set()

            while current:

                if current in seen:
                    break

                seen.add(
                    current
                )

                parent = (
                    self.class_parent.get(
                        current
                    )
                )

                if not parent:
                    break

                expanded.add(
                    parent
                )

                current = parent

        return expanded

    def concept_matches_classes(
        self,
        concept: dict,
        required_classes: list[str] | set[str],
    ) -> bool:

        required = set(
            required_classes
        )

        if not required:
            return True

        actual = (
            self.expanded_classes(
                concept
            )
        )

        return bool(
            actual
            & required
        )


# ============================================================
# Generator
# ============================================================

class SyntheticGeneratorV04:

    def __init__(
        self,
        *,
        resources: V04GenerationResources,
        renderer: V04Renderer,
        seed: int,
    ) -> None:

        self.resources = (
            resources
        )

        self.renderer = (
            renderer
        )

        self.rng = random.Random(
            seed
        )

        self.seed = seed

        self.verb_usage = Counter()

        self.frame_usage = Counter()

        self.scenario_usage = Counter()

    # ========================================================
    # Scenario
    # ========================================================

    def choose_scenario(
        self,
    ) -> str:

        weights = (
            self.resources
            .policy
            .get(
                "scenario_weights",
                {},
            )
        )

        scenarios = list(
            weights.keys()
        )

        values = [
            float(
                weights[
                    scenario
                ]
            )
            for scenario
            in scenarios
        ]

        return weighted_choice(
            self.rng,
            scenarios,
            values,
        )

    # ========================================================
    # Frame
    # ========================================================

    def choose_frame(
        self,
        scenario: str,
    ) -> dict:

        candidates = [
            frame
            for frame
            in self.resources.active_frames
            if scenario
            in frame.get(
                "scenario_tags",
                [],
            )
        ]

        if not candidates:

            raise RuntimeError(
                f"No active frame for "
                f"scenario={scenario}"
            )

        weights = []

        for frame in candidates:

            base_weight = float(
                frame.get(
                    "weight",
                    1.0,
                )
            )

            # Gentle inverse-frequency balancing.
            #
            # The goal is not perfect uniformity,
            # only to avoid one frame dominating
            # a small pilot.
            used = (
                self.frame_usage[
                    frame["id"]
                ]
            )

            adjusted = (
                base_weight
                / (
                    1.0
                    + 0.35 * used
                )
            )

            weights.append(
                adjusted
            )

        return weighted_choice(
            self.rng,
            candidates,
            weights,
        )

    # ========================================================
    # Polarity
    # ========================================================

    def choose_polarity(
        self,
        frame: dict,
    ) -> str:

        allowed = (
            frame.get(
                "features",
                {},
            ).get(
                "polarity"
            )
        )

        if not allowed:

            return "pos"

        policy_weights = (
            self.resources
            .policy
            .get(
                "polarity_weights",
                {
                    "pos": 0.7,
                    "neg": 0.3,
                },
            )
        )

        candidates = [
            item
            for item in allowed
            if item
            in {
                "pos",
                "neg",
            }
        ]

        if not candidates:

            return "pos"

        weights = [
            float(
                policy_weights.get(
                    item,
                    1.0,
                )
            )
            for item in candidates
        ]

        return weighted_choice(
            self.rng,
            candidates,
            weights,
        )

    # ========================================================
    # Tense
    # ========================================================

    def choose_policy_tense(
        self,
        allowed_tenses: list[str] | None = None,
    ) -> str:

        policy = (
            self.resources
            .policy
            .get(
                "tense_weights",
                {
                    "present": 0.6,
                    "future": 0.4,
                },
            )
        )

        if allowed_tenses:

            candidates = [
                tense
                for tense
                in allowed_tenses
                if tense in policy
            ]

        else:

            candidates = list(
                policy.keys()
            )

        if not candidates:

            raise RuntimeError(
                "No valid tense candidates."
            )

        weights = [
            float(
                policy.get(
                    tense,
                    1.0,
                )
            )
            for tense in candidates
        ]

        return weighted_choice(
            self.rng,
            candidates,
            weights,
        )

    def tense_from_time(
        self,
        concept_id: str | None,
    ) -> str | None:

        if not concept_id:

            return None

        concept = (
            self.resources
            .active_concepts
            .get(
                concept_id
            )
        )

        if not concept:
            return None

        value = (
            concept.get(
                "time_features",
                {},
            ).get(
                "tense_hint"
            )
        )

        if value in {
            "present",
            "future",
        }:

            return value

        return None

    # ========================================================
    # Generic concept candidates
    # ========================================================

    def concepts_of_type(
        self,
        concept_type: str,
    ) -> list[dict]:

        return list(
            self.resources
            .by_type
            .get(
                concept_type,
                []
            )
        )

    def choose_subject(
        self,
    ) -> str:

        candidates = (
            self.concepts_of_type(
                "person"
            )
        )

        if not candidates:

            raise RuntimeError(
                "No active person concepts."
            )

        return self.rng.choice(
            candidates
        )["id"]

    def choose_time(
        self,
        *,
        clock_frame: bool,
    ) -> str:

        candidates = (
            self.concepts_of_type(
                "time"
            )
        )

        if clock_frame:

            restricted = [
                concept
                for concept
                in candidates
                if concept[
                    "id"
                ]
                in CLOCK_DAY_IDS
            ]

            if restricted:
                candidates = (
                    restricted
                )

        if not candidates:

            raise RuntimeError(
                "No time concepts."
            )

        return self.rng.choice(
            candidates
        )["id"]

    def choose_clock(
        self,
    ) -> str:

        clock_policy = (
            self.resources
            .policy
            .get(
                "clock_policy",
                {},
            )
        )

        minute_step = int(
            clock_policy.get(
                "minute_step",
                15,
            )
        )

        min_hour = int(
            clock_policy.get(
                "min_hour",
                0,
            )
        )

        max_hour = int(
            clock_policy.get(
                "max_hour",
                23,
            )
        )

        hour = self.rng.randint(
            min_hour,
            max_hour,
        )

        valid_minutes = list(
            range(
                0,
                60,
                minute_step,
            )
        )

        minute = self.rng.choice(
            valid_minutes
        )

        return (
            f"{hour:02d}:"
            f"{minute:02d}"
        )

    # ========================================================
    # Frame verb filters
    # ========================================================

    def verb_matches_frame(
        self,
        verb: dict,
        frame: dict,
        tense: str,
    ) -> bool:

        if not is_enabled(
            verb
        ):
            return False

        if verb.get(
            "concept_type"
        ) != "verb":
            return False

        slots = frame.get(
            "slots",
            {},
        )

        verb_slot = slots.get(
            "verb",
            {},
        )

        # ----------------------------------------------------
        # ARRIVE requires destination for this corpus.
        # ----------------------------------------------------

        if (
            verb["id"]
            in VERBS_REQUIRING_DESTINATION
            and "destination"
            not in slots
        ):

            return False

        # ----------------------------------------------------
        # Frame semantic class requirement
        # ----------------------------------------------------

        required_classes = (
            verb_slot.get(
                "semantic_classes",
                [],
            )
        )

        if (
            required_classes
            and not self.resources
            .concept_matches_classes(
                verb,
                required_classes,
            )
        ):

            return False

        # ----------------------------------------------------
        # Frame transitivity requirement
        # ----------------------------------------------------

        slot_features = (
            verb_slot.get(
                "features",
                {},
            )
        )

        allowed_transitivity = (
            slot_features.get(
                "transitivity"
            )
        )

        if allowed_transitivity:

            transitivity = (
                verb.get(
                    "features",
                    {},
                ).get(
                    "transitivity"
                )
            )

            if transitivity not in (
                allowed_transitivity
            ):

                return False

        # ----------------------------------------------------
        # Verb tense capability
        # ----------------------------------------------------

        allowed_tenses = (
            verb.get(
                "features",
                {},
            ).get(
                "allowed_tenses"
            )
        )

        if (
            allowed_tenses
            and tense
            not in allowed_tenses
        ):

            return False

        return True

    def choose_verb(
        self,
        *,
        frame: dict,
        scenario: str,
        tense: str,
    ) -> str:

        verb_slot = (
            frame.get(
                "slots",
                {},
            ).get(
                "verb"
            )
        )

        if verb_slot is None:

            raise RuntimeError(
                f"Frame {frame['id']} "
                "does not contain verb slot."
            )

        fixed_id = (
            verb_slot.get(
                "fixed_concept_id"
            )
        )

        if fixed_id:

            verb = (
                self.resources
                .active_concepts
                .get(
                    fixed_id
                )
            )

            if not verb:

                raise RuntimeError(
                    f"Fixed active verb missing: "
                    f"{fixed_id}"
                )

            if not self.verb_matches_frame(
                verb,
                frame,
                tense,
            ):

                raise RuntimeError(
                    f"Fixed verb {fixed_id} "
                    f"cannot render tense={tense}"
                )

            return fixed_id

        all_candidates = [
            verb
            for verb
            in self.concepts_of_type(
                "verb"
            )
            if self.verb_matches_frame(
                verb,
                frame,
                tense,
            )
        ]

        if not all_candidates:

            raise RuntimeError(
                f"No compatible verb for "
                f"frame={frame['id']} "
                f"tense={tense}"
            )

        # Scenario tags are treated as preference,
        # not a hard semantic restriction.
        scenario_candidates = [
            verb
            for verb
            in all_candidates
            if scenario
            in verb.get(
                "scenario_tags",
                [],
            )
        ]

        candidates = (
            scenario_candidates
            or all_candidates
        )

        weights = []

        for verb in candidates:

            used = (
                self.verb_usage[
                    verb["id"]
                ]
            )

            # Stronger balancing than frame balancing
            # because V0.3.1 GO became dominant.
            weight = (
                1.0
                / (
                    1.0
                    + 0.60 * used
                )
            )

            weights.append(
                weight
            )

        chosen = weighted_choice(
            self.rng,
            candidates,
            weights,
        )

        return chosen[
            "id"
        ]

    # ========================================================
    # Semantic compatibility
    # ========================================================

    def role_rule(
        self,
        verb_id: str,
        role: str,
    ) -> dict | None:

        rules = (
            self.resources
            .compatibility
            .get(
                "verb_rules",
                {},
            )
            .get(
                verb_id,
                {},
            )
        )

        if not isinstance(
            rules,
            dict,
        ):
            return None

        rule = rules.get(
            role
        )

        if isinstance(
            rule,
            dict,
        ):
            return rule

        return None

    def is_explicitly_forbidden(
        self,
        *,
        verb_id: str,
        role: str,
        concept_id: str,
    ) -> bool:

        forbidden = (
            self.resources
            .compatibility
            .get(
                "explicit_forbidden",
                [],
            )
        )

        for item in forbidden:

            if not isinstance(
                item,
                dict,
            ):
                continue

            if item.get(
                "verb"
            ) != verb_id:

                continue

            # Current explicit forbidden resource
            # primarily uses "object".
            expected = item.get(
                role
            )

            if (
                expected is None
                and role == "object"
            ):

                expected = item.get(
                    "object"
                )

            if expected == concept_id:

                return True

        return False

    def concept_allowed_for_role(
        self,
        *,
        verb_id: str,
        role: str,
        concept: dict,
        frame_slot: dict,
    ) -> bool:

        # ----------------------------------------------------
        # Frame-level semantic classes
        # ----------------------------------------------------

        frame_classes = (
            frame_slot.get(
                "semantic_classes",
                [],
            )
        )

        if (
            frame_classes
            and not self.resources
            .concept_matches_classes(
                concept,
                frame_classes,
            )
        ):

            return False

        # ----------------------------------------------------
        # Verb compatibility rule
        # ----------------------------------------------------

        rule = self.role_rule(
            verb_id,
            role,
        )

        if rule is None:

            unknown_policy = (
                self.resources
                .compatibility
                .get(
                    "unknown_policy",
                    "reject",
                )
            )

            if unknown_policy == "reject":
                return False

        else:

            allowed_classes = (
                rule.get(
                    "allowed_classes",
                    [],
                )
            )

            if (
                allowed_classes
                and not self.resources
                .concept_matches_classes(
                    concept,
                    allowed_classes,
                )
            ):

                return False

        # ----------------------------------------------------
        # Explicit forbidden
        # ----------------------------------------------------

        if self.is_explicitly_forbidden(
            verb_id=verb_id,
            role=role,
            concept_id=concept["id"],
        ):

            return False

        return True

    def choose_role_concept(
        self,
        *,
        verb_id: str,
        role: str,
        frame_slot: dict,
        concept_type: str,
    ) -> str:

        candidates = [
            concept
            for concept
            in self.concepts_of_type(
                concept_type
            )
            if self.concept_allowed_for_role(
                verb_id=verb_id,
                role=role,
                concept=concept,
                frame_slot=frame_slot,
            )
        ]

        if not candidates:

            raise RuntimeError(
                f"No compatible {concept_type} "
                f"for verb={verb_id}, role={role}"
            )

        return self.rng.choice(
            candidates
        )["id"]

    # ========================================================
    # Semantic sample
    # ========================================================

    def create_candidate(
        self,
    ) -> dict:

        scenario = (
            self.choose_scenario()
        )

        frame = self.choose_frame(
            scenario
        )

        frame_id = frame[
            "id"
        ]

        frame_slots = (
            frame.get(
                "slots",
                {},
            )
        )

        slots: dict[
            str,
            str,
        ] = {}

        computed: dict[
            str,
            Any,
        ] = {}

        # ----------------------------------------------------
        # Subject
        # ----------------------------------------------------

        if "subject" in frame_slots:

            slots[
                "subject"
            ] = self.choose_subject()

        # ----------------------------------------------------
        # Fixed verb is useful when choosing tense
        # ----------------------------------------------------

        fixed_verb_id = None

        if "verb" in frame_slots:

            fixed_verb_id = (
                frame_slots[
                    "verb"
                ].get(
                    "fixed_concept_id"
                )
            )

        # ----------------------------------------------------
        # Time / day
        # ----------------------------------------------------

        if "time" in frame_slots:

            slots[
                "time"
            ] = self.choose_time(
                clock_frame=False,
            )

        if "day" in frame_slots:

            slots[
                "day"
            ] = self.choose_time(
                clock_frame=True,
            )

        # ----------------------------------------------------
        # Determine tense from explicit temporal concept.
        # ----------------------------------------------------

        tense = None

        for name in (
            "time",
            "day",
        ):

            if name in slots:

                tense = self.tense_from_time(
                    slots[
                        name
                    ]
                )

                if tense:
                    break

        # ----------------------------------------------------
        # If no temporal concept, use generation policy.
        #
        # For fixed verbs such as WANT, respect
        # allowed_tenses before choosing.
        # ----------------------------------------------------

        if tense is None:

            fixed_allowed = None

            if fixed_verb_id:

                fixed_verb = (
                    self.resources
                    .active_concepts
                    .get(
                        fixed_verb_id
                    )
                )

                if fixed_verb:

                    fixed_allowed = (
                        fixed_verb.get(
                            "features",
                            {},
                        ).get(
                            "allowed_tenses"
                        )
                    )

            tense = self.choose_policy_tense(
                fixed_allowed
            )

        # ----------------------------------------------------
        # Polarity
        # ----------------------------------------------------

        polarity = (
            self.choose_polarity(
                frame
            )
        )

        # ----------------------------------------------------
        # Verb
        # ----------------------------------------------------

        verb_id = None

        if "verb" in frame_slots:

            verb_id = self.choose_verb(
                frame=frame,
                scenario=scenario,
                tense=tense,
            )

            slots[
                "verb"
            ] = verb_id

        # ----------------------------------------------------
        # Remaining frame slots
        # ----------------------------------------------------

        for slot_name, slot_spec in (
            frame_slots.items()
        ):

            if slot_name in slots:
                continue

            if not isinstance(
                slot_spec,
                dict,
            ):
                continue

            fixed_id = (
                slot_spec.get(
                    "fixed_concept_id"
                )
            )

            if fixed_id:

                slots[
                    slot_name
                ] = fixed_id

                continue

            required = bool(
                slot_spec.get(
                    "required",
                    False,
                )
            )

            if not required:
                continue

            concept_types = (
                slot_spec.get(
                    "concept_types",
                    [],
                )
            )

            if not concept_types:

                raise RuntimeError(
                    f"Frame {frame_id}, "
                    f"slot {slot_name}: "
                    "no concept_types."
                )

            concept_type = (
                concept_types[0]
            )

            # ================================================
            # Object / destination are compatibility-aware.
            # ================================================

            if slot_name in {
                "object",
                "destination",
                "source",
                "location",
                "recipient",
            }:

                if not verb_id:

                    raise RuntimeError(
                        f"Frame {frame_id}: "
                        f"role {slot_name} "
                        "requires verb."
                    )

                slots[
                    slot_name
                ] = self.choose_role_concept(
                    verb_id=verb_id,
                    role=slot_name,
                    frame_slot=slot_spec,
                    concept_type=concept_type,
                )

            # ================================================
            # Generic place
            # ================================================

            elif concept_type == "place":

                candidates = (
                    self.concepts_of_type(
                        "place"
                    )
                )

                if not candidates:

                    raise RuntimeError(
                        "No place concepts."
                    )

                slots[
                    slot_name
                ] = self.rng.choice(
                    candidates
                )[
                    "id"
                ]

            # ================================================
            # Other generic type
            # ================================================

            else:

                candidates = (
                    self.concepts_of_type(
                        concept_type
                    )
                )

                if not candidates:

                    raise RuntimeError(
                        f"No concepts for "
                        f"type={concept_type}"
                    )

                slots[
                    slot_name
                ] = self.rng.choice(
                    candidates
                )[
                    "id"
                ]

        # ----------------------------------------------------
        # Clock
        # ----------------------------------------------------

        computed_fields = frame.get(
            "computed",
            [],
        )

        if (
            isinstance(
                computed_fields,
                list,
            )
            and "clock"
            in computed_fields
        ):

            computed[
                "clock"
            ] = self.choose_clock()

        # ----------------------------------------------------
        # Render
        # ----------------------------------------------------

        features = {
            "tense":
                tense,

            "polarity":
                polarity,
        }

        texts = self.renderer.render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            computed=computed,
        )

        return {
            "scenario":
                scenario,

            "frame_id":
                frame_id,

            "slots":
                slots,

            "features":
                features,

            "computed":
                computed,

            "texts":
                texts,
        }

    # ========================================================
    # Signature
    # ========================================================

    @staticmethod
    def semantic_signature(
        candidate: dict,
    ) -> str:

        payload = {
            "frame_id":
                candidate[
                    "frame_id"
                ],

            "slots":
                candidate[
                    "slots"
                ],

            "features":
                candidate[
                    "features"
                ],

            "computed":
                candidate[
                    "computed"
                ],
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        *,
        n: int,
        max_attempts: int,
    ) -> tuple[
        list[dict],
        dict,
    ]:

        rows = []

        signatures = set()

        attempts = 0

        render_errors = Counter()

        while (
            len(rows) < n
            and attempts < max_attempts
        ):

            attempts += 1

            try:

                candidate = (
                    self.create_candidate()
                )

            except (
                RuntimeError,
                RenderError,
            ) as exc:

                render_errors[
                    str(exc)
                ] += 1

                continue

            signature = (
                self.semantic_signature(
                    candidate
                )
            )

            if signature in signatures:
                continue

            signatures.add(
                signature
            )

            index = (
                len(rows)
                + 1
            )

            semantic_id = (
                f"sem_v04_"
                f"{index:08d}"
            )

            row = {
                "semantic_id":
                    semantic_id,

                "scenario":
                    candidate[
                        "scenario"
                    ],

                "frame_id":
                    candidate[
                        "frame_id"
                    ],

                "slots":
                    candidate[
                        "slots"
                    ],

                "features":
                    candidate[
                        "features"
                    ],

                "computed":
                    candidate[
                        "computed"
                    ],

                "texts":
                    candidate[
                        "texts"
                    ],

                "metadata": {
                    "resource_version":
                        RESOURCE_VERSION,

                    "generator_version":
                        GENERATOR_VERSION,

                    "renderer_version":
                        RENDERER_VERSION,

                    "compatibility_version":
                        self.resources
                        .compatibility
                        .get(
                            "version",
                            "0.4",
                        ),

                    "generation_policy_version":
                        self.resources
                        .policy
                        .get(
                            "version",
                            "0.4",
                        ),

                    "seed":
                        self.seed,

                    "generation_attempt":
                        attempts,

                    "generation_policy":
                        (
                            "scenario_first_"
                            "compatibility_aware"
                        ),
                },
            }

            rows.append(
                row
            )

            frame_id = (
                candidate[
                    "frame_id"
                ]
            )

            verb_id = (
                candidate[
                    "slots"
                ].get(
                    "verb"
                )
            )

            scenario = (
                candidate[
                    "scenario"
                ]
            )

            self.frame_usage[
                frame_id
            ] += 1

            self.scenario_usage[
                scenario
            ] += 1

            if verb_id:

                self.verb_usage[
                    verb_id
                ] += 1

            if (
                len(rows) % 10 == 0
                or len(rows) == n
            ):

                print(
                    f"{len(rows)}/{n}"
                    f" | attempts={attempts}"
                    f" | unique="
                    f"{len(signatures)}"
                )

        if len(rows) != n:

            top_errors = (
                render_errors
                .most_common(
                    10
                )
            )

            raise RuntimeError(
                "Failed to generate requested "
                f"{n} samples.\n"
                f"Generated={len(rows)}\n"
                f"Attempts={attempts}\n"
                f"Top errors={top_errors}"
            )

        stats = {
            "generated":
                len(rows),

            "attempts":
                attempts,

            "efficiency":
                (
                    len(rows)
                    / attempts
                    if attempts
                    else 0.0
                ),

            "duplicate_signatures":
                (
                    attempts
                    - len(rows)
                    - sum(
                        render_errors.values()
                    )
                ),

            "candidate_errors":
                sum(
                    render_errors.values()
                ),

            "top_candidate_errors":
                [
                    {
                        "reason":
                            reason,

                        "count":
                            count,
                    }
                    for reason, count
                    in render_errors.most_common(
                        20
                    )
                ],
        }

        return (
            rows,
            stats,
        )


# ============================================================
# Coverage
# ============================================================

def build_coverage_report(
    rows: list[dict],
    policy: dict,
) -> dict:

    counters = {
        "scenario":
            Counter(),

        "frame":
            Counter(),

        "verb":
            Counter(),

        "subject":
            Counter(),

        "object":
            Counter(),

        "destination":
            Counter(),

        "time":
            Counter(),

        "tense":
            Counter(),

        "polarity":
            Counter(),
    }

    for row in rows:

        counters[
            "scenario"
        ][
            row.get(
                "scenario",
                "NONE",
            )
        ] += 1

        counters[
            "frame"
        ][
            row.get(
                "frame_id",
                "NONE",
            )
        ] += 1

        slots = row.get(
            "slots",
            {},
        )

        features = row.get(
            "features",
            {},
        )

        for key in (
            "verb",
            "subject",
            "object",
            "destination",
            "time",
        ):

            value = slots.get(
                key
            )

            if value:

                counters[
                    key
                ][
                    value
                ] += 1

        # Clock frames use "day".
        day = slots.get(
            "day"
        )

        if day:

            counters[
                "time"
            ][
                day
            ] += 1

        counters[
            "tense"
        ][
            features.get(
                "tense",
                "NONE",
            )
        ] += 1

        counters[
            "polarity"
        ][
            features.get(
                "polarity",
                "NONE",
            )
        ] += 1

    total = len(
        rows
    )

    def ratios(
        counter: Counter,
    ) -> dict:

        return {
            key: (
                count
                / total
                if total
                else 0.0
            )
            for key, count
            in counter.items()
        }

    constraints = (
        policy.get(
            "coverage_constraints",
            {},
        )
    )

    violations = []

    checks = (
        (
            "verb",
            constraints.get(
                "max_single_verb_ratio"
            ),
        ),
        (
            "frame",
            constraints.get(
                "max_single_frame_ratio"
            ),
        ),
        (
            "scenario",
            constraints.get(
                "max_single_scenario_ratio"
            ),
        ),
    )

    for name, max_ratio in checks:

        if max_ratio is None:
            continue

        for key, ratio in ratios(
            counters[
                name
            ]
        ).items():

            if ratio > float(
                max_ratio
            ):

                violations.append({
                    "type":
                        name,

                    "id":
                        key,

                    "ratio":
                        ratio,

                    "max_ratio":
                        float(
                            max_ratio
                        ),
                })

    return {
        "samples":
            total,

        "counts": {
            name:
                dict(
                    counter.most_common()
                )
            for name, counter
            in counters.items()
        },

        "ratios": {
            name:
                ratios(
                    counter
                )
            for name, counter
            in counters.items()
        },

        "coverage_constraint_violations":
            violations,

        "note": (
            "Coverage constraint violations "
            "during a 100-sample smoke test "
            "are diagnostic only and do not "
            "automatically invalidate the corpus."
        ),
    }


# ============================================================
# Preview
# ============================================================

def print_preview(
    rows: list[dict],
    n: int,
) -> None:

    if n <= 0:
        return

    print()
    print(
        "=" * 100
    )
    print(
        "PREVIEW"
    )
    print(
        "=" * 100
    )

    for row in rows[
        :n
    ]:

        print()

        print(
            row[
                "semantic_id"
            ],
            "|",
            row[
                "scenario"
            ],
            "|",
            row[
                "frame_id"
            ],
        )

        print(
            "slots:",
            row[
                "slots"
            ],
        )

        print(
            "features:",
            row[
                "features"
            ],
        )

        print(
            "ZH:",
            row[
                "texts"
            ][
                "zh"
            ],
        )

        print(
            "EN:",
            row[
                "texts"
            ][
                "en"
            ],
        )

        print(
            "RU:",
            row[
                "texts"
            ][
                "ru"
            ],
        )

        print(
            "UZ:",
            row[
                "texts"
            ][
                "uz"
            ],
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "FourLang Synthetic "
            "Generator V0.4."
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
        default=2040,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--concepts",
        default=str(
            DEFAULT_CONCEPTS
        ),
    )

    parser.add_argument(
        "--frames",
        default=str(
            DEFAULT_FRAMES
        ),
    )

    parser.add_argument(
        "--compatibility",
        default=str(
            DEFAULT_COMPATIBILITY
        ),
    )

    parser.add_argument(
        "--policy",
        default=str(
            DEFAULT_POLICY
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    if args.n <= 0:

        raise ValueError(
            "--n must be > 0"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "semantic_v04_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v04_stats.json"
    )

    coverage_file = (
        output_dir
        / "semantic_v04_coverage.json"
    )

    resources = (
        V04GenerationResources(
            concepts_path=Path(
                args.concepts
            ),
            frames_path=Path(
                args.frames
            ),
            compatibility_path=Path(
                args.compatibility
            ),
            policy_path=Path(
                args.policy
            ),
        )
    )

    renderer = V04Renderer(
        concepts_path=args.concepts,
        frames_path=args.frames,
    )

    generator = (
        SyntheticGeneratorV04(
            resources=resources,
            renderer=renderer,
            seed=args.seed,
        )
    )

    max_attempts = (
        args.n
        * args.max_attempt_multiplier
    )

    print(
        "=" * 100
    )

    print(
        "SYNTHETIC GENERATOR V0.4"
    )

    print(
        "=" * 100
    )

    print(
        "Project root:",
        PROJECT_ROOT,
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
        "Resource version:",
        RESOURCE_VERSION,
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
        "Active concepts:",
        len(
            resources.active_concepts
        ),
    )

    print(
        "Active frames:",
        len(
            resources.active_frames
        ),
    )

    print(
        "Active verbs:",
        [
            concept["id"]
            for concept
            in resources.by_type.get(
                "verb",
                []
            )
        ],
    )

    print(
        "Output:",
        output_file,
    )

    print(
        "=" * 100
    )

    rows, stats = (
        generator.generate(
            n=args.n,
            max_attempts=max_attempts,
        )
    )

    coverage = (
        build_coverage_report(
            rows,
            resources.policy,
        )
    )

    write_jsonl(
        output_file,
        rows,
    )

    stats.update({
        "generator_version":
            GENERATOR_VERSION,

        "renderer_version":
            RENDERER_VERSION,

        "resource_version":
            RESOURCE_VERSION,

        "seed":
            args.seed,

        "output":
            str(
                output_file
            ),
    })

    write_json(
        stats_file,
        stats,
    )

    write_json(
        coverage_file,
        coverage,
    )

    print_preview(
        rows,
        args.preview,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "GENERATOR V0.4 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Generated:",
        stats[
            "generated"
        ],
    )

    print(
        "Attempts:",
        stats[
            "attempts"
        ],
    )

    print(
        "Efficiency:",
        (
            f"{stats['efficiency']:.2%}"
        ),
    )

    print(
        "Candidate errors:",
        stats[
            "candidate_errors"
        ],
    )

    print()

    print(
        "Verb distribution:"
    )

    for key, value in (
        coverage[
            "counts"
        ][
            "verb"
        ].items()
    ):

        print(
            f"{key:<15}"
            f"{value}"
        )

    print()

    print(
        "Frame distribution:"
    )

    for key, value in (
        coverage[
            "counts"
        ][
            "frame"
        ].items()
    ):

        print(
            f"{key:<25}"
            f"{value}"
        )

    print()

    print(
        "Coverage constraint violations:",
        len(
            coverage[
                "coverage_constraint_violations"
            ]
        ),
    )

    print()

    print(
        "Files:"
    )

    print(
        output_file
    )

    print(
        stats_file
    )

    print(
        coverage_file
    )


if __name__ == "__main__":

    main()