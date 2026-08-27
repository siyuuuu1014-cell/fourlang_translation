from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
    DEFAULT_CONCEPTS,
    DEFAULT_FRAMES,
    DEFAULT_COMPATIBILITY,
    V04GenerationResources,
    SyntheticGeneratorV04,
    build_coverage_report,
    print_preview,
    weighted_choice,
    write_json,
    write_jsonl,
)
from scripts.synthetic.renderer_v04 import V04Renderer


# ============================================================
# Version
# ============================================================

GENERATOR_VERSION = "0.4.1"
RENDERER_VERSION = "0.4"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v041.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v041_balance_100"
)


# ============================================================
# Quota helpers
# ============================================================

def build_quotas(
    ratios: dict[str, float],
    sample_count: int,
) -> dict[str, int]:
    """
    Largest-remainder allocation.

    Example:
        ratios sum to 1.00 -> allocate N samples
        ratios sum to 0.85 -> allocate roughly 0.85 * N
    """

    if sample_count <= 0:
        raise ValueError(
            "sample_count must be > 0"
        )

    if not ratios:
        return {}

    clean = {
        str(key): float(value)
        for key, value in ratios.items()
    }

    for key, value in clean.items():

        if value < 0:
            raise ValueError(
                f"Negative ratio: {key}={value}"
            )

    raw = {
        key:
            sample_count * ratio
        for key, ratio in clean.items()
    }

    quotas = {
        key:
            int(
                math.floor(value)
            )
        for key, value in raw.items()
    }

    desired_total = int(
        round(
            sum(
                raw.values()
            )
        )
    )

    current_total = sum(
        quotas.values()
    )

    remaining = (
        desired_total
        - current_total
    )

    fractional = sorted(
        clean.keys(),
        key=lambda key: (
            raw[key]
            - math.floor(
                raw[key]
            )
        ),
        reverse=True,
    )

    for key in fractional[
        :remaining
    ]:

        quotas[
            key
        ] += 1

    return quotas


def validate_target_ratios(
    name: str,
    ratios: dict[str, float],
    *,
    expected_total: float | None = None,
) -> None:

    if not isinstance(
        ratios,
        dict,
    ) or not ratios:

        raise RuntimeError(
            f"{name} must be a non-empty object."
        )

    total = sum(
        float(value)
        for value in ratios.values()
    )

    if expected_total is not None:

        if abs(
            total
            - expected_total
        ) > 1e-6:

            raise RuntimeError(
                f"{name} must sum to "
                f"{expected_total}, got {total}"
            )


# ============================================================
# Balanced generator
# ============================================================

class SyntheticGeneratorV041(
    SyntheticGeneratorV04
):

    def __init__(
        self,
        *,
        resources: V04GenerationResources,
        renderer: V04Renderer,
        seed: int,
    ) -> None:

        super().__init__(
            resources=resources,
            renderer=renderer,
            seed=seed,
        )

        self.scenario_quotas: dict[
            str,
            int,
        ] = {}

        self.frame_quotas: dict[
            str,
            int,
        ] = {}

        self.verb_quotas: dict[
            str,
            int,
        ] = {}

        self._pending_frame: dict | None = None

        self.balance_fallback_count = 0

    # ========================================================
    # Prepare quotas
    # ========================================================

    def prepare_balance(
        self,
        n: int,
    ) -> None:

        policy = self.resources.policy

        scenario_ratios = (
            policy.get(
                "scenario_weights",
                {},
            )
        )

        frame_ratios = (
            policy.get(
                "frame_target_ratios",
                {},
            )
        )

        verb_ratios = (
            policy.get(
                "verb_target_ratios",
                {},
            )
        )

        validate_target_ratios(
            "scenario_weights",
            scenario_ratios,
            expected_total=1.0,
        )

        validate_target_ratios(
            "frame_target_ratios",
            frame_ratios,
            expected_total=1.0,
        )

        validate_target_ratios(
            "verb_target_ratios",
            verb_ratios,
        )

        self.scenario_quotas = (
            build_quotas(
                scenario_ratios,
                n,
            )
        )

        self.frame_quotas = (
            build_quotas(
                frame_ratios,
                n,
            )
        )

        self.verb_quotas = (
            build_quotas(
                verb_ratios,
                n,
            )
        )

        active_frame_ids = {
            frame["id"]
            for frame
            in self.resources.active_frames
        }

        unknown_frames = (
            set(
                self.frame_quotas
            )
            - active_frame_ids
        )

        if unknown_frames:

            raise RuntimeError(
                "Frame target contains "
                "inactive/unknown frames: "
                f"{sorted(unknown_frames)}"
            )

        active_verbs = {
            concept["id"]
            for concept
            in self.resources.by_type.get(
                "verb",
                []
            )
        }

        unknown_verbs = (
            set(
                self.verb_quotas
            )
            - active_verbs
        )

        if unknown_verbs:

            raise RuntimeError(
                "Verb target contains "
                "inactive/unknown verbs: "
                f"{sorted(unknown_verbs)}"
            )

    # ========================================================
    # Quota remaining
    # ========================================================

    @staticmethod
    def remaining(
        quota: int,
        used: int,
    ) -> int:

        return max(
            quota - used,
            0,
        )

    # ========================================================
    # Verb/frame constraints
    # ========================================================

    def verb_matches_frame(
        self,
        verb: dict,
        frame: dict,
        tense: str,
    ) -> bool:

        if not super().verb_matches_frame(
            verb,
            frame,
            tense,
        ):

            return False

        constraints = (
            self.resources
            .policy
            .get(
                "verb_frame_constraints",
                {},
            )
            .get(
                verb["id"],
                {},
            )
        )

        allowed = constraints.get(
            "allowed_frame_ids"
        )

        if (
            isinstance(
                allowed,
                list,
            )
            and allowed
            and frame["id"]
            not in allowed
        ):

            return False

        blocked = constraints.get(
            "blocked_frame_ids"
        )

        if (
            isinstance(
                blocked,
                list,
            )
            and frame["id"]
            in blocked
        ):

            return False

        return True

    # ========================================================
    # Joint Scenario + Frame
    # ========================================================

    def choose_scenario(
        self,
    ) -> str:

        candidates = []

        # ----------------------------------------------------
        # First preference:
        # scenario AND frame both below quota
        # ----------------------------------------------------

        for scenario, scenario_quota in (
            self.scenario_quotas.items()
        ):

            scenario_remaining = (
                self.remaining(
                    scenario_quota,
                    self.scenario_usage[
                        scenario
                    ],
                )
            )

            if scenario_remaining <= 0:
                continue

            for frame in (
                self.resources.active_frames
            ):

                if scenario not in frame.get(
                    "scenario_tags",
                    [],
                ):
                    continue

                frame_id = frame[
                    "id"
                ]

                frame_quota = (
                    self.frame_quotas.get(
                        frame_id,
                        0,
                    )
                )

                frame_remaining = (
                    self.remaining(
                        frame_quota,
                        self.frame_usage[
                            frame_id
                        ],
                    )
                )

                if frame_remaining <= 0:
                    continue

                scenario_deficit = (
                    scenario_remaining
                    / max(
                        scenario_quota,
                        1,
                    )
                )

                frame_deficit = (
                    frame_remaining
                    / max(
                        frame_quota,
                        1,
                    )
                )

                score = (
                    scenario_deficit
                    * frame_deficit
                )

                candidates.append(
                    (
                        scenario,
                        frame,
                        score,
                    )
                )

        # ----------------------------------------------------
        # Fallback:
        # quota combination became temporarily infeasible
        # ----------------------------------------------------

        if not candidates:

            self.balance_fallback_count += 1

            for scenario, scenario_quota in (
                self.scenario_quotas.items()
            ):

                scenario_remaining = (
                    self.remaining(
                        scenario_quota,
                        self.scenario_usage[
                            scenario
                        ],
                    )
                )

                if scenario_remaining <= 0:
                    continue

                for frame in (
                    self.resources.active_frames
                ):

                    if scenario not in frame.get(
                        "scenario_tags",
                        [],
                    ):
                        continue

                    frame_id = frame[
                        "id"
                    ]

                    frame_quota = (
                        self.frame_quotas.get(
                            frame_id,
                            1,
                        )
                    )

                    frame_ratio = (
                        self.frame_usage[
                            frame_id
                        ]
                        / max(
                            frame_quota,
                            1,
                        )
                    )

                    score = (
                        1.0
                        / (
                            1.0
                            + frame_ratio
                        )
                    )

                    candidates.append(
                        (
                            scenario,
                            frame,
                            score,
                        )
                    )

        # ----------------------------------------------------
        # Last-resort fallback
        # ----------------------------------------------------

        if not candidates:

            self.balance_fallback_count += 1

            for frame in (
                self.resources.active_frames
            ):

                for scenario in frame.get(
                    "scenario_tags",
                    [],
                ):

                    candidates.append(
                        (
                            scenario,
                            frame,
                            1.0,
                        )
                    )

        if not candidates:

            raise RuntimeError(
                "No scenario/frame candidates."
            )

        selected = weighted_choice(
            self.rng,
            candidates,
            [
                item[2]
                for item
                in candidates
            ],
        )

        scenario = selected[
            0
        ]

        self._pending_frame = selected[
            1
        ]

        return scenario

    def choose_frame(
        self,
        scenario: str,
    ) -> dict:

        if (
            self._pending_frame
            is not None
        ):

            frame = (
                self._pending_frame
            )

            self._pending_frame = None

            if scenario not in frame.get(
                "scenario_tags",
                [],
            ):

                raise RuntimeError(
                    "Internal scenario/frame "
                    "pair mismatch."
                )

            return frame

        return super().choose_frame(
            scenario
        )

    # ========================================================
    # Balanced Verb choice
    # ========================================================

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
                "has no verb slot."
            )

        fixed_id = (
            verb_slot.get(
                "fixed_concept_id"
            )
        )

        # ----------------------------------------------------
        # Dedicated fixed verb such as WANT_OBJECT
        # ----------------------------------------------------

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
                    f"is not compatible with "
                    f"frame={frame['id']} "
                    f"tense={tense}"
                )

            return fixed_id

        # ----------------------------------------------------
        # Generic candidates
        # ----------------------------------------------------

        candidates = [
            verb
            for verb
            in self.concepts_of_type(
                "verb"
            )
            if (
                verb["id"]
                in self.verb_quotas
                and self.verb_matches_frame(
                    verb,
                    frame,
                    tense,
                )
            )
        ]

        if not candidates:

            raise RuntimeError(
                "No balanced verb candidate "
                f"for frame={frame['id']}, "
                f"tense={tense}"
            )

        # ----------------------------------------------------
        # Prefer verbs that remain below quota
        # ----------------------------------------------------

        under_quota = []

        for verb in candidates:

            verb_id = verb[
                "id"
            ]

            quota = (
                self.verb_quotas[
                    verb_id
                ]
            )

            used = (
                self.verb_usage[
                    verb_id
                ]
            )

            remaining = (
                self.remaining(
                    quota,
                    used,
                )
            )

            if remaining > 0:

                under_quota.append(
                    (
                        verb,
                        remaining,
                    )
                )

        if under_quota:

            items = [
                item[
                    0
                ]
                for item
                in under_quota
            ]

            weights = []

            for verb, remaining in (
                under_quota
            ):

                quota = (
                    self.verb_quotas[
                        verb["id"]
                    ]
                )

                deficit_ratio = (
                    remaining
                    / max(
                        quota,
                        1,
                    )
                )

                # Scenario tags remain a SOFT bonus,
                # never a hard filter.
                scenario_bonus = (
                    1.15
                    if scenario
                    in verb.get(
                        "scenario_tags",
                        [],
                    )
                    else 1.0
                )

                weights.append(
                    deficit_ratio
                    * scenario_bonus
                )

            chosen = weighted_choice(
                self.rng,
                items,
                weights,
            )

            return chosen[
                "id"
            ]

        # ----------------------------------------------------
        # All eligible verbs reached quota.
        # Choose the smallest usage/quota ratio.
        # ----------------------------------------------------

        scored = []

        for verb in candidates:

            verb_id = verb[
                "id"
            ]

            quota = max(
                self.verb_quotas[
                    verb_id
                ],
                1,
            )

            ratio = (
                self.verb_usage[
                    verb_id
                ]
                / quota
            )

            scored.append(
                (
                    ratio,
                    self.rng.random(),
                    verb,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        return scored[
            0
        ][
            2
        ][
            "id"
        ]

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        *,
        n: int,
        max_attempts: int,
    ):

        self.prepare_balance(
            n
        )

        rows, stats = super().generate(
            n=n,
            max_attempts=max_attempts,
        )

        # Parent is V0.4.0, so update version metadata.
        for row in rows:

            metadata = row.setdefault(
                "metadata",
                {},
            )

            metadata[
                "generator_version"
            ] = GENERATOR_VERSION

            metadata[
                "generation_policy_version"
            ] = (
                self.resources
                .policy
                .get(
                    "version",
                    "unknown",
                )
            )

            metadata[
                "generation_policy"
            ] = (
                "quota_deficit_balancing"
            )

        stats[
            "generator_version"
        ] = GENERATOR_VERSION
        stats[
            "generation_policy_version"
        ] = (
            self.resources
            .policy
            .get(
                "version",
                "unknown",
            )
        )
        stats[
            "balance_fallback_count"
        ] = (
            self.balance_fallback_count
        )

        stats[
            "scenario_quotas"
        ] = dict(
            self.scenario_quotas
        )

        stats[
            "frame_quotas"
        ] = dict(
            self.frame_quotas
        )

        stats[
            "verb_quotas"
        ] = dict(
            self.verb_quotas
        )

        return (
            rows,
            stats,
        )


# ============================================================
# Balance report
# ============================================================

def build_balance_report(
    *,
    rows: list[dict],
    stats: dict,
) -> dict:

    actual_scenarios = {}

    actual_frames = {}

    actual_verbs = {}

    for row in rows:

        scenario = row.get(
            "scenario"
        )

        frame_id = row.get(
            "frame_id"
        )

        verb_id = (
            row.get(
                "slots",
                {},
            ).get(
                "verb"
            )
        )

        if scenario:

            actual_scenarios[
                scenario
            ] = (
                actual_scenarios.get(
                    scenario,
                    0,
                )
                + 1
            )

        if frame_id:

            actual_frames[
                frame_id
            ] = (
                actual_frames.get(
                    frame_id,
                    0,
                )
                + 1
            )

        if verb_id:

            actual_verbs[
                verb_id
            ] = (
                actual_verbs.get(
                    verb_id,
                    0,
                )
                + 1
            )

    def compare(
        target: dict,
        actual: dict,
    ) -> dict:

        ids = (
            set(
                target
            )
            | set(
                actual
            )
        )

        return {
            key: {
                "target":
                    target.get(
                        key,
                        0,
                    ),

                "actual":
                    actual.get(
                        key,
                        0,
                    ),

                "delta":
                    (
                        actual.get(
                            key,
                            0,
                        )
                        - target.get(
                            key,
                            0,
                        )
                    ),
            }
            for key in sorted(
                ids
            )
        }

    return {
        "samples":
            len(
                rows
            ),

        "balance_fallback_count":
            stats.get(
                "balance_fallback_count",
                0,
            ),

        "scenario":
            compare(
                stats.get(
                    "scenario_quotas",
                    {},
                ),
                actual_scenarios,
            ),

        "frame":
            compare(
                stats.get(
                    "frame_quotas",
                    {},
                ),
                actual_frames,
            ),

        "verb":
            compare(
                stats.get(
                    "verb_quotas",
                    {},
                ),
                actual_verbs,
            ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "FourLang Synthetic "
            "Generator V0.4.1 "
            "with quota balancing."
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
        default=2042,
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
        default=15,
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
        / "semantic_v041_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v041_stats.json"
    )

    coverage_file = (
        output_dir
        / "semantic_v041_coverage.json"
    )

    balance_file = (
        output_dir
        / "semantic_v041_balance.json"
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
        SyntheticGeneratorV041(
            resources=resources,
            renderer=renderer,
            seed=args.seed,
        )
    )

    print(
        "=" * 100
    )

    print(
        "SYNTHETIC GENERATOR V0.4.1"
    )

    print(
        "=" * 100
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
        "Policy:",
        args.policy,
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
            max_attempts=(
                args.n
                * args.max_attempt_multiplier
            ),
        )
    )

    coverage = (
        build_coverage_report(
            rows,
            resources.policy,
        )
    )

    balance = (
        build_balance_report(
            rows=rows,
            stats=stats,
        )
    )

    write_jsonl(
        output_file,
        rows,
    )

    write_json(
        stats_file,
        stats,
    )

    write_json(
        coverage_file,
        coverage,
    )

    write_json(
        balance_file,
        balance,
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
        "GENERATOR V0.4.1 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Generated:",
        len(
            rows
        ),
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
            f"{stats.get('efficiency', 0):.2%}"
        ),
    )

    print(
        "Candidate errors:",
        stats.get(
            "candidate_errors"
        ),
    )

    print(
        "Balance fallbacks:",
        stats.get(
            "balance_fallback_count"
        ),
    )

    print()

    print(
        "Verb distribution:"
    )

    for verb_id, count in (
        coverage[
            "counts"
        ][
            "verb"
        ].items()
    ):

        print(
            f"{verb_id:<15}"
            f"{count}"
        )

    print()

    print(
        "Frame distribution:"
    )

    for frame_id, count in (
        coverage[
            "counts"
        ][
            "frame"
        ].items()
    ):

        print(
            f"{frame_id:<25}"
            f"{count}"
        )

    print()

    violations = (
        coverage[
            "coverage_constraint_violations"
        ]
    )

    print(
        "Coverage constraint violations:",
        len(
            violations
        ),
    )

    if violations:

        for item in violations:

            print(
                " -",
                item,
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

    print(
        balance_file
    )


if __name__ == "__main__":

    main()