from __future__ import annotations

import argparse
from pathlib import Path

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
    DEFAULT_FRAMES,
    DEFAULT_COMPATIBILITY,
    V04GenerationResources,
    build_coverage_report,
    print_preview,
    write_json,
    write_jsonl,
)

from scripts.synthetic.generate_synthetic_v041 import (
    SyntheticGeneratorV041,
    build_balance_report,
)

from scripts.synthetic.renderer_v04 import (
    V04Renderer,
)


# ============================================================
# Version
# ============================================================

GENERATOR_VERSION = "0.4.3"
RENDERER_VERSION = "0.4"


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v043.jsonl"
)

DEFAULT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v043.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v043_regression_200"
)


CLOCK_FRAME_IDS = {
    "TRANSITIVE_CLOCK",
    "MOTION_CLOCK",
}


# ============================================================
# Generator V0.4.3
# ============================================================

class SyntheticGeneratorV043(
    SyntheticGeneratorV041
):
    """
    V0.4.3 linguistic-cleanup generator.

    Changes compared with V0.4.1:

    1. ARRIVE is future-only through concepts_v043.jsonl.

    2. *_CLOCK frames use future temporal context.

    3. WHERE_PLACE is a verbless frame and therefore
       does not retain meaningless present/future metadata.
    """

    # ========================================================
    # Clock time policy
    # ========================================================

    def choose_time(
        self,
        *,
        clock_frame: bool,
    ) -> str:
        """
        Normal frames keep the inherited time sampling.

        V0.4.3 clock frames use only future-compatible
        temporal concepts configured in clock_policy.
        """

        if not clock_frame:

            return super().choose_time(
                clock_frame=False,
            )

        clock_policy = (
            self.resources
            .policy
            .get(
                "clock_policy",
                {},
            )
        )

        force_future = bool(
            clock_policy.get(
                "force_future",
                False,
            )
        )

        if not force_future:

            return super().choose_time(
                clock_frame=True,
            )

        allowed_ids = (
            clock_policy.get(
                "allowed_day_ids",
                [
                    "TIME_TOMORROW",
                ],
            )
        )

        candidates = []

        for concept in self.concepts_of_type(
            "time"
        ):

            concept_id = concept.get(
                "id"
            )

            if concept_id not in allowed_ids:

                continue

            tense_hint = self.tense_from_time(
                concept_id
            )

            if tense_hint != "future":

                continue

            candidates.append(
                concept
            )

        if not candidates:

            raise RuntimeError(
                "V0.4.3 clock policy requires "
                "at least one active future time concept. "
                f"allowed_day_ids={allowed_ids}"
            )

        chosen = self.rng.choice(
            candidates
        )

        return chosen[
            "id"
        ]

    # ========================================================
    # Linguistic cleanup
    # ========================================================

    def apply_linguistic_cleanup(
        self,
        candidate: dict,
    ) -> dict:
        """
        Apply V0.4.3 semantic metadata constraints after
        the base candidate has been built.

        Important:
        actual language rendering is still performed by the
        verified V0.4 renderer. We only clean up semantic policy
        here.
        """

        frame_id = candidate.get(
            "frame_id"
        )

        slots = candidate.setdefault(
            "slots",
            {},
        )

        features = candidate.setdefault(
            "features",
            {},
        )

        computed = candidate.setdefault(
            "computed",
            {},
        )

        # ----------------------------------------------------
        # 1. WHERE_PLACE
        #
        # "Where is Beijing?" is verbless from the synthetic
        # semantic-frame perspective.
        #
        # Random "present/future" metadata is meaningless and
        # confused the residual linguistic judge.
        # ----------------------------------------------------

        if frame_id == "WHERE_PLACE":

            features.pop(
                "tense",
                None,
            )

            # Keep polarity only as a structural property.
            features[
                "polarity"
            ] = "pos"

        # ----------------------------------------------------
        # 2. CLOCK frames
        #
        # A precise clock expression in this version represents
        # a scheduled future event.
        # ----------------------------------------------------

        if frame_id in CLOCK_FRAME_IDS:

            day_id = slots.get(
                "day"
            )

            if day_id != "TIME_TOMORROW":

                raise RuntimeError(
                    f"{frame_id}: V0.4.3 requires "
                    "TIME_TOMORROW for clock frames, "
                    f"got {day_id!r}"
                )

            if features.get(
                "tense"
            ) != "future":

                raise RuntimeError(
                    f"{frame_id}: V0.4.3 requires "
                    "future tense for clock frames, "
                    f"got {features.get('tense')!r}"
                )

            if not computed.get(
                "clock"
            ):

                raise RuntimeError(
                    f"{frame_id}: missing computed clock."
                )

        # ----------------------------------------------------
        # 3. ARRIVE
        #
        # concepts_v043.jsonl restricts ARRIVE to future.
        # This second guard ensures no present ARRIVE candidate
        # can silently enter the corpus.
        # ----------------------------------------------------

        if slots.get(
            "verb"
        ) == "ARRIVE":

            if features.get(
                "tense"
            ) != "future":

                raise RuntimeError(
                    "ARRIVE is future-only in V0.4.3."
                )

        return candidate

    # ========================================================
    # Candidate creation
    # ========================================================

    def create_candidate(
        self,
    ) -> dict:

        candidate = super().create_candidate()

        candidate = (
            self.apply_linguistic_cleanup(
                candidate
            )
        )

        return candidate

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        *,
        n: int,
        max_attempts: int,
    ):
        """
        Reuse the verified quota-balancing generator,
        then update metadata to V0.4.3.
        """

        rows, stats = super().generate(
            n=n,
            max_attempts=max_attempts,
        )

        policy_version = (
            self.resources
            .policy
            .get(
                "version",
                "unknown",
            )
        )

        for row in rows:

            metadata = row.setdefault(
                "metadata",
                {},
            )

            metadata[
                "generator_version"
            ] = GENERATOR_VERSION

            metadata[
                "renderer_version"
            ] = RENDERER_VERSION

            metadata[
                "generation_policy_version"
            ] = policy_version

            metadata[
                "generation_policy"
            ] = (
                "quota_deficit_balancing_"
                "linguistic_cleanup"
            )

        stats[
            "generator_version"
        ] = GENERATOR_VERSION

        stats[
            "renderer_version"
        ] = RENDERER_VERSION

        stats[
            "generation_policy_version"
        ] = policy_version

        return (
            rows,
            stats,
        )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "FourLang Synthetic Generator V0.4.3"
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=100,
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

    concepts_path = Path(
        args.concepts
    )

    frames_path = Path(
        args.frames
    )

    compatibility_path = Path(
        args.compatibility
    )

    policy_path = Path(
        args.policy
    )

    # --------------------------------------------------------
    # Resource existence
    # --------------------------------------------------------

    required_files = [
        concepts_path,
        frames_path,
        compatibility_path,
        policy_path,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required V0.4.3 resource "
                f"not found: {path}"
            )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "semantic_v043_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v043_stats.json"
    )

    coverage_file = (
        output_dir
        / "semantic_v043_coverage.json"
    )

    balance_file = (
        output_dir
        / "semantic_v043_balance.json"
    )

    # --------------------------------------------------------
    # Resources
    # --------------------------------------------------------

    resources = (
        V04GenerationResources(
            concepts_path=concepts_path,
            frames_path=frames_path,
            compatibility_path=compatibility_path,
            policy_path=policy_path,
        )
    )

    renderer = V04Renderer(
        concepts_path=concepts_path,
        frames_path=frames_path,
    )

    generator = (
        SyntheticGeneratorV043(
            resources=resources,
            renderer=renderer,
            seed=args.seed,
        )
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print(
        "=" * 100
    )

    print(
        "SYNTHETIC GENERATOR V0.4.3"
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
        "Concepts:",
        concepts_path,
    )

    print(
        "Frames:",
        frames_path,
    )

    print(
        "Compatibility:",
        compatibility_path,
    )

    print(
        "Policy:",
        policy_path,
    )

    print(
        "Output:",
        output_file,
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    rows, stats = (
        generator.generate(
            n=args.n,
            max_attempts=(
                args.n
                * args.max_attempt_multiplier
            ),
        )
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print_preview(
        rows,
        args.preview,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "GENERATOR V0.4.3 COMPLETE"
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
            "attempts",
            0,
        ),
    )

    print(
        "Efficiency:",
        f"{stats.get('efficiency', 0):.2%}",
    )

    print(
        "Candidate errors:",
        stats.get(
            "candidate_errors",
            0,
        ),
    )

    print(
        "Balance fallbacks:",
        stats.get(
            "balance_fallback_count",
            0,
        ),
    )

    print()

    print(
        "Verb distribution:"
    )

    print(
        "-" * 60
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

    print(
        "-" * 60
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