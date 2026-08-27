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
    build_balance_report,
)

from scripts.synthetic.generate_synthetic_v044 import (
    SyntheticGeneratorV044,
)

from scripts.synthetic.renderer_v0441 import (
    V0441Renderer,
)


GENERATOR_VERSION = "0.4.4.1"
RENDERER_VERSION = "0.4.4.1"


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_POLICY = (
    RESOURCE_DIR
    / "generation_policy_v044.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v0441_regression_200"
)


class SyntheticGeneratorV0441(
    SyntheticGeneratorV044
):

    def apply_motion_event_cleanup(
            self,
            candidate: dict,
    ) -> dict:

        """
        V0.4.4.1 final motion-event cleanup.

        IMPORTANT:
        Do NOT call V044.apply_motion_event_cleanup() here.

        V044 performs an intermediate render. Since self.renderer
        is already V0441Renderer, invalid transient habitual
        structures would be rejected before V0441 has a chance
        to normalize them.

        Therefore V044 + V0441 rules are combined here and the
        candidate is rendered exactly once at the end.
        """

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

        verb_id = slots.get(
            "verb"
        )

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity",
            "pos",
        )

        destination = slots.get(
            "destination"
        )

        # ========================================================
        # Detect time slot
        # ========================================================

        time_slot_name = None

        if "time" in slots:
            time_slot_name = "time"

        elif "day" in slots:
            time_slot_name = "day"

        time_id = (
            slots.get(
                time_slot_name
            )
            if time_slot_name
            else None
        )

        # ========================================================
        # 0. TIME_EVERY_DAY must NOT leak into non-motion verbs.
        #
        # TIME_EVERY_DAY was introduced specifically for
        # GO / COME habitual motion in V0.4.4.
        #
        # Example to eliminate:
        #
        # I do not drink water every day.
        #
        # This introduces frequency-negation scope issues that are
        # outside the current semantic schema.
        # ========================================================

        if verb_id not in {
            "GO",
            "COME",
        }:

            if (
                    time_id == "TIME_EVERY_DAY"
                    and time_slot_name is not None
            ):

                if tense == "future":

                    slots[
                        time_slot_name
                    ] = "TIME_TOMORROW"

                else:

                    slots[
                        time_slot_name
                    ] = "TIME_TODAY"

            candidate[
                "texts"
            ] = self.renderer.render(
                frame_id=candidate[
                    "frame_id"
                ],
                slots=slots,
                features=features,
                computed=computed,
            )

            return candidate

        # ========================================================
        # GO / COME
        # ========================================================

        if tense == "present":

            # ----------------------------------------------------
            # TIME_NOW
            #
            # Full progressive is not implemented yet.
            #
            # Positive motion can become habitual IF the semantic
            # structure remains useful.
            #
            # Negative habitual is not allowed.
            #
            # GO without destination is also not allowed as
            # habitual because "I go every day" is context-poor.
            # ----------------------------------------------------

            if time_id == "TIME_NOW":

                can_be_habitual = (
                        polarity == "pos"
                        and not (
                        verb_id == "GO"
                        and destination is None
                )
                )

                if can_be_habitual:

                    slots[
                        time_slot_name
                    ] = "TIME_EVERY_DAY"

                    features[
                        "tense"
                    ] = "present"

                    features[
                        "event_type"
                    ] = "habitual"

                else:

                    slots[
                        time_slot_name
                    ] = "TIME_TOMORROW"

                    features[
                        "tense"
                    ] = "future"

                    features[
                        "event_type"
                    ] = "planned"

            # ----------------------------------------------------
            # TIME_EVERY_DAY
            # ----------------------------------------------------

            elif time_id == "TIME_EVERY_DAY":

                can_be_habitual = (
                        polarity == "pos"
                        and not (
                        verb_id == "GO"
                        and destination is None
                )
                )

                if can_be_habitual:

                    features[
                        "tense"
                    ] = "present"

                    features[
                        "event_type"
                    ] = "habitual"

                else:

                    if time_slot_name is not None:
                        slots[
                            time_slot_name
                        ] = "TIME_TOMORROW"

                    features[
                        "tense"
                    ] = "future"

                    features[
                        "event_type"
                    ] = "planned"

            # ----------------------------------------------------
            # TIME_TODAY
            #
            # Treat as one-off planned event.
            # ----------------------------------------------------

            elif time_id == "TIME_TODAY":

                if time_slot_name is not None:
                    slots[

                        time_slot_name

                    ] = "TIME_TOMORROW"

                features[

                    "tense"

                ] = "future"

                features[

                    "event_type"

                ] = "planned"
            # ----------------------------------------------------
            # Bare present motion
            # ----------------------------------------------------

            elif time_id is None:

                features[
                    "tense"
                ] = "future"

                features[
                    "event_type"
                ] = "planned"

            # ----------------------------------------------------
            # Other present temporal context
            #
            # e.g. TIME_NEXT_WEEK should obviously be future.
            # ----------------------------------------------------

            else:

                features[
                    "tense"
                ] = "future"

                features[
                    "event_type"
                ] = "planned"

        # ========================================================
        # Existing future motion
        # ========================================================

        elif tense == "future":

            features[
                "event_type"
            ] = "planned"

            # TIME_EVERY_DAY is reserved for habitual in this
            # version. Do not allow planned + EVERY_DAY.

            if (
                    time_id == "TIME_EVERY_DAY"
                    and time_slot_name is not None
            ):
                slots[
                    time_slot_name
                ] = "TIME_TOMORROW"

        else:

            raise RuntimeError(
                f"Unsupported GO/COME tense: {tense}"
            )

        # ========================================================
        # Final render
        #
        # Render ONCE after all semantic normalization.
        # ========================================================

        candidate[
            "texts"
        ] = self.renderer.render(
            frame_id=candidate[
                "frame_id"
            ],
            slots=slots,
            features=features,
            computed=computed,
        )

        return candidate
    def generate(
        self,
        *,
        n: int,
        max_attempts: int,
    ):

        rows, stats = super().generate(
            n=n,
            max_attempts=max_attempts,
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

        stats[
            "generator_version"
        ] = GENERATOR_VERSION

        stats[
            "renderer_version"
        ] = RENDERER_VERSION

        return rows, stats


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "FourLang Synthetic Generator V0.4.4.1"
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
        default=2053,
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
        default=20,
    )

    parser.add_argument(
        "--max-attempt-multiplier",
        type=int,
        default=100,
    )

    return parser.parse_args()


def main() -> None:

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "semantic_v0441_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v0441_stats.json"
    )

    coverage_file = (
        output_dir
        / "semantic_v0441_coverage.json"
    )

    balance_file = (
        output_dir
        / "semantic_v0441_balance.json"
    )

    resources = V04GenerationResources(
        concepts_path=DEFAULT_CONCEPTS,
        frames_path=DEFAULT_FRAMES,
        compatibility_path=DEFAULT_COMPATIBILITY,
        policy_path=DEFAULT_POLICY,
    )

    renderer = V0441Renderer(
        concepts_path=DEFAULT_CONCEPTS,
        frames_path=DEFAULT_FRAMES,
    )

    generator = SyntheticGeneratorV0441(
        resources=resources,
        renderer=renderer,
        seed=args.seed,
    )

    print(
        "=" * 100
    )

    print(
        "SYNTHETIC GENERATOR V0.4.4.1"
    )

    print(
        "=" * 100
    )

    rows, stats = generator.generate(
        n=args.n,
        max_attempts=(
            args.n
            * args.max_attempt_multiplier
        ),
    )

    coverage = build_coverage_report(
        rows,
        resources.policy,
    )

    balance = build_balance_report(
        rows=rows,
        stats=stats,
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
        "GENERATOR V0.4.4.1 COMPLETE"
    )

    print(
        "=" * 100
    )

    print(
        "Generated:",
        len(rows),
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
        "Coverage constraint violations:",
        len(
            coverage[
                "coverage_constraint_violations"
            ]
        ),
    )

    print()

    print(
        output_file
    )


if __name__ == "__main__":
    main()