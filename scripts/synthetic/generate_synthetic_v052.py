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
# Paths / Versions
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.2"
FROZEN_PARENT_VERSION = "0.5.1"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

ACTIVATION_FILE = (
    RESOURCE_DIR
    / "activation_v052.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v052_smoke_100"
)


# =============================================================================
# V0.5.2 capability whitelist
# =============================================================================

NEW_FRAMES = {
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
}

NEW_VERBS = {
    "CALL",
    "WAIT",
}


# =============================================================================
# Future capability blacklist
#
# These resources may already exist, but V0.5.2 must NOT generate them.
# =============================================================================

BLOCKED_VERBS = {
    "GIVE",
    "BRING",
    "LOSE",
    "RETURN",
    "NEED",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "LOSE_OBJECT",
    "RETURN_PLACE",
    "NEED_OBJECT",
    "LEAVE_PLACE",
}


# =============================================================================
# Participant overlap policy
#
# Exact reflexive:
#
# PERSON_I -> PERSON_I
# PERSON_WE -> PERSON_WE
#
# was already forbidden.
#
# Qwen targeted audit exposed another semantic issue:
#
# PERSON_I -> PERSON_WE
# PERSON_WE -> PERSON_I
#
# because speaker belongs to the "we/us" participant set.
#
# For the controlled synthetic corpus we intentionally disallow these
# speaker-group overlap cases until reflexive/group semantics are explicitly
# modeled by a later version.
# =============================================================================

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


def person_target_allowed(
    subject_id: str | None,
    person_id: str | None,
) -> bool:
    """
    Return True only if the person target is semantically safe for the
    current controlled CALL_PERSON / WAIT_PERSON frames.
    """

    if not subject_id or not person_id:
        return False

    # Exact reflexive.
    if subject_id == person_id:
        return False

    # Speaker/group overlap.
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
        encoding="utf-8",
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
        encoding="utf-8",
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
            "Activation version mismatch."
        )

    if (
        activation.get(
            "frozen_parent"
        )
        != FROZEN_PARENT_VERSION
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
            "V0.5.2 activation has not "
            "passed validation."
        )

    generation_policy = activation.get(
        "generation_policy",
        {},
    )

    if not generation_policy.get(
        "v052_generation_allowed",
        False,
    ):
        raise RuntimeError(
            "V0.5.2 generation is disabled."
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
            "Activation enabled_verbs mismatch. "
            f"expected={sorted(NEW_VERBS)}, "
            f"actual={sorted(enabled_verbs)}"
        )

    if enabled_frames != NEW_FRAMES:
        raise RuntimeError(
            "Activation enabled_frames mismatch. "
            f"expected={sorted(NEW_FRAMES)}, "
            f"actual={sorted(enabled_frames)}"
        )

    return activation


# =============================================================================
# Frozen V0.5.1 generation
# =============================================================================

def generate_frozen_parent(
    *,
    n: int,
    seed: int,
    temp_dir: Path,
) -> list[dict]:

    if temp_dir.exists():
        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.generate_synthetic_v051",
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--output-dir",
        str(temp_dir),
    ]

    print()
    print("=" * 100)
    print("RUNNING FROZEN V0.5.1 GENERATOR")
    print("=" * 100)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

    exact_file = (
        temp_dir
        / "semantic_v051_raw.jsonl"
    )

    if exact_file.exists():
        parent_file = exact_file

    else:
        candidates = sorted(
            temp_dir.rglob(
                "semantic_*raw.jsonl"
            )
        )

        if not candidates:
            raise RuntimeError(
                "Frozen V0.5.1 generator completed "
                "but no raw JSONL output was found."
            )

        parent_file = candidates[0]

    rows = read_jsonl(
        parent_file
    )

    if len(rows) != n:
        raise RuntimeError(
            "Frozen parent row count mismatch: "
            f"expected={n}, "
            f"actual={len(rows)}"
        )

    print(
        "Frozen parent rows:",
        len(rows),
    )

    print(
        "Frozen parent source:",
        parent_file,
    )

    return rows


# =============================================================================
# Basic sampling policy
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

    # Approximately 70% positive / 30% negative.
    return (
        "pos"
        if rng.random() < 0.70
        else "neg"
    )


def semantic_signature(
    *,
    frame_id: str,
    slots: dict,
    features: dict,
) -> str:

    payload = {
        "frame_id": frame_id,
        "slots": slots,
        "features": features,
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )


def frame_targets(
    total_new: int,
) -> dict[str, int]:

    frames = [
        "CALL_PERSON",
        "WAIT_PERSON",
        "WAIT_AT_PLACE",
    ]

    base = (
        total_new
        // len(frames)
    )

    remainder = (
        total_new
        % len(frames)
    )

    result = {
        frame_id: base
        for frame_id in frames
    }

    for frame_id in frames[
        :remainder
    ]:
        result[
            frame_id
        ] += 1

    return result


# =============================================================================
# Generate new V0.5.2 rows
# =============================================================================

def build_new_rows(
    *,
    total_new: int,
    seed: int,
) -> tuple[list[dict], int]:

    from scripts.synthetic.renderer_v052 import (
        V052Renderer,
    )

    renderer = V052Renderer()

    rng = random.Random(
        seed
    )

    subject_ids = sorted(
        renderer.subjects.keys()
    )

    person_ids = sorted(
        renderer.person_targets.keys()
    )

    place_ids = sorted(
        renderer.place_targets.keys()
    )

    if not subject_ids:
        raise RuntimeError(
            "No subjects available."
        )

    if not person_ids:
        raise RuntimeError(
            "No person targets available."
        )

    if not place_ids:
        raise RuntimeError(
            "No place targets available."
        )

    targets = frame_targets(
        total_new
    )

    rows: list[dict] = []
    signatures: set[str] = set()

    attempts = 0

    max_attempts = max(
        5000,
        total_new * 300,
    )

    frame_counter = Counter()

    while (
        len(rows) < total_new
        and attempts < max_attempts
    ):
        attempts += 1

        unfinished_frames = [
            frame_id
            for frame_id, target
            in targets.items()
            if frame_counter[
                frame_id
            ] < target
        ]

        if not unfinished_frames:
            break

        # Prioritize the least-filled frame.
        unfinished_frames.sort(
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
            unfinished_frames[0]
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
            "tense": tense,
            "polarity": polarity,
        }

        # =====================================================================
        # CALL_PERSON
        # =====================================================================

        if frame_id == "CALL_PERSON":

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

            target_person = rng.choice(
                valid_people
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "CALL",

                "person":
                    target_person,
            }

            scenario = (
                "communication"
            )

        # =====================================================================
        # WAIT_PERSON
        # =====================================================================

        elif frame_id == "WAIT_PERSON":

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

            target_person = rng.choice(
                valid_people
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "WAIT",

                "person":
                    target_person,
            }

            scenario = rng.choice(
                [
                    "daily",
                    "travel",
                    "communication",
                ]
            )

        # =====================================================================
        # WAIT_AT_PLACE
        # =====================================================================

        elif frame_id == "WAIT_AT_PLACE":

            place_id = rng.choice(
                place_ids
            )

            slots = {
                "subject":
                    subject_id,

                "verb":
                    "WAIT",

                "place":
                    place_id,
            }

            scenario = rng.choice(
                [
                    "travel",
                    "transport",
                    "daily",
                ]
            )

        else:
            raise RuntimeError(
                f"Unexpected V0.5.2 frame: "
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
            # Renderer is also a defensive gate.
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
                    FROZEN_PARENT_VERSION,

                "generator_version":
                    VERSION,

                "renderer_version":
                    VERSION,

                "resource_version":
                    VERSION,

                "generation_source":
                    "v052_new_frame",

                "batch":
                    "Batch-2A",
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
            "Failed to generate requested "
            "V0.5.2 new rows. "
            f"requested={total_new}, "
            f"generated={len(rows)}, "
            f"attempts={attempts}"
        )

    return (
        rows,
        attempts,
    )


# =============================================================================
# Parent metadata normalization
# =============================================================================

def prepare_parent_rows(
    rows: list[dict],
) -> list[dict]:

    result = []

    for row in rows:

        output_row = copy.deepcopy(
            row
        )

        old_metadata = output_row.get(
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
            FROZEN_PARENT_VERSION,
        )

        metadata[
            "generator_version"
        ] = VERSION

        metadata[
            "generation_source"
        ] = "frozen_parent_v051"

        metadata[
            "frozen_parent_version"
        ] = FROZEN_PARENT_VERSION

        output_row[
            "metadata"
        ] = metadata

        result.append(
            output_row
        )

    return result


# =============================================================================
# Final generation contract
# =============================================================================

def validate_final_contract(
    rows: list[dict],
    expected_total: int,
) -> list[str]:

    errors: list[str] = []

    if len(rows) != expected_total:
        errors.append(
            f"Row count mismatch: "
            f"{len(rows)} != {expected_total}"
        )

    semantic_ids = [
        row.get(
            "semantic_id"
        )
        for row in rows
    ]

    if len(
        set(semantic_ids)
    ) != len(
        semantic_ids
    ):
        errors.append(
            "Duplicate semantic_id detected."
        )

    for row in rows:

        semantic_id = row.get(
            "semantic_id"
        )

        frame_id = row.get(
            "frame_id"
        )

        slots = row.get(
            "slots",
            {},
        )

        if not isinstance(
            slots,
            dict,
        ):
            errors.append(
                f"{semantic_id}: invalid slots"
            )
            continue

        verb_id = slots.get(
            "verb"
        )

        texts = row.get(
            "texts",
            {},
        )

        # ---------------------------------------------------------------------
        # Block future capabilities
        # ---------------------------------------------------------------------

        if verb_id in BLOCKED_VERBS:
            errors.append(
                f"{semantic_id}: "
                f"blocked verb leaked: "
                f"{verb_id}"
            )

        if frame_id in BLOCKED_FRAMES:
            errors.append(
                f"{semantic_id}: "
                f"blocked frame leaked: "
                f"{frame_id}"
            )

        # ---------------------------------------------------------------------
        # CALL / WAIT person role safety
        # ---------------------------------------------------------------------

        if frame_id in {
            "CALL_PERSON",
            "WAIT_PERSON",
        }:

            subject_id = slots.get(
                "subject"
            )

            person_id = slots.get(
                "person"
            )

            if not person_target_allowed(
                subject_id,
                person_id,
            ):
                errors.append(
                    f"{semantic_id}: "
                    "person target semantic overlap "
                    f"not allowed: "
                    f"{subject_id} -> {person_id}"
                )

        # ---------------------------------------------------------------------
        # Fixed frame / verb contract
        # ---------------------------------------------------------------------

        if frame_id == "CALL_PERSON":

            if verb_id != "CALL":
                errors.append(
                    f"{semantic_id}: "
                    "CALL_PERSON verb mismatch"
                )

        elif frame_id in {
            "WAIT_PERSON",
            "WAIT_AT_PLACE",
        }:

            if verb_id != "WAIT":
                errors.append(
                    f"{semantic_id}: "
                    f"{frame_id} verb mismatch"
                )

        # ---------------------------------------------------------------------
        # Four-language completeness
        # ---------------------------------------------------------------------

        if not isinstance(
            texts,
            dict,
        ):
            errors.append(
                f"{semantic_id}: "
                "invalid texts container"
            )

        elif set(
            texts.keys()
        ) != {
            "zh",
            "en",
            "ru",
            "uz",
        }:
            errors.append(
                f"{semantic_id}: "
                "missing language text"
            )

    return errors


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Generator "
            "V0.5.2 Batch-2A"
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
        default=2052,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--new-ratio",
        type=float,
        default=0.30,
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

    activation = load_activation()

    output_dir = (
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "semantic_v052_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v052_stats.json"
    )

    temp_parent_dir = (
        output_dir
        / "_frozen_parent_v051"
    )

    new_n = max(
        3,
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
        "SYNTHETIC GENERATOR V0.5.2"
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
        "Frozen parent rows:",
        parent_n,
    )

    print(
        "New V0.5.2 rows:",
        new_n,
    )

    print(
        "New ratio:",
        f"{new_n / args.n:.2%}",
    )

    print(
        "Activation:",
        ACTIVATION_FILE,
    )

    print(
        "Enabled frames:",
        activation[
            "enabled_frames"
        ],
    )

    print(
        "Participant overlap blocked:",
        sorted(
            PARTICIPANT_OVERLAP_PAIRS
        ),
    )

    # =========================================================================
    # Frozen parent
    # =========================================================================

    parent_rows_raw = generate_frozen_parent(
        n=parent_n,
        seed=args.seed,
        temp_dir=temp_parent_dir,
    )

    parent_rows = prepare_parent_rows(
        parent_rows_raw
    )

    # =========================================================================
    # V0.5.2 new rows
    # =========================================================================

    (
        new_rows,
        new_attempts,
    ) = build_new_rows(
        total_new=new_n,
        seed=args.seed + 1000,
    )

    rows = (
        parent_rows
        + new_rows
    )

    rng = random.Random(
        args.seed + 2000
    )

    rng.shuffle(
        rows
    )

    # Re-index the consolidated corpus.
    for index, row in enumerate(
        rows,
        start=1,
    ):
        row[
            "semantic_id"
        ] = (
            f"sem_v052_"
            f"{index:08d}"
        )

    # =========================================================================
    # Contract
    # =========================================================================

    contract_errors = validate_final_contract(
        rows,
        args.n,
    )

    if contract_errors:

        print()
        print("CONTRACT ERRORS")
        print("-" * 100)

        for error in contract_errors[
            :50
        ]:
            print(
                "ERROR:",
                error,
            )

        raise RuntimeError(
            f"V0.5.2 generation contract "
            f"failed with "
            f"{len(contract_errors)} errors."
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    frame_counter = Counter(
        row.get(
            "frame_id"
        )
        for row in rows
    )

    verb_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb",
            "NO_VERB",
        )
        for row in rows
    )

    source_counter = Counter(
        row.get(
            "metadata",
            {},
        ).get(
            "generation_source",
            "UNKNOWN",
        )
        for row in rows
    )

    new_frame_counter = Counter(
        row.get(
            "frame_id"
        )
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    )

    new_verb_counter = Counter(
        row.get(
            "slots",
            {},
        ).get(
            "verb"
        )
        for row in rows
        if row.get(
            "frame_id"
        ) in NEW_FRAMES
    )

    # =========================================================================
    # Write
    # =========================================================================

    write_jsonl(
        output_file,
        rows,
    )

    stats = {
        "generator_version":
            VERSION,

        "frozen_parent_version":
            FROZEN_PARENT_VERSION,

        "seed":
            args.seed,

        "samples":
            args.n,

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

        "frame_distribution":
            dict(
                frame_counter
            ),

        "verb_distribution":
            dict(
                verb_counter
            ),

        "source_distribution":
            dict(
                source_counter
            ),

        "new_frame_distribution":
            dict(
                new_frame_counter
            ),

        "new_verb_distribution":
            dict(
                new_verb_counter
            ),

        "activation_file":
            str(
                ACTIVATION_FILE
            ),
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
        "GENERATOR V0.5.2 COMPLETE"
    )
    print("=" * 100)

    print(
        "Generated:",
        len(rows),
    )

    print(
        "Frozen parent rows:",
        parent_n,
    )

    print(
        "New rows:",
        new_n,
    )

    print(
        "New generation attempts:",
        new_attempts,
    )

    print(
        "Contract errors:",
        len(
            contract_errors
        ),
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

    print("Files:")
    print(output_file)
    print(stats_file)

    print()
    print("=" * 100)
    print(
        "V0.5.2 GENERATION CONTRACT PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()