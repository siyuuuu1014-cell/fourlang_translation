from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.synthetic.generate_synthetic_v04 import PROJECT_ROOT
from scripts.synthetic.renderer_v051 import V051Renderer


# ============================================================
# Version
# ============================================================

GENERATOR_VERSION = "0.5.1"
RENDERER_VERSION = "0.5.1"
FROZEN_CORE_VERSION = "0.4.4.1"


# ============================================================
# Paths
# ============================================================

V05_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

DEFAULT_POLICY = (
    V05_RESOURCE_DIR
    / "generation_policy_v051.json"
)

DEFAULT_COMPATIBILITY = (
    V05_RESOURCE_DIR
    / "semantic_compatibility_v05.json"
)

DEFAULT_ARGUMENTS = (
    V05_RESOURCE_DIR
    / "argument_realization_v051.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
    / "v051_smoke_100"
)


# ============================================================
# V0.5.1 allowed contract
# ============================================================

NEW_VERB_TO_FRAME = {
    "SEE": "SEE_OBJECT",
    "TAKE": "TAKE_OBJECT",
}

ALLOWED_NEW_VERBS = set(
    NEW_VERB_TO_FRAME
)

ALLOWED_NEW_FRAMES = set(
    NEW_VERB_TO_FRAME.values()
)

BLOCKED_VERBS = {
    "LOSE",
    "CALL",
    "WAIT",
    "GIVE",
    "BRING",
    "RETURN",
    "NEED",
    "LEAVE",
}

BLOCKED_FRAMES = {
    "LOSE_OBJECT",
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
    "RETURN_PLACE",
    "NEED_OBJECT",
    "LEAVE_PLACE",
    "WHERE_OBJECT",
    "WHERE_PERSON",
}


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
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
            f"JSONL file not found: {path}"
        )

    rows: list[dict] = []

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
                raise RuntimeError(
                    f"Invalid JSONL: "
                    f"{path}:{line_no}"
                ) from exc

            if not isinstance(row, dict):
                raise RuntimeError(
                    f"{path}:{line_no} "
                    f"is not a JSON object"
                )

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


# ============================================================
# Base V0.4.4.1 generation
# ============================================================

def locate_base_raw_file(
    base_output_dir: Path,
) -> Path:

    preferred = [
        base_output_dir
        / "semantic_v0441_raw.jsonl",

        base_output_dir
        / "semantic_v04_raw.jsonl",
    ]

    for path in preferred:
        if path.exists():
            return path

    candidates = list(
        base_output_dir.glob(
            "semantic*_raw.jsonl"
        )
    )

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "V0.4.4.1 generator completed but "
            f"no semantic *_raw.jsonl was found "
            f"inside {base_output_dir}"
        )

    raise RuntimeError(
        "Multiple possible V0.4.4.1 raw files found: "
        + ", ".join(
            str(x)
            for x in candidates
        )
    )


def generate_frozen_core(
    *,
    n: int,
    seed: int,
    base_output_dir: Path,
) -> list[dict]:

    if n <= 0:
        return []

    # --------------------------------------------------------
    # Important:
    #
    # We execute the frozen generator as an external module
    # rather than importing / subclassing its internal class.
    #
    # This prevents accidental modification of V0.4.4.1 and
    # avoids circular-import regressions.
    # --------------------------------------------------------

    if base_output_dir.exists():
        shutil.rmtree(
            base_output_dir
        )

    base_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-m",
        "scripts.synthetic.generate_synthetic_v0441",

        "--n",
        str(n),

        "--seed",
        str(seed),

        "--output-dir",
        str(base_output_dir),

        "--preview",
        "0",
    ]

    print()
    print("=" * 100)
    print("GENERATING FROZEN V0.4.4.1 CORE")
    print("=" * 100)
    print(" ".join(command))
    print()

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Frozen V0.4.4.1 generator failed "
            f"with return code {result.returncode}"
        )

    raw_file = locate_base_raw_file(
        base_output_dir
    )

    rows = read_jsonl(
        raw_file
    )

    if len(rows) != n:
        raise RuntimeError(
            "Frozen core row count mismatch: "
            f"requested={n}, actual={len(rows)}"
        )

    return rows


# ============================================================
# Compatibility helper
# ============================================================

def build_allowed_objects(
    compatibility: dict,
) -> dict[str, list[str]]:

    semantic_classes = (
        compatibility.get(
            "semantic_classes",
            {}
        )
    )

    verb_rules = compatibility.get(
        "verb_rules",
        {}
    )

    result: dict[str, list[str]] = {}

    for verb_id in sorted(
        ALLOWED_NEW_VERBS
    ):

        rule = verb_rules.get(
            verb_id
        )

        if not isinstance(
            rule,
            dict,
        ):
            raise RuntimeError(
                f"Missing compatibility rule "
                f"for enabled verb: {verb_id}"
            )

        class_ids = rule.get(
            "object_classes",
            []
        )

        object_ids: list[str] = []

        for class_id in class_ids:

            members = semantic_classes.get(
                class_id
            )

            if not isinstance(
                members,
                list,
            ):
                raise RuntimeError(
                    f"{verb_id} references unknown "
                    f"semantic class: {class_id}"
                )

            object_ids.extend(
                members
            )

        # Deduplicate while preserving order.
        object_ids = list(
            dict.fromkeys(
                object_ids
            )
        )

        if not object_ids:
            raise RuntimeError(
                f"No compatible objects "
                f"resolved for {verb_id}"
            )

        result[
            verb_id
        ] = object_ids

    return result


# ============================================================
# New sample generator
# ============================================================

SCENARIO_BY_VERB = {
    "SEE": [
        "daily",
        "travel",
        "work_study",
    ],

    "TAKE": [
        "daily",
        "travel",
        "shopping",
    ],
}


def generate_new_samples(
    *,
    n: int,
    rng: random.Random,
    renderer: V051Renderer,
    allowed_objects: dict[str, list[str]],
    subject_ids: list[str],
) -> list[dict]:

    if n <= 0:
        return []

    rows: list[dict] = []

    seen_signatures: set[
        tuple[Any, ...]
    ] = set()

    attempts = 0

    max_attempts = max(
        n * 100,
        1000,
    )

    verb_counts = Counter()

    # --------------------------------------------------------
    # Balance SEE / TAKE rather than pure random sampling.
    # --------------------------------------------------------

    ordered_verbs = [
        "SEE",
        "TAKE",
    ]

    while len(rows) < n:

        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError(
                "Unable to generate enough unique "
                f"V0.5.1 samples. "
                f"Generated={len(rows)}, "
                f"requested={n}, "
                f"attempts={attempts}"
            )

        min_count = min(
            verb_counts.get(
                verb_id,
                0,
            )
            for verb_id in ordered_verbs
        )

        candidate_verbs = [
            verb_id
            for verb_id in ordered_verbs
            if verb_counts.get(
                verb_id,
                0,
            ) == min_count
        ]

        verb_id = rng.choice(
            candidate_verbs
        )

        frame_id = (
            NEW_VERB_TO_FRAME[
                verb_id
            ]
        )

        subject_id = rng.choice(
            subject_ids
        )

        object_id = rng.choice(
            allowed_objects[
                verb_id
            ]
        )

        tense = rng.choice([
            "present",
            "future",
        ])

        polarity = rng.choice([
            "pos",
            "neg",
        ])

        signature = (
            frame_id,
            subject_id,
            verb_id,
            object_id,
            tense,
            polarity,
        )

        if signature in seen_signatures:
            continue

        seen_signatures.add(
            signature
        )

        slots = {
            "subject":
                subject_id,

            "verb":
                verb_id,

            "object":
                object_id,
        }

        features = {
            "tense":
                tense,

            "polarity":
                polarity,
        }

        texts = renderer.render(
            frame_id=frame_id,
            slots=slots,
            features=features,
        )

        scenario = rng.choice(
            SCENARIO_BY_VERB[
                verb_id
            ]
        )

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
                    FROZEN_CORE_VERSION,

                "generator_version":
                    GENERATOR_VERSION,

                "renderer_version":
                    RENDERER_VERSION,

                "resource_version":
                    "0.5.1",

                "generation_source":
                    "v051_new_frame",

                "batch":
                    "Batch-1A",
            },
        }

        rows.append(
            row
        )

        verb_counts[
            verb_id
        ] += 1

    return rows


# ============================================================
# Normalize / merge
# ============================================================

def normalize_base_row(
    row: dict,
) -> dict:

    output = dict(
        row
    )

    old_id = output.get(
        "semantic_id"
    )

    metadata = output.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict,
    ):
        metadata = {}

    metadata = dict(
        metadata
    )

    metadata[
        "synthetic_core_version"
    ] = FROZEN_CORE_VERSION

    metadata[
        "v051_merge_source"
    ] = "frozen_v0441"

    metadata[
        "parent_semantic_id"
    ] = old_id

    output[
        "metadata"
    ] = metadata

    return output


def assign_v051_ids(
    rows: list[dict],
) -> None:

    for index, row in enumerate(
        rows,
        start=1,
    ):

        row[
            "semantic_id"
        ] = (
            f"sem_v051_"
            f"{index:08d}"
        )


# ============================================================
# Contract validation
# ============================================================

def validate_contract(
    rows: list[dict],
    expected_total: int,
) -> list[str]:

    errors: list[str] = []

    if len(rows) != expected_total:
        errors.append(
            f"Expected {expected_total} rows, "
            f"got {len(rows)}"
        )

    ids = [
        row.get(
            "semantic_id"
        )
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        errors.append(
            "Duplicate semantic_id detected."
        )

    for row in rows:

        slots = row.get(
            "slots",
            {}
        )

        verb = slots.get(
            "verb"
        )

        frame_id = row.get(
            "frame_id"
        )

        if verb in BLOCKED_VERBS:
            errors.append(
                f"{row.get('semantic_id')}: "
                f"blocked verb leaked: {verb}"
            )

        if frame_id in BLOCKED_FRAMES:
            errors.append(
                f"{row.get('semantic_id')}: "
                f"blocked frame leaked: {frame_id}"
            )

        if frame_id == "SEE_OBJECT":
            if verb != "SEE":
                errors.append(
                    f"{row.get('semantic_id')}: "
                    f"SEE_OBJECT verb={verb}"
                )

        if frame_id == "TAKE_OBJECT":
            if verb != "TAKE":
                errors.append(
                    f"{row.get('semantic_id')}: "
                    f"TAKE_OBJECT verb={verb}"
                )

    return errors


# ============================================================
# Statistics
# ============================================================

def count_distribution(
    rows: list[dict],
    key_func,
) -> dict[str, int]:

    counter = Counter()

    for row in rows:

        value = key_func(
            row
        )

        if value is None:
            value = "__NONE__"

        counter[
            str(value)
        ] += 1

    return dict(
        counter.most_common()
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Synthetic Generator V0.5.1 "
            "Batch-1A smoke generator"
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
        default=2060,
    )

    parser.add_argument(
        "--new-ratio",
        type=float,
        default=0.20,
        help=(
            "Fraction of corpus generated "
            "from SEE_OBJECT / TAKE_OBJECT."
        ),
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

    if not (
        0.0
        <= args.new_ratio
        <= 1.0
    ):
        raise ValueError(
            "--new-ratio must be "
            "between 0 and 1"
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_file = (
        output_dir
        / "semantic_v051_raw.jsonl"
    )

    stats_file = (
        output_dir
        / "semantic_v051_stats.json"
    )

    contract_file = (
        output_dir
        / "semantic_v051_contract.json"
    )

    base_output_dir = (
        output_dir
        / "_base_v0441"
    )

    policy = read_json(
        DEFAULT_POLICY
    )

    compatibility = read_json(
        DEFAULT_COMPATIBILITY
    )

    arguments = read_json(
        DEFAULT_ARGUMENTS
    )

    # ========================================================
    # Confirm activation policy
    # ========================================================

    enabled_verbs = set(
        policy.get(
            "enabled_new_verbs",
            []
        )
    )

    enabled_frames = set(
        policy.get(
            "enabled_new_frames",
            []
        )
    )

    if enabled_verbs != ALLOWED_NEW_VERBS:
        raise RuntimeError(
            "V0.5.1 activation policy mismatch. "
            f"Expected verbs="
            f"{sorted(ALLOWED_NEW_VERBS)}, "
            f"got={sorted(enabled_verbs)}"
        )

    if enabled_frames != ALLOWED_NEW_FRAMES:
        raise RuntimeError(
            "V0.5.1 activation policy mismatch. "
            f"Expected frames="
            f"{sorted(ALLOWED_NEW_FRAMES)}, "
            f"got={sorted(enabled_frames)}"
        )

    # ========================================================
    # Counts
    # ========================================================

    new_n = int(
        round(
            args.n
            * args.new_ratio
        )
    )

    if (
        args.new_ratio > 0
        and args.n > 1
    ):
        new_n = max(
            2,
            new_n,
        )

    new_n = min(
        new_n,
        args.n,
    )

    base_n = (
        args.n
        - new_n
    )

    print("=" * 100)
    print("SYNTHETIC GENERATOR V0.5.1")
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
        "Frozen V0.4.4.1 rows:",
        base_n,
    )

    print(
        "New V0.5.1 rows:",
        new_n,
    )

    print(
        "New ratio:",
        f"{args.new_ratio:.2%}",
    )

    print(
        "Enabled new verbs:",
        sorted(enabled_verbs),
    )

    print(
        "Enabled new frames:",
        sorted(enabled_frames),
    )

    print(
        "Output:",
        raw_file,
    )

    print("=" * 100)

    # ========================================================
    # 1. Frozen core
    # ========================================================

    base_rows = generate_frozen_core(
        n=base_n,
        seed=args.seed,
        base_output_dir=base_output_dir,
    )

    base_rows = [
        normalize_base_row(row)
        for row in base_rows
    ]

    # ========================================================
    # 2. New V0.5.1 samples
    # ========================================================

    rng = random.Random(
        args.seed + 1
    )

    renderer = V051Renderer()

    allowed_objects = (
        build_allowed_objects(
            compatibility
        )
    )

    subject_ids = list(
        arguments.get(
            "subjects",
            {}
        )
    )

    if not subject_ids:
        raise RuntimeError(
            "No subject IDs found in "
            "argument_realization_v051.json"
        )

    new_rows = generate_new_samples(
        n=new_n,
        rng=rng,
        renderer=renderer,
        allowed_objects=allowed_objects,
        subject_ids=subject_ids,
    )

    # ========================================================
    # 3. Merge + shuffle + IDs
    # ========================================================

    rows = (
        base_rows
        + new_rows
    )

    rng.shuffle(
        rows
    )

    assign_v051_ids(
        rows
    )

    # ========================================================
    # 4. Contract
    # ========================================================

    contract_errors = (
        validate_contract(
            rows,
            expected_total=args.n,
        )
    )

    new_rows_final = [
        row
        for row in rows
        if (
            row.get(
                "frame_id"
            )
            in ALLOWED_NEW_FRAMES
        )
    ]

    new_verb_counts = count_distribution(
        new_rows_final,
        lambda row:
            row.get(
                "slots",
                {},
            ).get(
                "verb"
            ),
    )

    new_frame_counts = count_distribution(
        new_rows_final,
        lambda row:
            row.get(
                "frame_id"
            ),
    )

    # Both SEE and TAKE must appear.
    for verb_id in ALLOWED_NEW_VERBS:

        if new_verb_counts.get(
            verb_id,
            0,
        ) <= 0:

            contract_errors.append(
                f"Enabled verb never appeared: "
                f"{verb_id}"
            )

    # ========================================================
    # 5. Save
    # ========================================================

    write_jsonl(
        raw_file,
        rows,
    )

    stats = {
        "generator_version":
            GENERATOR_VERSION,

        "renderer_version":
            RENDERER_VERSION,

        "frozen_core_version":
            FROZEN_CORE_VERSION,

        "seed":
            args.seed,

        "samples":
            len(rows),

        "base_rows":
            base_n,

        "new_rows":
            new_n,

        "new_ratio":
            args.new_ratio,

        "counts": {
            "frame":
                count_distribution(
                    rows,
                    lambda row:
                        row.get(
                            "frame_id"
                        ),
                ),

            "verb":
                count_distribution(
                    rows,
                    lambda row:
                        row.get(
                            "slots",
                            {},
                        ).get(
                            "verb"
                        ),
                ),

            "new_frame":
                new_frame_counts,

            "new_verb":
                new_verb_counts,

            "tense":
                count_distribution(
                    rows,
                    lambda row:
                        row.get(
                            "features",
                            {},
                        ).get(
                            "tense"
                        ),
                ),

            "polarity":
                count_distribution(
                    rows,
                    lambda row:
                        row.get(
                            "features",
                            {},
                        ).get(
                            "polarity"
                        ),
                ),
        },
    }

    contract_summary = {
        "version":
            GENERATOR_VERSION,

        "total":
            len(rows),

        "new_rows":
            len(new_rows_final),

        "blocked_verbs":
            sorted(
                BLOCKED_VERBS
            ),

        "blocked_frames":
            sorted(
                BLOCKED_FRAMES
            ),

        "new_verb_counts":
            new_verb_counts,

        "new_frame_counts":
            new_frame_counts,

        "errors":
            contract_errors,

        "pass":
            not contract_errors,
    }

    write_json(
        stats_file,
        stats,
    )

    write_json(
        contract_file,
        contract_summary,
    )

    # ========================================================
    # Preview
    # ========================================================

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
                row.get(
                    "semantic_id"
                ),
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
                {}
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

    # ========================================================
    # Final report
    # ========================================================

    print()
    print("=" * 100)
    print("GENERATOR V0.5.1 COMPLETE")
    print("=" * 100)

    print(
        "Generated:",
        len(rows),
    )

    print(
        "Frozen core rows:",
        base_n,
    )

    print(
        "New rows:",
        len(new_rows_final),
    )

    print()

    print(
        "New verb distribution:"
    )

    for key, value in (
        new_verb_counts.items()
    ):
        print(
            f"{key:<20}{value}"
        )

    print()

    print(
        "New frame distribution:"
    )

    for key, value in (
        new_frame_counts.items()
    ):
        print(
            f"{key:<25}{value}"
        )

    print()

    print(
        "Contract errors:",
        len(contract_errors),
    )

    if contract_errors:

        for error in contract_errors:
            print(
                "ERROR:",
                error,
            )

        print()
        print(
            "V0.5.1 GENERATION CONTRACT FAILED"
        )

        raise SystemExit(1)

    print(
        "V0.5.1 GENERATION CONTRACT PASS"
    )

    print()

    print(
        "Files:"
    )

    print(raw_file)
    print(stats_file)
    print(contract_file)


if __name__ == "__main__":
    main()