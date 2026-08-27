from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONCEPTS = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "concepts_v04.jsonl"
)

DEFAULT_FRAMES = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "frames_v04.json"
)


BATCH_VERSION = "0.4-batch1"

BATCH_IDS = {
    "COME",
    "ARRIVE",
    "LEAVE",
    "WANT",
    "NEED",
}


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Concept file not found: {path}"
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
                row = json.loads(line)

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
                    "is not a JSON object."
                )

            rows.append(row)

    return rows


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

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"{path} root must be JSON object."
        )

    return data


def atomic_write_jsonl(
    path: Path,
    rows: list[dict],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
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

    tmp.replace(path)


def atomic_write_json(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
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

    tmp.replace(path)


def make_backup(
    path: Path,
) -> Path:

    backup = path.with_suffix(
        path.suffix + ".before_batch1.bak"
    )

    if not backup.exists():

        shutil.copy2(
            path,
            backup,
        )

    return backup


# ============================================================
# Common metadata
# ============================================================

def active_meta() -> dict:

    return {
        "enabled": True,
        "status": "active",
        "version_added": "0.4",
        "batch_added": BATCH_VERSION,
    }


def planned_meta(
    reason: str,
) -> dict:

    return {
        "enabled": False,
        "status": "planned",
        "version_added": "0.4",
        "batch_added": BATCH_VERSION,
        "disabled_reason": reason,
    }


# ============================================================
# COME
# ============================================================

def build_come() -> dict:

    return {
        "id": "COME",

        "concept_type": "verb",

        "semantic_classes": [
            "MOTION_ACTION"
        ],

        "scenario_tags": [
            "daily",
            "travel",
            "transport",
            "hotel",
            "medical",
        ],

        "features": {
            "transitivity": "intransitive",
            "resultative": False,
            "modal": False,
            "allowed_tenses": [
                "present",
                "future",
            ],
        },

        "argument_schema": {
            "subject": {
                "required": True,
                "semantic_classes": [
                    "PERSON"
                ],
            },

            "destination": {
                "required": False,
                "semantic_classes": [
                    "PLACE"
                ],
            },
        },

        "render_policy": {
            "destination_role": "destination",

            "notes": (
                "COME may use the standard destination "
                "surface in the first V0.4 renderer."
            ),
        },

        "forms": {

            "zh": {
                "base": "来",
                "present_positive": "来",
                "present_negative": "不来",
                "future_positive": "会来",
                "future_negative": "不会来",
            },

            "en": {
                "base": "come",
                "present_3sg": "comes",
                "past": "came",
                "past_participle": "come",
                "gerund": "coming",
            },

            "ru": {
                "infinitive": "приходить",

                "aspect_policy": {
                    "present": "imperfective",
                    "future": "perfective",
                },

                "present": {
                    "1sg": "прихожу",
                    "2sg": "приходишь",
                    "3sg": "приходит",
                    "1pl": "приходим",
                    "2pl": "приходите",
                    "3pl": "приходят",
                },

                "future_strategy": "perfective",

                "imperfective": {
                    "lemma": "приходить",

                    "present_1sg": "прихожу",
                    "present_2sg": "приходишь",
                    "present_3sg": "приходит",
                    "present_1pl": "приходим",
                    "present_2pl": "приходите",
                    "present_3pl": "приходят",
                },

                "perfective": {
                    "lemma": "прийти",

                    "future_1sg": "приду",
                    "future_2sg": "придёшь",
                    "future_3sg": "придёт",
                    "future_1pl": "придём",
                    "future_2pl": "придёте",
                    "future_3pl": "придут",
                },
            },

            "uz": {
                "lemma": "kelmoq",
                "base": "kelmoq",

                "present_future": {
                    "1sg": "kelaman",
                    "2sg": "kelasan",
                    "3sg": "keladi",
                    "1pl": "kelamiz",
                    "2pl": "kelasiz",
                    "3pl": "keladilar",
                },

                "negative_present_future": {
                    "1sg": "kelmayman",
                    "2sg": "kelmaysan",
                    "3sg": "kelmaydi",
                    "1pl": "kelmaymiz",
                    "2pl": "kelmaysiz",
                    "3pl": "kelmaydilar",
                },
            },
        },

        "meta": active_meta(),
    }


# ============================================================
# ARRIVE
# ============================================================

def build_arrive() -> dict:

    return {
        "id": "ARRIVE",

        "concept_type": "verb",

        "semantic_classes": [
            "MOTION_ACTION"
        ],

        "scenario_tags": [
            "travel",
            "transport",
            "hotel",
        ],

        "features": {
            "transitivity": "intransitive",
            "resultative": True,
            "modal": False,
            "allowed_tenses": [
                "present",
                "future",
            ],
        },

        "argument_schema": {
            "subject": {
                "required": True,
                "semantic_classes": [
                    "PERSON"
                ],
            },

            "destination": {
                "required": False,
                "semantic_classes": [
                    "PLACE"
                ],
            },
        },

        # Important:
        # English cannot simply reuse "to the airport":
        #
        # go to the airport       OK
        # arrive to the airport   NOT our target
        #
        # ARRIVE therefore requires verb-aware
        # destination realization in renderer_v04.
        "render_policy": {
            "destination_role": "destination",

            "language_overrides": {
                "en": {
                    "strategy": "preposition_plus_base",
                    "preposition": "at",
                },

                "zh": {
                    "strategy": "base_place",
                },

                "ru": {
                    "strategy": "destination_form",
                },

                "uz": {
                    "strategy": "destination_form",
                },
            },
        },

        "forms": {

            "zh": {
                "base": "到达",
                "present_positive": "到达",
                "present_negative": "没到达",
                "future_positive": "会到达",
                "future_negative": "不会到达",
            },

            "en": {
                "base": "arrive",
                "present_3sg": "arrives",
                "past": "arrived",
                "past_participle": "arrived",
                "gerund": "arriving",
            },

            "ru": {
                "infinitive": "прибывать",

                "aspect_policy": {
                    "present": "imperfective",
                    "future": "perfective",
                },

                "present": {
                    "1sg": "прибываю",
                    "2sg": "прибываешь",
                    "3sg": "прибывает",
                    "1pl": "прибываем",
                    "2pl": "прибываете",
                    "3pl": "прибывают",
                },

                "future_strategy": "perfective",

                "imperfective": {
                    "lemma": "прибывать",

                    "present_1sg": "прибываю",
                    "present_2sg": "прибываешь",
                    "present_3sg": "прибывает",
                    "present_1pl": "прибываем",
                    "present_2pl": "прибываете",
                    "present_3pl": "прибывают",
                },

                "perfective": {
                    "lemma": "прибыть",

                    "future_1sg": "прибуду",
                    "future_2sg": "прибудешь",
                    "future_3sg": "прибудет",
                    "future_1pl": "прибудем",
                    "future_2pl": "прибудете",
                    "future_3pl": "прибудут",
                },
            },

            "uz": {
                "lemma": "yetib kelmoq",
                "base": "yetib kelmoq",

                "present_future": {
                    "1sg": "yetib kelaman",
                    "2sg": "yetib kelasan",
                    "3sg": "yetib keladi",
                    "1pl": "yetib kelamiz",
                    "2pl": "yetib kelasiz",
                    "3pl": "yetib keladilar",
                },

                "negative_present_future": {
                    "1sg": "yetib kelmayman",
                    "2sg": "yetib kelmaysan",
                    "3sg": "yetib kelmaydi",
                    "1pl": "yetib kelmaymiz",
                    "2pl": "yetib kelmaysiz",
                    "3pl": "yetib kelmaydilar",
                },
            },
        },

        "meta": active_meta(),
    }


# ============================================================
# LEAVE
#
# Added to schema, but deliberately disabled.
# It requires SOURCE rather than DESTINATION.
# ============================================================

def build_leave() -> dict:

    return {
        "id": "LEAVE",

        "concept_type": "verb",

        "semantic_classes": [
            "MOTION_ACTION"
        ],

        "scenario_tags": [
            "daily",
            "travel",
            "transport",
            "hotel",
        ],

        "features": {
            "transitivity": "intransitive",
            "resultative": False,
            "modal": False,
            "allowed_tenses": [
                "present",
                "future",
            ],
            "requires_special_frame": True,
        },

        "argument_schema": {
            "subject": {
                "required": True,
                "semantic_classes": [
                    "PERSON"
                ],
            },

            "source": {
                "required": False,
                "semantic_classes": [
                    "PLACE"
                ],
            },
        },

        "render_policy": {
            "required_frame_role": "source",

            "required_future_frame": (
                "MOTION_SOURCE"
            ),
        },

        "forms": {

            "zh": {
                "base": "离开",
                "present_positive": "离开",
                "present_negative": "不离开",
                "future_positive": "会离开",
                "future_negative": "不会离开",
            },

            "en": {
                "base": "leave",
                "present_3sg": "leaves",
                "past": "left",
                "past_participle": "left",
                "gerund": "leaving",
            },

            "ru": {
                "infinitive": "уходить",

                "aspect_policy": {
                    "present": "imperfective",
                    "future": "perfective",
                },

                "present": {
                    "1sg": "ухожу",
                    "2sg": "уходишь",
                    "3sg": "уходит",
                    "1pl": "уходим",
                    "2pl": "уходите",
                    "3pl": "уходят",
                },

                "future_strategy": "perfective",

                "perfective": {
                    "lemma": "уйти",

                    "future_1sg": "уйду",
                    "future_2sg": "уйдёшь",
                    "future_3sg": "уйдёт",
                    "future_1pl": "уйдём",
                    "future_2pl": "уйдёте",
                    "future_3pl": "уйдут",
                },
            },

            "uz": {
                "lemma": "ketmoq",
                "base": "ketmoq",

                "present_future": {
                    "1sg": "ketaman",
                    "2sg": "ketasan",
                    "3sg": "ketadi",
                    "1pl": "ketamiz",
                    "2pl": "ketasiz",
                    "3pl": "ketadilar",
                },

                "negative_present_future": {
                    "1sg": "ketmayman",
                    "2sg": "ketmaysan",
                    "3sg": "ketmaydi",
                    "1pl": "ketmaymiz",
                    "2pl": "ketmaysiz",
                    "3pl": "ketmaydilar",
                },
            },
        },

        "meta": planned_meta(
            (
                "LEAVE requires a dedicated "
                "MOTION_SOURCE frame and source-case "
                "rendering before generation."
            )
        ),
    }


# ============================================================
# WANT
# ============================================================

def build_want() -> dict:

    return {
        "id": "WANT",

        "concept_type": "verb",

        "semantic_classes": [
            "INFORMATION_ACTION"
        ],

        "scenario_tags": [
            "daily",
            "shopping",
            "restaurant",
            "hotel",
            "medical",
        ],

        "features": {
            "transitivity": "transitive",
            "resultative": False,
            "modal": False,

            # Batch 1 deliberately keeps WANT_OBJECT
            # in present/simple intent contexts.
            "allowed_tenses": [
                "present"
            ],
        },

        "argument_schema": {
            "subject": {
                "required": True,
                "semantic_classes": [
                    "PERSON"
                ],
            },

            "object": {
                "required": True,
                "semantic_classes": [
                    "PHYSICAL_OBJECT",
                    "EDIBLE",
                    "DRINKABLE",
                    "DEVICE",
                    "MEDICINE",
                    "TRAVEL_DOCUMENT",
                ],
            },
        },

        "forms": {

            "zh": {
                "base": "想要",
                "present_positive": "想要",
                "present_negative": "不想要",
            },

            "en": {
                "base": "want",
                "present_3sg": "wants",
                "past": "wanted",
                "past_participle": "wanted",
                "gerund": "wanting",
            },

            "ru": {
                "infinitive": "хотеть",

                "present": {
                    "1sg": "хочу",
                    "2sg": "хочешь",
                    "3sg": "хочет",
                    "1pl": "хотим",
                    "2pl": "хотите",
                    "3pl": "хотят",
                },

                "future_strategy": "disabled_v04_batch1",
            },

            "uz": {
                "lemma": "xohlamoq",
                "base": "xohlamoq",

                "present_future": {
                    "1sg": "xohlayman",
                    "2sg": "xohlaysan",
                    "3sg": "xohlaydi",
                    "1pl": "xohlaymiz",
                    "2pl": "xohlaysiz",
                    "3pl": "xohlaydilar",
                },

                "negative_present_future": {
                    "1sg": "xohlamayman",
                    "2sg": "xohlamaysan",
                    "3sg": "xohlamaydi",
                    "1pl": "xohlamaymiz",
                    "2pl": "xohlamaysiz",
                    "3pl": "xohlamaydilar",
                },
            },
        },

        "meta": active_meta(),
    }


# ============================================================
# NEED
#
# Deliberately disabled.
#
# Russian natural form:
#   Мне нужно лекарство.
#   Мне нужен паспорт.
#
# Uzbek natural form:
#   Menga dori kerak.
#
# These are NOT ordinary S + V + O constructions.
# ============================================================

def build_need() -> dict:

    return {
        "id": "NEED",

        "concept_type": "verb",

        "semantic_classes": [
            "INFORMATION_ACTION"
        ],

        "scenario_tags": [
            "daily",
            "travel",
            "hotel",
            "medical",
        ],

        "features": {
            "transitivity": "transitive",
            "resultative": False,
            "modal": False,
            "allowed_tenses": [
                "present"
            ],
            "requires_special_renderer": True,
        },

        "argument_schema": {
            "subject": {
                "required": True,
                "semantic_classes": [
                    "PERSON"
                ],
            },

            "object": {
                "required": True,
                "semantic_classes": [
                    "PHYSICAL_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "TRAVEL_DOCUMENT",
                    "DEVICE",
                    "MEDICINE",
                ],
            },
        },

        "render_policy": {

            "zh": {
                "construction": "subject_verb_object",
            },

            "en": {
                "construction": "subject_verb_object",
            },

            "ru": {
                "construction": (
                    "dative_subject_predicative_nuzhen"
                ),
                "example": (
                    "Мне нужно лекарство."
                ),
            },

            "uz": {
                "construction": (
                    "dative_subject_object_kerak"
                ),
                "example": (
                    "Menga dori kerak."
                ),
            },
        },

        "forms": {

            "zh": {
                "base": "需要",
                "present_positive": "需要",
                "present_negative": "不需要",
            },

            "en": {
                "base": "need",
                "present_3sg": "needs",
                "past": "needed",
                "past_participle": "needed",
                "gerund": "needing",
            },

            # Kept only as lexical metadata.
            # Do NOT activate generic Russian S+V+O rendering.
            "ru": {
                "infinitive": "нуждаться",

                "preferred_construction": (
                    "dative_predicative"
                ),
            },

            "uz": {
                "lemma": "kerak",
                "base": "kerak",

                "preferred_construction": (
                    "dative_predicative"
                ),
            },
        },

        "meta": planned_meta(
            (
                "NEED requires language-specific "
                "Russian and Uzbek predicative "
                "constructions before activation."
            )
        ),
    }


# ============================================================
# Batch resources
# ============================================================

def build_batch_concepts() -> list[dict]:

    return [
        build_come(),
        build_arrive(),
        build_leave(),
        build_want(),
        build_need(),
    ]


# ============================================================
# Concept merge
# ============================================================

def merge_concepts(
    existing: list[dict],
    batch: list[dict],
    force: bool,
) -> tuple[
    list[dict],
    list[str],
    list[str],
]:

    result = deepcopy(
        existing
    )

    index = {
        row.get("id"): i
        for i, row
        in enumerate(result)
        if row.get("id")
    }

    added = []
    replaced = []

    for concept in batch:

        cid = concept[
            "id"
        ]

        if cid in index:

            if not force:

                raise RuntimeError(
                    f"Concept {cid} already exists. "
                    "Use --force only if you intend "
                    "to replace Batch 1 concepts."
                )

            result[
                index[cid]
            ] = concept

            replaced.append(
                cid
            )

        else:

            index[
                cid
            ] = len(
                result
            )

            result.append(
                concept
            )

            added.append(
                cid
            )

    ids = [
        row.get(
            "id"
        )
        for row in result
    ]

    if len(ids) != len(
        set(ids)
    ):
        raise RuntimeError(
            "Duplicate Concept IDs after merge."
        )

    return (
        result,
        added,
        replaced,
    )


# ============================================================
# Frame activation policy
# ============================================================

def update_frames(
    frames_data: dict,
) -> tuple[
    dict,
    list[str],
    list[str],
]:

    result = deepcopy(
        frames_data
    )

    frames = result.get(
        "frames"
    )

    if not isinstance(
        frames,
        list,
    ):
        raise RuntimeError(
            "frames_v04.json: "
            "'frames' must be a list."
        )

    activated = []
    kept_disabled = []

    found_want = False
    found_need = False

    for frame in frames:

        if not isinstance(
            frame,
            dict,
        ):
            continue

        fid = frame.get(
            "id"
        )

        # ====================================================
        # WANT_OBJECT is safe to activate.
        # ====================================================

        if fid == "WANT_OBJECT":

            found_want = True

            meta = frame.setdefault(
                "meta",
                {},
            )

            meta[
                "enabled"
            ] = True

            meta[
                "status"
            ] = "active"

            meta[
                "version_modified"
            ] = "0.4"

            meta[
                "batch_activated"
            ] = BATCH_VERSION

            meta.pop(
                "disabled_reason",
                None,
            )

            activated.append(
                fid
            )

        # ====================================================
        # NEED_OBJECT remains disabled intentionally.
        # ====================================================

        elif fid == "NEED_OBJECT":

            found_need = True

            meta = frame.setdefault(
                "meta",
                {},
            )

            meta[
                "enabled"
            ] = False

            meta[
                "status"
            ] = "planned"

            meta[
                "version_modified"
            ] = "0.4"

            meta[
                "disabled_reason"
            ] = (
                "NEED requires dedicated "
                "Russian/Uzbek predicative rendering."
            )

            kept_disabled.append(
                fid
            )

    if not found_want:
        raise RuntimeError(
            "WANT_OBJECT frame not found."
        )

    if not found_need:
        raise RuntimeError(
            "NEED_OBJECT frame not found."
        )

    return (
        result,
        activated,
        kept_disabled,
    )


# ============================================================
# Summary helpers
# ============================================================

def count_concept_types(
    rows: list[dict],
) -> dict[str, int]:

    counts: dict[
        str,
        int,
    ] = {}

    for row in rows:

        ctype = row.get(
            "concept_type",
            "UNKNOWN",
        )

        counts[
            ctype
        ] = (
            counts.get(
                ctype,
                0,
            )
            + 1
        )

    return counts


def active_concept_ids(
    rows: list[dict],
) -> list[str]:

    output = []

    for row in rows:

        meta = row.get(
            "meta",
            {},
        )

        if (
            isinstance(meta, dict)
            and meta.get(
                "enabled",
                True,
            )
        ):

            output.append(
                row["id"]
            )

    return output


def disabled_concept_ids(
    rows: list[dict],
) -> list[str]:

    output = []

    for row in rows:

        meta = row.get(
            "meta",
            {},
        )

        if (
            isinstance(meta, dict)
            and not meta.get(
                "enabled",
                True,
            )
        ):

            output.append(
                row["id"]
            )

    return output


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Expand Synthetic V0.4 "
            "Coverage Batch 1."
        )
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
        "--force",
        action="store_true",
        help=(
            "Replace existing Batch 1 concept "
            "entries if they already exist."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate and preview changes "
            "without writing files."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> None:

    args = parse_args()

    concepts_path = Path(
        args.concepts
    )

    frames_path = Path(
        args.frames
    )

    print(
        "=" * 90
    )

    print(
        "SYNTHETIC V0.4 COVERAGE EXPANSION - BATCH 1"
    )

    print(
        "=" * 90
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
        "Force:",
        args.force,
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    print()

    # ========================================================
    # Load
    # ========================================================

    concepts = read_jsonl(
        concepts_path
    )

    frames = read_json(
        frames_path
    )

    old_count = len(
        concepts
    )

    old_type_counts = (
        count_concept_types(
            concepts
        )
    )

    # ========================================================
    # Build Batch 1
    # ========================================================

    batch = (
        build_batch_concepts()
    )

    batch_ids = {
        row[
            "id"
        ]
        for row in batch
    }

    if batch_ids != BATCH_IDS:

        raise RuntimeError(
            "Internal Batch 1 ID mismatch."
        )

    # ========================================================
    # Merge concepts
    # ========================================================

    (
        merged_concepts,
        added,
        replaced,
    ) = merge_concepts(
        concepts,
        batch,
        force=args.force,
    )

    # ========================================================
    # Frames
    # ========================================================

    (
        updated_frames,
        activated_frames,
        disabled_frames,
    ) = update_frames(
        frames
    )

    # ========================================================
    # Preview
    # ========================================================

    print(
        "Before concepts:",
        old_count,
    )

    print(
        "After concepts:",
        len(
            merged_concepts
        ),
    )

    print()

    print(
        "Before type counts:",
        old_type_counts,
    )

    print(
        "After type counts:",
        count_concept_types(
            merged_concepts
        ),
    )

    print()

    print(
        "Added:",
        added,
    )

    print(
        "Replaced:",
        replaced,
    )

    print()

    print(
        "Frames activated:",
        activated_frames,
    )

    print(
        "Frames kept disabled:",
        disabled_frames,
    )

    print()

    print(
        "Batch 1 active concepts:",
        [
            cid
            for cid
            in BATCH_IDS
            if cid
            in active_concept_ids(
                merged_concepts
            )
        ],
    )

    print(
        "Batch 1 planned concepts:",
        [
            cid
            for cid
            in BATCH_IDS
            if cid
            in disabled_concept_ids(
                merged_concepts
            )
        ],
    )

    # ========================================================
    # Dry run
    # ========================================================

    if args.dry_run:

        print()

        print(
            "=" * 90
        )

        print(
            "DRY RUN COMPLETE - NO FILES WRITTEN"
        )

        print(
            "=" * 90
        )

        return

    # ========================================================
    # Backup
    # ========================================================

    concepts_backup = (
        make_backup(
            concepts_path
        )
    )

    frames_backup = (
        make_backup(
            frames_path
        )
    )

    # ========================================================
    # Write
    # ========================================================

    atomic_write_jsonl(
        concepts_path,
        merged_concepts,
    )

    atomic_write_json(
        frames_path,
        updated_frames,
    )

    # ========================================================
    # Complete
    # ========================================================

    print()

    print(
        "Backups:"
    )

    print(
        concepts_backup
    )

    print(
        frames_backup
    )

    print()

    print(
        "=" * 90
    )

    print(
        "V0.4 COVERAGE BATCH 1 COMPLETE"
    )

    print(
        "=" * 90
    )

    print(
        "Next:"
    )

    print(
        "python -m "
        "scripts.synthetic."
        "validate_v04_resources"
    )


if __name__ == "__main__":
    main()