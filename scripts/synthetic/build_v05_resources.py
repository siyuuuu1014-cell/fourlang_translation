from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

V04_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

V05_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

BASE_CONCEPTS = (
    V04_DIR
    / "concepts_v044.jsonl"
)

BASE_FRAMES = (
    V04_DIR
    / "frames_v04.json"
)

BASE_COMPATIBILITY = (
    V04_DIR
    / "semantic_compatibility_v04.json"
)

BASE_POLICY = (
    V04_DIR
    / "generation_policy_v044.json"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}"
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
                    f"Invalid JSONL: "
                    f"{path}:{line_no}"
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


# ============================================================
# Helpers
# ============================================================

def lang(
    zh: str,
    en: str,
    ru: str,
    uz: str,
) -> dict:

    return {
        "zh": {
            "lemma": zh,
        },
        "en": {
            "lemma": en,
        },
        "ru": {
            "lemma": ru,
        },
        "uz": {
            "lemma": uz,
        },
    }


def make_concept(
    *,
    concept_id: str,
    concept_type: str,
    semantic_class: str,
    zh: str,
    en: str,
    ru: str,
    uz: str,
    operation: str = "add",
    generation_enabled: bool = False,
    renderer_strategy: str = "pending",
    allowed_frames: list[str] | None = None,
    notes: str = "",
) -> dict:

    return {
        "operation":
            operation,

        "id":
            concept_id,

        "type":
            concept_type,

        "semantic_class":
            semantic_class,

        "resource_status":
            "resource_ready",

        "generation_enabled":
            generation_enabled,

        "renderer_strategy":
            renderer_strategy,

        "scope": {
            "allowed_frames":
                allowed_frames or [],
        },

        "lang":
            lang(
                zh,
                en,
                ru,
                uz,
            ),

        "notes":
            notes,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    V05_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Make sure frozen resources exist.
    # --------------------------------------------------------

    required_base_files = [
        BASE_CONCEPTS,
        BASE_FRAMES,
        BASE_COMPATIBILITY,
        BASE_POLICY,
    ]

    for path in required_base_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Frozen V0.4.4.1 dependency "
                f"not found: {path}"
            )

    base_rows = read_jsonl(
        BASE_CONCEPTS
    )

    base_ids = {
        row.get("id")
        for row in base_rows
        if row.get("id")
    }

    # ========================================================
    # Manifest
    # ========================================================

    manifest = {
        "version":
            "0.5.0-batch1",

        "resource_phase":
            "coverage_expansion_batch1",

        "inherits": {
            "synthetic_core":
                "0.4.4.1",

            "concepts":
                "../v04/concepts_v044.jsonl",

            "frames":
                "../v04/frames_v04.json",

            "compatibility":
                "../v04/semantic_compatibility_v04.json",

            "generation_policy":
                "../v04/generation_policy_v044.json",
        },

        "design": {
            "type":
                "delta_resource_package",

            "generator_enabled":
                False,

            "reason":
                (
                    "Build and validate resource coverage "
                    "before implementing V0.5 renderer."
                ),
        },
    }

    # ========================================================
    # Concept additions
    # ========================================================

    concepts: list[dict] = []

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    object_specs = [
        (
            "ID_CARD",
            "IDENTITY_DOCUMENT",
            "身份证",
            "ID card",
            "удостоверение личности",
            "shaxsiy guvohnoma",
        ),
        (
            "MONEY",
            "MONEY",
            "钱",
            "money",
            "деньги",
            "pul",
        ),
        (
            "LUGGAGE",
            "PORTABLE_OBJECT",
            "行李",
            "luggage",
            "багаж",
            "bagaj",
        ),
        (
            "CHARGER",
            "DEVICE_ACCESSORY",
            "充电器",
            "charger",
            "зарядное устройство",
            "quvvatlagich",
        ),
        (
            "CLOTHES",
            "CLOTHING",
            "衣服",
            "clothes",
            "одежда",
            "kiyim",
        ),
        (
            "ADDRESS",
            "INFORMATION_OBJECT",
            "地址",
            "address",
            "адрес",
            "manzil",
        ),
        (
            "ROOM",
            "ROOM_OBJECT",
            "房间",
            "room",
            "комната",
            "xona",
        ),
        (
            "TABLE",
            "FURNITURE",
            "桌子",
            "table",
            "стол",
            "stol",
        ),
    ]

    for (
        concept_id,
        semantic_class,
        zh,
        en,
        ru,
        uz,
    ) in object_specs:

        concepts.append(
            make_concept(
                concept_id=concept_id,
                concept_type="object",
                semantic_class=semantic_class,
                zh=zh,
                en=en,
                ru=ru,
                uz=uz,
                generation_enabled=False,
                notes=(
                    "Batch-1 object; morphology "
                    "will be finalized in Renderer V0.5."
                ),
            )
        )

    # --------------------------------------------------------
    # Places
    # --------------------------------------------------------

    place_specs = [
        (
            "PHARMACY",
            "LOCAL_PLACE",
            "药店",
            "pharmacy",
            "аптека",
            "dorixona",
        ),
        (
            "SUPERMARKET",
            "LOCAL_PLACE",
            "超市",
            "supermarket",
            "супермаркет",
            "supermarket",
        ),
        (
            "SHOP",
            "LOCAL_PLACE",
            "商店",
            "shop",
            "магазин",
            "do'kon",
        ),
        (
            "CAFE",
            "LOCAL_PLACE",
            "咖啡馆",
            "cafe",
            "кафе",
            "kafe",
        ),
        (
            "POLICE_STATION",
            "LOCAL_PLACE",
            "警察局",
            "police station",
            "полицейский участок",
            "politsiya bo'limi",
        ),
        (
            "BUS_STOP",
            "TRANSPORT_PLACE",
            "公交站",
            "bus stop",
            "автобусная остановка",
            "avtobus bekati",
        ),
        (
            "SUBWAY_STATION",
            "TRANSPORT_PLACE",
            "地铁站",
            "subway station",
            "станция метро",
            "metro bekati",
        ),
        (
            "CITY_CENTER",
            "LOCAL_PLACE",
            "市中心",
            "city center",
            "центр города",
            "shahar markazi",
        ),
        (
            "TOILET",
            "LOCAL_PLACE",
            "洗手间",
            "toilet",
            "туалет",
            "hojatxona",
        ),
    ]

    for (
        concept_id,
        semantic_class,
        zh,
        en,
        ru,
        uz,
    ) in place_specs:

        concepts.append(
            make_concept(
                concept_id=concept_id,
                concept_type="place",
                semantic_class=semantic_class,
                zh=zh,
                en=en,
                ru=ru,
                uz=uz,
                generation_enabled=False,
            )
        )

    # --------------------------------------------------------
    # New verbs
    #
    # IMPORTANT:
    # Resource-ready != generation-enabled.
    #
    # We do NOT let the Generator use them until V0.5
    # morphology / syntax renderers exist.
    # --------------------------------------------------------

    verb_specs = [
        (
            "SEE",
            "PERCEPTION_VERB",
            "看见",
            "see",
            "видеть",
            "ko'rmoq",
            "simple_transitive",
        ),
        (
            "CALL",
            "COMMUNICATION_VERB",
            "打电话",
            "call",
            "звонить",
            "qo'ng'iroq qilmoq",
            "language_specific_argument_structure",
        ),
        (
            "WAIT",
            "WAIT_VERB",
            "等",
            "wait",
            "ждать",
            "kutmoq",
            "case_governed",
        ),
        (
            "GIVE",
            "TRANSFER_VERB",
            "给",
            "give",
            "давать",
            "bermoq",
            "ditransitive",
        ),
        (
            "TAKE",
            "TRANSFER_VERB",
            "拿",
            "take",
            "брать",
            "olmoq",
            "simple_transitive",
        ),
        (
            "BRING",
            "MOTION_TRANSFER_VERB",
            "带来",
            "bring",
            "приносить",
            "olib kelmoq",
            "transitive_destination",
        ),
        (
            "LOSE",
            "POSSESSION_CHANGE_VERB",
            "丢失",
            "lose",
            "терять",
            "yo'qotmoq",
            "simple_transitive",
        ),
        (
            "RETURN",
            "MOTION_VERB",
            "返回",
            "return",
            "возвращаться",
            "qaytmoq",
            "motion_place",
        ),
    ]

    for (
        concept_id,
        semantic_class,
        zh,
        en,
        ru,
        uz,
        strategy,
    ) in verb_specs:

        concepts.append(
            make_concept(
                concept_id=concept_id,
                concept_type="verb",
                semantic_class=semantic_class,
                zh=zh,
                en=en,
                ru=ru,
                uz=uz,
                generation_enabled=False,
                renderer_strategy=strategy,
                notes=(
                    "Do not enable before V0.5 "
                    "language-specific morphology is tested."
                ),
            )
        )

    # --------------------------------------------------------
    # NEED / LEAVE existed as disabled concepts in V0.4.
    # Use an override rather than silently duplicating them.
    # --------------------------------------------------------

    if "NEED" in base_ids:

        concepts.append({
            "operation":
                "override",

            "id":
                "NEED",

            "resource_status":
                "v05_candidate",

            "generation_enabled":
                False,

            "renderer_strategy":
                "language_specific_need_construction",

            "notes":
                (
                    "Russian and other languages require "
                    "construction-specific rendering."
                ),
        })

    else:

        concepts.append(
            make_concept(
                concept_id="NEED",
                concept_type="verb",
                semantic_class="NEED_VERB",
                zh="需要",
                en="need",
                ru="нуждаться",
                uz="kerak bo'lmoq",
                generation_enabled=False,
                renderer_strategy=(
                    "language_specific_need_construction"
                ),
            )
        )

    if "LEAVE" in base_ids:

        concepts.append({
            "operation":
                "override",

            "id":
                "LEAVE",

            "resource_status":
                "v05_candidate",

            "generation_enabled":
                False,

            "renderer_strategy":
                "motion_place",

            "notes":
                "Activate only after motion regression.",
        })

    else:

        concepts.append(
            make_concept(
                concept_id="LEAVE",
                concept_type="verb",
                semantic_class="MOTION_VERB",
                zh="离开",
                en="leave",
                ru="уходить",
                uz="ketmoq",
                generation_enabled=False,
                renderer_strategy="motion_place",
            )
        )

    # ========================================================
    # Frames
    # ========================================================

    frames = {
        "version":
            "0.5.0-batch1",

        "inherits":
            "../v04/frames_v04.json",

        "generation_enabled":
            False,

        "frames": [
            {
                "id":
                    "NEED_OBJECT",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "medical",
                    "shopping",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "NEED",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "SEE_OBJECT",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "work_study",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "SEE",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "CALL_PERSON",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "communication",
                    "daily",
                    "travel",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "CALL",
                    },
                    {
                        "name": "person",
                        "type": "person",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "WAIT_PERSON",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "transport",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "WAIT",
                    },
                    {
                        "name": "person",
                        "type": "person",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "WAIT_AT_PLACE",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "travel",
                    "transport",
                    "hotel",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "WAIT",
                    },
                    {
                        "name": "place",
                        "type": "place",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "GIVE_OBJECT_PERSON",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "hotel",
                    "restaurant",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "GIVE",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                    {
                        "name": "person",
                        "type": "person",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "TAKE_OBJECT",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "shopping",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "TAKE",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "BRING_OBJECT_DESTINATION",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "hotel",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "BRING",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                    {
                        "name": "destination",
                        "type": "place",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "LOSE_OBJECT",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "LOSE",
                    },
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "LEAVE_PLACE",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "travel",
                    "transport",
                    "daily",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "LEAVE",
                    },
                    {
                        "name": "place",
                        "type": "place",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "RETURN_PLACE",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "travel",
                    "daily",
                ],

                "slots": [
                    {
                        "name": "subject",
                        "type": "person",
                        "required": True,
                    },
                    {
                        "name": "verb",
                        "type": "verb",
                        "required": True,
                        "fixed": "RETURN",
                    },
                    {
                        "name": "destination",
                        "type": "place",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "WHERE_OBJECT",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "hotel",
                ],

                "slots": [
                    {
                        "name": "object",
                        "type": "object",
                        "required": True,
                    },
                ],
            },
            {
                "id":
                    "WHERE_PERSON",

                "generation_enabled":
                    False,

                "scenario_tags": [
                    "daily",
                    "travel",
                    "communication",
                ],

                "slots": [
                    {
                        "name": "person",
                        "type": "person",
                        "required": True,
                    },
                ],
            },
        ],
    }

    # ========================================================
    # Compatibility
    # ========================================================

    compatibility = {
        "version":
            "0.5.0-batch1",

        "inherits":
            "../v04/semantic_compatibility_v04.json",

        "generation_enabled":
            False,

        "semantic_classes": {
            "PORTABLE_OBJECT": [
                "LUGGAGE",
            ],
            "IDENTITY_DOCUMENT": [
                "ID_CARD",
            ],
            "MONEY": [
                "MONEY",
            ],
            "DEVICE_ACCESSORY": [
                "CHARGER",
            ],
            "CLOTHING": [
                "CLOTHES",
            ],
            "INFORMATION_OBJECT": [
                "ADDRESS",
            ],
            "ROOM_OBJECT": [
                "ROOM",
            ],
            "FURNITURE": [
                "TABLE",
            ],
            "LOCAL_PLACE": [
                "PHARMACY",
                "SUPERMARKET",
                "SHOP",
                "CAFE",
                "POLICE_STATION",
                "CITY_CENTER",
                "TOILET",
            ],
            "TRANSPORT_PLACE": [
                "BUS_STOP",
                "SUBWAY_STATION",
            ],
        },

        "verb_rules": {
            "NEED": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "MONEY",
                    "DEVICE_ACCESSORY",
                    "CLOTHING",
                    "INFORMATION_OBJECT",
                ],
            },
            "SEE": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "DEVICE_ACCESSORY",
                    "FURNITURE",
                    "ROOM_OBJECT",
                ],
            },
            "CALL": {
                "requires_person":
                    True,
            },
            "WAIT": {
                "allows_person":
                    True,

                "allows_place":
                    True,
            },
            "GIVE": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "MONEY",
                    "DEVICE_ACCESSORY",
                ],

                "requires_person":
                    True,
            },
            "TAKE": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "MONEY",
                    "DEVICE_ACCESSORY",
                    "CLOTHING",
                ],
            },
            "BRING": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "DEVICE_ACCESSORY",
                    "CLOTHING",
                ],

                "destination_classes": [
                    "LOCAL_PLACE",
                    "TRANSPORT_PLACE",
                ],
            },
            "LOSE": {
                "object_classes": [
                    "PORTABLE_OBJECT",
                    "IDENTITY_DOCUMENT",
                    "MONEY",
                    "DEVICE_ACCESSORY",
                ],
            },
            "LEAVE": {
                "place_classes": [
                    "LOCAL_PLACE",
                    "TRANSPORT_PLACE",
                ],
            },
            "RETURN": {
                "destination_classes": [
                    "LOCAL_PLACE",
                    "TRANSPORT_PLACE",
                ],
            },
        },
    }

    # ========================================================
    # Generation policy
    # ========================================================

    generation_policy = {
        "version":
            "0.5.0-batch1",

        "inherits":
            "../v04/generation_policy_v044.json",

        "generation_enabled":
            False,

        "activation_policy": {
            "new_concepts":
                "disabled_until_renderer_ready",

            "new_verbs":
                "disabled_until_morphology_regression",

            "new_frames":
                "disabled_until_frame_regression",

            "promotion_sequence": [
                "resource_validation",
                "renderer_unit_tests",
                "frame_unit_tests",
                "compatibility_validation",
                "100_sample_smoke",
                "500_sample_pilot",
                "qwen_targeted_audit",
            ],
        },

        "scenario_targets": {
            "daily":
                0.18,

            "travel":
                0.18,

            "transport":
                0.12,

            "shopping":
                0.12,

            "hotel":
                0.10,

            "restaurant":
                0.08,

            "medical":
                0.08,

            "communication":
                0.08,

            "work_study":
                0.06,
        },

        "coverage_limits": {
            "max_single_verb_ratio":
                0.12,

            "max_single_frame_ratio":
                0.12,

            "max_single_object_ratio":
                0.08,

            "max_single_place_ratio":
                0.08,
        },
    }

    # ========================================================
    # Save
    # ========================================================

    write_json(
        V05_DIR / "manifest_v05.json",
        manifest,
    )

    write_jsonl(
        V05_DIR / "concepts_v05.jsonl",
        concepts,
    )

    write_json(
        V05_DIR / "frames_v05.json",
        frames,
    )

    write_json(
        V05_DIR
        / "semantic_compatibility_v05.json",
        compatibility,
    )

    write_json(
        V05_DIR
        / "generation_policy_v05.json",
        generation_policy,
    )

    print("=" * 90)
    print("BUILD V0.5 BATCH-1 RESOURCES")
    print("=" * 90)

    print(
        "Frozen base concepts:",
        len(base_rows),
    )

    additions = [
        x
        for x in concepts
        if x["operation"] == "add"
    ]

    overrides = [
        x
        for x in concepts
        if x["operation"] == "override"
    ]

    print(
        "V0.5 additions:",
        len(additions),
    )

    print(
        "V0.5 overrides:",
        len(overrides),
    )

    print(
        "V0.5 new frames:",
        len(frames["frames"]),
    )

    print()

    for path in [
        V05_DIR / "manifest_v05.json",
        V05_DIR / "concepts_v05.jsonl",
        V05_DIR / "frames_v05.json",
        V05_DIR / "semantic_compatibility_v05.json",
        V05_DIR / "generation_policy_v05.json",
    ]:
        print(path)

    print()
    print("V0.5 BATCH-1 RESOURCE BUILD PASS")


if __name__ == "__main__":
    main()