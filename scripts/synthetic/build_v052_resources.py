from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

BASE_VERBS = (
    RESOURCE_DIR
    / "verb_realization_v051.json"
)

BASE_ARGUMENTS = (
    RESOURCE_DIR
    / "argument_realization_v051.json"
)

OUTPUT_VERBS = (
    RESOURCE_DIR
    / "verb_realization_v052.json"
)

OUTPUT_ARGUMENTS = (
    RESOURCE_DIR
    / "argument_realization_v052.json"
)

OUTPUT_MANIFEST = (
    RESOURCE_DIR
    / "manifest_v052.json"
)


VERSION = "0.5.2"


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required resource not found: {path}"
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
# CALL
# ============================================================

CALL_REALIZATION = {
    "zh": {
        "present_positive": "打电话",
        "present_negative": "不打电话",
        "future_positive": "会打电话",
        "future_negative": "不会打电话",
    },

    "en": {
        "present": {
            "1sg": "call",
            "2sg": "call",
            "3sg": "calls",
            "1pl": "call",
            "2pl": "call",
            "3pl": "call",
        },

        "present_negative": {
            "1sg": "do not call",
            "2sg": "do not call",
            "3sg": "does not call",
            "1pl": "do not call",
            "2pl": "do not call",
            "3pl": "do not call",
        },

        "future": "will call",
        "future_negative": "will not call",
    },

    "ru": {
        "present": {
            "1sg": "звоню",
            "2sg": "звонишь",
            "3sg": "звонит",
            "1pl": "звоним",
            "2pl": "звоните",
            "3pl": "звонят",
        },

        "future": {
            "1sg": "позвоню",
            "2sg": "позвонишь",
            "3sg": "позвонит",
            "1pl": "позвоним",
            "2pl": "позвоните",
            "3pl": "позвонят",
        },
    },

    "uz": {
        "present_future": {
            "1sg": "qo'ng'iroq qilaman",
            "2sg": "qo'ng'iroq qilasan",
            "3sg": "qo'ng'iroq qiladi",
            "1pl": "qo'ng'iroq qilamiz",
            "2pl": "qo'ng'iroq qilasiz",
            "3pl": "qo'ng'iroq qiladilar",
        },

        "negative": {
            "1sg": "qo'ng'iroq qilmayman",
            "2sg": "qo'ng'iroq qilmaysan",
            "3sg": "qo'ng'iroq qilmaydi",
            "1pl": "qo'ng'iroq qilmaymiz",
            "2pl": "qo'ng'iroq qilmaysiz",
            "3pl": "qo'ng'iroq qilmaydilar",
        },
    },
}


# ============================================================
# WAIT
# ============================================================

WAIT_REALIZATION = {
    "zh": {
        "present_positive": "等",
        "present_negative": "不等",
        "future_positive": "会等",
        "future_negative": "不会等",
    },

    "en": {
        "present": {
            "1sg": "wait",
            "2sg": "wait",
            "3sg": "waits",
            "1pl": "wait",
            "2pl": "wait",
            "3pl": "wait",
        },

        "present_negative": {
            "1sg": "do not wait",
            "2sg": "do not wait",
            "3sg": "does not wait",
            "1pl": "do not wait",
            "2pl": "do not wait",
            "3pl": "do not wait",
        },

        "future": "will wait",
        "future_negative": "will not wait",
    },

    "ru": {
        "present": {
            "1sg": "жду",
            "2sg": "ждёшь",
            "3sg": "ждёт",
            "1pl": "ждём",
            "2pl": "ждёте",
            "3pl": "ждут",
        },

        "future": {
            "1sg": "буду ждать",
            "2sg": "будешь ждать",
            "3sg": "будет ждать",
            "1pl": "будем ждать",
            "2pl": "будете ждать",
            "3pl": "будут ждать",
        },
    },

    "uz": {
        "present_future": {
            "1sg": "kutaman",
            "2sg": "kutasan",
            "3sg": "kutadi",
            "1pl": "kutamiz",
            "2pl": "kutasiz",
            "3pl": "kutadilar",
        },

        "negative": {
            "1sg": "kutmayman",
            "2sg": "kutmaysan",
            "3sg": "kutmaydi",
            "1pl": "kutmaymiz",
            "2pl": "kutmaysiz",
            "3pl": "kutmaydilar",
        },
    },
}


# ============================================================
# Person argument forms
#
# IMPORTANT:
# These are TARGET / OBJECT forms.
#
# subject == target will initially be blocked by Generator
# because:
#
# I call me
# Я звоню мне
#
# actually requires reflexive semantics in some situations.
# We postpone reflexive constructions.
# ============================================================

PERSON_TARGETS = {

    "PERSON_I": {
        "zh": "我",

        "en_object": "me",

        "ru_acc": "меня",
        "ru_dat": "мне",

        "uz_acc": "meni",
        "uz_dat": "menga",
    },

    "PERSON_YOU": {
        "zh": "你",

        "en_object": "you",

        "ru_acc": "тебя",
        "ru_dat": "тебе",

        "uz_acc": "seni",
        "uz_dat": "senga",
    },

    "PERSON_HE": {
        "zh": "他",

        "en_object": "him",

        "ru_acc": "его",
        "ru_dat": "ему",

        "uz_acc": "uni",
        "uz_dat": "unga",
    },

    "PERSON_SHE": {
        "zh": "她",

        "en_object": "her",

        "ru_acc": "её",
        "ru_dat": "ей",

        "uz_acc": "uni",
        "uz_dat": "unga",
    },

    "PERSON_WE": {
        "zh": "我们",

        "en_object": "us",

        "ru_acc": "нас",
        "ru_dat": "нам",

        "uz_acc": "bizni",
        "uz_dat": "bizga",
    },

    "PERSON_THEY": {
        "zh": "他们",

        "en_object": "them",

        "ru_acc": "их",
        "ru_dat": "им",

        "uz_acc": "ularni",
        "uz_dat": "ularga",
    },
}


# ============================================================
# Place locative forms
#
# Store complete language-specific location expressions instead
# of attempting generic morphology.
# ============================================================

PLACE_TARGETS = {

    "AIRPORT": {
        "zh_loc": "在机场",
        "en_loc": "at the airport",
        "ru_loc": "в аэропорту",
        "uz_loc": "aeroportda",
    },

    "STATION": {
        "zh_loc": "在车站",
        "en_loc": "at the station",
        "ru_loc": "на вокзале",
        "uz_loc": "vokzalda",
    },

    "BANK": {
        "zh_loc": "在银行",
        "en_loc": "at the bank",
        "ru_loc": "в банке",
        "uz_loc": "bankda",
    },

    "HOSPITAL": {
        "zh_loc": "在医院",
        "en_loc": "at the hospital",
        "ru_loc": "в больнице",
        "uz_loc": "shifoxonada",
    },

    "SCHOOL": {
        "zh_loc": "在学校",
        "en_loc": "at school",
        "ru_loc": "в школе",
        "uz_loc": "maktabda",
    },

    "OFFICE": {
        "zh_loc": "在办公室",
        "en_loc": "at the office",
        "ru_loc": "в офисе",
        "uz_loc": "ofisda",
    },

    "HOTEL": {
        "zh_loc": "在酒店",
        "en_loc": "at the hotel",
        "ru_loc": "в отеле",
        "uz_loc": "mehmonxonada",
    },

    "RESTAURANT": {
        "zh_loc": "在餐厅",
        "en_loc": "at the restaurant",
        "ru_loc": "в ресторане",
        "uz_loc": "restoranda",
    },

    "TASHKENT": {
        "zh_loc": "在塔什干",
        "en_loc": "in Tashkent",
        "ru_loc": "в Ташкенте",
        "uz_loc": "Toshkentda",
    },

    "BEIJING": {
        "zh_loc": "在北京",
        "en_loc": "in Beijing",
        "ru_loc": "в Пекине",
        "uz_loc": "Pekinda",
    },

    "MOSCOW": {
        "zh_loc": "在莫斯科",
        "en_loc": "in Moscow",
        "ru_loc": "в Москве",
        "uz_loc": "Moskvada",
    },
}


# ============================================================
# Frame contracts
# ============================================================

FRAME_CONTRACTS = {

    "CALL_PERSON": {
        "verb": "CALL",

        "required_slots": [
            "subject",
            "verb",
            "person",
        ],

        "forbidden_slots": [
            "object",
            "destination",
            "place",
            "time",
            "day",
        ],

        "argument_roles": {
            "person": {
                "en": "object",
                "ru": "dative",
                "uz": "dative",
            }
        },

        "constraints": [
            "subject_must_not_equal_person"
        ],

        "generation_enabled": False,
    },

    "WAIT_PERSON": {
        "verb": "WAIT",

        "required_slots": [
            "subject",
            "verb",
            "person",
        ],

        "forbidden_slots": [
            "object",
            "destination",
            "place",
            "time",
            "day",
        ],

        "argument_roles": {
            "person": {
                "en": "wait_for_object",
                "ru": "accusative",
                "uz": "accusative",
            }
        },

        "constraints": [
            "subject_must_not_equal_person"
        ],

        "generation_enabled": False,
    },

    "WAIT_AT_PLACE": {
        "verb": "WAIT",

        "required_slots": [
            "subject",
            "verb",
            "place",
        ],

        "forbidden_slots": [
            "object",
            "person",
            "destination",
            "time",
            "day",
        ],

        "argument_roles": {
            "place": {
                "zh": "locative",
                "en": "locative",
                "ru": "locative",
                "uz": "locative",
            }
        },

        "generation_enabled": False,
    },
}


# ============================================================
# Validation
# ============================================================

def validate_person_targets() -> list[str]:

    errors = []

    required = {
        "zh",
        "en_object",
        "ru_acc",
        "ru_dat",
        "uz_acc",
        "uz_dat",
    }

    for person_id, data in (
        PERSON_TARGETS.items()
    ):

        missing = (
            required
            - set(data)
        )

        if missing:

            errors.append(
                f"{person_id}: missing "
                f"{sorted(missing)}"
            )

    return errors


def validate_places() -> list[str]:

    errors = []

    required = {
        "zh_loc",
        "en_loc",
        "ru_loc",
        "uz_loc",
    }

    for place_id, data in (
        PLACE_TARGETS.items()
    ):

        missing = (
            required
            - set(data)
        )

        if missing:

            errors.append(
                f"{place_id}: missing "
                f"{sorted(missing)}"
            )

    return errors


def validate_verb(
    verb_id: str,
    data: dict,
) -> list[str]:

    errors = []

    for language in (
        "zh",
        "en",
        "ru",
        "uz",
    ):

        if language not in data:

            errors.append(
                f"{verb_id}: "
                f"missing language "
                f"{language}"
            )

    return errors


# ============================================================
# Build
# ============================================================

def main() -> None:

    print("=" * 100)
    print("BUILD V0.5.2 BATCH-2A RESOURCES")
    print("=" * 100)

    base_verbs = read_json(
        BASE_VERBS
    )

    base_arguments = read_json(
        BASE_ARGUMENTS
    )

    verbs = deepcopy(
        base_verbs
    )

    arguments = deepcopy(
        base_arguments
    )

    # --------------------------------------------------------
    # Basic schema protection
    # --------------------------------------------------------

    if "verbs" not in verbs:

        raise RuntimeError(
            "verb_realization_v051.json "
            "does not contain 'verbs'."
        )

    # --------------------------------------------------------
    # Never silently overwrite an already-existing CALL/WAIT
    # unless it is identical.
    # --------------------------------------------------------

    for verb_id, resource in {
        "CALL": CALL_REALIZATION,
        "WAIT": WAIT_REALIZATION,
    }.items():

        existing = verbs[
            "verbs"
        ].get(
            verb_id
        )

        if (
            existing is not None
            and existing != resource
        ):

            raise RuntimeError(
                f"{verb_id} already exists "
                "with a different definition."
            )

        verbs[
            "verbs"
        ][
            verb_id
        ] = resource

    # --------------------------------------------------------
    # Extend arguments without modifying old sections.
    # --------------------------------------------------------

    arguments[
        "person_targets"
    ] = deepcopy(
        PERSON_TARGETS
    )

    arguments[
        "place_targets"
    ] = deepcopy(
        PLACE_TARGETS
    )

    arguments[
        "frame_contracts_v052"
    ] = deepcopy(
        FRAME_CONTRACTS
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    verbs[
        "resource_version"
    ] = VERSION

    arguments[
        "resource_version"
    ] = VERSION

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    errors = []

    errors.extend(
        validate_person_targets()
    )

    errors.extend(
        validate_places()
    )

    errors.extend(
        validate_verb(
            "CALL",
            CALL_REALIZATION,
        )
    )

    errors.extend(
        validate_verb(
            "WAIT",
            WAIT_REALIZATION,
        )
    )

    if errors:

        print()
        print("ERRORS")
        print("-" * 100)

        for error in errors:
            print(
                "ERROR:",
                error,
            )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    write_json(
        OUTPUT_VERBS,
        verbs,
    )

    write_json(
        OUTPUT_ARGUMENTS,
        arguments,
    )

    manifest = {
        "version":
            VERSION,

        "frozen_parent":
            "0.5.1",

        "batch":
            "Batch-2A",

        "new_verbs": [
            "CALL",
            "WAIT",
        ],

        "new_frames": [
            "CALL_PERSON",
            "WAIT_PERSON",
            "WAIT_AT_PLACE",
        ],

        "generation_enabled": False,

        "new_argument_features": [
            "person_en_object",
            "person_ru_accusative",
            "person_ru_dative",
            "person_uz_accusative",
            "person_uz_dative",
            "place_locative",
        ],

        "constraints": [
            "CALL_PERSON subject != person",
            "WAIT_PERSON subject != person",
            "reflexive semantics postponed",
        ],

        "files": {
            "verbs":
                str(
                    OUTPUT_VERBS
                ),

            "arguments":
                str(
                    OUTPUT_ARGUMENTS
                ),
        },
    }

    write_json(
        OUTPUT_MANIFEST,
        manifest,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print(
        "Frozen parent: 0.5.1"
    )

    print(
        "New verbs:",
        [
            "CALL",
            "WAIT",
        ],
    )

    print(
        "New frames:",
        [
            "CALL_PERSON",
            "WAIT_PERSON",
            "WAIT_AT_PLACE",
        ],
    )

    print(
        "Person target forms:",
        len(
            PERSON_TARGETS
        ),
    )

    print(
        "Place locative forms:",
        len(
            PLACE_TARGETS
        ),
    )

    print()
    print("Generation enabled: False")
    print()

    print("Files:")
    print(OUTPUT_VERBS)
    print(OUTPUT_ARGUMENTS)
    print(OUTPUT_MANIFEST)

    print()
    print("=" * 100)
    print(
        "V0.5.2 BATCH-2A RESOURCE BUILD PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()