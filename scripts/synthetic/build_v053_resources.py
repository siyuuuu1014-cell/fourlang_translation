from __future__ import annotations

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

PARENT_VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v052.json"
)

PARENT_ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v052.json"
)

OUTPUT_VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v053.json"
)

OUTPUT_ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v053.json"
)


VERSION = "0.5.3"
FROZEN_PARENT = "0.5.2"


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected dict: {path}"
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


# =============================================================================
# GIVE morphology
# =============================================================================

GIVE_RESOURCE = {
    "semantic_type": "ditransitive",

    "zh": {
        "present_positive": "给",
        "present_negative": "不给",
        "future_positive": "会给",
        "future_negative": "不会给",
    },

    "en": {
        "lemma": "give",

        "present": {
            "1sg": "give",
            "2sg": "give",
            "3sg": "gives",
            "1pl": "give",
            "2pl": "give",
            "3pl": "give",
        },

        "present_negative": {
            "1sg": "do not give",
            "2sg": "do not give",
            "3sg": "does not give",
            "1pl": "do not give",
            "2pl": "do not give",
            "3pl": "do not give",
        },

        "future": "will give",
        "future_negative": "will not give",
    },

    "ru": {
        "present": {
            "1sg": "даю",
            "2sg": "даёшь",
            "3sg": "даёт",
            "1pl": "даём",
            "2pl": "даёте",
            "3pl": "дают",
        },

        "future": {
            "1sg": "дам",
            "2sg": "дашь",
            "3sg": "даст",
            "1pl": "дадим",
            "2pl": "дадите",
            "3pl": "дадут",
        },
    },

    "uz": {
        "present_future": {
            "1sg": "beraman",
            "2sg": "berasan",
            "3sg": "beradi",
            "1pl": "beramiz",
            "2pl": "berasiz",
            "3pl": "beradilar",
        },

        "negative": {
            "1sg": "bermayman",
            "2sg": "bermaysan",
            "3sg": "bermaydi",
            "1pl": "bermaymiz",
            "2pl": "bermaysiz",
            "3pl": "bermaydilar",
        },
    },
}


# =============================================================================
# BRING morphology
# =============================================================================

BRING_RESOURCE = {
    "semantic_type":
        "transitive_destination",

    "zh": {
        "present_positive": "带到",
        "present_negative": "不带到",
        "future_positive": "会带到",
        "future_negative": "不会带到",
    },

    "en": {
        "lemma": "bring",

        "present": {
            "1sg": "bring",
            "2sg": "bring",
            "3sg": "brings",
            "1pl": "bring",
            "2pl": "bring",
            "3pl": "bring",
        },

        "present_negative": {
            "1sg": "do not bring",
            "2sg": "do not bring",
            "3sg": "does not bring",
            "1pl": "do not bring",
            "2pl": "do not bring",
            "3pl": "do not bring",
        },

        "future": "will bring",
        "future_negative": "will not bring",
    },

    "ru": {
        "present": {
            "1sg": "приношу",
            "2sg": "приносишь",
            "3sg": "приносит",
            "1pl": "приносим",
            "2pl": "приносите",
            "3pl": "приносят",
        },

        "future": {
            "1sg": "принесу",
            "2sg": "принесёшь",
            "3sg": "принесёт",
            "1pl": "принесём",
            "2pl": "принесёте",
            "3pl": "принесут",
        },
    },

    # Controlled semantic meaning:
    # carry/take an object to a destination.
    "uz": {
        "present_future": {
            "1sg": "olib boraman",
            "2sg": "olib borasan",
            "3sg": "olib boradi",
            "1pl": "olib boramiz",
            "2pl": "olib borasiz",
            "3pl": "olib boradilar",
        },

        "negative": {
            "1sg": "olib bormayman",
            "2sg": "olib bormaysan",
            "3sg": "olib bormaydi",
            "1pl": "olib bormaymiz",
            "2pl": "olib bormaysiz",
            "3pl": "olib bormaydilar",
        },
    },
}


# =============================================================================
# Directional destinations
# =============================================================================

DIRECTIONAL_FORMS = {
    "AIRPORT": {
        "zh_dir": "机场",
        "en_dir": "to the airport",
        "ru_dir": "в аэропорт",
        "uz_dir": "aeroportga",
    },

    "STATION": {
        "zh_dir": "车站",
        "en_dir": "to the station",
        "ru_dir": "на вокзал",
        "uz_dir": "vokzalga",
    },

    "BANK": {
        "zh_dir": "银行",
        "en_dir": "to the bank",
        "ru_dir": "в банк",
        "uz_dir": "bankka",
    },

    "HOSPITAL": {
        "zh_dir": "医院",
        "en_dir": "to the hospital",
        "ru_dir": "в больницу",
        "uz_dir": "shifoxonaga",
    },

    "SCHOOL": {
        "zh_dir": "学校",
        "en_dir": "to the school",
        "ru_dir": "в школу",
        "uz_dir": "maktabga",
    },

    "OFFICE": {
        "zh_dir": "办公室",
        "en_dir": "to the office",
        "ru_dir": "в офис",
        "uz_dir": "ofisga",
    },

    "HOTEL": {
        "zh_dir": "酒店",
        "en_dir": "to the hotel",
        "ru_dir": "в отель",
        "uz_dir": "mehmonxonaga",
    },

    "RESTAURANT": {
        "zh_dir": "餐厅",
        "en_dir": "to the restaurant",
        "ru_dir": "в ресторан",
        "uz_dir": "restoranga",
    },

    "TASHKENT": {
        "zh_dir": "塔什干",
        "en_dir": "to Tashkent",
        "ru_dir": "в Ташкент",
        "uz_dir": "Toshkentga",
    },

    "BEIJING": {
        "zh_dir": "北京",
        "en_dir": "to Beijing",
        "ru_dir": "в Пекин",
        "uz_dir": "Pekinga",
    },

    "MOSCOW": {
        "zh_dir": "莫斯科",
        "en_dir": "to Moscow",
        "ru_dir": "в Москву",
        "uz_dir": "Moskvaga",
    },
}


GIVE_FRAME = {
    "verb": "GIVE",

    "required_slots": [
        "subject",
        "verb",
        "object",
        "person",
    ],

    "forbidden_slots": [
        "destination",
        "place",
        "time",
        "day",
    ],

    "argument_roles": {
        "object": {
            "en": "direct_object",
            "ru": "accusative",
            "uz": "accusative",
        },

        "person": {
            "en": "indirect_object",
            "ru": "dative",
            "uz": "dative",
        },
    },

    "constraints": [
        "subject_must_not_equal_person",
        "speaker_group_overlap_not_allowed",
        "object_person_roles_must_not_swap",
    ],

    "generation_enabled": False,
}


BRING_FRAME = {
    "verb": "BRING",

    "required_slots": [
        "subject",
        "verb",
        "object",
        "destination",
    ],

    "forbidden_slots": [
        "person",
        "place",
        "time",
        "day",
    ],

    "argument_roles": {
        "object": {
            "en": "direct_object",
            "ru": "accusative",
            "uz": "accusative",
        },

        "destination": {
            "zh": "directional",
            "en": "directional",
            "ru": "directional",
            "uz": "directional",
        },
    },

    "constraints": [
        "object_destination_roles_must_not_swap",
        "destination_must_use_directional_form",
    ],

    "generation_enabled": False,
}


def main() -> None:

    print("=" * 100)
    print("BUILD V0.5.3 RESOURCES")
    print("=" * 100)

    verb_parent = read_json(
        PARENT_VERB_FILE
    )

    arg_parent = read_json(
        PARENT_ARGUMENT_FILE
    )

    verb_data = copy.deepcopy(
        verb_parent
    )

    arg_data = copy.deepcopy(
        arg_parent
    )

    # -------------------------------------------------------------------------
    # Verb resource
    # -------------------------------------------------------------------------

    verb_data["version"] = VERSION
    verb_data["resource_version"] = VERSION
    verb_data["frozen_parent"] = FROZEN_PARENT

    verbs = verb_data.setdefault(
        "verbs",
        {},
    )

    verbs["GIVE"] = GIVE_RESOURCE
    verbs["BRING"] = BRING_RESOURCE

    # This means morphology exists, not that generator is enabled.
    verb_data["enabled_verbs"] = sorted(
        verbs.keys()
    )

    # -------------------------------------------------------------------------
    # Argument resource
    # -------------------------------------------------------------------------

    arg_data["version"] = VERSION
    arg_data["resource_version"] = VERSION
    arg_data["frozen_parent"] = FROZEN_PARENT

    places = arg_data.setdefault(
        "place_targets",
        {},
    )

    for place_id, directional in (
        DIRECTIONAL_FORMS.items()
    ):

        if place_id not in places:
            raise RuntimeError(
                f"Missing parent place: {place_id}"
            )

        places[
            place_id
        ].update(
            directional
        )

    contracts = arg_data.setdefault(
        "frame_contracts_v053",
        {},
    )

    # Preserve old contracts too.
    old_contracts = arg_data.get(
        "frame_contracts_v052",
        {},
    )

    for frame_id, contract in (
        old_contracts.items()
    ):
        contracts.setdefault(
            frame_id,
            copy.deepcopy(contract),
        )

    contracts[
        "GIVE_OBJECT_PERSON"
    ] = GIVE_FRAME

    contracts[
        "BRING_OBJECT_DESTINATION"
    ] = BRING_FRAME

    write_json(
        OUTPUT_VERB_FILE,
        verb_data,
    )

    write_json(
        OUTPUT_ARGUMENT_FILE,
        arg_data,
    )

    print(
        "Verb resource:",
        OUTPUT_VERB_FILE,
    )

    print(
        "Argument resource:",
        OUTPUT_ARGUMENT_FILE,
    )

    print(
        "Verb keys:",
        sorted(
            verb_data["verbs"].keys()
        ),
    )

    print(
        "V0.5.3 contracts:",
        sorted(
            contracts.keys()
        ),
    )

    print()
    print(
        "Directional destinations:",
        len(DIRECTIONAL_FORMS),
    )

    print()
    print("=" * 100)
    print("V0.5.3 RESOURCE BUILD PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()