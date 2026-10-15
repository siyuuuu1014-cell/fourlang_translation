from __future__ import annotations

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.4"
FROZEN_PARENT = "0.5.3"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

PARENT_VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v053.json"
)

PARENT_ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v053.json"
)

OUTPUT_VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v054.json"
)

OUTPUT_ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v054.json"
)


# =============================================================================
# Object morphology required by NEED
# =============================================================================

OBJECT_NEED_FORMS = {
    "ADDRESS": {
        "ru_nom": "адрес",
        "ru_gender": "m",
        "ru_number": "sg",
        "ru_need_agreement": "m",
        "uz_nom": "manzil",
    },

    "CHARGER": {
        "ru_nom": "зарядное устройство",
        "ru_gender": "n",
        "ru_number": "sg",
        "ru_need_agreement": "n",
        "uz_nom": "quvvatlagich",
    },

    "CLOTHES": {
        "ru_nom": "одежда",
        "ru_gender": "f",
        "ru_number": "sg",
        "ru_need_agreement": "f",
        "uz_nom": "kiyim",
    },

    "ID_CARD": {
        "ru_nom": "удостоверение личности",
        "ru_gender": "n",
        "ru_number": "sg",
        "ru_need_agreement": "n",
        "uz_nom": "shaxsiy guvohnoma",
    },

    "LUGGAGE": {
        "ru_nom": "багаж",
        "ru_gender": "m",
        "ru_number": "sg",
        "ru_need_agreement": "m",
        "uz_nom": "bagaj",
    },

    "MONEY": {
        "ru_nom": "деньги",
        "ru_gender": None,
        "ru_number": "pl",
        "ru_need_agreement": "pl",
        "uz_nom": "pul",
    },

    "ROOM": {
        "ru_nom": "комната",
        "ru_gender": "f",
        "ru_number": "sg",
        "ru_need_agreement": "f",
        "uz_nom": "xona",
    },

    "TABLE": {
        "ru_nom": "стол",
        "ru_gender": "m",
        "ru_number": "sg",
        "ru_need_agreement": "m",
        "uz_nom": "stol",
    },
}


# =============================================================================
# NEED
# =============================================================================

NEED_RESOURCE = {
    "semantic_type": "language_specific_need_construction",

    "zh": {
        "present_positive": "需要",
        "present_negative": "不需要",
        "future_positive": "会需要",
        "future_negative": "不会需要",
    },

    "en": {
        "lemma": "need",

        "present": {
            "1sg": "need",
            "2sg": "need",
            "3sg": "needs",
            "1pl": "need",
            "2pl": "need",
            "3pl": "need",
        },

        "present_negative": {
            "1sg": "do not need",
            "2sg": "do not need",
            "3sg": "does not need",
            "1pl": "do not need",
            "2pl": "do not need",
            "3pl": "do not need",
        },

        "future": "will need",
        "future_negative": "will not need",
    },

    "ru": {
        "construction": "dative_experiencer_predicative",

        "predicative": {
            "m": "нужен",
            "f": "нужна",
            "n": "нужно",
            "pl": "нужны",
        },

        "future_copula": {
            "sg": "будет",
            "pl": "будут",
        },
    },

    "uz": {
        "construction": "dative_experiencer_kerak",

        "present_positive": "kerak",
        "present_negative": "kerak emas",
        "future_positive": "kerak bo'ladi",
        "future_negative": "kerak bo'lmaydi",
    },
}


# =============================================================================
# LOSE
#
# IMPORTANT:
# V0.5.4 generation enables FUTURE ONLY.
# Present fields exist only for resource completeness and MUST NOT be generated.
# =============================================================================

LOSE_RESOURCE = {
    "semantic_type": "simple_transitive",

    "generation_tenses": [
        "future",
    ],

    "zh": {
        "present_positive": "弄丢",
        "present_negative": "没弄丢",
        "future_positive": "会弄丢",
        "future_negative": "不会弄丢",
    },

    "en": {
        "lemma": "lose",

        "present": {
            "1sg": "lose",
            "2sg": "lose",
            "3sg": "loses",
            "1pl": "lose",
            "2pl": "lose",
            "3pl": "lose",
        },

        "present_negative": {
            "1sg": "do not lose",
            "2sg": "do not lose",
            "3sg": "does not lose",
            "1pl": "do not lose",
            "2pl": "do not lose",
            "3pl": "do not lose",
        },

        "future": "will lose",
        "future_negative": "will not lose",
    },

    "ru": {
        "present": {
            "1sg": "теряю",
            "2sg": "теряешь",
            "3sg": "теряет",
            "1pl": "теряем",
            "2pl": "теряете",
            "3pl": "теряют",
        },

        "future": {
            "1sg": "потеряю",
            "2sg": "потеряешь",
            "3sg": "потеряет",
            "1pl": "потеряем",
            "2pl": "потеряете",
            "3pl": "потеряют",
        },
    },

    "uz": {
        "present_future": {
            "1sg": "yo'qotaman",
            "2sg": "yo'qotasan",
            "3sg": "yo'qotadi",
            "1pl": "yo'qotamiz",
            "2pl": "yo'qotasiz",
            "3pl": "yo'qotadilar",
        },

        "negative": {
            "1sg": "yo'qotmayman",
            "2sg": "yo'qotmaysan",
            "3sg": "yo'qotmaydi",
            "1pl": "yo'qotmaymiz",
            "2pl": "yo'qotmaysiz",
            "3pl": "yo'qotmaydilar",
        },
    },
}


# =============================================================================
# Frame contracts
# =============================================================================

NEED_FRAME = {
    "verb": "NEED",

    "required_slots": [
        "subject",
        "verb",
        "object",
    ],

    "forbidden_slots": [
        "person",
        "destination",
        "place",
        "time",
        "day",
    ],

    "allowed_tenses": [
        "present",
        "future",
    ],

    "allowed_polarities": [
        "pos",
        "neg",
    ],

    "argument_roles": {
        "subject": {
            "zh": "subject",
            "en": "subject",
            "ru": "dative_experiencer",
            "uz": "dative_experiencer",
        },

        "object": {
            "zh": "object",
            "en": "object",
            "ru": "nominative_need_argument",
            "uz": "bare_need_argument",
        },
    },

    "constraints": [
        "language_specific_need_construction",
        "ru_experiencer_dative",
        "ru_object_nominative",
        "ru_predicative_agreement",
        "uz_experiencer_dative",
        "uz_object_bare",
    ],

    "generation_enabled": False,
}


LOSE_FRAME = {
    "verb": "LOSE",

    "required_slots": [
        "subject",
        "verb",
        "object",
    ],

    "forbidden_slots": [
        "person",
        "destination",
        "place",
        "time",
        "day",
    ],

    # Critical V0.5.4 restriction.
    "allowed_tenses": [
        "future",
    ],

    "allowed_polarities": [
        "pos",
        "neg",
    ],

    "argument_roles": {
        "object": {
            "en": "direct_object",
            "ru": "accusative",
            "uz": "accusative",
        },
    },

    "constraints": [
        "future_only_generation",
        "past_not_yet_enabled",
        "present_generation_not_enabled",
    ],

    "generation_enabled": False,
}


# =============================================================================
# IO
# =============================================================================

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
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("BUILD V0.5.4 RESOURCES")
    print("=" * 100)

    parent_verbs = read_json(
        PARENT_VERB_FILE
    )

    parent_arguments = read_json(
        PARENT_ARGUMENT_FILE
    )

    verbs_data = copy.deepcopy(
        parent_verbs
    )

    arguments_data = copy.deepcopy(
        parent_arguments
    )

    # -------------------------------------------------------------------------
    # Version
    # -------------------------------------------------------------------------

    verbs_data["version"] = VERSION
    verbs_data["resource_version"] = VERSION
    verbs_data["frozen_parent"] = FROZEN_PARENT

    arguments_data["version"] = VERSION
    arguments_data["resource_version"] = VERSION
    arguments_data["frozen_parent"] = FROZEN_PARENT

    # -------------------------------------------------------------------------
    # Verbs
    # -------------------------------------------------------------------------

    verbs = verbs_data.setdefault(
        "verbs",
        {},
    )

    verbs["NEED"] = NEED_RESOURCE
    verbs["LOSE"] = LOSE_RESOURCE

    # realization_available, NOT generator activation
    verbs_data["enabled_verbs"] = sorted(
        verbs.keys()
    )

    # -------------------------------------------------------------------------
    # Objects
    # -------------------------------------------------------------------------

    objects = arguments_data.setdefault(
        "objects",
        {},
    )

    for object_id, forms in OBJECT_NEED_FORMS.items():
        if object_id not in objects:
            raise RuntimeError(
                f"Missing parent object: {object_id}"
            )

        objects[
            object_id
        ].update(
            forms
        )

    # -------------------------------------------------------------------------
    # Contracts
    # -------------------------------------------------------------------------

    contracts = {}

    parent_contracts = (
        arguments_data.get(
            "frame_contracts_v053",
            {}
        )
    )

    for frame_id, contract in parent_contracts.items():
        contracts[
            frame_id
        ] = copy.deepcopy(
            contract
        )

    contracts[
        "NEED_OBJECT"
    ] = NEED_FRAME

    contracts[
        "LOSE_OBJECT"
    ] = LOSE_FRAME

    arguments_data[
        "frame_contracts_v054"
    ] = contracts

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    write_json(
        OUTPUT_VERB_FILE,
        verbs_data,
    )

    write_json(
        OUTPUT_ARGUMENT_FILE,
        arguments_data,
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
        "NEED realization:",
        "PASS"
        if "NEED" in verbs
        else "FAIL",
    )

    print(
        "LOSE realization:",
        "PASS"
        if "LOSE" in verbs
        else "FAIL",
    )

    print(
        "Objects enriched:",
        len(OBJECT_NEED_FORMS),
    )

    print(
        "Contracts:",
        sorted(contracts.keys()),
    )

    print()
    print("=" * 100)
    print("V0.5.4 RESOURCE BUILD PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()