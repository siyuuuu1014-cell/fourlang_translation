from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
)

CONCEPT_FILE = RESOURCE_DIR / "concepts.jsonl"
FRAME_FILE = RESOURCE_DIR / "frames.json"


# ============================================================
# helpers
# ============================================================

def pronoun(
    cid,
    person,
    zh,
    en,
    ru,
    uz,
):
    return {
        "id": cid,
        "pos": "pronoun",
        "semantic_type": "person",
        "meta": {
            "person": person,
        },
        "forms": {
            "zh": {"base": zh},
            "en": {"base": en},
            "ru": {"base": ru},
            "uz": {"base": uz},
        },
    }


def time_concept(
    cid,
    tense,
    semantic_type,
    zh,
    en,
    ru,
    uz,
):
    return {
        "id": cid,
        "pos": "time",
        "semantic_type": semantic_type,
        "meta": {
            "tense": tense,
        },
        "forms": {
            "zh": {"base": zh},
            "en": {"base": en},
            "ru": {"base": ru},
            "uz": {"base": uz},
        },
    }


def place(
    cid,
    zh_base,
    en_base,
    en_destination,
    ru_base,
    ru_destination,
    uz_base,
    uz_destination,
):
    return {
        "id": cid,
        "pos": "noun",
        "semantic_type": "destination",
        "meta": {},
        "forms": {
            "zh": {
                "base": zh_base,
                "destination": zh_base,
            },
            "en": {
                "base": en_base,
                "destination": en_destination,
            },
            "ru": {
                "base": ru_base,
                "destination": ru_destination,
            },
            "uz": {
                "base": uz_base,
                "destination": uz_destination,
            },
        },
    }


def obj(
    cid,
    object_type,
    zh,
    en,
    ru,
    uz,
):
    return {
        "id": cid,
        "pos": "noun",
        "semantic_type": "object",
        "meta": {
            "object_type": object_type,
        },
        "forms": {
            "zh": {
                "base": zh,
                "object": zh,
            },
            "en": {
                "base": en,
                "object": en,
            },
            "ru": {
                "base": ru,
                "object": ru,
            },
            "uz": {
                "base": uz,
                "object": uz,
            },
        },
    }


def verb(
    cid,
    verb_type,
    allowed_object_types,
    zh_base,
    en_base,
    en_3sg,
    ru_inf,
    ru_present,
    uz_positive,
    uz_negative,
):
    return {
        "id": cid,
        "pos": "verb",
        "semantic_type": verb_type,
        "meta": {
            "allowed_object_types": allowed_object_types,
        },
        "forms": {
            "zh": {
                "base": zh_base,
            },
            "en": {
                "base": en_base,
                "present_3sg": en_3sg,
            },
            "ru": {
                "inf": ru_inf,
                **{
                    f"present_{k}": v
                    for k, v
                    in ru_present.items()
                },
            },
            "uz": {
                **{
                    f"finite_pos_{k}": v
                    for k, v
                    in uz_positive.items()
                },
                **{
                    f"finite_neg_{k}": v
                    for k, v
                    in uz_negative.items()
                },
            },
        },
    }


# ============================================================
# concept lexicon
# ============================================================

CONCEPTS = []


# ------------------------------------------------------------
# Pronouns
# ------------------------------------------------------------

CONCEPTS.extend([
    pronoun("PERSON_I", "1sg", "我", "I", "я", "men"),
    pronoun("PERSON_YOU", "2sg", "你", "you", "ты", "sen"),
    pronoun("PERSON_HE", "3sg", "他", "he", "он", "u"),
    pronoun("PERSON_SHE", "3sg", "她", "she", "она", "u"),
    pronoun("PERSON_WE", "1pl", "我们", "we", "мы", "biz"),
    pronoun("PERSON_THEY", "3pl", "他们", "they", "они", "ular"),
])


# ------------------------------------------------------------
# Time
# ------------------------------------------------------------

CONCEPTS.extend([
    time_concept(
        "TIME_NOW",
        "present",
        "time",
        "现在",
        "now",
        "сейчас",
        "hozir",
    ),
    time_concept(
        "TIME_TODAY",
        "present",
        "day",
        "今天",
        "today",
        "сегодня",
        "bugun",
    ),
    time_concept(
        "TIME_TOMORROW",
        "future",
        "day",
        "明天",
        "tomorrow",
        "завтра",
        "ertaga",
    ),
    time_concept(
        "TIME_NEXT_WEEK",
        "future",
        "time",
        "下周",
        "next week",
        "на следующей неделе",
        "keyingi hafta",
    ),
])


# ------------------------------------------------------------
# Destinations / entities
# ------------------------------------------------------------

CONCEPTS.extend([
    place(
        "AIRPORT",
        "机场",
        "the airport",
        "to the airport",
        "аэропорт",
        "в аэропорт",
        "aeroport",
        "aeroportga",
    ),
    place(
        "HOTEL",
        "酒店",
        "the hotel",
        "to the hotel",
        "отель",
        "в отель",
        "mehmonxona",
        "mehmonxonaga",
    ),
    place(
        "STATION",
        "车站",
        "the station",
        "to the station",
        "вокзал",
        "на вокзал",
        "vokzal",
        "vokzalga",
    ),
    place(
        "HOSPITAL",
        "医院",
        "the hospital",
        "to the hospital",
        "больница",
        "в больницу",
        "shifoxona",
        "shifoxonaga",
    ),
    place(
        "RESTAURANT",
        "餐厅",
        "the restaurant",
        "to the restaurant",
        "ресторан",
        "в ресторан",
        "restoran",
        "restoranga",
    ),
    place(
        "BANK",
        "银行",
        "the bank",
        "to the bank",
        "банк",
        "в банк",
        "bank",
        "bankka",
    ),
    place(
        "OFFICE",
        "办公室",
        "the office",
        "to the office",
        "офис",
        "в офис",
        "ofis",
        "ofisga",
    ),
    place(
        "SCHOOL",
        "学校",
        "the school",
        "to the school",
        "школа",
        "в школу",
        "maktab",
        "maktabga",
    ),
    place(
        "TASHKENT",
        "塔什干",
        "Tashkent",
        "to Tashkent",
        "Ташкент",
        "в Ташкент",
        "Toshkent",
        "Toshkentga",
    ),
    place(
        "MOSCOW",
        "莫斯科",
        "Moscow",
        "to Moscow",
        "Москва",
        "в Москву",
        "Moskva",
        "Moskvaga",
    ),
    place(
        "BEIJING",
        "北京",
        "Beijing",
        "to Beijing",
        "Пекин",
        "в Пекин",
        "Pekin",
        "Pekinga",
    ),
])


# ------------------------------------------------------------
# Objects
# ------------------------------------------------------------

CONCEPTS.extend([
    obj(
        "TICKET",
        "ticket",
        "票",
        "a ticket",
        "билет",
        "chiptani",
    ),
    obj(
        "WATER",
        "drink",
        "水",
        "water",
        "воду",
        "suvni",
    ),
    obj(
        "FOOD",
        "food",
        "食物",
        "food",
        "еду",
        "ovqatni",
    ),
    obj(
        "PHONE",
        "device",
        "手机",
        "a phone",
        "телефон",
        "telefonni",
    ),
    obj(
        "PASSPORT",
        "document",
        "护照",
        "a passport",
        "паспорт",
        "pasportni",
    ),
    obj(
        "BOOK",
        "book",
        "书",
        "a book",
        "книгу",
        "kitobni",
    ),
    obj(
        "COFFEE",
        "drink",
        "咖啡",
        "coffee",
        "кофе",
        "qahvani",
    ),
    obj(
        "MEDICINE",
        "medicine",
        "药",
        "medicine",
        "лекарство",
        "dorini",
    ),
])


# ------------------------------------------------------------
# Verbs
# ------------------------------------------------------------

CONCEPTS.append(
    verb(
        "GO",
        "motion_verb",
        [],
        "去",
        "go",
        "goes",
        "ехать",
        {
            "1sg": "еду",
            "2sg": "едешь",
            "3sg": "едет",
            "1pl": "едем",
            "3pl": "едут",
        },
        {
            "1sg": "boraman",
            "2sg": "borasan",
            "3sg": "boradi",
            "1pl": "boramiz",
            "3pl": "boradilar",
        },
        {
            "1sg": "bormayman",
            "2sg": "bormaysan",
            "3sg": "bormaydi",
            "1pl": "bormaymiz",
            "3pl": "bormaydilar",
        },
    )
)


CONCEPTS.append(
    verb(
        "BUY",
        "transitive_verb",
        [
            "ticket",
            "drink",
            "food",
            "device",
            "document",
            "book",
            "medicine",
        ],
        "买",
        "buy",
        "buys",
        "покупать",
        {
            "1sg": "покупаю",
            "2sg": "покупаешь",
            "3sg": "покупает",
            "1pl": "покупаем",
            "3pl": "покупают",
        },
        {
            "1sg": "sotib olaman",
            "2sg": "sotib olasan",
            "3sg": "sotib oladi",
            "1pl": "sotib olamiz",
            "3pl": "sotib oladilar",
        },
        {
            "1sg": "sotib olmayman",
            "2sg": "sotib olmaysan",
            "3sg": "sotib olmaydi",
            "1pl": "sotib olmaymiz",
            "3pl": "sotib olmaydilar",
        },
    )
)


CONCEPTS.append(
    verb(
        "FIND",
        "transitive_verb",
        [
            "ticket",
            "drink",
            "food",
            "device",
            "document",
            "book",
            "medicine",
        ],
        "找到",
        "find",
        "finds",
        "находить",
        {
            "1sg": "нахожу",
            "2sg": "находишь",
            "3sg": "находит",
            "1pl": "находим",
            "3pl": "находят",
        },
        {
            "1sg": "topaman",
            "2sg": "topasan",
            "3sg": "topadi",
            "1pl": "topamiz",
            "3pl": "topadilar",
        },
        {
            "1sg": "topmayman",
            "2sg": "topmaysan",
            "3sg": "topmaydi",
            "1pl": "topmaymiz",
            "3pl": "topmaydilar",
        },
    )
)


CONCEPTS.append(
    verb(
        "EAT",
        "transitive_verb",
        ["food"],
        "吃",
        "eat",
        "eats",
        "есть",
        {
            "1sg": "ем",
            "2sg": "ешь",
            "3sg": "ест",
            "1pl": "едим",
            "3pl": "едят",
        },
        {
            "1sg": "yeyman",
            "2sg": "yeysan",
            "3sg": "yeydi",
            "1pl": "yeymiz",
            "3pl": "yeydilar",
        },
        {
            "1sg": "yemayman",
            "2sg": "yemaysan",
            "3sg": "yemaydi",
            "1pl": "yemaymiz",
            "3pl": "yemaydilar",
        },
    )
)


CONCEPTS.append(
    verb(
        "DRINK",
        "transitive_verb",
        ["drink"],
        "喝",
        "drink",
        "drinks",
        "пить",
        {
            "1sg": "пью",
            "2sg": "пьёшь",
            "3sg": "пьёт",
            "1pl": "пьём",
            "3pl": "пьют",
        },
        {
            "1sg": "ichaman",
            "2sg": "ichasan",
            "3sg": "ichadi",
            "1pl": "ichamiz",
            "3pl": "ichadilar",
        },
        {
            "1sg": "ichmayman",
            "2sg": "ichmaysan",
            "3sg": "ichmaydi",
            "1pl": "ichmaymiz",
            "3pl": "ichmaydilar",
        },
    )
)


CONCEPTS.append(
    verb(
        "READ",
        "transitive_verb",
        [
            "book",
            "document",
        ],
        "读",
        "read",
        "reads",
        "читать",
        {
            "1sg": "читаю",
            "2sg": "читаешь",
            "3sg": "читает",
            "1pl": "читаем",
            "3pl": "читают",
        },
        {
            "1sg": "o'qiyman",
            "2sg": "o'qiysan",
            "3sg": "o'qiydi",
            "1pl": "o'qiymiz",
            "3pl": "o'qiydilar",
        },
        {
            "1sg": "o'qimayman",
            "2sg": "o'qimaysan",
            "3sg": "o'qimaydi",
            "1pl": "o'qimaymiz",
            "3pl": "o'qimaydilar",
        },
    )
)


# ============================================================
# Grammar frames
# ============================================================

FRAMES = {
    "version": "0.1",
    "frames": [

        {
            "id": "MOTION_TIME",
            "weight": 3,

            "slot_order": [
                "subject",
                "time",
                "destination",
            ],

            "fixed_slots": {
                "verb": "GO",
            },

            "slots": {
                "subject": {
                    "pos": "pronoun",
                },
                "time": {
                    "pos": "time",
                },
                "destination": {
                    "semantic_type": "destination",
                },
            },

            "features": {
                "polarity": [
                    "pos",
                    "neg",
                ],
            },

            "roles": {
                "subject": "base",
                "time": "base",
                "destination": "destination",
            },

            "templates": {
                "zh": "{subject}{time}{verb}{destination}。",
                "en": "{subject} {verb} {destination} {time}.",
                "ru": "{subject} {time} {verb} {destination}.",
                "uz": "{subject} {time} {destination} {verb}.",
            },
        },

        {
            "id": "MOTION_CLOCK",
            "weight": 3,

            "slot_order": [
                "subject",
                "day",
                "destination",
            ],

            "fixed_slots": {
                "verb": "GO",
            },

            "slots": {
                "subject": {
                    "pos": "pronoun",
                },
                "day": {
                    "semantic_type": "day",
                },
                "destination": {
                    "semantic_type": "destination",
                },
            },

            "features": {
                "polarity": [
                    "pos",
                    "neg",
                ],
            },

            "computed": [
                "clock",
            ],

            "roles": {
                "subject": "base",
                "day": "base",
                "destination": "destination",
            },

            "templates": {
                "zh": "{subject}{day}{clock}{verb}{destination}。",
                "en": "{subject} {verb} {destination} at {clock} {day}.",
                "ru": "{subject} {day} в {clock} {verb} {destination}.",
                "uz": "{subject} {day} soat {clock} da {destination} {verb}.",
            },
        },

        {
            "id": "TRANSITIVE_TIME",
            "weight": 4,

            "slot_order": [
                "subject",
                "time",
                "verb",
                "object",
            ],

            "slots": {
                "subject": {
                    "pos": "pronoun",
                },
                "time": {
                    "pos": "time",
                },
                "verb": {
                    "semantic_type": "transitive_verb",
                },
                "object": {
                    "semantic_type": "object",
                },
            },

            "features": {
                "polarity": [
                    "pos",
                    "neg",
                ],
            },

            "dependencies": {
                "object_allowed_by_verb": True,
            },

            "roles": {
                "subject": "base",
                "time": "base",
                "object": "object",
            },

            "templates": {
                "zh": "{subject}{time}{verb}{object}。",
                "en": "{subject} {verb} {object} {time}.",
                "ru": "{subject} {time} {verb} {object}.",
                "uz": "{subject} {time} {object} {verb}.",
            },
        },

        {
            "id": "TRANSITIVE_CLOCK",
            "weight": 4,

            "slot_order": [
                "subject",
                "day",
                "verb",
                "object",
            ],

            "slots": {
                "subject": {
                    "pos": "pronoun",
                },
                "day": {
                    "semantic_type": "day",
                },
                "verb": {
                    "semantic_type": "transitive_verb",
                },
                "object": {
                    "semantic_type": "object",
                },
            },

            "features": {
                "polarity": [
                    "pos",
                    "neg",
                ],
            },

            "dependencies": {
                "object_allowed_by_verb": True,
            },

            "computed": [
                "clock",
            ],

            "roles": {
                "subject": "base",
                "day": "base",
                "object": "object",
            },

            "templates": {
                "zh": "{subject}{day}{clock}{verb}{object}。",
                "en": "{subject} {verb} {object} at {clock} {day}.",
                "ru": "{subject} {day} в {clock} {verb} {object}.",
                "uz": "{subject} {day} soat {clock} da {object} {verb}.",
            },
        },

        {
            "id": "WHERE_PLACE",
            "weight": 1,

            "slot_order": [
                "place",
            ],

            "slots": {
                "place": {
                    "semantic_type": "destination",
                },
            },

            "features": {},

            "roles": {
                "place": "base",
            },

            "templates": {
                "zh": "{place}在哪里？",
                "en": "Where is {place}?",
                "ru": "Где {place}?",
                "uz": "{place} qayerda?",
            },
        },
    ],
}


# ============================================================
# write
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    RESOURCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        CONCEPT_FILE.exists()
        and
        not args.force
    ):
        raise FileExistsError(
            f"{CONCEPT_FILE} 已存在。\n"
            f"如果确定覆盖，运行 --force"
        )

    with open(
        CONCEPT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for concept in CONCEPTS:

            f.write(
                json.dumps(
                    concept,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(
        FRAME_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            FRAMES,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 80)
    print("Synthetic resources initialized")
    print("=" * 80)

    print(
        "Concepts:",
        len(CONCEPTS),
    )

    print(
        "Frames:",
        len(FRAMES["frames"]),
    )

    print(CONCEPT_FILE)
    print(FRAME_FILE)


if __name__ == "__main__":
    main()