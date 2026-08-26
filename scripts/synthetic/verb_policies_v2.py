from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "verb_policies_v2.json"
)


SUPPORTED_PERSONS = {
    "1sg",
    "2sg",
    "3sg",
    "1pl",
    "3pl",
}


# ============================================================
# Resources
# ============================================================

def load_verb_policies(
    path: Path = POLICY_FILE,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Verb policy file not found:\n"
            f"{path}"
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

        raise ValueError(
            "verb_policies_v2.json "
            "must contain a JSON object."
        )

    if "verbs" not in data:

        raise ValueError(
            "Missing field: verbs"
        )

    return data


# ============================================================
# Generic helpers
# ============================================================

def normalize_concept_id(
    concept_id: Any,
) -> str:

    return str(
        concept_id or ""
    ).strip().upper()


def is_find_concept(
    concept_id: Any,
) -> bool:

    """
    Compatible with IDs such as:

    FIND
    VERB_FIND
    ACTION_FIND
    """

    concept_id = normalize_concept_id(
        concept_id
    )

    if not concept_id:
        return False

    return (
        concept_id == "FIND"
        or
        concept_id.endswith("_FIND")
        or
        "FIND" in concept_id
    )


def get_policy_key(
    concept_id: Any,
) -> str | None:

    if is_find_concept(
        concept_id
    ):

        return "FIND"

    return None


def replace_once(
    text: str,
    old: str,
    new: str,
) -> str | None:

    text = str(text)
    old = str(old)
    new = str(new)

    if not old:
        return None

    if old not in text:
        return None

    return text.replace(
        old,
        new,
        1,
    )


# ============================================================
# Chinese
# ============================================================

def render_zh_verb_v2(
    *,
    verb_id: str,
    original_surface: str,
    tense: str | None,
    polarity: str | None,
    policies: dict,
) -> str:

    """
    V2 currently modifies only known cases.

    All other verbs fall back to original_surface.
    """

    policy_key = get_policy_key(
        verb_id
    )

    if policy_key is None:

        return original_surface


    verb_policy = (
        policies
        .get(
            "verbs",
            {},
        )
        .get(
            policy_key,
            {},
        )
    )


    zh_policy = verb_policy.get(
        "zh",
        {},
    )


    # ========================================================
    # FIND + negative
    #
    # Existing renderer:
    # 不找到
    #
    # V2:
    # 没找到
    # ========================================================

    if (
        policy_key == "FIND"
        and
        polarity == "neg"
    ):

        negative_surface = (
            zh_policy.get(
                "negative_surface"
            )
        )

        if negative_surface:

            return str(
                negative_surface
            )


    # Everything else stays unchanged for now.

    return original_surface


# ============================================================
# Russian
# ============================================================

def render_ru_verb_v2(
    *,
    verb_id: str,
    original_surface: str,
    person: str | None,
    tense: str | None,
    polarity: str | None,
    policies: dict,
) -> str:

    """
    V2 currently focuses on Russian FIND aspect.

    Present:
        keep current imperfective behavior

    Future:
        use perfective FIND forms

        я найду
        ты найдёшь
        он найдёт
        мы найдём
        они найдут

    Negative future:
        не найду
        не найдёшь
        ...
    """

    policy_key = get_policy_key(
        verb_id
    )

    if policy_key is None:

        return original_surface


    if person not in SUPPORTED_PERSONS:

        return original_surface


    verb_policy = (
        policies
        .get(
            "verbs",
            {},
        )
        .get(
            policy_key,
            {},
        )
    )


    ru_policy = verb_policy.get(
        "ru",
        {},
    )


    aspect_policy = ru_policy.get(
        "aspect_policy",
        {},
    )


    # ========================================================
    # Future FIND
    # ========================================================

    if (
        policy_key == "FIND"
        and
        tense == "future"
        and
        aspect_policy.get(
            "future"
        )
        == "perfective"
    ):

        perfective = ru_policy.get(
            "perfective",
            {},
        )


        expected = perfective.get(
            f"future_{person}"
        )


        if not expected:

            raise ValueError(
                "Missing Russian perfective "
                f"future form for "
                f"verb={verb_id}, "
                f"person={person}"
            )


        expected = str(
            expected
        )


        if polarity == "neg":

            return (
                "не "
                + expected
            )


        return expected


    # ========================================================
    # Present:
    #
    # Do not modify yet.
    # Existing present forms are retained.
    # ========================================================

    return original_surface