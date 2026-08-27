from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "verb_policies_v2.json"
)


# 当前项目 Semantic Frame 使用的人称集合。
#
# 如果后面增加 PERSON_YOU_PLURAL / 2pl，
# 再统一扩展这里和 verb_policies_v2.json。
SUPPORTED_PERSONS = {
    "1sg",
    "2sg",
    "3sg",
    "1pl",
    "3pl",
}


# ============================================================
# Resource loading
# ============================================================

def load_verb_policies(
    path: Path | str = POLICY_FILE,
) -> dict:
    """
    Load Renderer V2 verb policies.

    Expected file:

        data/synthetic/resources/verb_policies_v2.json

    Expected basic structure:

        {
          "version": "v2",
          "verbs": {
            "FIND": {
              ...
            }
          }
        }
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Verb policy file not found:\n"
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "verb_policies_v2.json must "
            "contain a JSON object."
        )

    verbs = data.get("verbs")

    if not isinstance(verbs, dict):
        raise ValueError(
            "verb_policies_v2.json is missing "
            "a valid 'verbs' object."
        )

    return data


# ============================================================
# Generic helpers
# ============================================================

def normalize_concept_id(
    concept_id: Any,
) -> str:
    """
    Normalize Concept ID.

    Examples:

        FIND
        find
        VERB_FIND
        ACTION_FIND
    """

    if concept_id is None:
        return ""

    return str(
        concept_id
    ).strip().upper()


def is_find_concept(
    concept_id: Any,
) -> bool:
    """
    Determine whether a concept represents FIND.

    Compatible with IDs such as:

        FIND
        VERB_FIND
        ACTION_FIND

    Current project data may use slightly different
    naming conventions, so V2 intentionally keeps
    this check tolerant.
    """

    normalized = normalize_concept_id(
        concept_id
    )

    if not normalized:
        return False

    if normalized == "FIND":
        return True

    if normalized.endswith("_FIND"):
        return True

    if normalized.startswith("FIND_"):
        return True

    return "FIND" in normalized


def get_policy_key(
    concept_id: Any,
) -> str | None:
    """
    Map project Concept IDs to a policy key.

    Currently Renderer V2 only contains FIND policy.
    """

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
    """
    Replace exactly the first occurrence.

    Returns None if the old surface does not exist.

    This is deliberately strict because rebuilding
    a sentence without finding its original trace
    surface means the Renderer state is inconsistent.
    """

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
# Policy helpers
# ============================================================

def get_verb_policy(
    verb_id: Any,
    policies: dict,
) -> dict | None:
    """
    Return the V2 policy for the given verb concept.
    """

    policy_key = get_policy_key(
        verb_id
    )

    if policy_key is None:
        return None

    verb_policies = policies.get(
        "verbs",
        {},
    )

    policy = verb_policies.get(
        policy_key
    )

    if not isinstance(
        policy,
        dict,
    ):
        return None

    return policy


# ============================================================
# Chinese Renderer V2
# ============================================================

def render_zh_verb_v2(
    *,
    verb_id: str,
    original_surface: str,
    tense: str,
    polarity: str,
    policies: dict,
) -> str:
    """
    Chinese verb rendering policy V2.1.

    Main purpose:
    correctly render resultative FIND in Chinese.

    FIND:
        present + positive -> 找到了
        present + negative -> 没找到
        future  + positive -> 会找到
        future  + negative -> 不会找到

    Other verbs keep the original V0.1 surface unless
    explicitly configured in the policy resource.
    """

    verb_id_norm = (
        str(verb_id)
        .strip()
        .upper()
    )

    tense_norm = (
        str(tense)
        .strip()
        .lower()
    )

    polarity_norm = (
        str(polarity)
        .strip()
        .lower()
    )

    # ========================================================
    # FIND
    # ========================================================

    if verb_id_norm == "FIND":

        verb_policy = (
            policies
            .get("FIND", {})
            .get("zh", {})
        )

        # ----------------------------------------------------
        # Future
        # ----------------------------------------------------

        if tense_norm == "future":

            if polarity_norm == "neg":

                return verb_policy.get(
                    "future_negative",
                    "不会找到",
                )

            return verb_policy.get(
                "future_positive",
                "会找到",
            )

        # ----------------------------------------------------
        # Present / non-future
        # ----------------------------------------------------

        if polarity_norm == "neg":

            return verb_policy.get(
                "present_negative",
                "没找到",
            )

        return verb_policy.get(
            "present_positive",
            "找到了",
        )

    # ========================================================
    # Other verbs:
    # preserve existing stable renderer output
    # ========================================================

    return original_surface
# ============================================================
# Russian helpers
# ============================================================

def get_ru_perfective_future_form(
    *,
    verb_id: str,
    person: str,
    policies: dict,
) -> str | None:
    """
    Retrieve the Russian perfective future form.

    For FIND / найти:

        1sg -> найду
        2sg -> найдёшь
        3sg -> найдёт
        1pl -> найдём
        3pl -> найдут
    """

    policy = get_verb_policy(
        verb_id,
        policies,
    )

    if policy is None:
        return None

    ru_policy = policy.get(
        "ru",
        {},
    )

    perfective = ru_policy.get(
        "perfective",
        {},
    )

    if not isinstance(
        perfective,
        dict,
    ):
        return None

    value = perfective.get(
        f"future_{person}"
    )

    if value is None:
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    return value


# ============================================================
# Russian Renderer V2
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
    Render Russian verb surface according to V2 policy.

    Confirmed problem:

        FIND + future

    Previous Renderer could produce:

        Они будут находить телефон.

    For a single completed future FIND event, V2 uses
    the perfective verb найти:

        Они найдут телефон.

    Negative example:

        Он не найдёт телефон.

    Present FIND remains untouched in this version.

    Important:
    This is intentionally a narrow patch.
    We are not yet globally rewriting Russian aspect.
    """

    policy = get_verb_policy(
        verb_id,
        policies,
    )

    if policy is None:
        return original_surface

    policy_key = get_policy_key(
        verb_id
    )

    if policy_key != "FIND":
        return original_surface

    if tense != "future":
        return original_surface

    if person not in SUPPORTED_PERSONS:
        return original_surface

    ru_policy = policy.get(
        "ru",
        {},
    )

    aspect_policy = ru_policy.get(
        "aspect_policy",
        {},
    )

    future_aspect = aspect_policy.get(
        "future"
    )

    # This policy explicitly tells us that future FIND
    # should use the perfective paradigm.
    if future_aspect != "perfective":
        return original_surface

    future_form = (
        get_ru_perfective_future_form(
            verb_id=verb_id,
            person=person,
            policies=policies,
        )
    )

    if not future_form:
        raise ValueError(
            "Missing Russian perfective future form: "
            f"verb={verb_id}, "
            f"person={person}"
        )

    # --------------------------------------------------------
    # Positive future
    #
    # Он найдёт.
    # Мы найдём.
    # --------------------------------------------------------

    if polarity == "pos":
        return future_form

    # --------------------------------------------------------
    # Negative future
    #
    # Он не найдёт.
    # Мы не найдём.
    # --------------------------------------------------------

    if polarity == "neg":
        return (
            "не "
            + future_form
        )

    # Unknown polarity:
    # safest behavior is to leave original untouched.

    return original_surface


# ============================================================
# Optional diagnostic helper
# ============================================================

def describe_v2_policy(
    verb_id: str,
    policies: dict,
) -> dict:
    """
    Small diagnostic helper.

    Not required by rebuild_semantic_v02.py,
    but useful when debugging resources.
    """

    key = get_policy_key(
        verb_id
    )

    if key is None:
        return {
            "matched":
                False,

            "verb_id":
                verb_id,

            "policy_key":
                None,
        }

    policy = (
        policies
        .get(
            "verbs",
            {},
        )
        .get(
            key,
            {},
        )
    )

    return {
        "matched":
            True,

        "verb_id":
            verb_id,

        "policy_key":
            key,

        "policy":
            policy,
    }


# ============================================================
# Self-test
# ============================================================

def _self_test() -> None:
    """
    Optional lightweight local test.

    Run:

        python -m scripts.synthetic.renderer_v2

    It does not alter any project data.
    """

    print("=" * 80)
    print("RENDERER V2 SELF TEST")
    print("=" * 80)

    policies = load_verb_policies()

    print(
        "\nPolicy file:"
    )
    print(
        POLICY_FILE
    )

    print(
        "\nFIND policy loaded:",
        get_policy_key("FIND")
    )

    # Chinese
    zh = render_zh_verb_v2(
        verb_id="FIND",
        original_surface="不找到",
        tense="present",
        polarity="neg",
        policies=policies,
    )

    print(
        "\nZH FIND negative:"
    )
    print(
        "input : 不找到"
    )
    print(
        "output:",
        zh
    )

    # Russian
    tests = [
        (
            "1sg",
            "pos",
        ),
        (
            "2sg",
            "pos",
        ),
        (
            "3sg",
            "pos",
        ),
        (
            "1pl",
            "pos",
        ),
        (
            "3pl",
            "pos",
        ),
        (
            "3sg",
            "neg",
        ),
    ]

    print(
        "\nRU FIND future:"
    )

    for (
        person,
        polarity,
    ) in tests:
        value = render_ru_verb_v2(
            verb_id="FIND",
            original_surface="будет находить",
            person=person,
            tense="future",
            polarity=polarity,
            policies=policies,
        )

        print(
            f"{person:<4}"
            f" {polarity:<3}"
            f" -> "
            f"{value}"
        )

    print(
        "\nSelf-test complete."
    )


if __name__ == "__main__":
    _self_test()