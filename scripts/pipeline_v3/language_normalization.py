from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from opencc import OpenCC


CYRILLIC_RE = re.compile(r"[\u0400-\u052f]")
WHITESPACE_RE = re.compile(r"\s+")
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.!?;:])")
APOSTROPHE_RE = re.compile(r"'{2,}")

APOSTROPHE_VARIANTS = ("’", "‘", "`", "ʻ", "ʼ", "ʹ", "՚", "´")
CYRILLIC_VOWELS = frozenset("аеёиоуўэюяАЕЁИОУЎЭЮЯ")
APOSTROPHE_LIKE = frozenset(("'", *APOSTROPHE_VARIANTS, "ъ", "Ъ"))

# Uzbek Cyrillic plus common Russian letters that occur in names and loanwords.
CYRILLIC_TO_UZBEK_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "'", "ы": "i", "ь": "", "э": "e",
    "ю": "yu", "я": "ya", "ў": "o'", "қ": "q", "ғ": "g'",
    "ҳ": "h",
}


@lru_cache(maxsize=1)
def _traditional_to_simplified() -> OpenCC:
    return OpenCC("t2s")


def _preserve_case(source: str, replacement: str) -> str:
    if not replacement or not source.isupper():
        return replacement
    return replacement.upper() if len(replacement) == 1 else replacement[0].upper() + replacement[1:]


def to_simplified_chinese(text: str) -> str:
    current = unicodedata.normalize("NFKC", str(text))
    current = WHITESPACE_RE.sub(" ", current).strip()

    converter = _traditional_to_simplified()
    seen: set[str] = set()

    for _ in range(8):
        if current in seen:
            raise ValueError(
                "OpenCC Simplified Chinese conversion entered a cycle."
            )
        seen.add(current)

        converted = converter.convert(current)
        converted = WHITESPACE_RE.sub(" ", converted).strip()

        if converted == current:
            return converted

        current = converted

    raise ValueError(
        "OpenCC Simplified Chinese conversion did not stabilize after 8 passes."
    )


def to_uzbek_latin(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text))
    result: list[str] = []
    for index, character in enumerate(normalized):
        lower = character.lower()
        if lower == "е":
            previous = normalized[index - 1] if index else ""
            replacement = (
                "ye"
                if not previous.isalpha()
                or previous in CYRILLIC_VOWELS
                or previous in APOSTROPHE_LIKE
                else "e"
            )
            result.append(_preserve_case(character, replacement))
        elif lower in CYRILLIC_TO_UZBEK_LATIN:
            result.append(_preserve_case(character, CYRILLIC_TO_UZBEK_LATIN[lower]))
        else:
            result.append(character)
    latin = "".join(result)
    for variant in APOSTROPHE_VARIANTS:
        latin = latin.replace(variant, "'")
    latin = WHITESPACE_RE.sub(" ", latin)
    latin = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", latin)
    latin = APOSTROPHE_RE.sub("'", latin).strip()
    remaining = sorted(set(CYRILLIC_RE.findall(latin)))
    if remaining:
        raise ValueError(
            "Uzbek text contains unsupported Cyrillic characters after transliteration: "
            + "".join(remaining)
        )
    return latin


def normalize_language_text(language: str, text: str) -> str:
    language = str(language).lower().strip()
    if language == "zh":
        return to_simplified_chinese(text)
    if language == "uz":
        return to_uzbek_latin(text)
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text))).strip()
