"""Conservative, CPU-only review hints; never rewrite or reject training rows."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from decimal import Decimal

# Retain the import gate's established signals. Context constrains the new phrase:
# e.g. 然而家人 / 而家长 must not be mistaken for the Cantonese word 而家.
CANTONESE_PATTERN = re.compile(
    r"佢哋|我哋|你哋|点解|邊個|边个|幾多|几多|呢個|呢个|呢段|"
    r"[佢嘅喺咗哋冇唔嗰咁咩啲嚟攞睇嘢乜俾噉噃]|"
    r"(?<!然)而家(?=[我你他她佢只都就仲正已要唔冇喺])"
)

RAW_DIGITS = re.compile(r"[0-9]+(?:[.,][0-9]+)*")
# Grouped Arabic digits and a bounded set of explicit multipliers. This is not
# a full number-word parser; unsupported spellings remain review hints.
NUMBER = re.compile(
    r"(?<![0-9])(?P<number>[+-]?(?:[0-9]{1,3}(?:[, ][0-9]{3})+(?:\.[0-9]+)?"
    r"|[0-9]+(?:[.,][0-9]+)?))"
    r"(?P<scale>\s*(?:万|萬|亿|億|千|百万|百萬|"
    r"million\b|milliard\b|ming\b))?",
    re.IGNORECASE,
)
SCALES = {
    "万": 10000,
    "萬": 10000,
    "亿": 100000000,
    "億": 100000000,
    "千": 1000,
    "百万": 1000000,
    "百萬": 1000000,
    "ming": 1000,
    "million": 1000000,
    "milliard": 1000000000,
}
# Units are compared in base units, with conservative tolerance for rounding.
# Attach only immediately following units, never compare unrelated unit sets.
UNITS = (
    (r"(?:平方英尺|kvadrat\s+fut\w*|sq\.?\s*ft\b|ft2\b)", "area", "0.09290304"),
    (r"(?:平方米|平方公尺|kvadrat\s+metr\w*|m2\b)", "area", "1"),
    (r"(?:平方公里|平方千米|kvadrat\s+kilometr\w*|km2\b)", "area", "1000000"),
    (r"(?:英里|mil(?:ga|dan|ni)?\b)", "length", "1609.344"),
    (r"(?:英尺|fut\w*\b|ft\b)", "length", "0.3048"),
    (r"(?:公里|千米|kilometr\w*|km\b)", "length", "1000"),
    (r"(?:米|公尺|metr\w*|m\b)", "length", "1"),
    (r"(?:公斤|千克|kilogramm\w*|kg\b)", "mass", "1000"),
    (r"(?:克|gramm\w*|g\b)", "mass", "1"),
    (r"(?:小时|小時|soat\w*)", "duration", "3600"),
    (r"(?:分钟|分鐘|daqiqa\w*)", "duration", "60"),
    (r"(?:秒|soniya\w*)", "duration", "1"),
    (r"(?:%|％)", "percent", "1"),
)
UNIT_PATTERNS = [(re.compile(r"\s*" + p, re.I), dim, Decimal(f)) for p, dim, f in UNITS]


def number_evidence(text):
    normalized = unicodedata.normalize("NFKC", text).replace("\u00a0", " ")
    values = []
    consumed_until = 0
    for match in NUMBER.finditer(normalized):
        if match.start() < consumed_until:
            continue
        raw = match["number"].replace(" ", "")
        if re.fullmatch(r"[+-]?[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?", raw):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(",", ".")
        scale = (match["scale"] or "").strip().lower()
        value = Decimal(raw) * SCALES.get(scale, 1)
        entry = {"text": match[0], "value": str(value), "unit": None}
        for pattern, dimension, factor in UNIT_PATTERNS:
            unit = pattern.match(normalized, match.end())
            if unit:
                entry["unit"] = {
                    "text": unit[0].strip(),
                    "dimension": dimension,
                    "base_value": str(value * factor),
                }
                consumed_until = unit.end()
                break
        values.append(entry)
    return values


def _quantities_match(source, target):
    if len(source) != len(target) or not source:
        return False

    # Sorting preserves repeated quantities; unlike sets it does not lose counts.
    def key(quantity):
        return quantity["dimension"], Decimal(quantity["base_value"])

    for left, right in zip(
        sorted(source, key=key), sorted(target, key=key), strict=True
    ):
        a, b = Decimal(left["base_value"]), Decimal(right["base_value"])
        if left["dimension"] != right["dimension"]:
            return False
        if abs(a - b) > max(abs(a), abs(b)) * Decimal("0.005"):
            return False
    return True


def numeric_review(source, target):
    """Evidence, not a semantic verdict; no inferred repairs or automatic drops."""
    left, right = number_evidence(source), number_evidence(target)
    lq = [v["unit"] for v in left if v["unit"]]
    rq = [v["unit"] for v in right if v["unit"]]
    converted_match = (
        len(lq) == len(left) and len(rq) == len(right) and _quantities_match(lq, rq)
    )
    value_match = Counter(Decimal(v["value"]) for v in left) == Counter(
        Decimal(v["value"]) for v in right
    )
    flags = []
    if not value_match and not converted_match:
        if set(RAW_DIGITS.findall(source)) != set(RAW_DIGITS.findall(target)):
            flags.append("digit_token_set_difference")
        flags.append("number_value_or_count_review")
    if (lq or rq) and not _quantities_match(lq, rq):
        flags.append("number_unit_review")
    return {
        "hints": flags,
        "source": left,
        "target": right,
        "status": "REVIEW_NEEDED" if flags else "NO_SUPPORTED_MISMATCH_FOUND",
        "limitations": "Arabic digits and listed units only; spelled numbers, signs, "
        "dates, locale-dependent separators and unlisted units need manual review. "
        "0.5% quantity tolerance is a hint rule, not translation approval.",
    }
