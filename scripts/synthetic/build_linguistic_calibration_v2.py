from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "semantic_v01_valid.jsonl"
)

CONCEPT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "concepts.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "audit"
    / "linguistic_v2"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "linguistic_calibration_v02.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "linguistic_calibration_v02_summary.json"
)


TARGET_PER_GROUP = 20

PERSONS = [
    "1sg",
    "2sg",
    "3sg",
    "1pl",
    "3pl",
]


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            rows.append(
                json.loads(line)
            )

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


def load_concepts() -> dict[str, dict]:

    if not CONCEPT_FILE.exists():

        raise FileNotFoundError(
            f"Concept file not found:\n"
            f"{CONCEPT_FILE}"
        )

    rows = read_jsonl(
        CONCEPT_FILE
    )

    concepts = {}

    for row in rows:

        concept_id = row.get(
            "id"
        )

        if not concept_id:
            continue

        concepts[
            concept_id
        ] = row

    return concepts


# ============================================================
# General helpers
# ============================================================

def get_semantic_id(
    row: dict,
) -> str | None:

    value = row.get(
        "semantic_id"
    )

    if value is None:
        return None

    return str(
        value
    )


def get_subject_person(
    row: dict,
    concepts: dict[str, dict],
) -> str | None:

    subject_id = (
        row
        .get(
            "slots",
            {},
        )
        .get(
            "subject"
        )
    )

    if not subject_id:
        return None


    concept = concepts.get(
        subject_id
    )

    if not concept:
        return None


    person = (
        concept
        .get(
            "meta",
            {},
        )
        .get(
            "person"
        )
    )


    if person not in PERSONS:
        return None


    return person


def replace_once(
    text: str,
    old: str,
    new: str,
) -> str | None:

    text = str(
        text
    )

    old = str(
        old
    )

    new = str(
        new
    )


    if not old:
        return None


    if old not in text:
        return None


    return text.replace(
        old,
        new,
        1,
    )


def get_verb_id(
    row: dict,
) -> str | None:

    return (
        row
        .get(
            "slots",
            {},
        )
        .get(
            "verb"
        )
    )


def get_verb_concept(
    row: dict,
    concepts: dict[str, dict],
) -> dict | None:

    verb_id = get_verb_id(
        row
    )

    if not verb_id:
        return None


    return concepts.get(
        verb_id
    )


# ============================================================
# Known risky renderer cases
# ============================================================

def contains_find_verb(
    row: dict,
) -> bool:

    """
    当前已经确认 FIND 存在两个 Renderer 风险：

    Chinese:
        他今天不找到食物。
        应更接近：
        他今天没找到食物。

    Russian future:
        будут находить
        对一次性 completed find 可能应使用 perfective найти。

    因此 V2 的 CLEAN control 暂时排除 FIND，
    避免把已知 Renderer 问题误算成 Judge false reject。
    """

    verb_id = str(
        get_verb_id(
            row
        )
        or ""
    ).upper()


    return (
        "FIND"
        in verb_id
    )


def safe_clean_candidate(
    row: dict,
) -> bool:

    if contains_find_verb(
        row
    ):

        return False


    texts = row.get(
        "texts",
        {},
    )


    for lang in [
        "zh",
        "en",
        "ru",
        "uz",
    ]:

        text = texts.get(
            lang
        )

        if (
            text is None
            or
            not str(text).strip()
        ):

            return False


    return True


# ============================================================
# Person corruption helpers
# ============================================================

def choose_wrong_person_general(
    current_person: str,
    rng: random.Random,
) -> str | None:

    candidates = [
        person
        for person in PERSONS
        if person != current_person
    ]


    if not candidates:
        return None


    return rng.choice(
        candidates
    )


def choose_wrong_person_uzbek(
    current_person: str,
    rng: random.Random,
) -> str | None:

    """
    IMPORTANT:

    Uzbek 中显式复数主语：

        Ular keladi.
        Ular keladilar.

    在实际语言中都可能出现。

    因此：

        3pl -> 3sg

    不能作为“必错”的 calibration corruption。

    V2 只制造明确的人称冲突。
    """

    if current_person == "3pl":

        # 明确不匹配：
        #
        # Ular + 1sg
        # Ular + 2sg
        # Ular + 1pl

        candidates = [
            "1sg",
            "2sg",
            "1pl",
        ]

    else:

        candidates = [
            "1sg",
            "2sg",
            "3sg",
            "1pl",
        ]

        candidates = [
            person
            for person in candidates
            if person != current_person
        ]


    if not candidates:
        return None


    return rng.choice(
        candidates
    )


# ============================================================
# ENGLISH CORRUPTION
# ============================================================

def corrupt_english_agreement(
    row: dict,
    concepts: dict[str, dict],
    rng: random.Random,
) -> dict | None:

    """
    只构造：
        present + positive

    Example:

        He eats food.
            ↓
        He eat food.

    或：

        We eat food.
            ↓
        We eats food.

    同时修改 trace，
    让 Hard Semantic Validator 看不出来。
    """

    features = row.get(
        "features",
        {},
    )


    if (
        features.get(
            "tense"
        )
        != "present"
    ):

        return None


    if (
        features.get(
            "polarity"
        )
        != "pos"
    ):

        return None


    person = get_subject_person(
        row,
        concepts,
    )

    if not person:
        return None


    verb = get_verb_concept(
        row,
        concepts,
    )

    if not verb:
        return None


    forms = (
        verb
        .get(
            "forms",
            {},
        )
        .get(
            "en",
            {},
        )
    )


    base_form = forms.get(
        "base"
    )

    third_form = forms.get(
        "present_3sg"
    )


    if (
        not base_form
        or
        not third_form
    ):

        return None


    correct_form = (
        row
        .get(
            "trace",
            {},
        )
        .get(
            "en",
            {},
        )
        .get(
            "verb"
        )
    )


    if not correct_form:
        return None


    if person == "3sg":

        wrong_form = (
            base_form
        )

        wrong_person = (
            "non_3sg"
        )

    else:

        wrong_form = (
            third_form
        )

        wrong_person = (
            "3sg"
        )


    if (
        wrong_form
        == correct_form
    ):

        return None


    original_text = (
        row
        .get(
            "texts",
            {},
        )
        .get(
            "en",
            "",
        )
    )


    new_text = replace_once(
        original_text,
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["en"] = new_text


    # --------------------------------------------------------
    # 故意同步 trace
    #
    # 这样 Hard Semantic Validator 会认为：
    #
    # generated surface == trace
    #
    # 从而真正把任务交给 Grammar Validator。
    # --------------------------------------------------------

    corrupted[
        "trace"
    ]["en"]["verb"] = (
        wrong_form
    )


    corrupted[
        "calibration_expected"
    ] = "REJECT"


    corrupted[
        "calibration_error_type"
    ] = (
        "en_agreement_error"
    )


    corrupted[
        "calibration_language"
    ] = "en"


    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "subject_verb_agreement",

        "language":
            "en",

        "subject_person":
            person,

        "wrong_verb_person":
            wrong_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# RUSSIAN CORRUPTION
# ============================================================

def corrupt_russian_agreement(
    row: dict,
    concepts: dict[str, dict],
    rng: random.Random,
) -> dict | None:

    """
    V2 当前只构造：

        Russian present positive
        subject ↔ verb person mismatch

    暂时不测试 future aspect。
    因为我们已经发现 FIND future aspect
    本身存在 Renderer 风险。

    Example:

        Мы едим.
            ↓
        Мы ест.
    """

    features = row.get(
        "features",
        {},
    )


    if (
        features.get(
            "tense"
        )
        != "present"
    ):

        return None


    if (
        features.get(
            "polarity"
        )
        != "pos"
    ):

        return None


    person = get_subject_person(
        row,
        concepts,
    )

    if not person:
        return None


    wrong_person = (
        choose_wrong_person_general(
            person,
            rng,
        )
    )


    if not wrong_person:
        return None


    verb = get_verb_concept(
        row,
        concepts,
    )

    if not verb:
        return None


    forms = (
        verb
        .get(
            "forms",
            {},
        )
        .get(
            "ru",
            {},
        )
    )


    correct_form = (
        row
        .get(
            "trace",
            {},
        )
        .get(
            "ru",
            {},
        )
        .get(
            "verb"
        )
    )


    wrong_form = forms.get(
        f"present_{wrong_person}"
    )


    if (
        not correct_form
        or
        not wrong_form
    ):

        return None


    if (
        correct_form
        == wrong_form
    ):

        return None


    original_text = (
        row
        .get(
            "texts",
            {},
        )
        .get(
            "ru",
            "",
        )
    )


    new_text = replace_once(
        original_text,
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["ru"] = new_text


    corrupted[
        "trace"
    ]["ru"]["verb"] = (
        wrong_form
    )


    corrupted[
        "calibration_expected"
    ] = "REJECT"


    corrupted[
        "calibration_error_type"
    ] = (
        "ru_agreement_error"
    )


    corrupted[
        "calibration_language"
    ] = "ru"


    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "verb_person_agreement",

        "language":
            "ru",

        "subject_person":
            person,

        "wrong_verb_person":
            wrong_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# UZBEK CORRUPTION
# ============================================================

def corrupt_uzbek_agreement(
    row: dict,
    concepts: dict[str, dict],
    rng: random.Random,
) -> dict | None:

    """
    构造 Uzbek subject ↔ finite verb person mismatch。

    与 V1 的关键区别：

    不再使用：

        3pl -> 3sg

    作为必错样本。

    Example:

        Biz boramiz.
            ↓
        Biz boradi.

    或：

        Sen ichasan.
            ↓
        Sen ichaman.
    """

    person = get_subject_person(
        row,
        concepts,
    )


    if not person:
        return None


    polarity = (
        row
        .get(
            "features",
            {},
        )
        .get(
            "polarity",
            "pos",
        )
    )


    if polarity not in {
        "pos",
        "neg",
    }:

        return None


    wrong_person = (
        choose_wrong_person_uzbek(
            person,
            rng,
        )
    )


    if not wrong_person:
        return None


    verb = get_verb_concept(
        row,
        concepts,
    )


    if not verb:
        return None


    forms = (
        verb
        .get(
            "forms",
            {},
        )
        .get(
            "uz",
            {},
        )
    )


    correct_form = (
        row
        .get(
            "trace",
            {},
        )
        .get(
            "uz",
            {},
        )
        .get(
            "verb"
        )
    )


    wrong_form = forms.get(
        f"finite_{polarity}_{wrong_person}"
    )


    if (
        not correct_form
        or
        not wrong_form
    ):

        return None


    if (
        correct_form
        == wrong_form
    ):

        return None


    # --------------------------------------------------------
    # 特别保险：
    #
    # 如果原 subject 是 3pl，
    # 明确禁止 wrong form 实际等于 3sg。
    # --------------------------------------------------------

    if person == "3pl":

        third_singular_form = (
            forms.get(
                f"finite_{polarity}_3sg"
            )
        )


        if (
            third_singular_form
            and
            wrong_form
            == third_singular_form
        ):

            return None


    original_text = (
        row
        .get(
            "texts",
            {},
        )
        .get(
            "uz",
            "",
        )
    )


    new_text = replace_once(
        original_text,
        correct_form,
        wrong_form,
    )


    if new_text is None:
        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["uz"] = new_text


    corrupted[
        "trace"
    ]["uz"]["verb"] = (
        wrong_form
    )


    corrupted[
        "calibration_expected"
    ] = "REJECT"


    corrupted[
        "calibration_error_type"
    ] = (
        "uz_agreement_error"
    )


    corrupted[
        "calibration_language"
    ] = "uz"


    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "verb_person_agreement",

        "language":
            "uz",

        "subject_person":
            person,

        "wrong_verb_person":
            wrong_person,

        "original":
            correct_form,

        "corrupted":
            wrong_form,
    }


    return corrupted


# ============================================================
# CHINESE CORRUPTION
# ============================================================

def corrupt_chinese_word_order(
    row: dict,
) -> dict | None:

    """
    目标：

    semantic surfaces 全部保留，
    但故意制造不自然中文语序。

    Example:

        你下周去车站。
            ↓
        你去车站下周。

    Hard Semantic Validator 仍然会 PASS。
    """

    if contains_find_verb(
        row
    ):

        # 避免 FIND 已知 negative renderer 问题
        return None


    trace = (
        row
        .get(
            "trace",
            {},
        )
        .get(
            "zh",
            {},
        )
    )


    subject = trace.get(
        "subject"
    )

    verb = trace.get(
        "verb"
    )

    obj = trace.get(
        "object"
    )

    destination = trace.get(
        "destination"
    )

    time_surface = (
        trace.get(
            "time"
        )
        or
        trace.get(
            "day"
        )
    )

    clock = trace.get(
        "clock"
    )


    if (
        not subject
        or
        not verb
        or
        not time_surface
    ):

        return None


    complement = (
        obj
        or
        destination
    )


    if not complement:
        return None


    # --------------------------------------------------------
    # 故意把时间移到句尾
    # --------------------------------------------------------

    if clock:

        bad_text = (
            f"{subject}"
            f"{verb}"
            f"{complement}"
            f"{time_surface}"
            f"{clock}"
            "。"
        )

    else:

        bad_text = (
            f"{subject}"
            f"{verb}"
            f"{complement}"
            f"{time_surface}"
            "。"
        )


    original_text = (
        row
        .get(
            "texts",
            {},
        )
        .get(
            "zh",
            "",
        )
    )


    if (
        bad_text
        == original_text
    ):

        return None


    corrupted = copy.deepcopy(
        row
    )


    corrupted[
        "texts"
    ]["zh"] = bad_text


    # 不修改 trace。
    #
    # 因为所有 semantic surface
    # 仍然存在，只是顺序改变。


    corrupted[
        "calibration_expected"
    ] = "REJECT"


    corrupted[
        "calibration_error_type"
    ] = (
        "zh_word_order_error"
    )


    corrupted[
        "calibration_language"
    ] = "zh"


    corrupted[
        "linguistic_corruption"
    ] = {
        "type":
            "unnatural_word_order",

        "language":
            "zh",

        "original":
            original_text,

        "corrupted":
            bad_text,
    }


    return corrupted


# ============================================================
# Candidate collection
# ============================================================

def collect_corruptions(
    rows: list[dict],
    corrupt_fn,
    target_n: int,
    used_ids: set[str],
    rng: random.Random,
    *extra_args,
) -> list[dict]:

    candidates = rows[:]

    rng.shuffle(
        candidates
    )


    results = []


    for row in candidates:

        semantic_id = get_semantic_id(
            row
        )


        if not semantic_id:
            continue


        if semantic_id in used_ids:
            continue


        corrupted = corrupt_fn(
            row,
            *extra_args,
        )


        if corrupted is None:
            continue


        results.append(
            corrupted
        )


        used_ids.add(
            semantic_id
        )


        if (
            len(results)
            >= target_n
        ):

            break


    return results


# ============================================================
# Group validation
# ============================================================

def verify_group_counts(
    calibration: list[dict],
    per_group: int,
) -> Counter:

    counts = Counter(
        row.get(
            "calibration_error_type",
            "UNKNOWN",
        )
        for row in calibration
    )


    expected_groups = [
        "none",
        "en_agreement_error",
        "ru_agreement_error",
        "uz_agreement_error",
        "zh_word_order_error",
    ]


    incomplete = []


    for group in expected_groups:

        actual = counts.get(
            group,
            0,
        )


        if actual != per_group:

            incomplete.append(
                (
                    group,
                    actual,
                )
            )


    if incomplete:

        print()
        print(
            "[ERROR] "
            "Linguistic Calibration V2 incomplete"
        )

        print("-" * 80)


        for group, actual in incomplete:

            print(
                f"{group:<28}"
                f"{actual}/"
                f"{per_group}"
            )


        raise RuntimeError(
            "Unable to generate exactly "
            f"{per_group} samples "
            "for every calibration group."
        )


    return counts


# ============================================================
# Extra validation for Uzbek V2
# ============================================================

def validate_uzbek_corruptions(
    calibration: list[dict],
) -> None:

    """
    确认 V2 里不存在：

        3pl -> 3sg

    这种有争议 corruption。
    """

    invalid = []


    for row in calibration:

        if (
            row.get(
                "calibration_error_type"
            )
            !=
            "uz_agreement_error"
        ):

            continue


        corruption = row.get(
            "linguistic_corruption",
            {},
        )


        subject_person = (
            corruption.get(
                "subject_person"
            )
        )


        wrong_person = (
            corruption.get(
                "wrong_verb_person"
            )
        )


        if (
            subject_person
            == "3pl"
            and
            wrong_person
            == "3sg"
        ):

            invalid.append(
                row.get(
                    "semantic_id"
                )
            )


    if invalid:

        raise RuntimeError(
            "Invalid Uzbek V2 corruption "
            "detected: 3pl -> 3sg\n"
            f"Samples: {invalid}"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        type=str,
        default=str(
            DEFAULT_INPUT
        ),
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=2027,
    )


    parser.add_argument(
        "--per-group",
        type=int,
        default=TARGET_PER_GROUP,
    )


    args = parser.parse_args()


    input_file = Path(
        args.input
    )


    if not input_file.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{input_file}"
        )


    rows = read_jsonl(
        input_file
    )


    concepts = load_concepts()


    rng = random.Random(
        args.seed
    )


    used_ids: set[str] = set()

    calibration: list[dict] = []


    # ========================================================
    # 1. CLEAN CONTROL
    # ========================================================

    clean_candidates = [
        row
        for row in rows
        if safe_clean_candidate(
            row
        )
    ]


    rng.shuffle(
        clean_candidates
    )


    clean_count = 0


    for row in clean_candidates:

        semantic_id = get_semantic_id(
            row
        )


        if not semantic_id:
            continue


        if semantic_id in used_ids:
            continue


        clean = copy.deepcopy(
            row
        )


        clean[
            "calibration_expected"
        ] = "ACCEPT"


        clean[
            "calibration_error_type"
        ] = "none"


        clean[
            "calibration_language"
        ] = "none"


        clean[
            "linguistic_corruption"
        ] = None


        calibration.append(
            clean
        )


        used_ids.add(
            semantic_id
        )


        clean_count += 1


        if (
            clean_count
            >= args.per_group
        ):

            break


    # ========================================================
    # 2. ENGLISH AGREEMENT
    # ========================================================

    en_rows = collect_corruptions(
        rows,
        corrupt_english_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )


    calibration.extend(
        en_rows
    )


    # ========================================================
    # 3. RUSSIAN AGREEMENT
    # ========================================================

    ru_rows = collect_corruptions(
        rows,
        corrupt_russian_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )


    calibration.extend(
        ru_rows
    )


    # ========================================================
    # 4. UZBEK AGREEMENT
    # ========================================================

    uz_rows = collect_corruptions(
        rows,
        corrupt_uzbek_agreement,
        args.per_group,
        used_ids,
        rng,
        concepts,
        rng,
    )


    calibration.extend(
        uz_rows
    )


    # ========================================================
    # 5. CHINESE WORD ORDER
    # ========================================================

    zh_rows = collect_corruptions(
        rows,
        corrupt_chinese_word_order,
        args.per_group,
        used_ids,
        rng,
    )


    calibration.extend(
        zh_rows
    )


    # ========================================================
    # Validate groups
    # ========================================================

    counts = verify_group_counts(
        calibration,
        args.per_group,
    )


    validate_uzbek_corruptions(
        calibration
    )


    # ========================================================
    # Shuffle final dataset
    # ========================================================

    rng.shuffle(
        calibration
    )


    for index, row in enumerate(
        calibration,
        start=1,
    ):

        row[
            "calibration_id"
        ] = (
            f"ling_v2_{index:04d}"
        )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    write_jsonl(
        OUTPUT_FILE,
        calibration,
    )


    label_counter = Counter(
        row[
            "calibration_expected"
        ]
        for row in calibration
    )


    language_counter = Counter(
        row[
            "calibration_language"
        ]
        for row in calibration
    )


    summary = {
        "version":
            "linguistic_calibration_v2",

        "source_file":
            str(input_file),

        "source_rows":
            len(rows),

        "seed":
            args.seed,

        "per_group":
            args.per_group,

        "total":
            len(calibration),

        "expected_labels":
            dict(
                label_counter
            ),

        "error_types":
            dict(
                counts
            ),

        "languages":
            dict(
                language_counter
            ),

        "design_notes": [
            (
                "Known FIND renderer risks "
                "are excluded from clean controls."
            ),
            (
                "Uzbek 3pl -> 3sg is NOT "
                "used as a mandatory error."
            ),
            (
                "Uzbek agreement corruptions "
                "use unambiguous person mismatches."
            ),
            (
                "EN/RU/UZ corrupted verb forms "
                "are copied into trace intentionally "
                "to bypass the semantic hard validator."
            ),
            (
                "ZH corruption keeps semantic surfaces "
                "but changes word order."
            ),
        ],
    }


    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )


    # ========================================================
    # Console report
    # ========================================================

    print("=" * 90)
    print("LINGUISTIC CALIBRATION V2")
    print("=" * 90)


    print(
        "Input:",
        input_file
    )


    print(
        "Source rows:",
        len(rows)
    )


    print(
        "Calibration rows:",
        len(calibration)
    )


    print(
        "\nExpected labels:"
    )


    for key, value in sorted(
        label_counter.items()
    ):

        print(
            f"{key:<15}"
            f"{value}"
        )


    print(
        "\nError types:"
    )


    for key, value in sorted(
        counts.items()
    ):

        print(
            f"{key:<28}"
            f"{value}"
        )


    print(
        "\nLanguages:"
    )


    for key, value in sorted(
        language_counter.items()
    ):

        print(
            f"{key:<10}"
            f"{value}"
        )


    print(
        "\nV2 safeguards:"
    )

    print(
        "Uzbek 3pl -> 3sg corruption: DISABLED"
    )

    print(
        "Known FIND clean controls: EXCLUDED"
    )


    print(
        "\nSaved:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )


if __name__ == "__main__":
    main()