from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "generated"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "semantic_v01_raw.jsonl"
)

STATS_FILE = (
    OUTPUT_DIR
    / "semantic_v01_stats.json"
)


LANGUAGES = [
    "zh",
    "en",
    "ru",
    "uz",
]


RU_FUTURE_AUX = {
    "1sg": "буду",
    "2sg": "будешь",
    "3sg": "будет",
    "1pl": "будем",
    "3pl": "будут",
}


# ============================================================
# loading
# ============================================================

def load_jsonl(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def load_resources():

    concepts = load_jsonl(
        CONCEPT_FILE
    )

    concepts_by_id = {
        row["id"]: row
        for row in concepts
    }

    with open(
        FRAME_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        frame_data = json.load(f)

    return (
        concepts,
        concepts_by_id,
        frame_data["frames"],
    )


# ============================================================
# filtering
# ============================================================

def match_filter(
    concept,
    spec,
):

    for key, value in spec.items():

        if key in {
            "pos",
            "semantic_type",
        }:

            if (
                concept.get(key)
                != value
            ):
                return False

    return True


def choose_concept(
    concepts,
    spec,
    rng,
):

    pool = [
        c
        for c in concepts
        if match_filter(
            c,
            spec,
        )
    ]

    if not pool:

        raise RuntimeError(
            f"找不到符合条件的 concept: "
            f"{spec}"
        )

    return rng.choice(pool)


# ============================================================
# verb generation
# ============================================================

def render_verb(
    concept,
    lang,
    tense,
    polarity,
    person,
):

    forms = concept["forms"][lang]

    # --------------------------------------------------------
    # Chinese
    # --------------------------------------------------------

    if lang == "zh":

        base = forms["base"]

        if polarity == "neg":
            return "不" + base

        return base


    # --------------------------------------------------------
    # English
    # --------------------------------------------------------

    if lang == "en":

        base = forms["base"]

        if tense == "future":

            if polarity == "neg":
                return f"will not {base}"

            return f"will {base}"


        # present

        if polarity == "neg":

            if person == "3sg":
                return f"does not {base}"

            return f"do not {base}"


        if person == "3sg":
            return forms["present_3sg"]

        return base


    # --------------------------------------------------------
    # Russian
    # --------------------------------------------------------

    if lang == "ru":

        inf = forms["inf"]

        if tense == "future":

            aux = RU_FUTURE_AUX[
                person
            ]

            if polarity == "neg":
                return (
                    f"не {aux} {inf}"
                )

            return (
                f"{aux} {inf}"
            )

        present = forms[
            f"present_{person}"
        ]

        if polarity == "neg":
            return (
                "не "
                + present
            )

        return present


    # --------------------------------------------------------
    # Uzbek
    # --------------------------------------------------------

    if lang == "uz":

        key = (
            f"finite_{polarity}_"
            f"{person}"
        )

        return forms[key]


    raise ValueError(
        f"Unsupported language: "
        f"{lang}"
    )


# ============================================================
# text functions
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+([,.!?])",
        r"\1",
        text,
    )

    return text.strip()


def build_clock(rng):

    hour = rng.randint(
        0,
        23,
    )

    minute = rng.choice(
        [
            0,
            15,
            30,
            45,
        ]
    )

    return (
        f"{hour:02d}:"
        f"{minute:02d}"
    )


# ============================================================
# sample generation
# ============================================================

def derive_tense(
    selected,
):

    for key in [
        "time",
        "day",
    ]:

        if key in selected:

            tense = (
                selected[key]
                .get("meta", {})
                .get("tense")
            )

            if tense:
                return tense

    return "present"


def choose_polarity(
    frame,
    rng,
):

    values = (
        frame
        .get("features", {})
        .get(
            "polarity",
            ["pos"],
        )
    )

    if values == [
        "pos",
        "neg",
    ]:

        return (
            "pos"
            if rng.random() < 0.70
            else "neg"
        )

    return rng.choice(values)


def select_slots(
    frame,
    concepts,
    concepts_by_id,
    rng,
):

    selected = {}

    for (
        slot_name,
        concept_id,
    ) in frame.get(
        "fixed_slots",
        {}
    ).items():

        selected[
            slot_name
        ] = concepts_by_id[
            concept_id
        ]


    for slot_name in frame.get(
        "slot_order",
        []
    ):

        spec = frame["slots"][
            slot_name
        ]


        # object depends on chosen verb
        if (
            slot_name == "object"
            and
            frame.get(
                "dependencies",
                {}
            ).get(
                "object_allowed_by_verb",
                False,
            )
        ):

            verb = selected[
                "verb"
            ]

            allowed = set(
                verb.get(
                    "meta",
                    {}
                ).get(
                    "allowed_object_types",
                    [],
                )
            )

            pool = [
                c
                for c in concepts
                if (
                    c.get(
                        "semantic_type"
                    )
                    == "object"
                    and
                    c.get(
                        "meta",
                        {}
                    ).get(
                        "object_type"
                    )
                    in allowed
                )
            ]

            if not pool:

                raise RuntimeError(
                    f"No allowed object "
                    f"for verb "
                    f"{verb['id']}"
                )

            selected[
                slot_name
            ] = rng.choice(
                pool
            )

            continue


        selected[
            slot_name
        ] = choose_concept(
            concepts,
            spec,
            rng,
        )


    return selected


def semantic_signature(
    frame,
    selected,
    features,
    computed,
):

    payload = {
        "frame": frame["id"],

        "slots": {
            key: value["id"]
            for key, value
            in sorted(
                selected.items()
            )
        },

        "features": features,

        "computed": computed,
    }

    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
    )


def render_sample(
    frame,
    selected,
    features,
    computed,
):

    texts = {}
    traces = {}

    subject = selected.get(
        "subject"
    )

    person = (
        subject
        .get("meta", {})
        .get(
            "person",
            "3sg",
        )
        if subject
        else "3sg"
    )

    tense = features.get(
        "tense",
        "present",
    )

    polarity = features.get(
        "polarity",
        "pos",
    )


    for lang in LANGUAGES:

        values = {}

        lang_trace = {}


        # regular slots
        for (
            slot_name,
            concept,
        ) in selected.items():

            if slot_name == "verb":
                continue

            role = (
                frame
                .get(
                    "roles",
                    {}
                )
                .get(
                    slot_name,
                    "base",
                )
            )

            surface = (
                concept[
                    "forms"
                ][lang][role]
            )

            values[
                slot_name
            ] = surface

            lang_trace[
                slot_name
            ] = surface


        # verb
        if "verb" in selected:

            verb_surface = (
                render_verb(
                    selected[
                        "verb"
                    ],
                    lang,
                    tense,
                    polarity,
                    person,
                )
            )

            values[
                "verb"
            ] = verb_surface

            lang_trace[
                "verb"
            ] = verb_surface


        # computed fields
        for key, value in (
            computed.items()
        ):

            values[key] = value

            lang_trace[
                key
            ] = value


        template = (
            frame[
                "templates"
            ][lang]
        )

        text = (
            template.format(
                **values
            )
        )

        text = clean_text(
            text
        )

        # sentence capitalization
        if (
            lang in {
                "en",
                "ru",
                "uz",
            }
            and
            text
        ):

            text = (
                text[0].upper()
                + text[1:]
            )

        texts[lang] = text

        traces[lang] = (
            lang_trace
        )


    return (
        texts,
        traces,
    )


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    args = parser.parse_args()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        concepts,
        concepts_by_id,
        frames,
    ) = load_resources()


    rng = random.Random(
        args.seed
    )


    frame_weights = [
        frame.get(
            "weight",
            1,
        )
        for frame in frames
    ]


    generated = []

    signatures = set()

    frame_counter = Counter()


    max_attempts = (
        args.n
        * 200
    )

    attempts = 0


    while (
        len(generated)
        < args.n
        and
        attempts
        < max_attempts
    ):

        attempts += 1


        frame = rng.choices(
            frames,
            weights=frame_weights,
            k=1,
        )[0]


        selected = select_slots(
            frame,
            concepts,
            concepts_by_id,
            rng,
        )


        tense = derive_tense(
            selected
        )

        polarity = (
            choose_polarity(
                frame,
                rng,
            )
        )


        features = {
            "tense": tense,
            "polarity": polarity,
        }


        computed = {}

        if (
            "clock"
            in frame.get(
                "computed",
                [],
            )
        ):

            computed[
                "clock"
            ] = build_clock(
                rng
            )


        signature = (
            semantic_signature(
                frame,
                selected,
                features,
                computed,
            )
        )


        if (
            signature
            in signatures
        ):
            continue


        signatures.add(
            signature
        )


        (
            texts,
            traces,
        ) = render_sample(
            frame,
            selected,
            features,
            computed,
        )


        sample = {
            "semantic_id":
                f"sem_{len(generated)+1:08d}",

            "frame_id":
                frame["id"],

            "slots": {
                key: value["id"]
                for key, value
                in selected.items()
            },

            "features":
                features,

            "computed":
                computed,

            "texts":
                texts,

            "trace":
                traces,

            "source_type":
                "grammar_synthetic",

            "resource_version":
                "0.1",
        }


        generated.append(
            sample
        )

        frame_counter[
            frame["id"]
        ] += 1


    if len(generated) < args.n:

        raise RuntimeError(
            f"只生成了 "
            f"{len(generated)} 条，"
            f"目标 {args.n}。"
        )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for row in generated:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


    stats = {
        "target_samples":
            args.n,

        "generated_samples":
            len(generated),

        "attempts":
            attempts,

        "seed":
            args.seed,

        "frame_distribution":
            dict(
                frame_counter
            ),
    }


    with open(
        STATS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stats,
            f,
            ensure_ascii=False,
            indent=2,
        )


    print("=" * 80)
    print("Synthetic generation completed")
    print("=" * 80)

    print(
        "Generated:",
        len(generated),
    )

    print(
        "Attempts:",
        attempts,
    )

    print(
        "\nFrames:"
    )

    for key, value in (
        frame_counter.items()
    ):

        print(
            f"{key:<24} "
            f"{value}"
        )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()