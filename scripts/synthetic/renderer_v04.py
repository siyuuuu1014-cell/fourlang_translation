from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONCEPTS = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "concepts_v04.jsonl"
)

DEFAULT_FRAMES = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "frames_v04.json"
)


LANGUAGES = (
    "zh",
    "en",
    "ru",
    "uz",
)


RU_FUTURE_AUX = {
    "1sg": "буду",
    "2sg": "будешь",
    "3sg": "будет",
    "1pl": "будем",
    "2pl": "будете",
    "3pl": "будут",
}


CITY_CONCEPTS = {
    "TASHKENT",
    "MOSCOW",
    "BEIJING",
}


class RenderError(RuntimeError):
    pass


# ============================================================
# IO
# ============================================================

def read_jsonl(
    path: Path,
) -> list[dict]:

    if not path.exists():
        raise FileNotFoundError(
            f"Concept file not found: {path}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                row = json.loads(
                    line
                )

            except json.JSONDecodeError as exc:

                raise RenderError(
                    f"Invalid JSONL at "
                    f"{path}:{line_no}: {exc}"
                ) from exc

            if not isinstance(
                row,
                dict,
            ):
                raise RenderError(
                    f"{path}:{line_no} "
                    "is not a JSON object."
                )

            rows.append(
                row
            )

    return rows


def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if not isinstance(
        data,
        dict,
    ):
        raise RenderError(
            f"{path} root must be object."
        )

    return data


# ============================================================
# Normalization
# ============================================================

def normalize_polarity(
    value: str | None,
) -> str:

    value = str(
        value or "pos"
    ).strip().lower()

    if value in {
        "pos",
        "positive",
        "affirmative",
    }:
        return "pos"

    if value in {
        "neg",
        "negative",
    }:
        return "neg"

    raise RenderError(
        f"Unsupported polarity: {value}"
    )


def normalize_tense(
    value: str | None,
) -> str:

    value = str(
        value or "present"
    ).strip().lower()

    if value in {
        "present",
        "future",
    }:
        return value

    raise RenderError(
        f"Unsupported tense: {value}"
    )


def is_enabled(
    item: dict,
) -> bool:

    meta = item.get(
        "meta",
        {},
    )

    if not isinstance(
        meta,
        dict,
    ):
        return True

    return bool(
        meta.get(
            "enabled",
            True,
        )
    )


def capitalize_first(
    text: str,
) -> str:

    if not text:
        return text

    return (
        text[0].upper()
        + text[1:]
    )


def normalize_spaces(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# Renderer
# ============================================================

class V04Renderer:

    def __init__(
        self,
        concepts_path: str | Path = DEFAULT_CONCEPTS,
        frames_path: str | Path = DEFAULT_FRAMES,
    ) -> None:

        self.concepts_path = Path(
            concepts_path
        )

        self.frames_path = Path(
            frames_path
        )

        concept_rows = read_jsonl(
            self.concepts_path
        )

        frames_data = read_json(
            self.frames_path
        )

        self.concepts = {
            row["id"]: row
            for row in concept_rows
        }

        frames = frames_data.get(
            "frames",
            [],
        )

        if not isinstance(
            frames,
            list,
        ):
            raise RenderError(
                "frames_v04.json: "
                "'frames' must be list."
            )

        self.frames = {
            frame["id"]: frame
            for frame in frames
            if isinstance(
                frame,
                dict,
            )
            and frame.get(
                "id"
            )
        }

    # ========================================================
    # Resource access
    # ========================================================

    def get_concept(
        self,
        concept_id: str,
        *,
        require_enabled: bool = True,
    ) -> dict:

        concept = self.concepts.get(
            concept_id
        )

        if concept is None:
            raise RenderError(
                f"Unknown concept: {concept_id}"
            )

        if (
            require_enabled
            and not is_enabled(
                concept
            )
        ):
            raise RenderError(
                f"Concept {concept_id} "
                "is disabled/planned."
            )

        return concept

    def get_frame(
        self,
        frame_id: str,
    ) -> dict:

        frame = self.frames.get(
            frame_id
        )

        if frame is None:
            raise RenderError(
                f"Unknown frame: {frame_id}"
            )

        if not is_enabled(
            frame
        ):
            raise RenderError(
                f"Frame {frame_id} "
                "is disabled/planned."
            )

        return frame

    # ========================================================
    # Person
    # ========================================================

    def get_person_code(
        self,
        person: dict,
        lang: str,
    ) -> str:

        forms = person.get(
            "forms",
            {},
        )

        lang_form = forms.get(
            lang,
            {},
        )

        code = None

        if isinstance(
            lang_form,
            dict,
        ):
            code = lang_form.get(
                "person_code"
            )

        if code:
            return str(
                code
            )

        features = person.get(
            "person_features",
            {},
        )

        person_number = features.get(
            "person"
        )

        number = features.get(
            "number"
        )

        if person_number not in {
            1,
            2,
            3,
        }:
            raise RenderError(
                f"Invalid person features: "
                f"{person.get('id')}"
            )

        suffix = (
            "sg"
            if number == "singular"
            else "pl"
        )

        return (
            f"{person_number}"
            f"{suffix}"
        )

    # ========================================================
    # Generic surfaces
    # ========================================================

    def get_base_surface(
        self,
        concept_id: str,
        lang: str,
    ) -> str:

        concept = self.get_concept(
            concept_id
        )

        forms = concept.get(
            "forms",
            {},
        )

        lang_form = forms.get(
            lang,
            {},
        )

        if not isinstance(
            lang_form,
            dict,
        ):
            raise RenderError(
                f"{concept_id}: "
                f"missing forms.{lang}"
            )

        value = (
            lang_form.get(
                "surface"
            )
            or lang_form.get(
                "base"
            )
            or lang_form.get(
                "lemma"
            )
        )

        if not value:
            raise RenderError(
                f"{concept_id}: "
                f"no base surface for {lang}"
            )

        return str(
            value
        )

    def get_object_surface(
        self,
        concept_id: str,
        lang: str,
    ) -> str:

        concept = self.get_concept(
            concept_id
        )

        forms = concept.get(
            "forms",
            {},
        )

        lang_form = forms.get(
            lang,
            {},
        )

        if not isinstance(
            lang_form,
            dict,
        ):
            raise RenderError(
                f"{concept_id}: "
                f"invalid forms.{lang}"
            )

        value = (
            lang_form.get(
                "object"
            )
            or lang_form.get(
                "acc"
            )
        )

        cases = lang_form.get(
            "cases",
            {},
        )

        if (
            not value
            and isinstance(
                cases,
                dict,
            )
        ):
            value = cases.get(
                "acc"
            )

        value = (
            value
            or lang_form.get(
                "surface"
            )
            or lang_form.get(
                "base"
            )
            or lang_form.get(
                "lemma"
            )
        )

        if not value:
            raise RenderError(
                f"{concept_id}: "
                f"no object surface for {lang}"
            )

        return str(
            value
        )

    # ========================================================
    # Destination realization
    # ========================================================

    def get_destination_surface(
        self,
        concept_id: str,
        lang: str,
        verb_id: str,
    ) -> str:

        concept = self.get_concept(
            concept_id
        )

        forms = concept.get(
            "forms",
            {},
        )

        lang_form = forms.get(
            lang,
            {},
        )

        if not isinstance(
            lang_form,
            dict,
        ):
            raise RenderError(
                f"{concept_id}: "
                f"invalid forms.{lang}"
            )

        # ----------------------------------------------------
        # ARRIVE English:
        #
        # arrive at the airport
        # arrive at the hotel
        # arrive in Moscow
        # ----------------------------------------------------

        if (
            lang == "en"
            and verb_id == "ARRIVE"
        ):

            base = (
                lang_form.get(
                    "surface"
                )
                or lang_form.get(
                    "base"
                )
            )

            if not base:
                raise RenderError(
                    f"{concept_id}: "
                    "missing English base."
                )

            preposition = (
                "in"
                if concept_id
                in CITY_CONCEPTS
                else "at"
            )

            return (
                f"{preposition} {base}"
            )

        # ----------------------------------------------------
        # Chinese destination does not need a preposition.
        # ----------------------------------------------------

        if lang == "zh":

            return str(
                lang_form.get(
                    "destination"
                )
                or lang_form.get(
                    "surface"
                )
                or lang_form.get(
                    "base"
                )
            )

        # ----------------------------------------------------
        # Existing resources already contain:
        #
        # to the airport
        # в аэропорт
        # aeroportga
        # ----------------------------------------------------

        value = (
            lang_form.get(
                "destination"
            )
        )

        if value:
            return str(
                value
            )

        # V0.4 future schema fallback
        if lang == "ru":

            value = (
                lang_form.get(
                    "destination_phrase"
                )
            )

        elif lang == "uz":

            value = (
                lang_form.get(
                    "destination"
                )
            )

        if value:
            return str(
                value
            )

        raise RenderError(
            f"{concept_id}: no destination "
            f"surface for {lang}"
        )

    # ========================================================
    # Chinese verb
    # ========================================================

    def render_zh_verb(
        self,
        verb: dict,
        tense: str,
        polarity: str,
    ) -> str:

        forms = verb.get(
            "forms",
            {},
        ).get(
            "zh",
            {},
        )

        if not isinstance(
            forms,
            dict,
        ):
            raise RenderError(
                f"{verb['id']}: "
                "invalid Chinese verb forms."
            )

        base = forms.get(
            "base"
        )

        if not base:
            raise RenderError(
                f"{verb['id']}: "
                "missing zh.base"
            )

        if tense == "future":

            if polarity == "neg":

                return str(
                    forms.get(
                        "future_negative"
                    )
                    or (
                        "不会"
                        + str(base)
                    )
                )

            return str(
                forms.get(
                    "future_positive"
                )
                or (
                    "会"
                    + str(base)
                )
            )

        if polarity == "neg":

            return str(
                forms.get(
                    "present_negative"
                )
                or forms.get(
                    "negative"
                )
                or (
                    "不"
                    + str(base)
                )
            )

        return str(
            forms.get(
                "present_positive"
            )
            or forms.get(
                "positive"
            )
            or base
        )

    # ========================================================
    # English verb
    # ========================================================

    def render_en_verb(
        self,
        verb: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        forms = verb.get(
            "forms",
            {},
        ).get(
            "en",
            {},
        )

        base = forms.get(
            "base"
        )

        if not base:
            raise RenderError(
                f"{verb['id']}: "
                "missing en.base"
            )

        base = str(
            base
        )

        if tense == "future":

            if polarity == "neg":
                return (
                    f"will not {base}"
                )

            return (
                f"will {base}"
            )

        third_singular = (
            person_code == "3sg"
        )

        if polarity == "neg":

            auxiliary = (
                "does not"
                if third_singular
                else "do not"
            )

            return (
                f"{auxiliary} {base}"
            )

        if third_singular:

            return str(
                forms.get(
                    "present_3sg"
                )
                or (
                    base + "s"
                )
            )

        return base

    # ========================================================
    # Russian verb
    # ========================================================

    def _ru_present_form(
        self,
        forms: dict,
        person_code: str,
    ) -> str | None:

        present = forms.get(
            "present",
            {},
        )

        if isinstance(
            present,
            dict,
        ):

            value = present.get(
                person_code
            )

            if value:
                return str(
                    value
                )

        imperfective = forms.get(
            "imperfective",
            {},
        )

        if isinstance(
            imperfective,
            dict,
        ):

            value = imperfective.get(
                f"present_{person_code}"
            )

            if value:
                return str(
                    value
                )

        value = forms.get(
            f"present_{person_code}"
        )

        if value:
            return str(
                value
            )

        return None

    def _ru_perfective_future_form(
        self,
        forms: dict,
        person_code: str,
    ) -> str | None:

        perfective = forms.get(
            "perfective",
            {},
        )

        if not isinstance(
            perfective,
            dict,
        ):
            return None

        value = perfective.get(
            f"future_{person_code}"
        )

        if value:
            return str(
                value
            )

        future = perfective.get(
            "future",
            {},
        )

        if isinstance(
            future,
            dict,
        ):

            value = future.get(
                person_code
            )

            if value:
                return str(
                    value
                )

        return None

    def render_ru_verb(
        self,
        verb: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        forms = verb.get(
            "forms",
            {},
        ).get(
            "ru",
            {},
        )

        if not isinstance(
            forms,
            dict,
        ):
            raise RenderError(
                f"{verb['id']}: "
                "invalid Russian forms."
            )

        infinitive = forms.get(
            "infinitive"
        )

        if not infinitive:

            raise RenderError(
                f"{verb['id']}: "
                "missing ru.infinitive"
            )

        if tense == "present":

            surface = (
                self._ru_present_form(
                    forms,
                    person_code,
                )
            )

            if not surface:

                raise RenderError(
                    f"{verb['id']}: "
                    f"missing Russian present "
                    f"form for {person_code}"
                )

            if polarity == "neg":

                return (
                    f"не {surface}"
                )

            return surface

        strategy = forms.get(
            "future_strategy",
            "analytic",
        )

        # ----------------------------------------------------
        # Perfective future
        # ----------------------------------------------------

        if strategy == "perfective":

            surface = (
                self._ru_perfective_future_form(
                    forms,
                    person_code,
                )
            )

            if not surface:

                raise RenderError(
                    f"{verb['id']}: "
                    f"missing perfective future "
                    f"for {person_code}"
                )

            if polarity == "neg":

                return (
                    f"не {surface}"
                )

            return surface

        # ----------------------------------------------------
        # Analytic imperfective future
        #
        # буду пить
        # не буду пить
        # ----------------------------------------------------

        if strategy in {
            "analytic",
            None,
        }:

            auxiliary = (
                RU_FUTURE_AUX.get(
                    person_code
                )
            )

            if not auxiliary:

                raise RenderError(
                    "No Russian future auxiliary "
                    f"for {person_code}"
                )

            if polarity == "neg":

                return (
                    f"не {auxiliary} "
                    f"{infinitive}"
                )

            return (
                f"{auxiliary} "
                f"{infinitive}"
            )

        raise RenderError(
            f"{verb['id']}: unsupported "
            f"Russian future_strategy="
            f"{strategy!r}"
        )

    # ========================================================
    # Uzbek verb
    # ========================================================

    def render_uz_verb(
        self,
        verb: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        forms = verb.get(
            "forms",
            {},
        ).get(
            "uz",
            {},
        )

        if not isinstance(
            forms,
            dict,
        ):
            raise RenderError(
                f"{verb['id']}: "
                "invalid Uzbek forms."
            )

        key = (
            "negative_present_future"
            if polarity == "neg"
            else "present_future"
        )

        person_forms = forms.get(
            key,
            {},
        )

        if not isinstance(
            person_forms,
            dict,
        ):
            raise RenderError(
                f"{verb['id']}: "
                f"missing Uzbek {key}"
            )

        surface = person_forms.get(
            person_code
        )

        # Uzbek 3pl can sometimes use 3sg
        # as a fallback if an explicit plural
        # form is intentionally unavailable.
        if (
            not surface
            and person_code == "3pl"
        ):

            surface = person_forms.get(
                "3sg"
            )

        if not surface:

            raise RenderError(
                f"{verb['id']}: "
                f"missing Uzbek {key} "
                f"form for {person_code}"
            )

        return str(
            surface
        )

    # ========================================================
    # Unified verb
    # ========================================================

    def render_verb(
        self,
        verb_id: str,
        subject_id: str,
        lang: str,
        tense: str,
        polarity: str,
    ) -> str:

        verb = self.get_concept(
            verb_id
        )

        if verb.get(
            "concept_type"
        ) != "verb":

            raise RenderError(
                f"{verb_id} is not a verb."
            )

        subject = self.get_concept(
            subject_id
        )

        person_code = (
            self.get_person_code(
                subject,
                lang,
            )
        )

        allowed_tenses = (
            verb.get(
                "features",
                {},
            ).get(
                "allowed_tenses"
            )
        )

        if (
            allowed_tenses
            and tense
            not in allowed_tenses
        ):

            raise RenderError(
                f"{verb_id} does not allow "
                f"tense={tense}. "
                f"Allowed={allowed_tenses}"
            )

        if lang == "zh":

            return self.render_zh_verb(
                verb,
                tense,
                polarity,
            )

        if lang == "en":

            return self.render_en_verb(
                verb,
                person_code,
                tense,
                polarity,
            )

        if lang == "ru":

            return self.render_ru_verb(
                verb,
                person_code,
                tense,
                polarity,
            )

        if lang == "uz":

            return self.render_uz_verb(
                verb,
                person_code,
                tense,
                polarity,
            )

        raise RenderError(
            f"Unsupported language: {lang}"
        )

    # ========================================================
    # Tense derivation
    # ========================================================

    def derive_tense(
        self,
        slots: dict,
        features: dict,
    ) -> str:

        explicit = features.get(
            "tense"
        )

        if explicit:

            return normalize_tense(
                explicit
            )

        for slot_name in (
            "time",
            "day",
        ):

            concept_id = slots.get(
                slot_name
            )

            if not concept_id:
                continue

            concept = self.get_concept(
                concept_id
            )

            hint = (
                concept.get(
                    "time_features",
                    {},
                ).get(
                    "tense_hint"
                )
            )

            if hint:

                return normalize_tense(
                    hint
                )

        return "present"

    # ========================================================
    # Fixed concepts
    # ========================================================

    def resolve_slots(
        self,
        frame: dict,
        supplied_slots: dict,
    ) -> dict:

        slots = dict(
            supplied_slots
        )

        definitions = frame.get(
            "slots",
            {},
        )

        for slot_name, spec in (
            definitions.items()
        ):

            if not isinstance(
                spec,
                dict,
            ):
                continue

            fixed_id = spec.get(
                "fixed_concept_id"
            )

            if fixed_id:

                slots[
                    slot_name
                ] = fixed_id

            if (
                spec.get(
                    "required",
                    False,
                )
                and slot_name
                not in slots
            ):

                raise RenderError(
                    f"Frame {frame['id']}: "
                    f"missing required slot "
                    f"{slot_name}"
                )

        return slots

    # ========================================================
    # Main rendering
    # ========================================================

    def render(
        self,
        frame_id: str,
        slots: dict,
        features: dict | None = None,
        computed: dict | None = None,
    ) -> dict[str, str]:

        frame = self.get_frame(
            frame_id
        )

        features = dict(
            features or {}
        )

        computed = dict(
            computed or {}
        )

        slots = self.resolve_slots(
            frame,
            slots,
        )

        polarity = normalize_polarity(
            features.get(
                "polarity"
            )
        )

        tense = self.derive_tense(
            slots,
            features,
        )

        verb_id = slots.get(
            "verb"
        )

        subject_id = slots.get(
            "subject"
        )

        # A frame with a verb always needs a subject
        # in the current V0.4 design.
        if (
            verb_id
            and not subject_id
        ):

            raise RenderError(
                f"Frame {frame_id}: "
                "verb exists but subject missing."
            )

        texts: dict[
            str,
            str,
        ] = {}

        templates = frame.get(
            "render_template",
            {},
        )

        for lang in LANGUAGES:

            template = templates.get(
                lang
            )

            if not template:

                raise RenderError(
                    f"Frame {frame_id}: "
                    f"missing template {lang}"
                )

            values: dict[
                str,
                Any,
            ] = {}

            # -----------------------------------------------
            # subject
            # -----------------------------------------------

            if "subject" in slots:

                values[
                    "subject"
                ] = self.get_base_surface(
                    slots[
                        "subject"
                    ],
                    lang,
                )

            # -----------------------------------------------
            # verb
            # -----------------------------------------------

            if verb_id:

                values[
                    "verb"
                ] = self.render_verb(
                    verb_id,
                    subject_id,
                    lang,
                    tense,
                    polarity,
                )

            # -----------------------------------------------
            # object
            # -----------------------------------------------

            if "object" in slots:

                values[
                    "object"
                ] = self.get_object_surface(
                    slots[
                        "object"
                    ],
                    lang,
                )

            # -----------------------------------------------
            # destination
            # -----------------------------------------------

            if "destination" in slots:

                if not verb_id:

                    raise RenderError(
                        "Destination requires verb "
                        "in current V0.4 renderer."
                    )

                values[
                    "destination"
                ] = (
                    self.get_destination_surface(
                        slots[
                            "destination"
                        ],
                        lang,
                        verb_id,
                    )
                )

            # -----------------------------------------------
            # place
            # -----------------------------------------------

            if "place" in slots:

                values[
                    "place"
                ] = self.get_base_surface(
                    slots[
                        "place"
                    ],
                    lang,
                )

            # -----------------------------------------------
            # time
            # -----------------------------------------------

            if "time" in slots:

                values[
                    "time"
                ] = self.get_base_surface(
                    slots[
                        "time"
                    ],
                    lang,
                )

            # -----------------------------------------------
            # day
            # -----------------------------------------------

            if "day" in slots:

                values[
                    "day"
                ] = self.get_base_surface(
                    slots[
                        "day"
                    ],
                    lang,
                )

            # -----------------------------------------------
            # clock
            # -----------------------------------------------

            if "clock" in computed:

                values[
                    "clock"
                ] = str(
                    computed[
                        "clock"
                    ]
                )

            # -----------------------------------------------
            # Format
            # -----------------------------------------------

            try:

                sentence = template.format(
                    **values
                )

            except KeyError as exc:

                raise RenderError(
                    f"Frame {frame_id}, "
                    f"language {lang}: "
                    f"missing template value "
                    f"{exc}"
                ) from exc

            sentence = normalize_spaces(
                sentence
            )

            if lang in {
                "en",
                "ru",
                "uz",
            }:

                sentence = capitalize_first(
                    sentence
                )

            texts[
                lang
            ] = sentence

        return texts

    # ========================================================
    # Diagnostics
    # ========================================================

    def active_verb_ids(
        self,
    ) -> list[str]:

        return sorted(
            cid
            for cid, concept
            in self.concepts.items()
            if (
                concept.get(
                    "concept_type"
                )
                == "verb"
                and is_enabled(
                    concept
                )
            )
        )

    def disabled_verb_ids(
        self,
    ) -> list[str]:

        return sorted(
            cid
            for cid, concept
            in self.concepts.items()
            if (
                concept.get(
                    "concept_type"
                )
                == "verb"
                and not is_enabled(
                    concept
                )
            )
        )


# ============================================================
# Simple CLI diagnostic
# ============================================================

def main() -> None:

    renderer = V04Renderer()

    print(
        "=" * 80
    )

    print(
        "V0.4 RENDERER RESOURCE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Concepts:",
        len(
            renderer.concepts
        ),
    )

    print(
        "Frames:",
        len(
            renderer.frames
        ),
    )

    print(
        "Active verbs:",
        renderer.active_verb_ids(),
    )

    print(
        "Disabled verbs:",
        renderer.disabled_verb_ids(),
    )

    print()

    print(
        "RENDERER LOAD PASS"
    )


if __name__ == "__main__":
    main()