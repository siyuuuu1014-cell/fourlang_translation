from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Version / Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.2"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

VERB_RESOURCE = (
    RESOURCE_DIR
    / "verb_realization_v052.json"
)

ARGUMENT_RESOURCE = (
    RESOURCE_DIR
    / "argument_realization_v052.json"
)


NEW_FRAMES = {
    "CALL_PERSON",
    "WAIT_PERSON",
    "WAIT_AT_PLACE",
}

NEW_VERBS = {
    "CALL",
    "WAIT",
}


# =============================================================================
# Participant overlap policy
# =============================================================================

PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
}


# =============================================================================
# Subject fallback
#
# Parent V0.5.1 normally supplies subject resources. These mappings are only
# defensive fallbacks for the three V0.5.2 frames.
# =============================================================================

SUBJECT_FALLBACK = {
    "PERSON_I": {
        "person_code": "1sg",
        "zh": "我",
        "en": "I",
        "ru": "Я",
        "uz": "Men",
    },

    "PERSON_YOU": {
        "person_code": "2sg",
        "zh": "你",
        "en": "You",
        "ru": "Ты",
        "uz": "Sen",
    },

    "PERSON_HE": {
        "person_code": "3sg",
        "zh": "他",
        "en": "He",
        "ru": "Он",
        "uz": "U",
    },

    "PERSON_SHE": {
        "person_code": "3sg",
        "zh": "她",
        "en": "She",
        "ru": "Она",
        "uz": "U",
    },

    "PERSON_WE": {
        "person_code": "1pl",
        "zh": "我们",
        "en": "We",
        "ru": "Мы",
        "uz": "Biz",
    },

    "PERSON_THEY": {
        "person_code": "3pl",
        "zh": "他们",
        "en": "They",
        "ru": "Они",
        "uz": "Ular",
    },
}


# =============================================================================
# Deterministic morphology tables for CALL / WAIT
#
# These tables deliberately model only the controlled V0.5.2 capability.
# =============================================================================

RU_CALL_PRESENT = {
    "1sg": "звоню",
    "2sg": "звонишь",
    "3sg": "звонит",
    "1pl": "звоним",
    "3pl": "звонят",
}

RU_CALL_FUTURE = {
    "1sg": "позвоню",
    "2sg": "позвонишь",
    "3sg": "позвонит",
    "1pl": "позвоним",
    "3pl": "позвонят",
}


RU_WAIT_PRESENT = {
    "1sg": "жду",
    "2sg": "ждёшь",
    "3sg": "ждёт",
    "1pl": "ждём",
    "3pl": "ждут",
}

RU_BE_FUTURE = {
    "1sg": "буду",
    "2sg": "будешь",
    "3sg": "будет",
    "1pl": "будем",
    "3pl": "будут",
}


UZ_CALL_POSITIVE = {
    "1sg": "qo'ng'iroq qilaman",
    "2sg": "qo'ng'iroq qilasan",
    "3sg": "qo'ng'iroq qiladi",
    "1pl": "qo'ng'iroq qilamiz",
    "3pl": "qo'ng'iroq qiladilar",
}

UZ_CALL_NEGATIVE = {
    "1sg": "qo'ng'iroq qilmayman",
    "2sg": "qo'ng'iroq qilmaysan",
    "3sg": "qo'ng'iroq qilmaydi",
    "1pl": "qo'ng'iroq qilmaymiz",
    "3pl": "qo'ng'iroq qilmaydilar",
}


UZ_WAIT_POSITIVE = {
    "1sg": "kutaman",
    "2sg": "kutasan",
    "3sg": "kutadi",
    "1pl": "kutamiz",
    "3pl": "kutadilar",
}

UZ_WAIT_NEGATIVE = {
    "1sg": "kutmayman",
    "2sg": "kutmaysan",
    "3sg": "kutmaydi",
    "1pl": "kutmaymiz",
    "3pl": "kutmaydilar",
}


# =============================================================================
# IO
# =============================================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required resource not found: "
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
        raise RuntimeError(
            f"Expected JSON object: "
            f"{path}"
        )

    return data


# =============================================================================
# Import frozen V0.5.1 renderer
# =============================================================================

try:
    from scripts.synthetic.renderer_v051 import (
        V051Renderer,
    )

except ImportError as exc:
    raise ImportError(
        "V0.5.2 requires the frozen "
        "scripts.synthetic.renderer_v051.V051Renderer"
    ) from exc


# =============================================================================
# Renderer
# =============================================================================

class V052Renderer(
    V051Renderer
):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.version = VERSION

        # ---------------------------------------------------------------------
        # Load V0.5.2 resources
        # ---------------------------------------------------------------------

        self.verb_resource = read_json(
            VERB_RESOURCE
        )

        self.argument_resource = read_json(
            ARGUMENT_RESOURCE
        )

        self.v052_verbs = (
            self.verb_resource.get(
                "verbs",
                {},
            )
        )

        self.frame_contracts = (
            self.argument_resource.get(
                "frame_contracts_v052",
                {},
            )
        )

        self.person_targets = (
            self.argument_resource.get(
                "person_targets",
                {},
            )
        )

        self.place_targets = (
            self.argument_resource.get(
                "place_targets",
                {},
            )
        )

        # Parent renderer normally provides this.
        if not hasattr(
            self,
            "subjects",
        ):
            self.subjects = {}

        # Preserve existing subject records while ensuring the new renderer
        # has the person_code field required by Grammar Hard V0.5.2.
        for subject_id, fallback in (
            SUBJECT_FALLBACK.items()
        ):

            current = self.subjects.get(
                subject_id
            )

            if not isinstance(
                current,
                dict,
            ):
                self.subjects[
                    subject_id
                ] = dict(
                    fallback
                )

                continue

            merged = dict(
                fallback
            )

            merged.update(
                current
            )

            if not merged.get(
                "person_code"
            ):
                merged[
                    "person_code"
                ] = fallback[
                    "person_code"
                ]

            self.subjects[
                subject_id
            ] = merged

        self._validate_resource_contract()


    # =========================================================================
    # Resource checks
    # =========================================================================

    def _validate_resource_contract(
        self,
    ) -> None:

        if not isinstance(
            self.person_targets,
            dict,
        ) or not self.person_targets:

            raise RuntimeError(
                "V0.5.2 person_targets "
                "resource is empty."
            )

        if not isinstance(
            self.place_targets,
            dict,
        ) or not self.place_targets:

            raise RuntimeError(
                "V0.5.2 place_targets "
                "resource is empty."
            )

        for frame_id in NEW_FRAMES:

            if frame_id not in (
                self.frame_contracts
            ):
                raise RuntimeError(
                    "Missing V0.5.2 frame "
                    f"contract: {frame_id}"
                )

        for verb_id in NEW_VERBS:

            if (
                self.v052_verbs
                and verb_id not in self.v052_verbs
            ):
                raise RuntimeError(
                    "Missing V0.5.2 verb "
                    f"resource: {verb_id}"
                )


    # =========================================================================
    # Participant safety
    # =========================================================================

    @staticmethod
    def _validate_person_target(
        subject_id: str,
        person_id: str,
    ) -> None:

        if not subject_id:
            raise ValueError(
                "Missing subject."
            )

        if not person_id:
            raise ValueError(
                "Missing person target."
            )

        # Exact reflexive.
        if subject_id == person_id:

            raise ValueError(
                "Reflexive person target "
                "is not enabled in V0.5.2: "
                f"{subject_id} -> {person_id}"
            )

        # Speaker / group overlap.
        if (
            subject_id,
            person_id,
        ) in PARTICIPANT_OVERLAP_PAIRS:

            raise ValueError(
                "Speaker-group participant overlap "
                "is not enabled in V0.5.2: "
                f"{subject_id} -> {person_id}"
            )


    # =========================================================================
    # Subject helpers
    # =========================================================================

    def _subject_record(
        self,
        subject_id: str,
    ) -> dict:

        record = self.subjects.get(
            subject_id
        )

        if isinstance(
            record,
            dict,
        ):
            return record

        fallback = SUBJECT_FALLBACK.get(
            subject_id
        )

        if fallback:
            return fallback

        raise ValueError(
            f"Unknown subject: "
            f"{subject_id}"
        )


    def _subject_surface(
        self,
        subject_id: str,
        language: str,
    ) -> str:

        record = self._subject_record(
            subject_id
        )

        # Standard key.
        value = record.get(
            language
        )

        if isinstance(
            value,
            str,
        ) and value.strip():

            return value.strip()

        # Common resource variants.
        alternatives = [
            f"{language}_surface",
            f"surface_{language}",
        ]

        for key in alternatives:

            value = record.get(
                key
            )

            if isinstance(
                value,
                str,
            ) and value.strip():

                return value.strip()

        fallback = SUBJECT_FALLBACK.get(
            subject_id,
            {},
        ).get(
            language
        )

        if fallback:
            return fallback

        raise ValueError(
            "Missing subject surface: "
            f"{subject_id}/{language}"
        )


    def _person_code(
        self,
        subject_id: str,
    ) -> str:

        record = self._subject_record(
            subject_id
        )

        person_code = record.get(
            "person_code"
        )

        if person_code:
            return str(
                person_code
            )

        fallback = SUBJECT_FALLBACK.get(
            subject_id,
            {},
        ).get(
            "person_code"
        )

        if fallback:
            return fallback

        raise ValueError(
            f"Missing person_code: "
            f"{subject_id}"
        )


    # =========================================================================
    # Morphology APIs
    #
    # Grammar Hard V0.5.2 calls these methods directly.
    # =========================================================================

    def _en_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        verb_id = str(
            verb_id
        ).upper()

        tense = str(
            tense
        ).lower()

        polarity = str(
            polarity
        ).lower()

        if verb_id == "CALL":
            base = "call"

        elif verb_id == "WAIT":
            base = "wait"

        else:
            raise ValueError(
                "V0.5.2 _en_verb supports "
                f"CALL/WAIT only, got "
                f"{verb_id}"
            )

        # Future
        if tense == "future":

            if polarity == "neg":
                return (
                    f"will not {base}"
                )

            return (
                f"will {base}"
            )

        # Present negative
        if polarity == "neg":

            if person_code == "3sg":
                return (
                    f"does not {base}"
                )

            return (
                f"do not {base}"
            )

        # Present positive
        if person_code == "3sg":
            return (
                f"{base}s"
            )

        return base


    def _ru_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        verb_id = str(
            verb_id
        ).upper()

        tense = str(
            tense
        ).lower()

        polarity = str(
            polarity
        ).lower()

        # ---------------------------------------------------------------------
        # CALL
        # ---------------------------------------------------------------------

        if verb_id == "CALL":

            if tense == "future":
                verb = RU_CALL_FUTURE.get(
                    person_code
                )

            else:
                verb = RU_CALL_PRESENT.get(
                    person_code
                )

            if not verb:
                raise ValueError(
                    "Unsupported RU CALL "
                    f"person_code: "
                    f"{person_code}"
                )

            if polarity == "neg":
                return (
                    f"не {verb}"
                )

            return verb

        # ---------------------------------------------------------------------
        # WAIT
        # ---------------------------------------------------------------------

        if verb_id == "WAIT":

            if tense == "future":

                auxiliary = (
                    RU_BE_FUTURE.get(
                        person_code
                    )
                )

                if not auxiliary:
                    raise ValueError(
                        "Unsupported RU WAIT "
                        f"person_code: "
                        f"{person_code}"
                    )

                if polarity == "neg":
                    return (
                        f"не {auxiliary} ждать"
                    )

                return (
                    f"{auxiliary} ждать"
                )

            verb = RU_WAIT_PRESENT.get(
                person_code
            )

            if not verb:
                raise ValueError(
                    "Unsupported RU WAIT "
                    f"person_code: "
                    f"{person_code}"
                )

            if polarity == "neg":
                return (
                    f"не {verb}"
                )

            return verb

        raise ValueError(
            "V0.5.2 _ru_verb supports "
            f"CALL/WAIT only, got "
            f"{verb_id}"
        )


    def _uz_verb(
        self,
        *,
        verb_id: str,
        person_code: str,
        polarity: str,
        tense: str | None = None,
    ) -> str:
        """
        Uzbek controlled V0.5.2 uses present-future morphology.

        Therefore the same finite forms can express planned/future meaning,
        and we intentionally do NOT force an artificial English-style future
        auxiliary.

        This is also why Qwen's occasional TENSE_ERROR on forms such as
        'qilmaydi' is treated as a judge false reject.
        """

        verb_id = str(
            verb_id
        ).upper()

        polarity = str(
            polarity
        ).lower()

        if verb_id == "CALL":

            table = (
                UZ_CALL_NEGATIVE
                if polarity == "neg"
                else UZ_CALL_POSITIVE
            )

        elif verb_id == "WAIT":

            table = (
                UZ_WAIT_NEGATIVE
                if polarity == "neg"
                else UZ_WAIT_POSITIVE
            )

        else:
            raise ValueError(
                "V0.5.2 _uz_verb supports "
                f"CALL/WAIT only, got "
                f"{verb_id}"
            )

        value = table.get(
            person_code
        )

        if not value:
            raise ValueError(
                "Unsupported UZ person_code: "
                f"{person_code}"
            )

        return value


    # =========================================================================
    # Chinese morphology
    # =========================================================================

    @staticmethod
    def _zh_call(
        *,
        subject: str,
        target: str,
        tense: str,
        polarity: str,
    ) -> str:

        if tense == "future":

            if polarity == "neg":
                return (
                    f"{subject}不会给"
                    f"{target}打电话。"
                )

            return (
                f"{subject}会给"
                f"{target}打电话。"
            )

        if polarity == "neg":
            return (
                f"{subject}不给"
                f"{target}打电话。"
            )

        return (
            f"{subject}给"
            f"{target}打电话。"
        )


    @staticmethod
    def _zh_wait_person(
        *,
        subject: str,
        target: str,
        tense: str,
        polarity: str,
    ) -> str:

        if tense == "future":

            if polarity == "neg":
                return (
                    f"{subject}不会等"
                    f"{target}。"
                )

            return (
                f"{subject}会等"
                f"{target}。"
            )

        if polarity == "neg":
            return (
                f"{subject}不等"
                f"{target}。"
            )

        return (
            f"{subject}等"
            f"{target}。"
        )


    @staticmethod
    def _zh_wait_place(
        *,
        subject: str,
        place: str,
        tense: str,
        polarity: str,
    ) -> str:

        if tense == "future":

            if polarity == "neg":
                return (
                    f"{subject}不会"
                    f"{place}等。"
                )

            return (
                f"{subject}会"
                f"{place}等。"
            )

        if polarity == "neg":
            return (
                f"{subject}不"
                f"{place}等。"
            )

        return (
            f"{subject}"
            f"{place}等。"
        )


    # =========================================================================
    # English renderers
    # =========================================================================

    def _render_en_call(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "en",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "en_object"
        )

        if not target:
            raise ValueError(
                f"Missing en_object: "
                f"{person_id}"
            )

        verb = self._en_verb(
            verb_id="CALL",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} "
            f"{target}."
        )


    def _render_en_wait_person(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "en",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "en_object"
        )

        if not target:
            raise ValueError(
                f"Missing en_object: "
                f"{person_id}"
            )

        verb = self._en_verb(
            verb_id="WAIT",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} for "
            f"{target}."
        )


    def _render_en_wait_place(
        self,
        *,
        subject_id: str,
        place_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "en",
        )

        person_code = self._person_code(
            subject_id
        )

        place = self.place_targets[
            place_id
        ].get(
            "en_loc"
        )

        if not place:
            raise ValueError(
                f"Missing en_loc: "
                f"{place_id}"
            )

        verb = self._en_verb(
            verb_id="WAIT",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} "
            f"{place}."
        )


    # =========================================================================
    # Russian renderers
    # =========================================================================

    def _render_ru_call(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "ru",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "ru_dat"
        )

        if not target:
            raise ValueError(
                f"Missing ru_dat: "
                f"{person_id}"
            )

        verb = self._ru_verb(
            verb_id="CALL",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} "
            f"{target}."
        )


    def _render_ru_wait_person(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "ru",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "ru_acc"
        )

        if not target:
            raise ValueError(
                f"Missing ru_acc: "
                f"{person_id}"
            )

        verb = self._ru_verb(
            verb_id="WAIT",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} "
            f"{target}."
        )


    def _render_ru_wait_place(
        self,
        *,
        subject_id: str,
        place_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "ru",
        )

        person_code = self._person_code(
            subject_id
        )

        place = self.place_targets[
            place_id
        ].get(
            "ru_loc"
        )

        if not place:
            raise ValueError(
                f"Missing ru_loc: "
                f"{place_id}"
            )

        verb = self._ru_verb(
            verb_id="WAIT",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        return (
            f"{subject} "
            f"{verb} "
            f"{place}."
        )


    # =========================================================================
    # Uzbek renderers
    # =========================================================================

    def _render_uz_call(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "uz",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "uz_dat"
        )

        if not target:
            raise ValueError(
                f"Missing uz_dat: "
                f"{person_id}"
            )

        verb = self._uz_verb(
            verb_id="CALL",
            person_code=person_code,
            polarity=polarity,
            tense=tense,
        )

        return (
            f"{subject} "
            f"{target} "
            f"{verb}."
        )


    def _render_uz_wait_person(
        self,
        *,
        subject_id: str,
        person_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "uz",
        )

        person_code = self._person_code(
            subject_id
        )

        target = self.person_targets[
            person_id
        ].get(
            "uz_acc"
        )

        if not target:
            raise ValueError(
                f"Missing uz_acc: "
                f"{person_id}"
            )

        verb = self._uz_verb(
            verb_id="WAIT",
            person_code=person_code,
            polarity=polarity,
            tense=tense,
        )

        return (
            f"{subject} "
            f"{target} "
            f"{verb}."
        )


    def _render_uz_wait_place(
        self,
        *,
        subject_id: str,
        place_id: str,
        tense: str,
        polarity: str,
    ) -> str:

        subject = self._subject_surface(
            subject_id,
            "uz",
        )

        person_code = self._person_code(
            subject_id
        )

        place = self.place_targets[
            place_id
        ].get(
            "uz_loc"
        )

        if not place:
            raise ValueError(
                f"Missing uz_loc: "
                f"{place_id}"
            )

        verb = self._uz_verb(
            verb_id="WAIT",
            person_code=person_code,
            polarity=polarity,
            tense=tense,
        )

        return (
            f"{subject} "
            f"{place} "
            f"{verb}."
        )


    # =========================================================================
    # V0.5.2 frame renderer
    # =========================================================================

    def _render_v052(
        self,
        *,
        frame_id: str,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        subject_id = slots.get(
            "subject"
        )

        verb_id = slots.get(
            "verb"
        )

        tense = features.get(
            "tense",
            "present",
        )

        polarity = features.get(
            "polarity",
            "pos",
        )

        if tense not in {
            "present",
            "future",
        }:
            raise ValueError(
                f"Unsupported tense: "
                f"{tense}"
            )

        if polarity not in {
            "pos",
            "neg",
        }:
            raise ValueError(
                f"Unsupported polarity: "
                f"{polarity}"
            )

        # =====================================================================
        # CALL_PERSON
        # =====================================================================

        if frame_id == "CALL_PERSON":

            if verb_id != "CALL":
                raise ValueError(
                    "CALL_PERSON requires "
                    "verb=CALL."
                )

            person_id = slots.get(
                "person"
            )

            self._validate_person_target(
                subject_id,
                person_id,
            )

            if person_id not in (
                self.person_targets
            ):
                raise ValueError(
                    f"Unknown person target: "
                    f"{person_id}"
                )

            zh_subject = self._subject_surface(
                subject_id,
                "zh",
            )

            zh_target = self.person_targets[
                person_id
            ].get(
                "zh"
            )

            if not zh_target:
                raise ValueError(
                    f"Missing zh target: "
                    f"{person_id}"
                )

            return {
                "zh":
                    self._zh_call(
                        subject=zh_subject,
                        target=zh_target,
                        tense=tense,
                        polarity=polarity,
                    ),

                "en":
                    self._render_en_call(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "ru":
                    self._render_ru_call(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "uz":
                    self._render_uz_call(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),
            }

        # =====================================================================
        # WAIT_PERSON
        # =====================================================================

        if frame_id == "WAIT_PERSON":

            if verb_id != "WAIT":
                raise ValueError(
                    "WAIT_PERSON requires "
                    "verb=WAIT."
                )

            person_id = slots.get(
                "person"
            )

            self._validate_person_target(
                subject_id,
                person_id,
            )

            if person_id not in (
                self.person_targets
            ):
                raise ValueError(
                    f"Unknown person target: "
                    f"{person_id}"
                )

            zh_subject = self._subject_surface(
                subject_id,
                "zh",
            )

            zh_target = self.person_targets[
                person_id
            ].get(
                "zh"
            )

            if not zh_target:
                raise ValueError(
                    f"Missing zh target: "
                    f"{person_id}"
                )

            return {
                "zh":
                    self._zh_wait_person(
                        subject=zh_subject,
                        target=zh_target,
                        tense=tense,
                        polarity=polarity,
                    ),

                "en":
                    self._render_en_wait_person(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "ru":
                    self._render_ru_wait_person(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "uz":
                    self._render_uz_wait_person(
                        subject_id=subject_id,
                        person_id=person_id,
                        tense=tense,
                        polarity=polarity,
                    ),
            }

        # =====================================================================
        # WAIT_AT_PLACE
        # =====================================================================

        if frame_id == "WAIT_AT_PLACE":

            if verb_id != "WAIT":
                raise ValueError(
                    "WAIT_AT_PLACE requires "
                    "verb=WAIT."
                )

            place_id = slots.get(
                "place"
            )

            if not place_id:
                raise ValueError(
                    "WAIT_AT_PLACE requires "
                    "place."
                )

            if place_id not in (
                self.place_targets
            ):
                raise ValueError(
                    f"Unknown place target: "
                    f"{place_id}"
                )

            zh_subject = self._subject_surface(
                subject_id,
                "zh",
            )

            zh_place = self.place_targets[
                place_id
            ].get(
                "zh_loc"
            )

            if not zh_place:
                raise ValueError(
                    f"Missing zh_loc: "
                    f"{place_id}"
                )

            return {
                "zh":
                    self._zh_wait_place(
                        subject=zh_subject,
                        place=zh_place,
                        tense=tense,
                        polarity=polarity,
                    ),

                "en":
                    self._render_en_wait_place(
                        subject_id=subject_id,
                        place_id=place_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "ru":
                    self._render_ru_wait_place(
                        subject_id=subject_id,
                        place_id=place_id,
                        tense=tense,
                        polarity=polarity,
                    ),

                "uz":
                    self._render_uz_wait_place(
                        subject_id=subject_id,
                        place_id=place_id,
                        tense=tense,
                        polarity=polarity,
                    ),
            }

        raise ValueError(
            f"Unsupported V0.5.2 frame: "
            f"{frame_id}"
        )


    # =========================================================================
    # Public API
    # =========================================================================

    def render(
        self,
        *,
        frame_id: str,
        slots: dict,
        features: dict,
        **kwargs: Any,
    ) -> dict[str, str]:

        # V0.5.2 owns only its new frames.
        if frame_id in NEW_FRAMES:

            return self._render_v052(
                frame_id=frame_id,
                slots=slots,
                features=features,
            )

        # Everything else remains frozen V0.5.1 behavior.
        return super().render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            **kwargs,
        )


# =============================================================================
# Resource check CLI
# =============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "RENDERER V0.5.2 RESOURCE CHECK"
    )
    print("=" * 100)

    renderer = V052Renderer()

    print(
        "Renderer version:",
        VERSION,
    )

    print(
        "Verb resource:",
        VERB_RESOURCE,
    )

    print(
        "Argument resource:",
        ARGUMENT_RESOURCE,
    )

    print(
        "Supported new frames:",
        sorted(
            NEW_FRAMES
        ),
    )

    print(
        "Person targets:",
        len(
            renderer.person_targets
        ),
    )

    print(
        "Place targets:",
        len(
            renderer.place_targets
        ),
    )

    print(
        "Participant overlap blocked:",
        sorted(
            PARTICIPANT_OVERLAP_PAIRS
        ),
    )

    print()
    print(
        "RENDERER V0.5.2 LOAD PASS"
    )


if __name__ == "__main__":
    main()