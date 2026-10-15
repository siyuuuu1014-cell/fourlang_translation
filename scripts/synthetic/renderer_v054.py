from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.synthetic.renderer_v053 import (
    V053Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.4"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v054.json"
)

ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v054.json"
)

NEW_FRAMES = {
    "NEED_OBJECT",
    "LOSE_OBJECT",
}


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


def capitalize_first(text: str) -> str:
    if not text:
        return text

    return (
        text[0].upper()
        + text[1:]
    )


class V054Renderer(
    V053Renderer
):

    def __init__(self) -> None:
        super().__init__()

        self.version = VERSION

        self.v054_verb_resource = read_json(
            VERB_FILE
        )

        self.v054_argument_resource = read_json(
            ARGUMENT_FILE
        )

        self.v054_verbs = (
            self.v054_verb_resource[
                "verbs"
            ]
        )

        self.v054_objects = (
            self.v054_argument_resource[
                "objects"
            ]
        )

        self.v054_contracts = (
            self.v054_argument_resource[
                "frame_contracts_v054"
            ]
        )

        self._validate_v054_resources()


    # =========================================================================
    # Resource validation
    # =========================================================================

    def _validate_v054_resources(
        self,
    ) -> None:

        for verb_id in (
            "NEED",
            "LOSE",
        ):
            if verb_id not in self.v054_verbs:
                raise RuntimeError(
                    f"Missing V0.5.4 verb: "
                    f"{verb_id}"
                )

        for frame_id in NEW_FRAMES:
            if frame_id not in self.v054_contracts:
                raise RuntimeError(
                    f"Missing V0.5.4 frame: "
                    f"{frame_id}"
                )

        for object_id, obj in (
            self.v054_objects.items()
        ):
            for field in (
                "ru_nom",
                "ru_number",
                "ru_need_agreement",
                "uz_nom",
            ):
                if field not in obj:
                    raise RuntimeError(
                        f"{object_id}: "
                        f"missing {field}"
                    )


    # =========================================================================
    # Helpers
    # =========================================================================

    def _v054_object(
        self,
        object_id: str,
    ) -> dict:

        if object_id not in self.v054_objects:
            raise ValueError(
                f"Unknown object: {object_id}"
            )

        return self.v054_objects[
            object_id
        ]


    def _experiencer_target(
        self,
        subject_id: str,
    ) -> dict:

        if subject_id not in self.person_targets:
            raise ValueError(
                f"Unknown semantic subject: "
                f"{subject_id}"
            )

        return self.person_targets[
            subject_id
        ]


    # =========================================================================
    # NEED
    # =========================================================================

    def _render_need(
        self,
        *,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        if slots.get("verb") != "NEED":
            raise ValueError(
                "NEED_OBJECT requires verb=NEED."
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        if not subject_id:
            raise ValueError(
                "NEED_OBJECT missing subject."
            )

        if not object_id:
            raise ValueError(
                "NEED_OBJECT missing object."
            )

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        if tense not in {
            "present",
            "future",
        }:
            raise ValueError(
                f"NEED invalid tense: {tense}"
            )

        if polarity not in {
            "pos",
            "neg",
        }:
            raise ValueError(
                f"NEED invalid polarity: "
                f"{polarity}"
            )

        obj = self._v054_object(
            object_id
        )

        experiencer = (
            self._experiencer_target(
                subject_id
            )
        )

        resource = self.v054_verbs[
            "NEED"
        ]

        person_code = self._person_code(
            subject_id
        )

        # ---------------------------------------------------------------------
        # ZH
        # ---------------------------------------------------------------------

        zh_key = (
            f"{tense}_"
            f"{'positive' if polarity == 'pos' else 'negative'}"
        )

        zh_verb = resource[
            "zh"
        ][
            zh_key
        ]

        zh_subject = (
            self._subject_surface(
                subject_id,
                "zh",
            )
        )

        zh = (
            f"{zh_subject}"
            f"{zh_verb}"
            f"{obj['zh']}。"
        )

        # ---------------------------------------------------------------------
        # EN
        # ---------------------------------------------------------------------

        en_subject = (
            self._subject_surface(
                subject_id,
                "en",
            )
        )

        if tense == "future":
            en_verb = (
                resource["en"]["future"]
                if polarity == "pos"
                else resource["en"][
                    "future_negative"
                ]
            )

        elif polarity == "neg":
            en_verb = (
                resource["en"][
                    "present_negative"
                ][person_code]
            )

        else:
            en_verb = (
                resource["en"][
                    "present"
                ][person_code]
            )

        en = (
            f"{en_subject} "
            f"{en_verb} "
            f"{obj['en']}."
        )

        # ---------------------------------------------------------------------
        # RU
        #
        # semantic subject becomes dative experiencer.
        # object becomes grammatical nominative argument.
        # ---------------------------------------------------------------------

        ru_dat = capitalize_first(
            experiencer[
                "ru_dat"
            ]
        )

        agreement = obj[
            "ru_need_agreement"
        ]

        predicative = (
            resource["ru"][
                "predicative"
            ][agreement]
        )

        number = obj[
            "ru_number"
        ]

        if tense == "present":

            if polarity == "pos":
                ru = (
                    f"{ru_dat} "
                    f"{predicative} "
                    f"{obj['ru_nom']}."
                )

            else:
                ru = (
                    f"{ru_dat} "
                    f"не {predicative} "
                    f"{obj['ru_nom']}."
                )

        else:

            copula_key = (
                "pl"
                if number == "pl"
                else "sg"
            )

            copula = (
                resource["ru"][
                    "future_copula"
                ][copula_key]
            )

            if polarity == "pos":
                ru = (
                    f"{ru_dat} "
                    f"{copula} "
                    f"{predicative} "
                    f"{obj['ru_nom']}."
                )

            else:
                ru = (
                    f"{ru_dat} "
                    f"не {copula} "
                    f"{predicative} "
                    f"{obj['ru_nom']}."
                )

        # ---------------------------------------------------------------------
        # UZ
        # ---------------------------------------------------------------------

        uz_dat = capitalize_first(
            experiencer[
                "uz_dat"
            ]
        )

        uz_key = (
            f"{tense}_"
            f"{'positive' if polarity == 'pos' else 'negative'}"
        )

        uz_predicate = (
            resource["uz"][
                uz_key
            ]
        )

        uz = (
            f"{uz_dat} "
            f"{obj['uz_nom']} "
            f"{uz_predicate}."
        )

        return {
            "zh": zh,
            "en": en,
            "ru": ru,
            "uz": uz,
        }


    # =========================================================================
    # LOSE
    # =========================================================================

    def _render_lose(
        self,
        *,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        if slots.get("verb") != "LOSE":
            raise ValueError(
                "LOSE_OBJECT requires verb=LOSE."
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        if not subject_id:
            raise ValueError(
                "LOSE_OBJECT missing subject."
            )

        if not object_id:
            raise ValueError(
                "LOSE_OBJECT missing object."
            )

        tense = features.get(
            "tense"
        )

        polarity = features.get(
            "polarity"
        )

        # Critical V0.5.4 rule.
        if tense != "future":
            raise ValueError(
                "LOSE_OBJECT V0.5.4 "
                "supports future tense only."
            )

        if polarity not in {
            "pos",
            "neg",
        }:
            raise ValueError(
                f"LOSE invalid polarity: "
                f"{polarity}"
            )

        obj = self._v054_object(
            object_id
        )

        resource = self.v054_verbs[
            "LOSE"
        ]

        person_code = self._person_code(
            subject_id
        )

        # ZH
        zh_subject = (
            self._subject_surface(
                subject_id,
                "zh",
            )
        )

        zh_verb = (
            resource["zh"][
                "future_positive"
            ]
            if polarity == "pos"
            else resource["zh"][
                "future_negative"
            ]
        )

        zh = (
            f"{zh_subject}"
            f"{zh_verb}"
            f"{obj['zh']}。"
        )

        # EN
        en_subject = (
            self._subject_surface(
                subject_id,
                "en",
            )
        )

        en_verb = (
            resource["en"]["future"]
            if polarity == "pos"
            else resource["en"][
                "future_negative"
            ]
        )

        en = (
            f"{en_subject} "
            f"{en_verb} "
            f"{obj['en']}."
        )

        # RU
        ru_subject = (
            self._subject_surface(
                subject_id,
                "ru",
            )
        )

        ru_verb = (
            resource["ru"][
                "future"
            ][person_code]
        )

        if polarity == "neg":
            ru_verb = (
                f"не {ru_verb}"
            )

        ru = (
            f"{ru_subject} "
            f"{ru_verb} "
            f"{obj['ru_acc']}."
        )

        # UZ
        uz_subject = (
            self._subject_surface(
                subject_id,
                "uz",
            )
        )

        uz_table = (
            "present_future"
            if polarity == "pos"
            else "negative"
        )

        uz_verb = (
            resource["uz"][
                uz_table
            ][person_code]
        )

        uz = (
            f"{uz_subject} "
            f"{obj['uz_acc']} "
            f"{uz_verb}."
        )

        return {
            "zh": zh,
            "en": en,
            "ru": ru,
            "uz": uz,
        }


    # =========================================================================
    # Public
    # =========================================================================

    def render(
        self,
        *,
        frame_id: str,
        slots: dict,
        features: dict,
        **kwargs: Any,
    ) -> dict[str, str]:

        if frame_id == "NEED_OBJECT":
            return self._render_need(
                slots=slots,
                features=features,
            )

        if frame_id == "LOSE_OBJECT":
            return self._render_lose(
                slots=slots,
                features=features,
            )

        # Frozen V0.5.3 fallback.
        return super().render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            **kwargs,
        )


def main() -> None:
    print("=" * 100)
    print("RENDERER V0.5.4 RESOURCE CHECK")
    print("=" * 100)

    renderer = V054Renderer()

    print(
        "Renderer version:",
        renderer.version,
    )

    print(
        "Supported new frames:",
        sorted(NEW_FRAMES),
    )

    print(
        "Objects:",
        len(
            renderer.v054_objects
        ),
    )

    print()
    print(
        "LOSE generation tenses:",
        renderer.v054_verbs[
            "LOSE"
        ].get(
            "generation_tenses"
        ),
    )

    print()
    print(
        "RENDERER V0.5.4 LOAD PASS"
    )


if __name__ == "__main__":
    main()