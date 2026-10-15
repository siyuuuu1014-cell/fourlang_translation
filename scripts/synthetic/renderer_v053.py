from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.synthetic.renderer_v052 import (
    V052Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERSION = "0.5.3"

RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

VERB_FILE = (
    RESOURCE_DIR
    / "verb_realization_v053.json"
)

ARGUMENT_FILE = (
    RESOURCE_DIR
    / "argument_realization_v053.json"
)


NEW_FRAMES = {
    "GIVE_OBJECT_PERSON",
    "BRING_OBJECT_DESTINATION",
}


PARTICIPANT_OVERLAP_PAIRS = {
    ("PERSON_I", "PERSON_WE"),
    ("PERSON_WE", "PERSON_I"),
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


class V053Renderer(
    V052Renderer
):

    def __init__(self) -> None:
        super().__init__()

        self.version = VERSION

        self.v053_verb_resource = (
            read_json(
                VERB_FILE
            )
        )

        self.v053_argument_resource = (
            read_json(
                ARGUMENT_FILE
            )
        )

        self.v053_verbs = (
            self.v053_verb_resource[
                "verbs"
            ]
        )

        self.objects = (
            self.v053_argument_resource[
                "objects"
            ]
        )

        self.person_targets = (
            self.v053_argument_resource[
                "person_targets"
            ]
        )

        self.place_targets = (
            self.v053_argument_resource[
                "place_targets"
            ]
        )

        self.frame_contracts_v053 = (
            self.v053_argument_resource[
                "frame_contracts_v053"
            ]
        )

        self._validate_v053_resources()


    def _validate_v053_resources(
        self,
    ) -> None:

        for verb_id in [
            "GIVE",
            "BRING",
        ]:
            if verb_id not in self.v053_verbs:
                raise RuntimeError(
                    f"Missing verb: {verb_id}"
                )

        for frame_id in NEW_FRAMES:
            if (
                frame_id
                not in self.frame_contracts_v053
            ):
                raise RuntimeError(
                    f"Missing frame contract: "
                    f"{frame_id}"
                )

        for place_id, place in (
            self.place_targets.items()
        ):

            for key in [
                "zh_dir",
                "en_dir",
                "ru_dir",
                "uz_dir",
            ]:
                if not place.get(key):
                    raise RuntimeError(
                        f"{place_id}: "
                        f"missing {key}"
                    )


    # =========================================================================
    # Helpers
    # =========================================================================

    def _validate_give_person(
        self,
        subject_id: str,
        person_id: str,
    ) -> None:

        if not subject_id:
            raise ValueError(
                "Missing subject."
            )

        if not person_id:
            raise ValueError(
                "Missing person."
            )

        if subject_id == person_id:
            raise ValueError(
                "Reflexive GIVE person "
                "target is not enabled."
            )

        if (
            subject_id,
            person_id,
        ) in PARTICIPANT_OVERLAP_PAIRS:

            raise ValueError(
                "Speaker-group participant "
                "overlap is not enabled."
            )


    def _object(
        self,
        object_id: str,
    ) -> dict:

        if object_id not in self.objects:
            raise ValueError(
                f"Unknown object: {object_id}"
            )

        return self.objects[
            object_id
        ]


    def _person_target(
        self,
        person_id: str,
    ) -> dict:

        if (
            person_id
            not in self.person_targets
        ):
            raise ValueError(
                f"Unknown person: {person_id}"
            )

        return self.person_targets[
            person_id
        ]


    def _destination(
        self,
        destination_id: str,
    ) -> dict:

        if (
            destination_id
            not in self.place_targets
        ):
            raise ValueError(
                f"Unknown destination: "
                f"{destination_id}"
            )

        return self.place_targets[
            destination_id
        ]


    @staticmethod
    def _form(
        resource: dict,
        language: str,
        *,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        lang = resource[
            language
        ]

        # Chinese
        if language == "zh":

            key = (
                f"{tense}_"
                f"{'positive' if polarity == 'pos' else 'negative'}"
            )

            value = lang.get(key)

            if not value:
                raise ValueError(
                    f"Missing ZH form: {key}"
                )

            return value

        # English
        if language == "en":

            if tense == "future":

                key = (
                    "future"
                    if polarity == "pos"
                    else "future_negative"
                )

                return lang[key]

            if polarity == "neg":
                return lang[
                    "present_negative"
                ][person_code]

            return lang[
                "present"
            ][person_code]

        # Russian
        if language == "ru":

            value = lang[
                tense
            ][person_code]

            if polarity == "neg":
                return (
                    f"не {value}"
                )

            return value

        # Uzbek
        if language == "uz":

            table = (
                "present_future"
                if polarity == "pos"
                else "negative"
            )

            return lang[
                table
            ][person_code]

        raise ValueError(
            f"Unsupported language: "
            f"{language}"
        )


    # =========================================================================
    # GIVE
    # =========================================================================

    def _render_give(
        self,
        *,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        if slots.get(
            "verb"
        ) != "GIVE":
            raise ValueError(
                "GIVE_OBJECT_PERSON "
                "requires verb=GIVE."
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        person_id = slots.get(
            "person"
        )

        if not object_id:
            raise ValueError(
                "Missing object."
            )

        self._validate_give_person(
            subject_id,
            person_id,
        )

        obj = self._object(
            object_id
        )

        person = self._person_target(
            person_id
        )

        tense = features.get(
            "tense",
            "present",
        )

        polarity = features.get(
            "polarity",
            "pos",
        )

        person_code = self._person_code(
            subject_id
        )

        resource = self.v053_verbs[
            "GIVE"
        ]

        zh_v = self._form(
            resource,
            "zh",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        en_v = self._form(
            resource,
            "en",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        ru_v = self._form(
            resource,
            "ru",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        uz_v = self._form(
            resource,
            "uz",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        zh_subject = (
            self._subject_surface(
                subject_id,
                "zh",
            )
        )

        en_subject = (
            self._subject_surface(
                subject_id,
                "en",
            )
        )

        ru_subject = (
            self._subject_surface(
                subject_id,
                "ru",
            )
        )

        uz_subject = (
            self._subject_surface(
                subject_id,
                "uz",
            )
        )

        return {
            "zh": (
                f"{zh_subject}"
                f"{zh_v}"
                f"{person['zh']}"
                f"{obj['zh']}。"
            ),

            "en": (
                f"{en_subject} "
                f"{en_v} "
                f"{person['en_object']} "
                f"{obj['en']}."
            ),

            "ru": (
                f"{ru_subject} "
                f"{ru_v} "
                f"{person['ru_dat']} "
                f"{obj['ru_acc']}."
            ),

            "uz": (
                f"{uz_subject} "
                f"{person['uz_dat']} "
                f"{obj['uz_acc']} "
                f"{uz_v}."
            ),
        }


    # =========================================================================
    # BRING
    # =========================================================================

    def _render_bring(
        self,
        *,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        if slots.get(
            "verb"
        ) != "BRING":
            raise ValueError(
                "BRING_OBJECT_DESTINATION "
                "requires verb=BRING."
            )

        subject_id = slots.get(
            "subject"
        )

        object_id = slots.get(
            "object"
        )

        destination_id = slots.get(
            "destination"
        )

        if not subject_id:
            raise ValueError(
                "Missing subject."
            )

        if not object_id:
            raise ValueError(
                "Missing object."
            )

        if not destination_id:
            raise ValueError(
                "Missing destination."
            )

        obj = self._object(
            object_id
        )

        destination = (
            self._destination(
                destination_id
            )
        )

        tense = features.get(
            "tense",
            "present",
        )

        polarity = features.get(
            "polarity",
            "pos",
        )

        person_code = (
            self._person_code(
                subject_id
            )
        )

        resource = self.v053_verbs[
            "BRING"
        ]

        en_v = self._form(
            resource,
            "en",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        ru_v = self._form(
            resource,
            "ru",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        uz_v = self._form(
            resource,
            "uz",
            person_code=person_code,
            tense=tense,
            polarity=polarity,
        )

        zh_subject = (
            self._subject_surface(
                subject_id,
                "zh",
            )
        )

        en_subject = (
            self._subject_surface(
                subject_id,
                "en",
            )
        )

        ru_subject = (
            self._subject_surface(
                subject_id,
                "ru",
            )
        )

        uz_subject = (
            self._subject_surface(
                subject_id,
                "uz",
            )
        )

        # Explicit Chinese 把 construction makes
        # object/destination roles unambiguous.
        if tense == "future":

            if polarity == "neg":
                zh = (
                    f"{zh_subject}不会把"
                    f"{obj['zh']}带到"
                    f"{destination['zh_dir']}。"
                )

            else:
                zh = (
                    f"{zh_subject}会把"
                    f"{obj['zh']}带到"
                    f"{destination['zh_dir']}。"
                )

        else:

            if polarity == "neg":
                zh = (
                    f"{zh_subject}不把"
                    f"{obj['zh']}带到"
                    f"{destination['zh_dir']}。"
                )

            else:
                zh = (
                    f"{zh_subject}把"
                    f"{obj['zh']}带到"
                    f"{destination['zh_dir']}。"
                )

        return {
            "zh": zh,

            "en": (
                f"{en_subject} "
                f"{en_v} "
                f"{obj['en']} "
                f"{destination['en_dir']}."
            ),

            "ru": (
                f"{ru_subject} "
                f"{ru_v} "
                f"{obj['ru_acc']} "
                f"{destination['ru_dir']}."
            ),

            "uz": (
                f"{uz_subject} "
                f"{obj['uz_acc']} "
                f"{destination['uz_dir']} "
                f"{uz_v}."
            ),
        }


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

        if frame_id == (
            "GIVE_OBJECT_PERSON"
        ):
            return self._render_give(
                slots=slots,
                features=features,
            )

        if frame_id == (
            "BRING_OBJECT_DESTINATION"
        ):
            return self._render_bring(
                slots=slots,
                features=features,
            )

        # Frozen V0.5.2 behavior.
        return super().render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            **kwargs,
        )


def main() -> None:

    print("=" * 100)
    print(
        "RENDERER V0.5.3 RESOURCE CHECK"
    )
    print("=" * 100)

    renderer = V053Renderer()

    print(
        "Renderer version:",
        VERSION,
    )

    print(
        "Supported new frames:",
        sorted(NEW_FRAMES),
    )

    print(
        "Objects:",
        len(renderer.objects),
    )

    print(
        "Person targets:",
        len(
            renderer.person_targets
        ),
    )

    print(
        "Directional targets:",
        len(
            renderer.place_targets
        ),
    )

    print()
    print(
        "RENDERER V0.5.3 LOAD PASS"
    )


if __name__ == "__main__":
    main()