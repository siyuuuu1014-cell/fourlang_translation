from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
)

from scripts.synthetic.renderer_v04 import (
    DEFAULT_FRAMES,
    RenderError,
)

from scripts.synthetic.renderer_v0441 import (
    V0441Renderer,
)


# ============================================================
# Version
# ============================================================

RENDERER_VERSION = "0.5.1"


# ============================================================
# Resource paths
# ============================================================

V04_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

V05_RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v05"
)

DEFAULT_CONCEPTS = (
    V04_RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_VERB_REALIZATION = (
    V05_RESOURCE_DIR
    / "verb_realization_v051.json"
)

DEFAULT_ARGUMENT_REALIZATION = (
    V05_RESOURCE_DIR
    / "argument_realization_v051.json"
)


# ============================================================
# Supported V0.5.1 frames
# ============================================================

FRAME_TO_VERB = {
    "SEE_OBJECT": "SEE",
    "TAKE_OBJECT": "TAKE",
    "LOSE_OBJECT": "LOSE",
}


# ============================================================
# IO
# ============================================================

def read_json(
    path: Path,
) -> dict:

    if not path.exists():

        raise FileNotFoundError(
            f"Resource not found: {path}"
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
            f"Resource must be JSON object: {path}"
        )

    return data


# ============================================================
# Renderer
# ============================================================

class V051Renderer(
    V0441Renderer
):

    def __init__(
        self,
        concepts_path: str | Path = DEFAULT_CONCEPTS,
        frames_path: str | Path = DEFAULT_FRAMES,
        verb_realization_path: str | Path = DEFAULT_VERB_REALIZATION,
        argument_realization_path: str | Path = DEFAULT_ARGUMENT_REALIZATION,
    ) -> None:

        # ----------------------------------------------------
        # Frozen V0.4.4.1 renderer remains the fallback for
        # every existing frame.
        # ----------------------------------------------------

        super().__init__(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

        self.verb_realization_path = Path(
            verb_realization_path
        )

        self.argument_realization_path = Path(
            argument_realization_path
        )

        self.verb_realization = read_json(
            self.verb_realization_path
        )

        self.argument_realization = read_json(
            self.argument_realization_path
        )

        self._validate_resources()

    # ========================================================
    # Resource validation
    # ========================================================

    def _validate_resources(
        self,
    ) -> None:

        verbs = self.verb_realization.get(
            "verbs",
            {},
        )

        for verb_id in FRAME_TO_VERB.values():

            if verb_id not in verbs:

                raise RuntimeError(
                    f"Missing V0.5.1 verb realization: "
                    f"{verb_id}"
                )

        subjects = self.argument_realization.get(
            "subjects",
            {},
        )

        required_subjects = {
            "PERSON_I",
            "PERSON_YOU",
            "PERSON_HE",
            "PERSON_SHE",
            "PERSON_WE",
            "PERSON_THEY",
        }

        missing_subjects = (
            required_subjects
            - set(subjects)
        )

        if missing_subjects:

            raise RuntimeError(
                "Missing V0.5.1 subject realizations: "
                f"{sorted(missing_subjects)}"
            )

        objects = self.argument_realization.get(
            "objects",
            {},
        )

        if not objects:

            raise RuntimeError(
                "No V0.5.1 object realizations."
            )

        required_object_fields = {
            "zh",
            "en",
            "ru_acc",
            "uz_acc",
        }

        for object_id, object_data in objects.items():

            missing = (
                required_object_fields
                - set(object_data)
            )

            if missing:

                raise RuntimeError(
                    f"{object_id}: missing object fields "
                    f"{sorted(missing)}"
                )

    # ========================================================
    # Helpers
    # ========================================================

    def get_subject(
        self,
        subject_id: str,
    ) -> dict:

        subjects = self.argument_realization.get(
            "subjects",
            {},
        )

        subject = subjects.get(
            subject_id
        )

        if subject is None:

            raise RenderError(
                f"Unsupported V0.5.1 subject: "
                f"{subject_id}"
            )

        return subject

    def get_object(
        self,
        object_id: str,
    ) -> dict:

        objects = self.argument_realization.get(
            "objects",
            {},
        )

        obj = objects.get(
            object_id
        )

        if obj is None:

            raise RenderError(
                f"Unsupported V0.5.1 object: "
                f"{object_id}"
            )

        return obj

    def get_verb_policy(
        self,
        verb_id: str,
    ) -> dict:

        verbs = self.verb_realization.get(
            "verbs",
            {},
        )

        policy = verbs.get(
            verb_id
        )

        if policy is None:

            raise RenderError(
                f"Unsupported V0.5.1 verb: "
                f"{verb_id}"
            )

        return policy

    # ========================================================
    # Verb surface
    # ========================================================

    def get_zh_verb(
        self,
        verb_policy: dict,
        tense: str,
        polarity: str,
    ) -> str:

        zh = verb_policy[
            "zh"
        ]

        key = (
            f"{tense}_"
            f"{'positive' if polarity == 'pos' else 'negative'}"
        )

        surface = zh.get(
            key
        )

        if not surface:

            raise RenderError(
                f"Missing Chinese verb surface: "
                f"{key}"
            )

        return surface

    def get_en_verb(
        self,
        verb_policy: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        en = verb_policy[
            "en"
        ]

        if tense == "future":

            key = (
                "future"
                if polarity == "pos"
                else "future_negative"
            )

            surface = en.get(
                key
            )

        elif tense == "present":

            key = (
                "present"
                if polarity == "pos"
                else "present_negative"
            )

            surface = (
                en.get(
                    key,
                    {},
                )
                .get(
                    person_code
                )
            )

        else:

            raise RenderError(
                f"Unsupported English tense: {tense}"
            )

        if not surface:

            raise RenderError(
                "Missing English realization: "
                f"tense={tense}, "
                f"polarity={polarity}, "
                f"person={person_code}"
            )

        return surface

    def get_ru_verb(
        self,
        verb_policy: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        ru = verb_policy[
            "ru"
        ]

        forms = ru.get(
            tense
        )

        if not isinstance(
            forms,
            dict,
        ):

            raise RenderError(
                f"Missing Russian tense group: "
                f"{tense}"
            )

        surface = forms.get(
            person_code
        )

        if not surface:

            raise RenderError(
                "Missing Russian verb form: "
                f"tense={tense}, "
                f"person={person_code}"
            )

        if polarity == "neg":

            return (
                "не "
                + surface
            )

        return surface

    def get_uz_verb(
        self,
        verb_policy: dict,
        person_code: str,
        tense: str,
        polarity: str,
    ) -> str:

        # ----------------------------------------------------
        # Uzbek -adi / -maydi family is a present-future
        # paradigm.
        #
        # The same morphology can support present/future
        # interpretation depending on semantic context.
        # ----------------------------------------------------

        if tense not in {
            "present",
            "future",
        }:

            raise RenderError(
                f"Unsupported Uzbek tense: {tense}"
            )

        uz = verb_policy[
            "uz"
        ]

        group_name = (
            "present_future"
            if polarity == "pos"
            else "negative"
        )

        surface = (
            uz.get(
                group_name,
                {},
            )
            .get(
                person_code
            )
        )

        if not surface:

            raise RenderError(
                "Missing Uzbek verb form: "
                f"group={group_name}, "
                f"person={person_code}"
            )

        return surface

    # ========================================================
    # V0.5.1 transitive realization
    # ========================================================

    def render_v051_transitive(
        self,
        *,
        frame_id: str,
        slots: dict,
        features: dict,
    ) -> dict[str, str]:

        expected_verb = FRAME_TO_VERB.get(
            frame_id
        )

        if expected_verb is None:

            raise RenderError(
                f"Unsupported V0.5.1 frame: "
                f"{frame_id}"
            )

        subject_id = slots.get(
            "subject"
        )

        verb_id = slots.get(
            "verb"
        )

        object_id = slots.get(
            "object"
        )

        if not subject_id:

            raise RenderError(
                f"{frame_id}: missing subject"
            )

        if not object_id:

            raise RenderError(
                f"{frame_id}: missing object"
            )

        if verb_id != expected_verb:

            raise RenderError(
                f"{frame_id}: expected verb "
                f"{expected_verb}, got {verb_id}"
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

            raise RenderError(
                f"{frame_id}: unsupported tense "
                f"{tense}"
            )

        if polarity not in {
            "pos",
            "neg",
        }:

            raise RenderError(
                f"{frame_id}: unsupported polarity "
                f"{polarity}"
            )

        subject = self.get_subject(
            subject_id
        )

        obj = self.get_object(
            object_id
        )

        verb_policy = (
            self.get_verb_policy(
                verb_id
            )
        )

        person_code = subject[
            "person_code"
        ]

        zh_verb = self.get_zh_verb(
            verb_policy,
            tense,
            polarity,
        )

        en_verb = self.get_en_verb(
            verb_policy,
            person_code,
            tense,
            polarity,
        )

        ru_verb = self.get_ru_verb(
            verb_policy,
            person_code,
            tense,
            polarity,
        )

        uz_verb = self.get_uz_verb(
            verb_policy,
            person_code,
            tense,
            polarity,
        )

        # ====================================================
        # Language realization plans
        # ====================================================

        # Chinese:
        # SUBJECT + VERB + OBJECT
        zh = (
            f"{subject['zh']}"
            f"{zh_verb}"
            f"{obj['zh']}。"
        )

        # English:
        # SUBJECT + VERB + OBJECT
        en = (
            f"{subject['en']} "
            f"{en_verb} "
            f"{obj['en']}."
        )

        # Russian:
        # SUBJECT + VERB + OBJECT.ACC
        ru = (
            f"{subject['ru']} "
            f"{ru_verb} "
            f"{obj['ru_acc']}."
        )

        # Uzbek:
        # SUBJECT + OBJECT.ACC + VERB
        uz = (
            f"{subject['uz']} "
            f"{obj['uz_acc']} "
            f"{uz_verb}."
        )

        return {
            "zh": zh,
            "en": en,
            "ru": ru,
            "uz": uz,
        }

    # ========================================================
    # Public render
    # ========================================================

    def render(
        self,
        frame_id: str,
        slots: dict,
        features: dict | None = None,
        computed: dict | None = None,
    ) -> dict[str, str]:

        features = dict(
            features or {}
        )

        # ----------------------------------------------------
        # New V0.5.1 frames
        # ----------------------------------------------------

        if frame_id in FRAME_TO_VERB:

            return (
                self.render_v051_transitive(
                    frame_id=frame_id,
                    slots=slots,
                    features=features,
                )
            )

        # ----------------------------------------------------
        # Everything from V0.4.4.1 remains untouched.
        # ----------------------------------------------------

        return super().render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            computed=computed,
        )


# ============================================================
# Resource check
# ============================================================

def main() -> None:

    renderer = V051Renderer()

    print("=" * 90)
    print("RENDERER V0.5.1 RESOURCE CHECK")
    print("=" * 90)

    print(
        "Renderer version:",
        RENDERER_VERSION,
    )

    print(
        "Verb resource:",
        renderer.verb_realization_path,
    )

    print(
        "Argument resource:",
        renderer.argument_realization_path,
    )

    print(
        "Supported frames:",
        sorted(
            FRAME_TO_VERB
        ),
    )

    print()

    print(
        "RENDERER V0.5.1 LOAD PASS"
    )


if __name__ == "__main__":
    main()