from __future__ import annotations

import json
import re
from pathlib import Path

from scripts.synthetic.renderer_v04 import (
    DEFAULT_CONCEPTS,
    DEFAULT_FRAMES,
    RenderError,
    V04Renderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MOTION_FORMS = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
    / "motion_event_forms_v044.json"
)


def replace_surface_once(
    text: str,
    old: str,
    new: str,
) -> str:

    if old == new:
        return text

    pattern = (
        r"(?<![\w'’ʻ])"
        + re.escape(old)
        + r"(?![\w'’ʻ])"
    )

    result, count = re.subn(
        pattern,
        new,
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    if count != 1:
        raise RenderError(
            "Unable to replace Russian motion form: "
            f"old={old!r}, text={text!r}"
        )

    return result


class V044Renderer(V04Renderer):

    def __init__(
        self,
        concepts_path: str | Path = DEFAULT_CONCEPTS,
        frames_path: str | Path = DEFAULT_FRAMES,
        motion_forms_path: str | Path = DEFAULT_MOTION_FORMS,
    ) -> None:

        super().__init__(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

        self.motion_forms_path = Path(
            motion_forms_path
        )

        if not self.motion_forms_path.exists():
            raise FileNotFoundError(
                f"Motion-event forms missing: "
                f"{self.motion_forms_path}"
            )

        with self.motion_forms_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            self.motion_forms = json.load(f)

    def render_ru_habitual(
        self,
        *,
        verb_id: str,
        person_code: str,
        polarity: str,
    ) -> str:

        try:
            surface = (
                self.motion_forms[
                    "verbs"
                ][
                    verb_id
                ][
                    "ru"
                ][
                    "habitual"
                ][
                    person_code
                ]
            )

        except KeyError as exc:
            raise RenderError(
                f"No habitual Russian form: "
                f"verb={verb_id}, "
                f"person={person_code}"
            ) from exc

        if polarity == "neg":
            return f"не {surface}"

        return surface

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

        texts = super().render(
            frame_id=frame_id,
            slots=slots,
            features=features,
            computed=computed,
        )

        verb_id = slots.get(
            "verb"
        )

        if verb_id not in {
            "GO",
            "COME",
        }:
            return texts

        event_type = features.get(
            "event_type"
        )

        if event_type != "habitual":
            return texts

        tense = features.get(
            "tense"
        )

        if tense != "present":
            raise RenderError(
                "habitual motion currently requires "
                "tense=present"
            )

        subject_id = slots.get(
            "subject"
        )

        if not subject_id:
            raise RenderError(
                "habitual motion requires subject"
            )

        subject = self.get_concept(
            subject_id
        )

        person_code = self.get_person_code(
            subject,
            "ru",
        )

        polarity = str(
            features.get(
                "polarity",
                "pos",
            )
        )

        verb = self.get_concept(
            verb_id
        )

        old_surface = super().render_ru_verb(
            verb,
            person_code,
            "present",
            polarity,
        )

        new_surface = self.render_ru_habitual(
            verb_id=verb_id,
            person_code=person_code,
            polarity=polarity,
        )

        texts[
            "ru"
        ] = replace_surface_once(
            texts["ru"],
            old_surface,
            new_surface,
        )

        return texts


def main() -> None:

    renderer = V044Renderer()

    print(
        "=" * 80
    )

    print(
        "RENDERER V0.4.4 RESOURCE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "Motion morphology version:",
        renderer.motion_forms.get(
            "version"
        ),
    )

    print(
        "Habitual verbs:",
        list(
            renderer.motion_forms
            .get(
                "verbs",
                {}
            )
            .keys()
        ),
    )

    print()

    print(
        "RENDERER V0.4.4 LOAD PASS"
    )


if __name__ == "__main__":
    main()