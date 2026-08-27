from __future__ import annotations

import json
from pathlib import Path

from scripts.synthetic.generate_synthetic_v04 import (
    PROJECT_ROOT,
)

from scripts.synthetic.renderer_v04 import (
    DEFAULT_FRAMES,
    RenderError,
)

from scripts.synthetic.renderer_v044 import (
    V044Renderer,
    replace_surface_once,
)


RESOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "resources"
    / "v04"
)

DEFAULT_CONCEPTS = (
    RESOURCE_DIR
    / "concepts_v044.jsonl"
)

DEFAULT_LEXICALIZATION = (
    RESOURCE_DIR
    / "ru_motion_lexicalization_v0441.json"
)


class V0441Renderer(
    V044Renderer
):

    def __init__(
        self,
        concepts_path: str | Path = DEFAULT_CONCEPTS,
        frames_path: str | Path = DEFAULT_FRAMES,
        lexicalization_path: str | Path = DEFAULT_LEXICALIZATION,
    ) -> None:

        super().__init__(
            concepts_path=concepts_path,
            frames_path=frames_path,
        )

        self.lexicalization_path = Path(
            lexicalization_path
        )

        if not self.lexicalization_path.exists():

            raise FileNotFoundError(
                f"Missing Russian motion lexicalization: "
                f"{self.lexicalization_path}"
            )

        with self.lexicalization_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            self.ru_motion_lexicalization = (
                json.load(f)
            )

    # ========================================================
    # Destination classification
    # ========================================================

    def get_destination_class(
        self,
        destination_id: str | None,
    ) -> str:

        classes = (
            self.ru_motion_lexicalization
            .get(
                "destination_classes",
                {},
            )
        )

        if destination_id is None:

            return "local"

        for class_name, destination_ids in (
            classes.items()
        ):

            if destination_id in destination_ids:

                return class_name

        raise RenderError(
            "Russian habitual motion destination "
            f"is not classified: {destination_id}"
        )

    # ========================================================
    # Desired Russian form
    # ========================================================

    def get_ru_motion_surface(
        self,
        *,
        verb_id: str,
        destination_class: str,
        person_code: str,
    ) -> str:

        try:

            return (
                self.ru_motion_lexicalization[
                    "verbs"
                ][
                    verb_id
                ][
                    destination_class
                ][
                    person_code
                ]
            )

        except KeyError as exc:

            raise RenderError(
                "Missing V0.4.4.1 Russian motion form: "
                f"verb={verb_id}, "
                f"class={destination_class}, "
                f"person={person_code}"
            ) from exc

    # ========================================================
    # Render
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

        if features.get(
            "event_type"
        ) != "habitual":

            return texts

        if features.get(
            "tense"
        ) != "present":

            raise RenderError(
                "habitual motion requires tense=present"
            )

        # ----------------------------------------------------
        # Habitual negative is deliberately not supported
        # in V0.4.4.1.
        # ----------------------------------------------------

        if features.get(
            "polarity"
        ) != "pos":

            raise RenderError(
                "V0.4.4.1 habitual motion "
                "supports positive polarity only."
            )

        destination_id = slots.get(
            "destination"
        )

        # GO without destination is too weak:
        #
        # You go every day.
        #
        # Do not allow it.
        if (
            verb_id == "GO"
            and destination_id is None
        ):

            raise RenderError(
                "Habitual GO requires destination."
            )

        subject_id = slots.get(
            "subject"
        )

        if not subject_id:

            raise RenderError(
                "Habitual motion requires subject."
            )

        subject = self.get_concept(
            subject_id
        )

        person_code = self.get_person_code(
            subject,
            "ru",
        )

        destination_class = (
            self.get_destination_class(
                destination_id
            )
        )

        desired_surface = (
            self.get_ru_motion_surface(
                verb_id=verb_id,
                destination_class=destination_class,
                person_code=person_code,
            )
        )

        # ----------------------------------------------------
        # V044Renderer has already generated its habitual form.
        # Replace that verified old habitual surface with the
        # destination-sensitive V0.4.4.1 surface.
        # ----------------------------------------------------

        old_surface = (
            super().render_ru_habitual(
                verb_id=verb_id,
                person_code=person_code,
                polarity="pos",
            )
        )

        texts[
            "ru"
        ] = replace_surface_once(
            texts["ru"],
            old_surface,
            desired_surface,
        )

        return texts


def main() -> None:

    renderer = V0441Renderer()

    print(
        "=" * 90
    )

    print(
        "RENDERER V0.4.4.1 RESOURCE CHECK"
    )

    print(
        "=" * 90
    )

    print(
        "Lexicalization version:",
        renderer
        .ru_motion_lexicalization
        .get(
            "version"
        ),
    )

    print(
        "Destination classes:",
        renderer
        .ru_motion_lexicalization
        .get(
            "destination_classes"
        ),
    )

    print()

    print(
        "RENDERER V0.4.4.1 LOAD PASS"
    )


if __name__ == "__main__":
    main()