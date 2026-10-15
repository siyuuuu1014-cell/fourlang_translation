from pathlib import Path

from scripts.pipeline_v3 import preview_acceptance_v3_sources as preview
from scripts.pipeline_v3 import freeze_acceptance_v3 as freeze


def test_candidate_path_allowlist():
    assert preview.candidate_path_allowed(
        Path("data/pipeline_v2/zh_ru/splits/test.jsonl")
    )
    assert preview.candidate_path_allowed(Path("data/clean/en_uz/exp2/test.jsonl"))
    assert not preview.candidate_path_allowed(
        Path("data/benchmark/fourlang/flores_devtest.parquet")
    )
    assert not preview.candidate_path_allowed(
        Path("data/synthetic/generated/v03_test/semantic_v03_raw.jsonl")
    )


def test_stable_selection_is_repeatable_and_bounded():
    rows = [
        {"src_lang": "uz", "src_text": f"sentence {index}"}
        for index in range(20)
    ]
    first = preview.stable_select(rows, 5, 2026, "uz")
    assert first == preview.stable_select(rows, 5, 2026, "uz")
    assert len(first) == 5


def test_near_duplicate_selection_replaces_protected_and_candidate_matches():
    candidates = {language: {} for language in preview.LANGUAGES}
    protected = {language: set() for language in preview.LANGUAGES}
    for language in preview.LANGUAGES:
        protected[language].add("protected sentence alpha")
        texts = [
            "protected sentence alpha!",
            "candidate sentence beta",
            "candidate sentence beta!",
            "different sentence gamma",
            "another distinct sentence delta",
        ]
        candidates[language] = {
            str(index): {"src_lang": language, "src_text": text}
            for index, text in enumerate(texts)
        }
    selected, available, rejected = freeze.select_near_clean(
        candidates, protected, 2, 2026, shingle_size=3, threshold=0.8
    )
    assert all(sum(r["src_lang"] == language for r in selected) == 2 for language in preview.LANGUAGES)
    assert all(available[language] >= 2 for language in preview.LANGUAGES)
    assert sum(rejected.values()) >= len(preview.LANGUAGES)


def test_build_artifacts_has_complete_direction_coverage():
    selected = [
        {
            "src_lang": language,
            "src_text": f"distinct source for {language}",
            "origin": "data/splits/example/test.jsonl",
            "origin_kind": "AUTHENTIC_HELDOUT_CORPUS_NOT_USER_COLLECTED",
            "usage": "ACCEPTANCE_ONLY_NOT_FOR_TRAINING",
            "source_review_status": "PENDING_NATIVE_REVIEW",
        }
        for language in preview.LANGUAGES
    ]
    sources, tasks = freeze.build_artifacts(selected)
    assert len(sources) == 4
    assert len(tasks) == 12
    assert all(task["src_lang"] != task["tgt_lang"] for task in tasks)
    assert all(task["reference_text"] is None for task in tasks)


def test_frozen_publication_is_idempotent_and_never_overwrites(tmp_path):
    output = tmp_path / "acceptance-v3"
    files = {"sources.jsonl": b"source\n", "tasks.jsonl": b"task\n"}
    manifest = {"files": {"sources.jsonl": "a", "tasks.jsonl": "b"}}
    freeze._publish_locked(output, files, manifest)
    freeze._publish_locked(output, files, manifest)
    changed = {**files, "tasks.jsonl": b"changed\n"}
    try:
        freeze._publish_locked(output, changed, manifest)
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("Frozen artifacts must never be overwritten")
