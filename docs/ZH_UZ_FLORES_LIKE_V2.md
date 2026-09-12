# ZH-UZ FLORES-like v2 incremental runbook

This branch changes data selection and quality control only. It does not change
Student training, evaluation, or deployment logic. Version 1 remains an immutable
baseline.

## What v2 changes

- selects 14,000 sources per direction instead of 12,000;
- raises the Wikipedia ceiling to 65%;
- limits `v2_bronze` to one percent;
- reuses source judgments and NLLB translations only after exact input checks;
- sends only first-pass `MINOR` rows without any semantic error flag to an
  independent binary second review;
- gives rescued MINOR rows lower weights (`HIGH=0.5`, `MEDIUM=0.4`).

## Environment

```bash
cd /root/autodl-tmp/fourlang_translation

PY=/root/autodl-tmp/venvs/small100_student/bin/python
JUDGE_PY=/root/autodl-tmp/venvs/qwen3_judge/bin/python
CFG=configs/directions/zh_uz_flores_like_v2.toml

export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models
export FOURLANG_QWEN_MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B
```

## 1. Rebuild selection and reuse source judgments

```bash
"$PY" scripts/pipeline_v3/flores_like_data.py validate --config "$CFG"
"$PY" scripts/pipeline_v3/flores_like_data.py stage --config "$CFG"
"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py source --config "$CFG"
"$PY" scripts/pipeline_v3/flores_like_data.py select --config "$CFG"
```

The source Judge command should report reused compatible judgments. It loads Qwen
only if a staged source has no compatible v1 judgment.

## 2. Generate only missing Teacher translations

```bash
"$PY" scripts/pipeline_v2/seq2seq_flow.py generate_teacher --config "$CFG"
```

The command prints the exact number of compatible v1 translations reused before
loading NLLB. New v2 checkpoint shards are isolated under the v2 namespace.

## 3. Calibrate first and second review

```bash
"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py teacher \
  --config "$CFG" --calibration

"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py teacher_second \
  --config "$CFG" --calibration

"$PY" scripts/pipeline_v3/flores_like_data.py teacher_calibration \
  --config "$CFG"
```

Continue only when `full_teacher_audit_recommended` is `true`. The projection
counts original strict PASS rows plus only independently confirmed, unflagged
MINOR rows.

## 4. Incremental full audit and assembly

```bash
"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py teacher --config "$CFG"
"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py teacher_second --config "$CFG"

"$PY" scripts/pipeline_v3/flores_like_data.py assemble --config "$CFG"
"$PY" scripts/pipeline_v3/flores_like_data.py status --config "$CFG"
```

The full first review automatically imports compatible calibration judgments.
Do not run this section when the calibration capacity report says `NOT_READY`.

Expected final status:

```text
FLORES_LIKE_V2_BUILT_NOT_TRAINED
```

This means the dataset is ready. It does not mean a Student model has been
trained.
