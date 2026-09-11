# ZH-UZ FLORES-like KD experiment

This is an isolated data experiment. It does not change `zh_uz/v4`, the
SMaLL-100 training implementation, the existing Exp1/Exp2 checkpoints, or the
FLORES benchmark files.

## Intended mixture

The final bidirectional training file has equal effective loss mass in both
directions. Within each direction it contains:

- 40% human replay;
- 30% existing general-domain KD from `zh_uz/v4`;
- 30% newly generated FLORES-like KD.

The builder keeps every unique row and adjusts row weights. It does not repeat
human rows merely to obtain the requested ratio.

## Server setup

Run from the repository root with the same environment and model locations used
for the completed specialist experiments:

```bash
PY=/root/autodl-tmp/venvs/small100_student/bin/python
JUDGE_PY=/root/autodl-tmp/venvs/qwen3_judge/bin/python
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models
export FOURLANG_QWEN_MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B
export HF_HOME=/root/autodl-tmp/huggingface
CFG=configs/directions/zh_uz_flores_like_v1.toml
```

If the Qwen directory has a different name, change only
`FOURLANG_QWEN_MODEL_PATH`. The NLLB teacher remains
`$FOURLANG_MODEL_ROOT/nllb-200-3.3B`.

## 1. Validate and profile

```bash
$PY scripts/pipeline_v3/flores_like_data.py validate --config $CFG
$PY scripts/pipeline_v3/flores_like_data.py profile --config $CFG
```

Expected status: `PASS`. Profiling reads FLORES dev only to derive structural
length and sentence-feature quotas. It does not copy FLORES rows into training.

## 2. Extend with independent Wikipedia sentences

Collect 15,000 new Chinese and 8,000 new Uzbek Wikipedia sentences. The
operation checkpoints every 250 accepted rows and resumes safely after a
network or process interruption:

```bash
$PY scripts/pipeline_v3/flores_like_data.py extend --config $CFG
```

Expected report:

```text
reports/experiments/zh_uz_flores_like_v1/wikipedia_extension.json
```

The report must have `status: PASS`, `zh: 15000`, and `uz: 8000`. The source is
the dated `wikimedia/wikipedia` 20231101 configuration; collected rows retain
their source ID and CC BY-SA/GFDL provenance.

## 3. Stage unused independent source sentences

```bash
$PY scripts/pipeline_v3/flores_like_data.py stage --config $CFG
```

This reads the unused part of the existing v3 monolingual collection, excludes
v3-selected sources, v4, validation, FLORES dev and devtest, and applies
near-duplicate protection. It deliberately does not reuse the old source-review
coverage, because those reviewed rows substantially overlap the sources already
used by v3.

Expected output:

```text
data/pipeline_v2/zh_uz_flores_like_v1/monolingual_candidates.jsonl
reports/experiments/zh_uz_flores_like_v1/source_staging.json
```

Rerun this step after the Wikipedia extension. It merges the extension with the
old safe remainder and rewrites only the isolated source-review input.

## 4. Audit the newly staged sources with Qwen

This experiment uses the isolated `natural_sentence_entities_allowed_v2`
policy. It still rejects fragments, SEO/advertising copy, wrong-language text
and unnatural machine translation, but it does not reject an otherwise natural
sentence merely for containing proper names, acronyms, numbers or ordinary
Unicode punctuation. Cached source verdicts from another policy are not reused.

```bash
$JUDGE_PY scripts/pipeline_v2/qwen_judge.py source --config $CFG --calibration
```

Build a capacity estimate from that calibration before running the expensive
full audit:

```bash
$PY scripts/pipeline_v3/flores_like_data.py source_calibration --config $CFG
```

Continue only when `full_source_audit_recommended` is `true`. `READY` means the
lower 95% estimate still reaches 12,000 rows; `LIKELY_READY` means the point
estimate reaches it but the conservative estimate does not. After checking this
report, run the full source audit:

```bash
$JUDGE_PY scripts/pipeline_v2/qwen_judge.py source --config $CFG
```

Then make the final FLORES-shaped selection from Qwen `PASS` rows:

```bash
$PY scripts/pipeline_v3/flores_like_data.py select --config $CFG
```

Expected outputs:

```text
data/distillation/zh_uz/flores_like_v1/selected_sources.jsonl
data/pipeline_v2/zh_uz_flores_like_v1/kd_candidates.jsonl
reports/experiments/zh_uz_flores_like_v1/source_selection.json
```

Both directions should report `selected: 12000`. If selection is short, inspect
the persisted `source_selection.json`; do not lower leakage or Qwen-quality
requirements merely to reach the target.

## 5. Generate Teacher translations

The generic, checkpointed Teacher implementation is reused unchanged:

```bash
$PY scripts/pipeline_v2/seq2seq_flow.py generate_teacher --config $CFG
```

Expected output:

```text
data/pipeline_v2/zh_uz_flores_like_v1/teacher_generated.parquet
```

The command is resumable. Rerun the same command after an interruption; do not
delete checkpoint shards unless the input/configuration was intentionally
changed.

## 6. Audit Teacher output with Qwen

Run the small calibration first:

```bash
$JUDGE_PY scripts/pipeline_v2/qwen_judge.py teacher --config $CFG --calibration
```

Inspect the label/usefulness distribution. If it is sensible, run the complete
audit:

```bash
$JUDGE_PY scripts/pipeline_v2/qwen_judge.py teacher --config $CFG
```

Expected output:

```text
data/pipeline_v2/zh_uz_flores_like_v1/teacher_judged.parquet
```

Only parseable `PASS` rows with usefulness `HIGH` or `MEDIUM` are eligible.

## 7. Assemble the 40/30/30 dataset

```bash
$PY scripts/pipeline_v3/flores_like_data.py assemble --config $CFG
$PY scripts/pipeline_v3/flores_like_data.py status --config $CFG
```

Expected final files:

```text
data/distillation/zh_uz/flores_like_v1/train.jsonl
data/distillation/zh_uz/flores_like_v1/validation.jsonl
data/distillation/zh_uz/flores_like_v1/manifest.json
reports/experiments/zh_uz_flores_like_v1/assembly.json
```

`assemble` refuses to publish if either direction has fewer than 8,000 accepted
new Teacher rows. The manifest must say:

```json
{
  "status": "FLORES_LIKE_V1_BUILT_NOT_TRAINED",
  "flores_rows_in_training": 0,
  "existing_v4_modified": false
}
```

At that point the data experiment is ready, but no model has been trained yet.
Keep this distinction: a successful assembly report is not a completed student
experiment.

## Safety notes

- Never use `flores_devtest.parquet` for training or repeated model selection.
- Do not overwrite `data/distillation/zh_uz/v4`.
- Do not point the normal six-pair Exp2 command at this dataset, because its
  result directory belongs to the completed baseline.
- Commit only code/configuration/reports intended for Git. Large model, dataset,
  checkpoint and generated result files stay on the server.
