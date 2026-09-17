# ZH→UZ DeepSeek partial v1

This experiment freezes the already completed portion of the resumable
DeepSeek checkpoint and uses only freshly re-audited, unique `zh→uz` rows.
The live checkpoint is never modified, so API generation can resume later.

## Build the immutable training mix

Run only while no `deepseek_teacher.py generate` process is writing the
checkpoint:

```bash
PY=/root/autodl-tmp/venvs/small100_student/bin/python
DEEPSEEK_CFG=configs/directions/zh_uz_deepseek_teacher_v1.toml
TRAIN_CFG=configs/specialists/weak_pair_ablation.toml

"$PY" scripts/pipeline_v3/build_deepseek_partial_zh_uz.py \
  --config "$DEEPSEEK_CFG"
```

The builder:

- validates every checkpoint row against the frozen generation fingerprint;
- verifies that the checkpoint does not change while it is read;
- re-runs translation and source filters offline;
- keeps only usable `zh→uz` rows;
- preserves human targets, replaces duplicate legacy Teacher KD targets with
  DeepSeek translations, and adds genuinely new sources;
- assigns DeepSeek rows weight `1.5`;
- refuses to build with fewer than 4,000 unique usable rows;
- writes a versioned report and hashes without changing the checkpoint.

Outputs:

```text
data/distillation/zh_uz/deepseek_partial_v1/train.jsonl
data/distillation/zh_uz/deepseek_partial_v1/accepted_deepseek.jsonl
data/distillation/zh_uz/deepseek_partial_v1/reaudit.parquet
data/distillation/zh_uz/deepseek_partial_v1/build_report.json
```

## Prepare, train, and evaluate

```bash
"$PY" scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_deepseek_partial_v1 \
  --direction zh-uz

"$PY" scripts/pipeline_v3/weak_pair_ablation.py train \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_deepseek_partial_v1 \
  --direction zh-uz

"$PY" scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_deepseek_partial_v1 \
  --direction zh-uz
```

The experiment starts from the current `flores_relaxed_8k_ep3` model, trains
for one epoch at `2e-6`, and writes to an isolated result directory. Do not run
FLORES devtest until the dev result is selected.
