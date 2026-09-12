# ZH-UZ relaxed 8k ablation

This is an intentionally relaxed, isolated experiment. It does not change the
Student trainer, the v4 dataset, the v2 artifacts, or any existing model. It
selects exactly 8,000 newly judged Teacher rows per direction before the usual
leakage checks, then trains under the existing weak-pair settings.

The selection order is deterministic:

1. first-review `PASS`, `HIGH` before `MEDIUM`;
2. first-review `MINOR` only when semantic consistency is true and every
   explicit semantic error flag is false;
3. relaxed MINOR rows use weights `0.2` (`HIGH`) and `0.1` (`MEDIUM`).

This branch is for measuring downstream FLORES dev impact, not for approving
the relaxed data as production-quality KD.

## Environment

```bash
cd /root/autodl-tmp/fourlang_translation

PY=/root/autodl-tmp/venvs/small100_student/bin/python
JUDGE_PY=/root/autodl-tmp/venvs/qwen3_judge/bin/python
AUDIT_CFG=configs/directions/zh_uz_flores_like_v2.toml
DATA_CFG=configs/directions/zh_uz_flores_relaxed_8k.toml
TRAIN_CFG=configs/specialists/weak_pair_ablation.toml

export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models
export FOURLANG_QWEN_MODEL_PATH=/root/autodl-tmp/models/Qwen3-8B
```

## 1. Finish first review only

The command reuses the existing 1,000-row calibration and resumes its output.
Do not run the full `teacher_second` stage for this ablation.

```bash
"$JUDGE_PY" scripts/pipeline_v2/qwen_judge.py teacher --config "$AUDIT_CFG"
```

## 2. Assemble the isolated relaxed dataset

```bash
"$PY" scripts/pipeline_v3/flores_like_data.py validate --config "$DATA_CFG"
"$PY" scripts/pipeline_v3/flores_like_data.py assemble --config "$DATA_CFG"
"$PY" scripts/pipeline_v3/flores_like_data.py status --config "$DATA_CFG"
```

The assembly report must show `8,000` new rows for both `zh-uz` and `uz-zh`.
If either direction is short after leakage protection, assembly stops instead
of silently reducing the requested experiment.

## 3. Train and evaluate without overwriting earlier runs

```bash
"$PY" scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k

"$PY" scripts/pipeline_v3/weak_pair_ablation.py train \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k

"$PY" scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k

"$PY" scripts/pipeline_v3/weak_pair_ablation.py compare \
  --config "$TRAIN_CFG" \
  --pair zh_uz
```

Artifacts are isolated under:

```text
data/distillation/zh_uz/flores_relaxed_8k/
data/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/
results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/
results/evaluation/weak_pair_ablation/zh_uz/flores_relaxed_8k.json
```

Make the decision on FLORES `dev`. Keep `devtest` untouched until a final
candidate has been selected.
