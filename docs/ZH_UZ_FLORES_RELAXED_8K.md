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

## 4. Isolated three-epoch follow-up

Run this only after the two-epoch model has been evaluated. It starts again
from the same Exp1 model and uses the same prepared source data, learning rate,
batch size, seed, and optimizer. Only the maximum epoch count changes from two
to three, and all artifacts remain separate.

```bash
"$PY" scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k_ep3

"$PY" scripts/pipeline_v3/weak_pair_ablation.py train \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k_ep3

"$PY" scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant flores_relaxed_8k_ep3

"$PY" scripts/pipeline_v3/weak_pair_ablation.py compare \
  --config "$TRAIN_CFG" \
  --pair zh_uz
```

The three-epoch model is stored under
`results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/` and never
overwrites the two-epoch run.

## 5. One-time protected final evaluation

After the FLORES `dev` comparison selects `flores_relaxed_8k_ep3` in both
directions, run the frozen final evaluation exactly once:

```bash
"$PY" scripts/pipeline_v3/weak_pair_ablation.py final_evaluate \
  --config "$TRAIN_CFG" \
  --pair zh_uz
```

The command refuses a candidate that is not the recorded `dev` winner. It
evaluates the configured Exp1 baseline and frozen candidate on FLORES
`devtest`, writes the per-direction BLEU/chrF2 gate to
`results/evaluation/weak_pair_ablation/zh_uz/final_devtest.json`, and reuses
that immutable report without loading either model if invoked again. Do not
change the candidate or tune training after viewing this report.
