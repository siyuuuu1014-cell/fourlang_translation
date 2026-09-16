# ZH→UZ existing-data directional fine-tune

This experiment tests whether direction-specific continuation improves `zh→uz`
without waiting for newly generated teacher data.

## Contract

- Source model: `results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared`
- Training data: `data/distillation/zh_uz/flores_relaxed_8k/train.jsonl`
- Selected direction: `zh-uz` only
- Epochs: `1`
- Learning rate: `2e-6`
- Prepared data: `data/experiments/weak_pair_ablation/zh_uz/directional_existing_v1__zh_uz/`
- Output model: `results/experiments/weak_pair_ablation/zh_uz/directional_existing_v1__zh_uz/best_model/zh_uz/`
- FLORES dev report: `results/evaluation/weak_pair_ablation/zh_uz/directional_existing_v1__zh_uz.json`

The existing bidirectional model is read-only input to this run and is not
overwritten.

## Run

```bash
cd /root/autodl-tmp/fourlang_translation
PY=/root/autodl-tmp/venvs/small100_student/bin/python
TRAIN_CFG=configs/specialists/weak_pair_ablation.toml

"$PY" scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_existing_v1 \
  --direction zh-uz

"$PY" scripts/pipeline_v3/weak_pair_ablation.py train \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_existing_v1 \
  --direction zh-uz

"$PY" scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --config "$TRAIN_CFG" \
  --pair zh_uz \
  --variant directional_existing_v1 \
  --direction zh-uz
```

The `prepare`, `train`, and `evaluate` stages are restart-safe. Re-running an
identical completed training run reuses its exported model.

## Decision rule

Compare the new FLORES dev result with the current `flores_relaxed_8k_ep3`
result. Promote only if `zh→uz` chrF2 improves by at least `0.30` and BLEU does
not regress. Do not run FLORES devtest until the dev decision is frozen.
