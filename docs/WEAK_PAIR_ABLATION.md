# Weak-pair student ablation

This experiment diagnoses why the existing NLLB-200 3.3B KD data is not being
fully absorbed by the lightweight SMaLL-100 students for `zh_uz` and `uz_ru`.
It does not modify `pair_specialist_flow.py`, the six-pair configuration, the KD
datasets, or any promoted model.

## Controlled question

Phase 1 changes only data amount/layout while preserving the existing Exp2
settings (`2` epochs, learning rate `5e-6`, effective batch size `32`):

- selected current baseline;
- one bidirectional model trained with every unique eligible KD row;
- two directional models, each trained with every eligible row for its direction.

All selection evaluation uses FLORES `dev`. FLORES `devtest` is included in
leakage protection but is not evaluated by this experiment.

## Server preflight

From the repository root:

```bash
PY=/root/miniconda3/envs/small100_student/bin/python
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models

$PY scripts/pipeline_v3/weak_pair_ablation.py validate
$PY scripts/pipeline_v3/weak_pair_ablation.py status
```

If the Python executable is elsewhere, keep using the `$PY` value already used
for the six-pair runs.

## Run `zh_uz`

First evaluate both existing checkpoints on FLORES dev. The selected baseline is
Exp1, but retaining Exp2 makes the comparison auditable.

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant baseline_exp1
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant baseline_exp2
```

Train the full-data bidirectional control:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --pair zh_uz --variant bidir_full
$PY scripts/pipeline_v3/weak_pair_ablation.py train \
  --pair zh_uz --variant bidir_full
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant bidir_full
```

Train the two directional controls sequentially on the same GPU:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --pair zh_uz --variant directional_full --direction zh-uz
$PY scripts/pipeline_v3/weak_pair_ablation.py train \
  --pair zh_uz --variant directional_full --direction zh-uz
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant directional_full --direction zh-uz

$PY scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --pair zh_uz --variant directional_full --direction uz-zh
$PY scripts/pipeline_v3/weak_pair_ablation.py train \
  --pair zh_uz --variant directional_full --direction uz-zh
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant directional_full --direction uz-zh
```

Build the comparison:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py compare --pair zh_uz
python -m json.tool \
  results/evaluation/weak_pair_ablation/zh_uz/comparison.json
```

## Decision rule

- `bidir_full` wins by at least `1.0` chrF++: the old 10K cap was the main
  bottleneck.
- directional runs win by at least `1.0` chrF++: bidirectional interference is
  material; keep one logical pair route but test direction-specific weights.
- gains between `0.3` and `1.0`: rerun the winning structure with another seed
  before changing the formal route.
- gains below `0.3`: treat the layout/data change as flat and proceed to the
  learning-rate ablation.

Do not promote any artifact from this diagnostic directory. After `zh_uz` has a
clear result, repeat the same commands for `uz_ru`; its selected baseline is
Exp2.

## Phase 1b after the initial `zh_uz` result

The first full-data run contains about 71% Teacher rows and 29% human replay,
whereas the original controlled Exp2 used 60/40. Run these two additional
bidirectional controls; existing runs are reused and do not need to be repeated.

`full_weighted_60_40` keeps every unique row but rescales Teacher row weights per
direction so that Teacher KD contributes exactly 60% of the effective training
loss:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --pair zh_uz --variant full_weighted_60_40
$PY scripts/pipeline_v3/weak_pair_ablation.py train \
  --pair zh_uz --variant full_weighted_60_40
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant full_weighted_60_40
```

`full_native_lr2e6` preserves the natural full-data mixture and tests a smaller
update size:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py prepare \
  --pair zh_uz --variant full_native_lr2e6
$PY scripts/pipeline_v3/weak_pair_ablation.py train \
  --pair zh_uz --variant full_native_lr2e6
$PY scripts/pipeline_v3/weak_pair_ablation.py evaluate \
  --pair zh_uz --variant full_native_lr2e6
```

Rebuild the comparison after both evaluations:

```bash
$PY scripts/pipeline_v3/weak_pair_ablation.py compare --pair zh_uz
python -m json.tool \
  results/evaluation/weak_pair_ablation/zh_uz/comparison.json
```
