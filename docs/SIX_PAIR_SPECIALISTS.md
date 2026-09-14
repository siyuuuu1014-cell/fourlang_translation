# Six bidirectional pair specialists

This branch is an independent comparison against the shared four-language
Student. It does not replace or modify the unified-model pipeline.

## Experimental contract

- One independent SMaLL-100 instance is assigned to each unordered language
  pair: EN-ZH, EN-UZ, EN-RU, ZH-UZ, ZH-RU, and UZ-RU.
- Every instance supports both directions in its pair.
- Exp1 uses audited human parallel data.
- Exp2 starts from that pair's Exp1 and uses Teacher KD plus human replay.
- The backbone, seed, row budget, optimizer settings, and FLORES evaluation are
  held constant so routing/specialization is not confounded with architecture.
- A model is eligible for routing only if Exp2 matches or beats Exp1 in BLEU and
  chrF2 in both directions.

The experiment manifest is `configs/specialists/six_pair.toml`. The orchestration
entry point is `scripts/pipeline_v3/pair_specialist_flow.py`; it delegates model
training to the existing `train_model` implementation.

## Original experiment plan (historical)

This table records the initial plan, not the current training status. Selected
models are now listed in `configs/specialists/current_pair_models.json` and
`docs/CURRENT_MODEL_USAGE.md`. Do not rerun the historical training commands
below as part of repository organization.

| Pair | Mode | Meaning |
| --- | --- | --- |
| EN-UZ | `reuse_exp2` | Reuse the already accepted bidirectional Exp2 model. |
| EN-RU | `resume_exp2` | Reuse the completed Exp1 and continue with Exp2. |
| EN-ZH | `train_full` | Train a new bidirectional pair Student. Existing Marian models remain controls. |
| ZH-UZ | `train_full` | Blocked until the diagnosed KD-quality issues are resolved. |
| ZH-RU | `train_full` | Train Exp1 and then Exp2. |
| UZ-RU | `train_full` | Train Exp1 and then Exp2. |

## Server workflow

Set the existing model root first:

```bash
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models
cd /root/autodl-tmp/fourlang_translation
```

Inspect all six pairs without starting training:

```bash
python scripts/pipeline_v3/pair_specialist_flow.py status
```

For a new pair, prepare and train Exp1 before Exp2:

```bash
python scripts/pipeline_v3/pair_specialist_flow.py prepare --pair zh_ru --experiment exp1
python scripts/pipeline_v3/pair_specialist_flow.py train --pair zh_ru --experiment exp1
python scripts/pipeline_v3/pair_specialist_flow.py evaluate --pair zh_ru --experiment exp1

python scripts/pipeline_v3/pair_specialist_flow.py prepare --pair zh_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py train --pair zh_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py evaluate --pair zh_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py gate --pair zh_ru
```

For EN-RU, prepare both data stages but train only Exp2 because its original Exp1
is retained by the manifest:

```bash
python scripts/pipeline_v3/pair_specialist_flow.py prepare --pair en_ru --experiment exp1
python scripts/pipeline_v3/pair_specialist_flow.py prepare --pair en_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py evaluate --pair en_ru --experiment exp1
python scripts/pipeline_v3/pair_specialist_flow.py train --pair en_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py evaluate --pair en_ru --experiment exp2
python scripts/pipeline_v3/pair_specialist_flow.py gate --pair en_ru
```

After both directions pass, promotion atomically freezes the model and updates
only those two inference routes:

```bash
python scripts/pipeline_v3/pair_specialist_flow.py promote --pair zh_ru
```

Do not change `zh_uz.kd_quality_status` to `approved` merely because its KD file
exists. Point the manifest at the reviewed, frozen replacement first, then run
the same preparation and training sequence.
