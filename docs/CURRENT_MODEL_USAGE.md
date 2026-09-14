# Current FourLang model usage

The unified test entry point maps all 12 directions onto the six currently
selected bidirectional specialists:

The machine-readable source of truth is
`configs/specialists/current_pair_models.json`. It records each model's code,
training/validation data, evaluation evidence, and protected shared-model paths.

The older one-way Marian models at `models/final_specialists/en_zh_v1` and
`models/final_specialists/zh_en_v1` remain protected because the existing model
registry still references them. The unified tester uses the newer bidirectional
EN-ZH Exp2; do not remove the Marian copies until the registry is formally
migrated and verified.

| Directions | Model path |
|---|---|
| `en-zh`, `zh-en` | `results/student/pair_specialists/en_zh/exp2/best_model/shared` |
| `en-uz`, `uz-en` | `models/final_specialists/en_uz_small100_v1` |
| `en-ru`, `ru-en` | `models/final_pair_specialists/en_ru_v1` |
| `zh-uz`, `uz-zh` | `results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/best_model/shared` |
| `zh-ru`, `ru-zh` | `results/student/pair_specialists/zh_ru/exp2/best_model/shared` |
| `uz-ru`, `ru-uz` | `results/student/pair_specialists/uz_ru/exp2/best_model/shared` |

Inspect every resolved absolute path without loading model weights:

```bash
PY=/root/autodl-tmp/venvs/small100_student/bin/python
"$PY" scripts/pipeline_v3/translate_current_models.py --list-models
```

Translate one sentence:

```bash
"$PY" scripts/pipeline_v3/translate_current_models.py \
  --direction ru-uz \
  --text "На улице идёт дождь."
```

Omit `--text` to load the selected model once and test multiple sentences
interactively. Enter `/quit` to exit:

```bash
"$PY" scripts/pipeline_v3/translate_current_models.py --direction zh-en
```

The default runtime uses FP16 and five-beam decoding. Use `--json` for structured
output. An explicit `--model PATH` overrides the built-in route for one run.
Alternatively set a per-direction environment variable such as
`FOURLANG_ZH_UZ_MODEL_PATH`; an explicit `--model` takes priority.

## Inventory and cleanup

Deletion is currently suspended at the user's request. Even `--apply` refuses
to run until the experiment record and evidence archive have been completed and
reviewed. The older command example below is not currently executable.

Create a timestamped evidence snapshot and experiment record on the server:

```bash
"$PY" scripts/pipeline_v3/archive_pair_experiments.py
```

The output lives under `reports/experiment_archive/<UTC timestamp>/` and includes
`EXPERIMENT_RECORD.md`, original code/config/report copies in `evidence/`, a file
inventory, omissions, checksums, and a compressed evidence copy. Weights and large
datasets stay in their original locations; this is not a full weight backup.

Create a current inventory, including path availability and disk usage:

```bash
"$PY" scripts/pipeline_v3/manage_pair_model_artifacts.py inventory
```

Preview cleanup of only the explicitly abandoned ZH-UZ ablation artifacts:

```bash
"$PY" scripts/pipeline_v3/manage_pair_model_artifacts.py cleanup
```

The preview does not delete anything. Review its exact paths and reclaimable
bytes first. Applying cleanup requires all six selected models to exist with a
`config.json`, plus an explicit confirmation token:

```bash
"$PY" scripts/pipeline_v3/manage_pair_model_artifacts.py cleanup \
  --apply \
  --confirm DELETE_DISCARDED_PAIR_EXPERIMENTS
```

The cleanup allowlist excludes the shared four-language model, its results and
multilingual data, all six selected pair models, evaluation evidence, and the
active DeepSeek Teacher output. Exp1 controls and source datasets are retained
for provenance; tracked source code is not deleted because it consumes little
space and is needed to reproduce the selected models.
