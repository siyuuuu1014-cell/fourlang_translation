# Current FourLang model usage

The unified test entry point maps all 12 directions onto the six currently
selected bidirectional specialists:

| Directions | Model path |
|---|---|
| `en-zh`, `zh-en` | `results/student/pair_specialists/en_zh/exp2/best_model/shared` |
| `en-uz`, `uz-en` | `models/final_specialists/en_uz_small100_v1` |
| `en-ru`, `ru-en` | `results/student/pair_specialists/en_ru/exp2/best_model/shared` |
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
