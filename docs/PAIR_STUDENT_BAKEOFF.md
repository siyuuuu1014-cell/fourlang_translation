# Pair-level bidirectional Student bake-off

This stage starts only after the controlled six-pair SMaLL-100 baseline is
complete. It selects a promising backbone for a fixed-budget Exp1 pilot; it
does not train, promote, or update inference routes.

## Contract

- One candidate must serve both directions of a language pair.
- Selection uses only the shared four-language FLORES `dev` file. The existing
  `devtest` gate remains final evaluation evidence and must not be used to tune
  the architecture.
- Candidates must be commercially eligible and fit the configured 1.3B
  parameter ceiling.
- Ranking maximizes the weaker direction's chrF2, then macro chrF2, macro BLEU,
  and lower mean latency.
- A zero-shot winner is not a final model. It advances only to a controlled,
  fixed-budget Exp1 pilot.
- Direction-specific Marian models may be reported separately as controls but
  are not eligible under the one-bidirectional-model-per-pair contract.
- NLLB is excluded from deployable candidates because of its non-commercial
  license. MADLAD-3B remains a Teacher/ceiling control, not a small Student.

## Before running on the server

Free old optimizer checkpoints after preserving exported `best_model` weights,
training reports, evaluations, gates, and lightweight run metadata. A complete
run of one candidate evaluates both directions and loads one model at a time.

```bash
cd /root/autodl-tmp/fourlang_translation
export FOURLANG_MODEL_ROOT=/root/autodl-tmp/models
PY=/root/autodl-tmp/venvs/small100_student/bin/python
```

Freeze the completed SMaLL-100 baseline report:

```bash
$PY scripts/pipeline_v3/summarize_pair_specialists.py
```

Inspect bake-off readiness without loading a model:

```bash
$PY scripts/pipeline_v3/pair_student_bakeoff.py status
```

Run the highest-priority pair one candidate at a time. Each completed candidate
is saved atomically, so interruption does not discard earlier candidates.
Repeating a command reuses an `ok` result with the same benchmark, model,
configuration, code, and dependency fingerprint; add `--force` only for an
intentional fresh measurement:

```bash
$PY scripts/pipeline_v3/pair_student_bakeoff.py evaluate \
  --pair zh_uz --candidate-id small100

$PY scripts/pipeline_v3/pair_student_bakeoff.py evaluate \
  --pair zh_uz --candidate-id m2m100_418m

$PY scripts/pipeline_v3/pair_student_bakeoff.py evaluate \
  --pair zh_uz --candidate-id m2m100_1_2b
```

Only after all candidates finish:

```bash
$PY scripts/pipeline_v3/pair_student_bakeoff.py rank --pair zh_uz
```

Or evaluate all configured candidates and rank in one foreground run:

```bash
$PY scripts/pipeline_v3/pair_student_bakeoff.py run --pair zh_uz
```

Results are written under:

```text
results/model_selection/pair_students/<pair>/scores.json
results/model_selection/pair_students/<pair>/ranking.json
```

Run pairs in diagnostic priority order: `zh_uz`, `uz_ru`, `zh_ru`, `en_zh`,
`en_uz`, then `en_ru`. Do not use these zero-shot rankings to overwrite the
current model registry. First define and run the separate fixed-budget pilot.
