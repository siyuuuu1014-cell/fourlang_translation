# ZH-UZ DeepSeek API Teacher pilot

This isolated pilot tests one API-generated translation per source row. It does
not modify the existing NLLB Teacher output, relaxed-8k dataset, trained models,
or protected benchmarks.

The API key is read only from `DEEPSEEK_API_KEY`. Never put it in a TOML file,
shell history, source file, issue, or Git commit. Revoke any key that has been
posted in chat before running this pipeline.

## Pilot

```bash
cd /root/autodl-tmp/fourlang_translation
PY=/root/autodl-tmp/venvs/small100_student/bin/python
CFG=configs/directions/zh_uz_deepseek_teacher_v1.toml

read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY

"$PY" scripts/pipeline_v3/deepseek_teacher.py plan --config "$CFG"
"$PY" scripts/pipeline_v3/deepseek_teacher.py generate --config "$CFG"
"$PY" scripts/pipeline_v3/deepseek_teacher.py status --config "$CFG"
unset DEEPSEEK_API_KEY
```

The default pilot deterministically selects 200 rows per direction. Requests
contain ten same-direction rows and run with four workers. Completed responses
are appended to `generations.jsonl`, so the same command resumes after network
or process failure. Outputs are isolated under:

If a few batches repeatedly fail with an incomplete/chunked network response,
keep the same config and checkpoint but split only the pending work into single
requests:

```bash
"$PY" scripts/pipeline_v3/deepseek_teacher.py generate \
  --config "$CFG" \
  --batch-size 1 \
  --concurrency 1
```

These runtime-only overrides do not change the generation fingerprint. Existing
completed rows are reused.

```text
data/pipeline_v2/zh_uz_deepseek_teacher_v1/pilot/
```

Local hard checks reject empty/copied output, Arabic-number mismatches,
excessive repetition, and target-script violations. These checks do not prove
semantic correctness. Review the pilot before any full generation.

## Full generation

Only after the pilot passes semantic review:

```bash
"$PY" scripts/pipeline_v3/deepseek_teacher.py plan --config "$CFG" --full
"$PY" scripts/pipeline_v3/deepseek_teacher.py generate --config "$CFG" --full
```

`--full` is intentionally mandatory for the paid full run. It uses a separate
checkpoint directory and cannot overwrite the pilot.
