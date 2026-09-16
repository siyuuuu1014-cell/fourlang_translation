# M2M100 ru→uz routed LoRA repair

This experiment keeps the Exp4 model unchanged. A small LoRA adapter is trained
only for `ru→uz`; runtime routing loads it only for that direction. The other 11
directions use the byte-identical Exp4 base, so the adapter cannot regress them.

The adapter is trained from the existing approved `exp3_v2` corpus filtered to
`ru→uz`. It is evaluated on the fixed 1,012-row FLORES direction. The combined
12-direction report reuses the exact Exp4 metrics for the 11 routes on which the
adapter is not loaded and records that provenance explicitly.

No candidate is promoted automatically.

## Run

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_ru_uz_lora_v1.sh preflight

nohup bash scripts/pipeline_v3/run_fourlang_m2m100_ru_uz_lora_v1.sh run-all \
  > fourlang_m2m100_ru_uz_lora_v1.log 2>&1 &

tail -n 80 -f fourlang_m2m100_ru_uz_lora_v1.log
```

## Inspect the decision

```bash
python -m json.tool \
  results/evaluation/fourlang_m2m100/ru_uz_lora_v1_comparison.json
```

`CANDIDATE_PASSED` means the routed system passed the configured hard gates. It
still does not replace or delete Exp4.

## One-off routed translation

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_ru_uz_lora_v1.sh translate \
  --source ru --target uz --text "Сегодня погода очень хорошая."
```

The JSON output includes `adapter_active`. It is true only for `ru→uz`.
