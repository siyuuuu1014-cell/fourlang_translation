# Balanced target-Uzbek LoRA experiment

This experiment trains one adapter on all existing `en→uz` and `zh→uz` rows.
No rows are discarded. Existing example weights are rescaled so both directions
have equal total training mass. The already-passed dedicated `ru→uz` adapter is
not changed or included.

Hard gates require `zh→uz` to improve by at least 0.30 chrF2 while `en→uz` may
regress by no more than 0.10. All other directions remain exact Exp4 routes.

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_target_uz_lora_v1.sh preflight

nohup bash scripts/pipeline_v3/run_fourlang_m2m100_target_uz_lora_v1.sh run-all \
  > fourlang_m2m100_target_uz_lora_v1.log 2>&1 &

tail -n 80 -f fourlang_m2m100_target_uz_lora_v1.log
```

```bash
python -m json.tool \
  results/evaluation/fourlang_m2m100/target_uz_lora_v1_comparison.json
```

No candidate is promoted automatically.
