# M2M100 zh→uz routed LoRA experiment

This is an isolated A/B experiment using the same LoRA capacity and training
settings as the successful `ru→uz` repair. It starts from Exp4, trains only on
existing `zh→uz` rows from `exp3_v2`, and activates the adapter only for
`zh→uz`. The other 11 routes remain byte-identical Exp4 inference routes.

No candidate is promoted automatically.

```bash
bash scripts/pipeline_v3/run_fourlang_m2m100_zh_uz_lora_v1.sh preflight

nohup bash scripts/pipeline_v3/run_fourlang_m2m100_zh_uz_lora_v1.sh run-all \
  > fourlang_m2m100_zh_uz_lora_v1.log 2>&1 &

tail -n 80 -f fourlang_m2m100_zh_uz_lora_v1.log
```

Inspect the hard-gate decision:

```bash
python -m json.tool \
  results/evaluation/fourlang_m2m100/zh_uz_lora_v1_comparison.json
```
