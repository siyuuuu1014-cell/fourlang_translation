# M2M100 zh→uz validation-only decoding calibration

This experiment does not train or modify model weights. It selects one of seven
decoding configurations using only the existing `exp3_v2` `zh→uz` validation
rows. After the setting is frozen, it evaluates that setting once on FLORES
devtest and applies the existing hard promotion gates.

```bash
bash scripts/pipeline_v3/run_m2m100_zh_uz_decode_calibration_v1.sh preflight

nohup bash scripts/pipeline_v3/run_m2m100_zh_uz_decode_calibration_v1.sh run-all \
  > fourlang_m2m100_zh_uz_decode_calibration_v1.log 2>&1 &

tail -n 80 -f fourlang_m2m100_zh_uz_decode_calibration_v1.log
```

Inspect the final decision:

```bash
python -m json.tool \
  results/evaluation/fourlang_m2m100/zh_uz_decode_calibration_v1_comparison.json
```

No candidate is promoted automatically, and the earlier LoRA and Exp4 artifacts
are never overwritten.
