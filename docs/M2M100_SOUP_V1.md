# M2M100 regression-gated weight soup

This experiment linearly interpolates the Exp4 error-replay model and Targeted-v2
at Exp4 weights 0.25, 0.50, and 0.75. It performs no gradient training, overwrites no
source model, deletes no candidate, and never promotes a candidate automatically.

Candidates are ranked by macro chrF2 minus direction-regression penalties and must
also pass hard limits for maximum regression, ru-to-uz recovery, zh-to-uz retention,
macro chrF2, and worst-direction chrF2.

```bash
nohup bash scripts/pipeline_v3/run_fourlang_m2m100_soup_v1.sh run-all \
  > fourlang_m2m100_soup_v1.log 2>&1 &
```
