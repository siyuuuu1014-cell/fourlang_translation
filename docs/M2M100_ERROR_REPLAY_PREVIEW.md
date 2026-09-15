# M2M100 Exp4 error-replay preview

This pipeline inventories and scores the union of Exp1, Exp2, and Exp3_v2. It is
preview-only: it cannot write a formal training dataset and has no training action.

The approved invariants are:

- retain every physical source record;
- normalize duplicate-group contribution without deleting duplicate records;
- give zh-to-uz a slight provisional priority while preserving all 12 directions;
- assign every record a positive proposed weight and visit all records in a full epoch.

The 30/50/15/5 difficulty bands and all numerical multipliers are proposals. Review
`preview.json` and `review_samples.jsonl` before authorizing a formal dataset build.

Server preview command:

```bash
nohup bash scripts/pipeline_v3/run_m2m100_error_replay_preview_v1.sh run-preview \
  > fourlang_m2m100_error_replay_preview_v1.log 2>&1 &
```

Progress can be viewed with:

```bash
tail -n 80 -f fourlang_m2m100_error_replay_preview_v1.log
```
