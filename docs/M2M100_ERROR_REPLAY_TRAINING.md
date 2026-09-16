# M2M100 Exp4 error-replay training

The approved experiment builds a versioned 324,000-row weighted dataset from the
frozen Exp1 + Exp2 + Exp3_v2 score shards. It never overwrites those source datasets.

Approved settings:

- source model: `m2m100_kd_from_human_v1/best_model/shared`;
- epochs: 1;
- learning rate: 2e-6;
- effective batch size: 32;
- maximum retained checkpoints: 1;
- minimum free disk before training: 16 GiB;
- fixed validation: Exp2 validation;
- final evaluation: all 12 FLORES devtest directions.

Run build and preflight separately before starting the background training job:

```bash
bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh build
bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh preflight
```

Then run the full train, evaluate, and comparison sequence:

```bash
nohup bash scripts/pipeline_v3/run_m2m100_error_replay_v1.sh run-all \
  > fourlang_m2m100_error_replay_v1.log 2>&1 &
```

The build is content-addressed and reusable; `run-all` will verify and reuse the
same formal dataset rather than overwrite it.
