# 未完成事项

1. 下列缺失项不得用自动重生成代替历史数据；先查备份与训练指纹。
2. 四语原路径恢复由共同依赖协调后执行；本导航只引用备份。
3. 历史代码的完整可重跑验证未完成，尤其缺失preview_manual_zh_uz_isolation.py的旧构建链。
4. 当前参数与历史参数可能不同，实际运行以train_report和指纹为准。
5. 没有进行语音ASR项目重构、路由迁移、权重搬迁或全仓库改名。

- 01_pair_models/en_ru / 03_data/en_ru__exp1__trainer_input.jsonl：`/root/autodl-tmp/fourlang_translation/data/specialists/en_ru/exp1/train.jsonl`
- 01_pair_models/en_ru / 03_data/en_ru__exp1__trainer_validation.jsonl：`/root/autodl-tmp/fourlang_translation/data/specialists/en_ru/exp1/validation.jsonl`
- 02_shared_fourlang/exp1 / 03_data/fourlang__exp1__train.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp1/train.jsonl`
- 02_shared_fourlang/exp1 / 03_data/fourlang__exp1__validation.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp1/validation.jsonl`
- 02_shared_fourlang/exp2 / 03_data/fourlang__exp2__train.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/train.jsonl`
- 02_shared_fourlang/exp2 / 03_data/fourlang__exp2__validation.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp2/validation.jsonl`
- 02_shared_fourlang/exp3 / 03_data/fourlang__exp3__train.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3/train.jsonl`
- 02_shared_fourlang/exp3 / 03_data/fourlang__exp3__validation.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3/validation.jsonl`
- 02_shared_fourlang/exp3 / 04_model：`/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3/best_model/shared`
- 02_shared_fourlang/exp3 / 05_train_report.json：`/root/autodl-tmp/fourlang_translation/results/student/fourlang/exp3/train_report.json`
- 02_shared_fourlang/exp3 / 06_evaluation.json：`/root/autodl-tmp/fourlang_translation/results/evaluation/fourlang/exp3/metrics.json`
- 02_shared_fourlang/exp3_v2 / 03_data/fourlang__exp3_v2__train.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/train.jsonl`
- 02_shared_fourlang/exp3_v2 / 03_data/fourlang__exp3_v2__validation.jsonl：`/root/autodl-tmp/fourlang_translation/data/multilingual/fourlang/exp3_v2/validation.jsonl`
