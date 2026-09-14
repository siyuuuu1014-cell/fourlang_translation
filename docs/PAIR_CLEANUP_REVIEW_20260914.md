# 六组专用模型删除前核验与最终建议

服务器：`/root/autodl-tmp/fourlang_translation`。核验结果保存在
`reports/cleanup_verification/20260914T054957Z/verification.json`。
本地原始副本：[verification.json](PAIR_CLEANUP_VERIFICATION_20260914.json)。

没有删除任何文件，清理入口继续禁用。本表取代此前按整个落选实验目录清理的粗略建议。

## 模型完整性与实际测试

六组模型均具有 config、自定义 SMaLL-100 tokenizer、词表和 SentencePiece 文件；已逐组在 V100 上用 FP16、5 beams 实际加载权重并分别执行两个方向。12 次解码均完成。下表的加载成功仅代表模型可运行。

| 模型 | 加载/双向解码 | 本次例句人工观察 |
|---|---|---|
| en_zh Exp2 | 成功 | 会议/九点信息保留 |
| en_uz 冻结 Exp2 | 成功 | 核心意思保留，乌语有空格问题 |
| en_ru 冻结 Exp2 | 成功 | 会议/九点信息保留 |
| zh_ru Exp2 | 成功 | 会议/九点信息保留 |
| zh_uz ep3 | 成功 | 明显错译，不能标作质量通过 |
| uz_ru Exp2 | 成功 | 明显错译，不能标作质量通过 |

中乌实测：输入“会议九点开始。”，输出 `Sayohat to'rtada boshladi.`；反向输入 `Uchrashuv soat to'qqizda boshlanadi.`，输出“收听始于八点钟.”。乌俄同一句输出 `Выбор начинается в полном порядке.`。详细双向原文和译文均在 JSON 中。

这只有每方向一条例句，不能估算总体准确率，也尚未定位是模型能力还是运行/解码一致性问题。因此保留中乌和乌俄旧对照，便于后续与训练评测入口对比。已通过的历史 FLORES 门槛不等于本次语义检查通过。

## 副本与归档核验

EN-RU `results/student/pair_specialists/en_ru/exp2/best_model/shared` 中每个文件已与 `models/final_pair_specialists/en_ru_v1` 同名文件逐个比较 SHA256，全部相同。此结论为源目录文件全部在冻结副本中一致，不要求冻结目录没有额外模型卡。

所有本轮实际存在候选目录的 JSON/JSONL/TOML/MD/TXT/LOG/CSV/Parquet，均找到相同 SHA256 的归档副本，所检查类型的归档缺口为 0。二进制 optimizer、RNG、权重不属于元数据备份范围；删除仍会丧失完整断点恢复能力。

字面引用扫描范围为 configs、scripts、inference、service。旧中乌 Exp2 被 `configs/specialists/weak_pair_ablation.toml` 明确引用。EN-RU 源路径和消融输出也会通过 pair/variant 名动态构造，不能因字面引用数为 0 判定无依赖：`pair_specialist_flow.py` 和 `weak_pair_ablation.py` 会重新访问这些路径。

## 建议第一阶段仅评估两个 checkpoint 目录

以下路径相对服务器项目根目录。大小是 `du -sb` 逻辑字节数，不等于精确可释放物理磁盘量。

| 精确目录 | 字节 | 约 GiB | 条件与影响 |
|---|---:|---:|---|
| `results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints` | 11998025937 | 11.174 | 2 epoch 已导出 best_model；删除后不能从这些 optimizer/RNG 断点原样续训 |
| `results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints` | 11998034712 | 11.174 | 当前 ep3 已能加载并双向解码；删除后同样失去精确续训和 checkpoint 级别比较 |

合计约 22.35 GiB。只有确定不再使用这些断点时才适合执行。两个实验的 best_model、训练报告、trainer_state 归档、数据和评估结果都应保留。

## 其余目录继续保留

- EN-RU 重复源模型约 1.245 GiB：内容一致，但训练评测入口仍可能访问源路径；路径迁移前保留。
- 中乌旧 Exp2 约 1.245 GiB：配置明确引用，且本次发现语义错误，保留作对照。
- 中乌其余五个落选消融各约 1.245 GiB：保留权重以便重评，不以一次 dev 胜负推定失去诊断价值。
- 所有 `data/experiments/weak_pair_ablation/zh_uz/*`：体积小且有复现用途，保留。
- 六个当前模型、旧 Marian 注册模型、Exp1 对照、原始数据、DeepSeek 数据及四语集合模型与共享代码全部保留。

## 执行状态

本轮仅完成只读核验和写入报告。清理仍处于暂停状态；上面两个 checkpoint 是缩小后的可审阅候选，不是已经删除的结果。后续若决定清理，应在执行前重新确认对应实验没有恢复训练，并保存逐文件删除记录。
